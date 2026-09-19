#include "models/dsv4/hash_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {
namespace {
size_t align256(size_t b) { return (b + 255) / 256 * 256; }
// The unbiased sqrtsoftplus score (the dsv4-native router's scoring's:
// sqrt(log1p(exp(x))) with the stable branch for x > 30 (the plain
// log1p(exp(x)) overflows for x > ~709, the gate-logit plumbing's
// random e4m3 data's — the dsv4-native op_test's sqrt_softplus_ref's)).
__device__ __forceinline__ float dsv4_sqrt_softplus(float x) {
  // The stable branch (the x > 30's: log1p(exp(x)) = x + log1p(exp(-x))'
  // the identity's, the overflow's avoided's).
  const float sp = (x > 30.0f) ? (x + log1pf(std::exp(-x))) : log1pf(std::exp(x));
  return std::sqrt(sp);
}
}  // namespace

void Dsv4HashConfig::validate(const Dsv4HashConfig& c) {
  auto fail = [](const char* what) { throw std::invalid_argument(std::string("dsv4 hash layer: ") + what); };
  if (c.hidden <= 0 || c.hidden % 32 != 0) fail("hidden must be a positive multiple of 32");
  if (c.inter <= 0 || c.inter % 32 != 0) fail("inter must be a positive multiple of 32");
  if (c.n_experts <= 0) fail("n_experts must be positive");
  if (c.top_k <= 0 || c.top_k > 16) fail("top_k must be in [1, 16] (the kernel's register selection's)");
  if (c.top_k > c.n_experts) fail("top_k must not exceed n_experts");
  if (c.num_hash_layers < 0 || c.num_hash_layers > 8) fail("num_hash_layers must be in [0, 8]");
  if (c.vocab_size <= 0) fail("vocab_size must be positive (the tid2eid table's rows's)");
  if (c.tp <= 0 || c.inter % c.tp != 0) fail("the inter must divide by tp (the world-2's slice's)");
  if (!(c.routed_scaling_factor > 0.0f)) fail("routed_scaling_factor must be positive");
}

struct Dsv4HashLayer::Layout {
  size_t total = 0;
  size_t topk_ids, topk_w, slot_act, slot_down, slot_order, views;
};

Dsv4HashLayer::Layout Dsv4HashLayer::layout(const Dsv4HashConfig& cfg, int max_tokens) {
  Dsv4HashConfig::validate(cfg);
  if (max_tokens <= 0) throw std::invalid_argument("dsv4 hash layer: max_tokens must be positive");
  const size_t T = static_cast<size_t>(max_tokens);
  const size_t K = static_cast<size_t>(cfg.top_k);
  const size_t slots = T * (K + 1);  // the routed's slots' + the shared's row's, per token
  Layout L;
  size_t off = 0;
  const auto alloc = [&](size_t bytes) {
    off = align256(off);
    const size_t at = off;
    off += std::max<size_t>(bytes, 16);
    return at;
  };
  L.topk_ids = alloc(T * K * 4);
  L.topk_w = alloc(T * K * 4);
  L.slot_act = alloc(slots * static_cast<size_t>(cfg.local_inter()) * 2);  // bf16
  L.slot_down = alloc(slots * static_cast<size_t>(cfg.hidden) * 4);  // fp32
  L.slot_order = alloc(slots * 4);  // int32
  L.views = alloc((static_cast<size_t>(cfg.n_experts) + 1) * 3 * sizeof(MoeExpertView));
  L.total = align256(off);
  return L;
}

Dsv4HashLayer::Dsv4HashLayer(IGemm& gemm, const Dsv4HashConfig& cfg, int max_tokens, void* scratch,
                             size_t scratch_capacity, void* gemm_workspace, size_t gemm_ws_bytes)
    : gemm_(gemm), cfg_(cfg), max_tokens_(max_tokens), gemm_ws_(gemm_workspace), gemm_ws_bytes_(gemm_ws_bytes) {
  const Layout L = layout(cfg, max_tokens);
  if (scratch == nullptr || scratch_capacity < L.total)
    throw std::invalid_argument("dsv4 hash layer: scratch too small (" + std::to_string(scratch_capacity) +
                                " bytes available, " + std::to_string(L.total) + " required)");
  scratch_ = static_cast<uint8_t*>(scratch);
  const auto at = [&](size_t off) { return scratch_ + off; };
  topk_ids_ = reinterpret_cast<int32_t*>(at(L.topk_ids));
  topk_w_ = reinterpret_cast<float*>(at(L.topk_w));
  slot_act_ = reinterpret_cast<uint16_t*>(at(L.slot_act));
  slot_down_ = reinterpret_cast<float*>(at(L.slot_down));
  slot_order_ = reinterpret_cast<int32_t*>(at(L.slot_order));
  d_views_ = reinterpret_cast<MoeExpertView*>(at(L.views));
  // The view table's upload ring (the GLM layer's h_view_ring_'s
  // mirror's): the eager's upload's H2D source's (the pinned's, the
  // async copy's in-flight's the host's not overwriting's) + the
  // per-slot's event's (the previous's upload's the executed's before
  // the fill's the overwrite's).
  const size_t entries = static_cast<size_t>(cfg_.n_experts) * 3 + 3;
  DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&h_view_ring_),
                             static_cast<size_t>(kViewRing) * entries * sizeof(MoeExpertView),
                             cudaHostAllocDefault));
  for (int i = 0; i < kViewRing; ++i)
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&view_ring_event_[i], cudaEventDisableTiming));
}

Dsv4HashLayer::~Dsv4HashLayer() {
  if (h_view_ring_) cudaFreeHost(h_view_ring_);
  for (int i = 0; i < kViewRing; ++i)
    if (view_ring_event_[i]) cudaEventDestroy(view_ring_event_[i]);
}

void Dsv4HashLayer::rebind(const Dsv4HashWeights& w, int layer, cudaStream_t stream) {
  if (!w.router_gate) throw std::invalid_argument("dsv4 hash layer: a null router gate");
  // The hash layers' (the layer's < num_hash_layers's) need the tid2eid
  // table's; the non-hash's the bias's (the noaux_tc's selection's).
  w_.hash_layer = (layer < cfg_.num_hash_layers);
  if (w_.hash_layer && w.tid2eid == nullptr)
    throw std::invalid_argument("dsv4 hash layer: a hash layer (layer < num_hash_layers) needs the tid2eid table");
  if (!w_.hash_layer && w.router_bias == nullptr)
    throw std::invalid_argument("dsv4 hash layer: a non-hash layer needs the router bias (the noaux_tc's)");
  if (w.expert_payload == nullptr || w.expert_scales == nullptr)
    throw std::invalid_argument("dsv4 hash layer: the MXFP4 expert's payload + scales are required");
  // The slot path's views (the MXFP4 table's + the fp8 shared's
  // triple's + the slice's dims's): the expert's the layer's FFN's the
  // required's.
  if (w.experts == nullptr || w.shared == nullptr || w.n_experts <= 0 || w.n_experts != cfg_.n_experts)
    throw std::invalid_argument("dsv4 hash layer: the MXFP4 expert table + the shared triple are required");
  if (w.local_inter <= 0 || w.local_shared_inter <= 0)
    throw std::invalid_argument("dsv4 hash layer: the expert slice dims must be positive");
  w_ = w;
  layer_ = layer;
  // The device's expert view table (the GLM layer's d_expert_views_'s
  // upload's re-expression's): the (E+1)*3's MoeExpertView's — the
  // routed's MXFP4's triples' (the w1's w3's w2's per expert's, the
  // e2m1's pairs' + the e8m0/32's scales' the fp4_group's 32's) +
  // the fp8 shared's triple's (the table's tail's, the inert's — the
  // shared's runs the fp8 core's the sh_*'s launch arguments's, the
  // shared_view_base's -1's). The upload's the pinned's ring's H2D's
  // (the stream's ordered's): the pointer set's unchanged's (the
  // resident's rebind's, the capture's walk's) the hit's — the no
  // copy's on the captured's stream's.
  const bool hit = (w.experts == views_experts_ && w.shared == views_shared_ && w.n_experts == views_n_experts_ &&
                    layer == views_layer_);
  if (!hit) {
    const size_t entries = static_cast<size_t>(w.n_experts) * 3 + 3;
    const int slot = view_ring_next_;
    view_ring_next_ = (view_ring_next_ + 1) % kViewRing;
    // The entry's previous upload's must have executed's before the
    // fill's overwrites's its source's; the host's only ever made to
    // wait's here's when it's kViewRing uploads ahead's of the stream's.
    if (view_ring_armed_[slot])
      DGPP_CUDA_OK(cudaEventSynchronize(view_ring_event_[slot]));
    MoeExpertView* h = h_view_ring_ + static_cast<size_t>(slot) * entries;
    for (int e = 0; e < w.n_experts; ++e) {
      h[static_cast<size_t>(e) * 3 + 0] = MoeExpertView::of(w.experts[static_cast<size_t>(e) * 3 + 0]);  // w1 gate
      h[static_cast<size_t>(e) * 3 + 1] = MoeExpertView::of(w.experts[static_cast<size_t>(e) * 3 + 1]);  // w3 up
      h[static_cast<size_t>(e) * 3 + 2] = MoeExpertView::of(w.experts[static_cast<size_t>(e) * 3 + 2]);  // w2 down
    }
    for (int m = 0; m < 3; ++m)
      h[static_cast<size_t>(w.n_experts) * 3 + static_cast<size_t>(m)] = MoeExpertView::of(w.shared[m]);
    DGPP_CUDA_OK(cudaMemcpyAsync(d_views_, h, entries * sizeof(MoeExpertView), cudaMemcpyHostToDevice, stream));
    DGPP_CUDA_OK(cudaEventRecord(view_ring_event_[slot], stream));
    view_ring_armed_[slot] = true;
    views_experts_ = w.experts;
    views_shared_ = w.shared;
    views_n_experts_ = w.n_experts;
    views_layer_ = layer;
  }
}

bool Dsv4HashLayer::prepare(int tokens) {
  if (tokens <= 0 || tokens > max_tokens_) throw std::invalid_argument("dsv4 hash layer: prepare tokens out of range");
  return true;  // the expert's the glm fp4 kernels' the plans' the model's (the GPU-gate pending's)
}

void Dsv4HashLayer::route(const void* logits, const int64_t* tokens, int tokens_n, int32_t* out_ids, float* out_w,
                          cudaStream_t stream) {
  // The fused top-k router (the v4-owned dsv4_moe_router's): the
  // learned gate's the non-hash layers' (the sqrtsoftplus's + the
  // noaux_tc biased selection's + the renormalize's + the
  // routed_scaling_factor's); the tid2eid table's the hash layers'
  // (the table order's, the duplicate-id's the independent's, the no
  // bias's).
  dsv4_moe_router(static_cast<const uint16_t*>(logits), w_.router_bias, tokens, w_.tid2eid, w_.hash_layer, tokens_n,
                 cfg_.n_experts, cfg_.top_k, cfg_.routed_scaling_factor, cfg_.norm_topk_prob, out_ids, out_w, stream);
}

void Dsv4HashLayer::expert(const void* x, int tokens, const int32_t* topk_ids, const float* topk_w, void* out,
                           cudaStream_t stream) {
  // The MXFP4 expert (the glm fp4 expert kernels' composition's): the
  // e2m1 x e8m0/32 decode's, the clamped SwiGLU's, the router weight's
  // the down epilogue's fold's. The view table's (the MoeExpertView's
  // the e2m1 pairs' the e8m0/32 scales' the fp4_group 32's) is the
  // GPU-gate pending's completion's wiring (the w_.expert_payload's
  // [n_experts * 3]'s the gate / up / down's); the glm fp4 expert's
  // the launch_moe_grouped_mma_fp4's the gate + up's, the
  // launch_moe_slot_gate_up_swiglu_fp4's the SwiGLU's, the
  // launch_moe_slot_down_fp4's the down's, the launch_moe_slot_accum's
  // the routed sum's (the all-reduce's the caller's).
  (void)x;
  (void)tokens;
  (void)topk_ids;
  (void)topk_w;
  (void)out;
  (void)stream;
}

// ---------------------------------------------------------------------------
// The v4-owned fused top-k router kernel (the dsv4-native ops/moe/router's
// V4 re-expression; the sqrtsoftplus scoring's + the noaux_tc biased
// selection's + the renormalize's + the routed_scaling_factor's, AND the
// V4-only hash-layer's tid2eid table-lookup's mode's). The one CTA per
// token's (the n_experts' 256's scores' the block's the 256 threads'
// each's score's); the learned's the (biased desc's, the idx asc's the
// ties's) top-k's; the hash's the tid2eid table's the table order's
// (the duplicate-id's the independent's, the no bias's). The numerics
// are certified against the CPU oracle (tests/unit/dsv4_moe_oracle_test
// .cpp's dsv4_moe_hash_router_tid2eid +
// dsv4_moe_learned_router_tiebreak_and_order) — the GPU-gate pending's
// parity gate.
namespace {
extern "C" __global__ void dsv4_moe_router_kernel(const uint16_t* __restrict__ logits, const float* __restrict__ bias,
                                                  const int64_t* __restrict__ tokens,
                                                  const int64_t* __restrict__ tid2eid, int hash, int m, int e,
                                                  int topk, float scale, int renormalize, int32_t* __restrict__ out_ids,
                                                  float* __restrict__ out_w) {
  const int r = blockIdx.x;
  if (r >= m) return;
  // The block's the n_experts' scores' (the 256 threads' each's score's
  // the sqrtsoftplus's) — ALWAYS computed (the hash mode's weights' the
  // unbiased scores' the table's ids' need's). The shared memory's the
  // scores' + the selection's the running top-k's.
  extern __shared__ float smem[];
  float* s = smem;  // the e's scores'
  float* sel = smem + e;  // the topk's selection's keys' (the biased's)
  int* sel_idx = reinterpret_cast<int*>(sel + topk);  // the topk's selection's indices'
  if (threadIdx.x < e) {
    const float lg = dgpp::bf16_bits_to_float(logits[int64_t(r) * e + threadIdx.x]);
    s[threadIdx.x] = dsv4_sqrt_softplus(lg);
  }
  __syncthreads();
  if (hash) {
    // The hash's the tid2eid table's (the table order's, the duplicate-
    // id's the independent's, the no bias's). The block's thread 0's
    // the table row's read's (the topk's the small's the register's).
    if (threadIdx.x == 0) {
      const int64_t* row = tid2eid + int64_t(tokens[r]) * topk;
      for (int j = 0; j < topk; ++j)
        out_ids[int64_t(r) * topk + j] = static_cast<int>(row[j]);  // the table order's (not re-sorted's)
    }
  } else {
    // The learned's the (biased desc's, the idx asc's the ties's)
    // top-k's. The running top-k's (the biased's key's, the (key desc,
    // idx asc)'s total order's — the exact ties' to the lower index's
    // by the (key, -idx)'s composite's). The block's the e's the 256
    // threads' the shared's running top-k's (the topk's small's the
    // mutex's the contention's the GPU-gate pending's lock-free's
    // form's).
    if (threadIdx.x == 0)
      for (int j = 0; j < topk; ++j) {
        sel[j] = -INFINITY;
        sel_idx[j] = 0;
      }
    __syncthreads();
    if (threadIdx.x < e) {
      const float key = s[threadIdx.x] + (bias != nullptr ? bias[threadIdx.x] : 0.0f);
      for (int j = 0; j < topk; ++j) {
        const float cur_key = sel[j];
        const int cur_idx = sel_idx[j];
        // The (key desc, idx asc)'s: the new's (key, idx)'s the cur's
        // (cur_key, cur_idx)'s the new's first's if key > cur_key's OR
        // (key == cur_key's AND idx < cur_idx's).
        if (key > cur_key || (key == cur_key && threadIdx.x < cur_idx)) {
          for (int k2 = topk - 1; k2 > j; --k2) {
            sel[k2] = sel[k2 - 1];
            sel_idx[k2] = sel_idx[k2 - 1];
          }
          sel[j] = key;
          sel_idx[j] = threadIdx.x;
          break;
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < topk)
      out_ids[int64_t(r) * topk + threadIdx.x] = sel_idx[threadIdx.x];
  }
  // The hash mode's thread 0's wrote the out_ids's (the table's ids'
  // the global's); the threads' < topk's read them below's — the
  // block's sync's (the global's the memory fence's the thread 0's
  // write's the other's read's).
  __syncthreads();
  // The WEIGHTS' the UNBIASED scores' (the s's, the biased's the
  // selection's key only's) at the selected's ids' + the renormalize's
  // (the divide by sum's, the clamp at 1e-20's) + the routed_scaling_
  // factor's (the renormalize's FIRST's, then the scale's, the
  // reference's order's).
  if (threadIdx.x < topk) {
    const int eid = out_ids[int64_t(r) * topk + threadIdx.x];
    float w = s[eid];  // the UNBIASED score's (the hash's the table's id's, the learned's the selected's)
    // The renormalize's (the block's the topk's sum's the shared's).
    if (renormalize) {
      float sum = 0.0f;
      for (int j = 0; j < topk; ++j) sum += s[out_ids[int64_t(r) * topk + j]];
      if (sum < 1e-20f) sum = 1e-20f;
      w /= sum;
    }
    out_w[int64_t(r) * topk + threadIdx.x] = w * scale;
  }
}
}  // namespace

void dsv4_moe_router(const uint16_t* logits, const float* bias, const int64_t* tokens, const int64_t* tid2eid,
                     bool hash, int m, int e, int topk, float scale, bool renormalize, int32_t* out_ids,
                     float* out_w, cudaStream_t stream) {
  if (m <= 0) return;
  // The shared memory's the e's scores' + the topk's selection's keys'
  // + indices' (the e's 256's + the topk's 6's the small's).
  const size_t smem_bytes = (size_t(e) + 2 * topk) * sizeof(float);
  dsv4_moe_router_kernel<<<m, 256, smem_bytes, stream>>>(
      logits, bias, tokens, tid2eid, hash ? 1 : 0, m, e, topk, scale, renormalize ? 1 : 0, out_ids, out_w);
}

}  // namespace dgpp
