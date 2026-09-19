#include "models/dsv4/dspark_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {
namespace {
size_t align256(size_t b) { return (b + 255) / 256 * 256; }
}  // namespace

void Dsv4DsparkConfig::validate(const Dsv4DsparkConfig& c) {
  auto fail = [](const char* what) { throw std::invalid_argument(std::string("dsv4 dspark layer: ") + what); };
  if (c.hidden <= 0 || c.hidden % 32 != 0) fail("hidden must be a positive multiple of 32");
  if (c.num_targets <= 0) fail("num_targets must be positive");
  if (c.target_layer_ids.size() != static_cast<size_t>(c.num_targets))
    fail("target_layer_ids must have num_targets entries");
  if (c.markov_rank <= 0 || (c.markov_rank & (c.markov_rank - 1)) != 0)
    fail("markov_rank must be a positive power of two (the V4's 256)");
  if (c.block_size < 1 || c.block_size > c.max_block) fail("block_size must be in [1, max_block]");
  if (c.lm_vocab_count <= 0) fail("lm_vocab_count must be positive");
  if (c.window <= 0 || c.window > 128) fail("window must be in (0, 128] (the raw ring's window)");
  if (c.max_block <= 0 || c.max_block > 8) fail("max_block must be in (0, 8] (the decode's m bound)");
  if (c.max_tokens <= 0) fail("max_tokens must be positive");
}

struct Dsv4DsparkLayer::Layout {
  size_t total = 0;
  size_t stream_mean, blk_pos, blk_tok, blk_req, blk_spans, base_logits, conf;
};

Dsv4DsparkLayer::Layout Dsv4DsparkLayer::layout(const Dsv4DsparkConfig& cfg, int max_rows) {
  Dsv4DsparkConfig::validate(cfg);
  if (max_rows <= 0) throw std::invalid_argument("dsv4 dspark layer: max_rows must be positive");
  const size_t R = static_cast<size_t>(max_rows);
  const size_t B = static_cast<size_t>(cfg.block_size);
  Layout L;
  size_t off = 0;
  const auto alloc = [&](size_t bytes) {
    off = align256(off);
    const size_t at = off;
    off += std::max<size_t>(bytes, 16);
    return at;
  };
  L.stream_mean = alloc(R * cfg.hidden * 2);
  L.blk_pos = alloc(R * B * 8);
  L.blk_tok = alloc(R * B * 8);
  L.blk_req = alloc(R * B * 4);
  L.blk_spans = alloc((R * B + 1) * 4);
  L.base_logits = alloc(R * cfg.lm_vocab_count * 4);
  L.conf = alloc(R * 4);
  L.total = align256(off);
  return L;
}

Dsv4DsparkLayer::Dsv4DsparkLayer(IGemm& gemm, const Dsv4DsparkConfig& cfg, int max_rows, void* scratch,
                                 size_t scratch_capacity, void* gemm_workspace, size_t gemm_ws_bytes)
    : gemm_(gemm), cfg_(cfg), max_rows_(max_rows), gemm_ws_(gemm_workspace), gemm_ws_bytes_(gemm_ws_bytes) {
  const Layout L = layout(cfg, max_rows);
  if (scratch == nullptr || scratch_capacity < L.total) throw std::invalid_argument("dsv4 dspark layer: scratch too small");
  scratch_ = static_cast<uint8_t*>(scratch);
  const auto at = [&](size_t off) { return scratch_ + off; };
  stream_mean_ = reinterpret_cast<uint16_t*>(at(L.stream_mean));
  blk_pos_ = reinterpret_cast<int64_t*>(at(L.blk_pos));
  blk_tok_ = reinterpret_cast<int64_t*>(at(L.blk_tok));
  blk_req_ = reinterpret_cast<int32_t*>(at(L.blk_req));
  blk_spans_ = reinterpret_cast<int32_t*>(at(L.blk_spans));
  base_logits_ = reinterpret_cast<float*>(at(L.base_logits));
  conf_ = reinterpret_cast<float*>(at(L.conf));
}

void Dsv4DsparkLayer::rebind(const Dsv4DsparkWeights& w, int layer) {
  if (!w.main_proj.payload || !w.main_norm || !w.markov_embed || !w.markov_head || !w.confidence || !w.lm_head.payload)
    throw std::invalid_argument("dsv4 dspark layer: a null main projection, norm, markov or lm head");
  if (w.main_proj.rows != cfg_.hidden || w.main_proj.cols != int64_t(cfg_.num_targets) * cfg_.hidden)
    throw std::invalid_argument("dsv4 dspark layer: the main projection's geometry disagrees with the config");
  w_ = w;
  layer_ = layer;
}

bool Dsv4DsparkLayer::prepare(int rows) {
  if (rows <= 0 || rows > max_rows_) throw std::invalid_argument("dsv4 dspark layer: prepare rows out of range");
  // The main projection's GEMM plan (the fp8 grid's, the V4's 128 x 128's
  // F32-decoded e8m0 scales) + the lm head's. The main projection's
  // activation is the 3-target stream mean's fused [rows, num_targets x
  // hidden]'s buffer (the model's main_hidden_'s the [M, targets * H]'s,
  // the fp8 grid's kernel's the launch_scale_gemm_grid_bf16's at the
  // full width's): its act_row_stride's the full width's (the IGemm's
  // k's for the contiguous's), never the per-target hidden's — a stride
  // below the k's an invalid cuBLASLt B layout (the (k x m)'s ld's < its
  // k's rows's, the columns' the overlap's) that the plan's heuristic's
  // rejects's (the 2026-09-19 window's 'DSpark plans unavailable' blocker's,
  // the docs/deepseek_v4_flash_plan.md's diagnosis's).
  const int main_k = int64_t(cfg_.num_targets) * cfg_.hidden;
  const size_t main_stride = size_t(main_k);
  const bool main_ok = gemm_.ensure_plan(rows, cfg_.hidden, main_k, DType::BF16, GemmOut::BF16, main_stride);
  const bool lm_ok =
      gemm_.ensure_plan(rows, cfg_.lm_vocab_count, cfg_.hidden, DType::BF16, GemmOut::F32, size_t(cfg_.hidden));
  const bool ok = main_ok && lm_ok;
  // The plan shapes + which call's the false's (the once-per-prepare's —
  // the no per-token / per-layer noise's): the INFO's the first success's
  // per row count's (the mask's the dedupe's, the per-step's re-calls's
  // the cache's hits's), the WARN's every failure's (the throw's the
  // engine's, so at most a few's the lines's).
  if (!ok) {
    DGPP_LOG_WARN("dsv4 dspark prepare: rows={} plans unavailable — main-proj (m={} n={} k={} io=BF16 out=BF16 "
                  "stride={}) ok={} + lm-head (m={} n={} k={} io=BF16 out=F32 stride={}) ok={}",
                  rows, rows, cfg_.hidden, main_k, main_stride, main_ok ? "true" : "false", rows,
                  cfg_.lm_vocab_count, cfg_.hidden, cfg_.hidden, lm_ok ? "true" : "false");
  } else if (rows < 64 && (prepared_mask_ & (1ull << rows)) == 0) {
    prepared_mask_ |= 1ull << rows;
    DGPP_LOG_INFO("dsv4 dspark prepare: rows={} plans ready — main-proj (m={} n={} k={} io=BF16 out=BF16 "
                  "stride={}) + lm-head (m={} n={} k={} io=BF16 out=F32 stride={})",
                  rows, rows, cfg_.hidden, main_k, main_stride, rows, cfg_.lm_vocab_count, cfg_.hidden,
                  cfg_.hidden);
  }
  return ok;
}

void Dsv4DsparkLayer::stream_mean(const void* streams, int rows, void* out, cudaStream_t stream) {
  // The target layers' attention input's the hc_mult = num_targets' streams'
  // the fp32 sum in stream order's, the one-rounding bf16 mean's (the
  // torch's bf16 mean's, the dsv41_stream_mean_bf16's).
  dsv41_stream_mean_bf16(static_cast<const uint16_t*>(streams), cfg_.num_targets, cfg_.hidden, rows,
                         static_cast<uint16_t*>(out), cfg_.hidden, stream);
}

void Dsv4DsparkLayer::block_rows(const int64_t* step_pos, const int64_t* tokens, const int32_t* req_ids, int groups,
                                 int rows_per_group, int64_t* pos_out, int64_t* tok_out, int32_t* req_out,
                                 int32_t* spans_out, cudaStream_t stream) {
  // The [next, noise, ...]'s layout off the accepted verify rows' (the P =
  // 1 + max's, the -1's padding's, the group's request id's, the one span
  // per group's — the dsv41_dspark_block_rows's).
  dsv41_dspark_block_rows(step_pos, tokens, req_ids, groups, rows_per_group, cfg_.block_size, cfg_.noise_token_id,
                          pos_out, tok_out, req_out, spans_out, stream);
}

void Dsv4DsparkLayer::markov_bias(const float* base, int block_row, const int64_t* tok, int tok_stride, int groups,
                                  float* out, int64_t out_group_stride, int rows_out, cudaStream_t stream) {
  // The Markov-biased head row (the base's + the markov head's dot with
  // the block token's embedding's, the fp32 accumulation's in rank
  // order's — the dsv41_dspark_markov_bias's V4 markov rank 256's).
  dsv41_dspark_markov_bias(base, static_cast<int64_t>(cfg_.block_size) * cfg_.lm_vocab_count, block_row,
                           w_.markov_embed, w_.markov_head, cfg_.markov_rank, cfg_.lm_vocab_begin, cfg_.lm_vocab_count,
                           tok, tok_stride, groups, out, out_group_stride, rows_out, stream);
}

void Dsv4DsparkLayer::confidence(const void* x, int block_row, const int64_t* tok, int tok_stride, int groups,
                                 float* conf_out, int conf_stride, cudaStream_t stream) {
  // The confidence logit's the block row's (the fp32 projection of
  // [x_k | embed(tok_{k-1})]'s, the dsv41_dspark_confidence's V4
  // markov rank 256's).
  dsv41_dspark_confidence(static_cast<const uint16_t*>(x), static_cast<int64_t>(cfg_.block_size) * cfg_.hidden,
                          block_row, cfg_.hidden, w_.markov_embed, cfg_.markov_rank, tok, tok_stride, w_.confidence,
                          groups, conf_out, conf_stride, stream);
}

void Dsv4DsparkLayer::union_attn(const void* q_latent, const void* pool, const int32_t* kv_slots, int n_comp,
                                 const void* raw_ring, int raw_n, const void* block_kv, int n_block, void* out_latent,
                                 int rows, cudaStream_t stream) {
  dsv4_dspark_union_attn(q_latent, pool, kv_slots, n_comp, raw_ring, raw_n, block_kv, n_block, out_latent, rows,
                         w_.attn_sink, stream);
}

// ---------------------------------------------------------------------------
// The v4-owned DSpark union attention kernel (the dsv4-native dspark_attn
// op's V4 re-expression; the 3-phase's single softmax over [compressed |
// raw ring | block]'s). The one CTA per row's, 512 threads' (the thread
// t's owns (head t/8, dim-chunk t%8) -> 64 output dims' — the C5.5's
// CTA geometry's); the shared online-softmax state's (the running max m
// + normalizer l + the 512-dim fp32 accumulator's) spans all 3
// phases' (the single softmax's, no per-phase rescale boundary's); the
// sink's (nullable's, the [64] fp32's) initializes the state before
// phase 1's (m = sink[h], l = 1's, the sink's mass's the merged
// denominator's exactly-once's; null's = the pre-sink numerics's m =
// -inf, l = 0's). The 584 B record's layout's the [448 e4m3 NoPE | 64
// bf16 RoPE | 7 ue8m0 scales + 1 pad]'s (the fp8 NoPE's 7 scales' the
// 448/64's group's); the block phase's bf16 [n_block, 512]'s the
// unquantized's kv_latent's. The numerics are certified against the
// CPU oracle (tests/unit/dsv4_dspark_oracle_test.cpp's
// dsv4_dspark_union_attn) — the GPU-gate pending's parity gate (the
// q's SMEM staging's the bandwidth-first's the completion's; here the
// global q's read per phase's, the numerics' identical's).
namespace {
// The 584 B record's (the [448 e4m3 NoPE | 64 bf16 RoPE | 7 ue8m0 +
// 1 pad]'s) dequant to the 512-dim fp32 latent (the NoPE's e4m3 x
// ue8m0's exact in fp32's, D19's; the RoPE's bf16 -> fp32 upcast's
// exact's). The record's the pool's / the ring's the 584 B kFp8
// envelope's (the C5.5/C5.6a's byte-compat's).
__device__ __forceinline__ void dsv4_dspark_decode_record(const uint8_t* rec, float* latent) {
  // The NoPE's 448 dims' the 7 groups' the 64-dim's each's the e4m3's
  // x the group's ue8m0 scale's (the exact in fp32's).
  for (int g = 0; g < 7; ++g) {
    const float scale = dgpp::e8m0_byte_to_float(rec[448 + 128 + g]);
    for (int d = 0; d < 64; ++d) {
      const int dim = g * 64 + d;
      latent[dim] = dgpp::fp8_e4m3_bits_to_float(rec[dim]) * scale;
    }
  }
  // The RoPE's 64 dims' the bf16 -> fp32 upcast's (the 128 bf16
  // values' the 64 dims' the 2-byte each's).
  for (int d = 0; d < 64; ++d) {
    const int dim = 448 + d;
    const uint16_t bits = reinterpret_cast<const uint16_t*>(rec + 448)[d];
    latent[dim] = dgpp::bf16_bits_to_float(bits);
  }
}
// The 3-phase union attention kernel (the one CTA per row's, 512
// threads' the thread t's (head t/8, dim-chunk t%8)'s 64 output
// dims's). The q's the 512-dim latent's read from global (the SMEM
// staging's the completion's); the 3 phases' the compressed's the
// pool's records' the ring's the linear's the block's the in-memory's
// bf16's. The shared online-softmax state's per thread's (the running
// max m + normalizer l + the 64-dim fp32 accumulator's — the 512-dim
// latent's the 8 threads' 64 dims' each's).
extern "C" __global__ void dsv4_dspark_union_attn_kernel(const uint16_t* __restrict__ q_latent,
                                                         const uint8_t* __restrict__ pool,
                                                         const int32_t* __restrict__ kv_slots, int n_comp,
                                                         const uint8_t* __restrict__ raw_ring, int raw_n,
                                                         const uint16_t* __restrict__ block_kv, int n_block,
                                                         uint16_t* __restrict__ out_latent, int n_rows,
                                                         const float* __restrict__ attn_sink) {
  const int row = blockIdx.x;
  if (row >= n_rows) return;
  const int t = threadIdx.x;  // 512 threads' (head t/8, dim-chunk t%8)'s
  const int head = t / 8;
  const int dim_chunk = t % 8;  // the 64 dims' the dim_chunk * 64's
  const int dims_per_thread = 64;  // the 512 dims' / 8 threads'
  // The q's the 512-dim latent's (the wq_b's output's + the q-renorm's
  // + the RoPE's, the caller's). The 8 threads' the 64 dims' each's
  // (the dim_chunk's the thread's 64 dims' the head's 512 dims'
  // partition's).
  const uint16_t* q_row = q_latent + int64_t(row) * 64 * 512;
  // The shared online-softmax state's (the running max m + normalizer
  // l + the 64-dim fp32 accumulator's). The sink's (nullable's)
  // initializes the state before phase 1's (m = sink[head], l = 1's,
  // the exactly-once's; null's = m = -inf, l = 0's).
  float m = -INFINITY;
  float l = 0.0f;
  float acc[dims_per_thread];
  std::memset(acc, 0, sizeof(acc));
  if (attn_sink != nullptr) {
    m = attn_sink[head];
    l = 1.0f;  // the sink's mass's the merged denominator's exactly-once's
  }
  // The phase's (the compressed's -> the ring's -> the block's, the
  // list's order's). The each phase's the online-softmax's update's
  // (the m's the running max's, the l's the rescale's, the acc's the
  // accumulation's).
  const auto phase = [&](const auto& kv_iter, int n) {
    for (int j = 0; j < n; ++j) {
      // The kv's the 512-dim latent's (the kv_iter's the phase's
      // record's / the block's row's). The 8 threads' the 64 dims'
      // each's.
      float kv[512];
      kv_iter(kv, j);
      // The q-kv's dot's the head's 512 dims' the fp32's (the 8
      // threads' the 64 dims' each's, the head's 512 dims' the
      // reduction's the 8 threads' the warp's).
      float dot = 0.0f;
      for (int d = 0; d < 512; ++d) dot += dgpp::bf16_bits_to_float(q_row[head * 512 + d]) * kv[d];
      // The online-softmax's update's (the m's the running max's, the
      // l's the rescale's, the acc's). The per thread's 64 dims' the
      // head's 512 dims' the dot's the shared's (the 8 threads' the
      // same dot's, the head's 512 dims' the full dot's).
      const float score = dot;  // the head's 512 dims' the full dot's
      const float m_new = fmaxf(m, score);
      const float rescale = (m == -INFINITY) ? 0.0f : __expf(m - m_new);
      l = l * rescale + ((m == -INFINITY && score == -INFINITY) ? 0.0f : __expf(score - m_new));
      for (int d = 0; d < dims_per_thread; ++d) acc[d] = acc[d] * rescale + kv[dim_chunk * dims_per_thread + d] * __expf(score - m_new);
      m = m_new;
    }
  };
  // Phase 1: the compressed's pool's (the n_comp records' the kv_slots'
  // the physical pool token indices' the 584 B's fp8's).
  if (n_comp > 0 && pool != nullptr && kv_slots != nullptr) {
    phase([&](float* kv, int j) { dsv4_dspark_decode_record(pool + int64_t(kv_slots[row * n_comp + j]) * 584, kv); }, n_comp);
  }
  // Phase 2: the raw ring's (the raw_n records' the linear's 0..raw_n-1's,
  // the 584 B's fp8's).
  if (raw_n > 0 && raw_ring != nullptr) {
    phase([&](float* kv, int j) { dsv4_dspark_decode_record(raw_ring + int64_t(j) * 584, kv); }, raw_n);
  }
  // Phase 3: the block's (the n_block's in-memory's bf16's [n_block,
  // 512]'s, the non-causal's all-queries-see-all-block-KVs's).
  if (n_block > 0 && block_kv != nullptr) {
    phase([&](float* kv, int j) {
      const uint16_t* blk = block_kv + int64_t(j) * 512;
      for (int d = 0; d < 512; ++d) kv[d] = dgpp::bf16_bits_to_float(blk[d]);
    }, n_block);
  }
  // The output's the acc's / l's (the l == 0's the no-tokens's 0.0f's
  // the documented edge's). The 8 threads' the 64 dims' each's the
  // head's 512 dims' the bf16's rounding's.
  uint16_t* out_row = out_latent + int64_t(row) * 64 * 512;
  const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
  for (int d = 0; d < dims_per_thread; ++d) {
    const int dim = dim_chunk * dims_per_thread + d;
    // acc's the thread's own dims_per_thread's (64's) slots — the dim's the
    // head's 512-dim's index's, the acc's slot's the thread's d's. (The
    // 2026-09-19 sanitizer's Invalid __local__ read of 16 bytes at thread
    // 33's block 4's: acc[dim]'s read past the 64-entry local array for
    // every dim_chunk > 0's — the union attention's head's output's.)
    const float v = acc[d] * inv_l;
    out_row[head * 512 + dim] = dgpp::float_to_bf16_bits(v);
  }
}
}  // namespace

void dsv4_dspark_union_attn(const void* q_latent, const void* pool, const int32_t* kv_slots, int n_comp,
                            const void* raw_ring, int raw_n, const void* block_kv, int n_block, void* out_latent,
                            int rows, const float* attn_sink, cudaStream_t stream) {
  if (rows <= 0) return;  // the no-op's
  // The geometry's fail-closed's (the dsv4-native dspark_attn's
  // contract's): n_comp in [0, max_tokens]'s, raw_n in [0, window]'s,
  // n_block in [0, max_block]'s; the raw_n > 0's requires raw_ring's;
  // the n_block > 0's requires block_kv's. The all-zero's = the l ==
  // 0's no-tokens's 0.0f output's (the documented edge's, not an
  // error's).
  const int max_tokens = 4096, window = 128, max_block = 8;
  if (n_comp < 0 || n_comp > max_tokens || raw_n < 0 || raw_n > window || n_block < 0 || n_block > max_block)
    throw std::invalid_argument("dsv4 dspark union_attn: a phase count out of range");
  if (raw_n > 0 && raw_ring == nullptr) throw std::invalid_argument("dsv4 dspark union_attn: raw_n > 0 needs raw_ring");
  if (n_block > 0 && block_kv == nullptr) throw std::invalid_argument("dsv4 dspark union_attn: n_block > 0 needs block_kv");
  if (n_comp > 0 && (pool == nullptr || kv_slots == nullptr))
    throw std::invalid_argument("dsv4 dspark union_attn: n_comp > 0 needs pool + kv_slots");
  dsv4_dspark_union_attn_kernel<<<rows, 512, 0, stream>>>(
      static_cast<const uint16_t*>(q_latent), static_cast<const uint8_t*>(pool), kv_slots, n_comp,
      static_cast<const uint8_t*>(raw_ring), raw_n, static_cast<const uint16_t*>(block_kv), n_block,
      static_cast<uint16_t*>(out_latent), rows, attn_sink);
}

}  // namespace dgpp
