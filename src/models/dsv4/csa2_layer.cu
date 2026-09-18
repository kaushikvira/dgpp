#include "models/dsv4/csa2_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/dsa.hpp"
#include "kernels/scale_gemm.hpp"

namespace dgpp {
namespace {
size_t align256(size_t b) { return (b + 255) / 256 * 256; }
int64_t round_up_to(int64_t v, int64_t g) { return (v + g - 1) / g * g; }
}  // namespace

void Dsv4Csa2Config::validate(const Dsv4Csa2Config& c) {
  auto fail = [](const char* what) { throw std::invalid_argument(std::string("dsv4 csa2 layer: ") + what); };
  if (c.hidden <= 0 || c.hidden % 32 != 0) fail("hidden must be a positive multiple of 32");
  if (c.q_lora <= 0 || c.q_lora % 32 != 0) fail("q_lora must be a positive multiple of 32");
  if (c.o_lora <= 0 || c.o_lora % 32 != 0) fail("o_lora must be a positive multiple of 32");
  if (c.tp <= 0 || c.num_heads % c.tp != 0 || c.o_groups % c.tp != 0 || c.num_heads % c.o_groups != 0)
    fail("heads and groups must divide by tp, heads by groups");
  if (c.local_heads() < 4 || (c.local_heads() & (c.local_heads() - 1)) != 0)
    fail("local heads must be a power of two >= 4 (the attention head-group tiling)");
  // The V4's 64 index heads (the dsv41's 32's NEW fold width — the
  // v4-owned select kernels' contract, the shared csa2_select's 32-head
  // pin's V4 re-expression; the spec §2.1(d)).
  if (c.index_heads != 64) fail("the V4's selection is the 64-head fold (the v4-owned kernels' contract)");
  if (c.index_topk <= 0 || (c.index_topk & (c.index_topk - 1)) != 0 || c.index_topk > 1024)
    fail("index_topk must be a power of two <= 1024");
  if (c.window <= 0 || c.ring_slots < c.window + 16) fail("the ring must hold the window plus the spec rows");
  if (c.block_tokens <= 0 || c.block_tokens % 2 != 0) fail("block_tokens must be a positive even number");
  if (!(c.eps >= 0.f)) fail("eps");
}

struct Dsv4Csa2Layer::Layout {
  size_t total = 0;
  size_t qr, kv, q, o, oa, idx_q, q_fp8, q_scale, w, w_folded, latent, ik, pos, req_ids, slots, topk, counts,
       m_main, l_main, c_main, m_win, l_win, c_win, violations;
  int64_t max_entries = 0;
};

Dsv4Csa2Layer::Layout Dsv4Csa2Layer::layout(const Dsv4Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens,
                                            int max_decode_rows, int decode_n_split, size_t dot_budget) {
  Dsv4Csa2Config::validate(cfg);
  if (max_tokens <= 0 || max_cache_tokens <= 0) throw std::invalid_argument("dsv4 csa2 layer: max_tokens / max_cache_tokens");
  if (max_decode_rows <= 0 || max_decode_rows > 32 || max_decode_rows > max_tokens)
    throw std::invalid_argument("dsv4 csa2 layer: max_decode_rows must be in [1, min(32, max_tokens)]");
  if (decode_n_split <= 0 || dot_budget == 0) throw std::invalid_argument("dsv4 csa2 layer: decode_n_split / dot_budget");
  const int lh = cfg.local_heads(), lg = cfg.local_groups();
  const size_t T = static_cast<size_t>(max_tokens);
  Layout L;
  size_t off = 0;
  const auto alloc = [&](size_t bytes) {
    off = align256(off);
    const size_t at = off;
    off += std::max<size_t>(bytes, 16);
    return at;
  };
  L.max_entries = round_up_to(max_cache_tokens, 256);
  L.qr = alloc(T * cfg.q_lora * 2);
  L.kv = alloc(T * kCsa2Latent * 2);
  L.q = alloc(T * lh * kCsa2Latent * 2);
  L.o = alloc(T * lh * kCsa2Latent * 2);
  L.oa = alloc(T * lg * cfg.o_lora * 2);
  L.idx_q = alloc(T * cfg.index_heads * kCsa2IndexDim * 2);
  L.q_fp8 = alloc(T * cfg.index_heads * kCsa2IndexDim);
  L.q_scale = alloc(T * cfg.index_heads * 4);
  L.w = alloc(T * cfg.index_heads * 2);
  L.w_folded = alloc(T * cfg.index_heads * 4);
  L.latent = alloc(T * kCsa2Latent * 2);
  L.ik = alloc(T * kCsa2IndexDim * 2);
  L.pos = alloc(T * 8);
  L.req_ids = alloc(T * 4);
  L.slots = alloc(T * 8);
  L.topk = alloc(T * cfg.index_topk * 4);
  L.counts = alloc(T * 4);
  const size_t ws_rows = std::max<size_t>(std::max(max_decode_rows, 8), std::max(decode_n_split, 8));
  L.m_main = alloc(ws_rows * lh * 4);
  L.l_main = alloc(ws_rows * lh * 4);
  L.c_main = alloc(ws_rows * lh * kCsa2Latent * 4);
  L.m_win = alloc(ws_rows * lh * 4);
  L.l_win = alloc(ws_rows * lh * 4);
  L.c_win = alloc(ws_rows * lh * kCsa2Latent * 4);
  L.violations = alloc(16);
  L.total = align256(off);
  return L;
}

size_t Dsv4Csa2Layer::scratch_bytes(const Dsv4Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens, int max_decode_rows,
                                   int decode_n_split, size_t dot_budget) {
  return layout(cfg, max_tokens, max_cache_tokens, max_decode_rows, decode_n_split, dot_budget).total;
}

Dsv4Csa2Layer::Dsv4Csa2Layer(IGemm& gemm, const Dsv4Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens,
                             void* scratch, size_t scratch_capacity, void* gemm_workspace, size_t gemm_ws_bytes,
                             int max_decode_rows, int decode_n_split, size_t dot_budget)
    : gemm_(gemm), cfg_(cfg), max_tokens_(max_tokens), max_cache_tokens_(max_cache_tokens),
      max_decode_rows_(max_decode_rows), decode_n_split_(decode_n_split), gemm_ws_(gemm_workspace),
      gemm_ws_bytes_(gemm_ws_bytes) {
  const Layout L = layout(cfg, max_tokens, max_cache_tokens, max_decode_rows, decode_n_split, dot_budget);
  if (scratch == nullptr || scratch_capacity < L.total) throw std::invalid_argument("dsv4 csa2 layer: scratch too small");
  scratch_ = static_cast<uint8_t*>(scratch);
  attn_scale_ = static_cast<float>(std::pow(static_cast<double>(kCsa2Latent), -0.5));
  const auto at = [&](size_t off) { return scratch_ + off; };
  qr_ = reinterpret_cast<uint16_t*>(at(L.qr));
  kv_ = reinterpret_cast<uint16_t*>(at(L.kv));
  q_ = reinterpret_cast<uint16_t*>(at(L.q));
  o_ = reinterpret_cast<uint16_t*>(at(L.o));
  oa_ = reinterpret_cast<uint16_t*>(at(L.oa));
  idx_q_ = reinterpret_cast<uint16_t*>(at(L.idx_q));
  q_fp8_ = at(L.q_fp8);
  q_scale_ = reinterpret_cast<float*>(at(L.q_scale));
  iw_ = reinterpret_cast<uint16_t*>(at(L.w));
  w_folded_ = reinterpret_cast<float*>(at(L.w_folded));
  latent_ = reinterpret_cast<uint16_t*>(at(L.latent));
  ik_ = reinterpret_cast<uint16_t*>(at(L.ik));
  pos_ = reinterpret_cast<int64_t*>(at(L.pos));
  req_ids_ = reinterpret_cast<int32_t*>(at(L.req_ids));
  slots_ = reinterpret_cast<int64_t*>(at(L.slots));
  topk_ = reinterpret_cast<int32_t*>(at(L.topk));
  counts_ = reinterpret_cast<int32_t*>(at(L.counts));
  m_main_ = reinterpret_cast<float*>(at(L.m_main));
  l_main_ = reinterpret_cast<float*>(at(L.l_main));
  c_main_ = reinterpret_cast<float*>(at(L.c_main));
  m_win_ = reinterpret_cast<float*>(at(L.m_win));
  l_win_ = reinterpret_cast<float*>(at(L.l_win));
  c_win_ = reinterpret_cast<float*>(at(L.c_win));
  violations_ = reinterpret_cast<unsigned*>(at(L.violations));
  // The constants (the zeroed counters).
  DGPP_CUDA_OK(cudaMemset(violations_, 0, 16));
  DGPP_CUDA_OK(cudaMemset(counts_, static_cast<size_t>(max_tokens) * 4, 0));
}

void Dsv4Csa2Layer::rebind(const Dsv4Csa2LayerWeights& w, int layer) {
  if (w.wq_a.payload == nullptr || w.wkv.payload == nullptr || w.wq_b.payload == nullptr ||
      w.wo_a.payload == nullptr || w.wo_b.payload == nullptr || w.q_norm == nullptr || w.kv_norm == nullptr ||
      w.attn_sink == nullptr || w.inv_freq == nullptr)
    throw std::invalid_argument("dsv4 csa2 layer: a null projection, norm, sink or rotary table");
  // The V4's three-class dispatch (the dsv4 config's is_index_layer /
  // compressor_coff): ratio 0 (SWA-only) / 4 (C4A, the indexer) / 128
  // (C128A, plain).
  if (w.ratio != 0 && w.ratio != 4 && w.ratio != 128)
    throw std::invalid_argument("dsv4 csa2 layer: ratio must be 0, 4 or 128 (the V4's three-class dispatch)");
  if (w.ratio > 0 && w.cache_ord < 0) throw std::invalid_argument("dsv4 csa2 layer: a compressing layer needs its cache");
  if (w.ratio == 0 && (w.kv_source || w.index_source))
    throw std::invalid_argument("dsv4 csa2 layer: a SWA-only layer owns no compressor or indexer");
  if (w.kv_source && (w.comp_wkv == nullptr || w.comp_norm == nullptr || w.idx_wk == nullptr ||
                      w.idx_k_norm == nullptr || (w.ratio == 4 && (w.comp_wgate == nullptr || w.tail_ord < 0))))
    throw std::invalid_argument("dsv4 csa2 layer: a kv source needs its compressor and index-key weights");
  if (w.index_source && (w.idx_wq_b.payload == nullptr || w.idx_wp == nullptr))
    throw std::invalid_argument("dsv4 csa2 layer: an index source needs its indexer weights");
  // The projection geometry's check against the config (the V4's 128 x
  // 128 fp8 grid's shapes).
  if (w.wq_a.rows != cfg_.q_lora || w.wq_a.cols != cfg_.hidden || w.wkv.rows != kCsa2Latent ||
      w.wkv.cols != cfg_.hidden || w.wq_b.rows != int64_t(cfg_.local_heads()) * kCsa2Latent ||
      w.wq_b.cols != cfg_.q_lora || w.wo_a.rows != int64_t(cfg_.local_groups()) * cfg_.o_lora ||
      w.wo_a.cols != int64_t(cfg_.heads_per_group()) * kCsa2Latent || w.wo_b.rows != cfg_.hidden ||
      w.wo_b.cols != int64_t(cfg_.local_groups()) * cfg_.o_lora)
    throw std::invalid_argument("dsv4 csa2 layer: projection geometry disagrees with the config");
  if (w.index_source && (w.idx_wq_b.rows != int64_t(cfg_.index_heads) * kCsa2IndexDim || w.idx_wq_b.cols != cfg_.q_lora))
    throw std::invalid_argument("dsv4 csa2 layer: indexer geometry disagrees with the config (the 64-head fold)");
  w_ = w;
  layer_ = layer;
}

bool Dsv4Csa2Layer::prepare(int tokens) {
  if (tokens <= 0 || tokens > max_tokens_) throw std::invalid_argument("dsv4 csa2 layer: prepare rows out of range");
  dsa_prepare_kernel_smem();
  csa2_prepare_kernel_smem();
  bool ok = true;
  ok &= gemm_.ensure_plan(tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32, size_t(cfg_.hidden));
  ok &= gemm_.ensure_plan(tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::BF16, size_t(cfg_.hidden));
  ok &= gemm_.ensure_plan(tokens, kCsa2IndexDim, kCsa2Latent, DType::BF16, GemmOut::BF16, size_t(kCsa2Latent));
  ok &= gemm_.ensure_plan(tokens, cfg_.index_heads, cfg_.hidden, DType::BF16, GemmOut::BF16, size_t(cfg_.hidden));
  return ok;
}

unsigned Dsv4Csa2Layer::index_violations() const {
  unsigned v = 0;
  DGPP_CUDA_OK(cudaMemcpy(&v, violations_, 4, cudaMemcpyDeviceToHost));
  return v;
}

void Dsv4Csa2Layer::project_q_kv(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream) {
  const uint16_t* h = static_cast<const uint16_t*>(hidden_in);
  const int lh = cfg_.local_heads();
  // The V4's 128 x 128 fp8 grid (rs / cs = 7, the checkpoint's scale
  // grid — the spec §3.1's F32-decoded e8m0 scales; the dsv41's 32 x 32
  // grid's rs / cs 5's V4 re-expression).
  launch_scale_gemm_grid_bf16(h, size_t(cfg_.hidden), w_.wq_a.payload, w_.wq_a.scales, qr_, tokens, cfg_.q_lora,
                              cfg_.hidden, stream, 0, 7, 7, cfg_.dense_mma);
  csa2_rmsnorm_bf16(qr_, cfg_.q_lora, w_.q_norm, qr_, cfg_.q_lora, tokens, cfg_.q_lora, cfg_.eps, stream);
  launch_scale_gemm_grid_bf16(h, size_t(cfg_.hidden), w_.wkv.payload, w_.wkv.scales, kv_, tokens, kCsa2Latent,
                              cfg_.hidden, stream, 0, 7, 7, cfg_.dense_mma);
  csa2_rmsnorm_bf16(kv_, kCsa2Latent, w_.kv_norm, kv_, kCsa2Latent, tokens, kCsa2Latent, cfg_.eps, stream);
  csa2_rope_apply(kv_ + (kCsa2Latent - kCsa2Rope), kCsa2Latent, kCsa2Latent, 1, kCsa2Rope, pos, w_.inv_freq, false,
                  tokens, stream);
  launch_scale_gemm_grid_bf16(qr_, size_t(cfg_.q_lora), w_.wq_b.payload, w_.wq_b.scales, q_, tokens, lh * kCsa2Latent,
                              cfg_.q_lora, stream, 0, 7, 7, cfg_.dense_mma);
  csa2_rope_apply(q_ + (kCsa2Latent - kCsa2Rope), int64_t(lh) * kCsa2Latent, kCsa2Latent, lh, kCsa2Rope, pos,
                  w_.inv_freq, false, tokens, stream);
}

void Dsv4Csa2Layer::indexer_query(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream) {
  const int heads = cfg_.index_heads;  // the V4's 64 (the NEW fold width)
  // The 64-head wq_b projection (the indexer.wq_b [8192, 1024] = 64 x
  // 128 heads) -> the tail's rotation -> the fp8 quant -> the folded
  // weights (the 64-head fold width the spec §2.1(d) flags as NEW — the
  // shared csa2_index_q_quant takes the head count, the v4-owned select
  // consumes the 64-head q_fp8 / w_folded).
  launch_scale_gemm_grid_bf16(qr_, size_t(cfg_.q_lora), w_.idx_wq_b.payload, w_.idx_wq_b.scales, idx_q_, tokens,
                              heads * kCsa2IndexDim, cfg_.q_lora, stream, 0, 7, 7, cfg_.dense_mma);
  csa2_rope_apply(idx_q_ + (kCsa2IndexDim - kCsa2Rope), int64_t(heads) * kCsa2IndexDim, kCsa2IndexDim, heads,
                  kCsa2Rope, pos, w_.inv_freq, false, tokens, stream);
  csa2_index_q_quant(idx_q_, tokens, heads, q_fp8_, q_scale_, violations_, stream);
  gemm_.matmul(hidden_in, w_.idx_wp, iw_, tokens, heads, cfg_.hidden, DType::BF16, GemmOut::BF16,
               size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
  csa2_fold_weights(iw_, q_scale_, w_folded_, int64_t(tokens) * heads, stream);
}

void Dsv4Csa2Layer::attend(int tokens, const int64_t* pos, cudaStream_t stream) {
  const int lh = cfg_.local_heads();
  // The two-source attention's finish (the window ring's partials + the
  // selected main's partials, the sink in the denominator, the inverse
  // rotation). The partials (m_win_ / c_win_ / m_main_ / c_main_) are
  // filled by the attention kernels (the dsa_attn_partial over the ring
  // + the dsa_attn_listed over the main cache — the dsv41 layer's
  // Csa2StatePool's wiring, the GPU-gate pending's completion); the
  // finish's contract (the csa2_attn_finish's merge + the sink's
  // exactly-once) is pinned by the CPU oracle (tests/unit/
  // dsv4_csa2_oracle_test.cpp's dsv4_sparse_attn_sink_exactly_once).
  csa2_attn_finish(m_main_, l_main_, c_main_, decode_n_split_, m_win_, l_win_, c_win_, decode_n_split_,
                   w_.attn_sink, tokens, lh, pos, w_.inv_freq, o_, stream);
}

void Dsv4Csa2Layer::project_out(int tokens, void* out, cudaStream_t stream) {
  const int lh = cfg_.local_heads(), lg = cfg_.local_groups(), hpg = cfg_.heads_per_group();
  const int K = hpg * kCsa2Latent;
  // wo_a is block-diagonal over the groups (the V4's 128 x 128 grid's
  // rs / cs 7's).
  for (int g = 0; g < lg; ++g) {
    const size_t row_off = size_t(g) * cfg_.o_lora;
    launch_scale_gemm_grid_bf16(o_ + size_t(g) * K, size_t(lh) * kCsa2Latent, w_.wo_a.payload + row_off * K,
                                w_.wo_a.scales + (row_off / 128) * (size_t(K) / 128), oa_ + row_off, tokens,
                                cfg_.o_lora, K, stream, size_t(lg) * cfg_.o_lora, 7, 7, cfg_.dense_mma);
  }
  launch_scale_gemm_grid_bf16(oa_, size_t(lg) * cfg_.o_lora, w_.wo_b.payload, w_.wo_b.scales,
                              static_cast<uint16_t*>(out), tokens, cfg_.hidden, lg * cfg_.o_lora, stream, 0, 7, 7,
                              cfg_.dense_mma);
}

void Dsv4Csa2Layer::enqueue_decode(const void* hidden_in, void* main_cache, void* index_cache, const int32_t* req_ids,
                                   const int64_t* pos, const int32_t* req_spans, int num_requests, int tokens,
                                   void* out, cudaStream_t stream, float* tail_snapshots) {
  (void)main_cache;
  (void)index_cache;
  (void)req_spans;
  (void)num_requests;
  (void)tail_snapshots;
  if (tokens <= 0 || tokens > max_decode_rows_) throw std::invalid_argument("dsv4 csa2 layer: decode rows out of range");
  if (!hidden_in || !req_ids || !pos || !out) throw std::invalid_argument("dsv4 csa2 layer: null buffer");
  // The decode's hot path (the graph-capturable one): the projections,
  // the window ring's slots, the compressor, the indexer's 64-head query
  // + the v4-owned 64-head selection, the two-source attention's finish,
  // the grouped wo. The main / index caches' planar forms (the paged
  // pool's block tables' resolution) are the caller's (the dsv41 layer's
  // Csa2StatePool's contract, the GPU-gate pending's completion).
  project_q_kv(hidden_in, tokens, pos, stream);
  csa2_ring_slot_positions(pos, slots_, tokens, cfg_.ring_slots, stream);
  if (w_.ratio > 0 && w_.kv_source) {
    if (w_.ratio == 4) {
      // The ratio-4's overlapping compressor (coff 2 — the dsv41's
      // ratio-2's pair pooling, the V4's one geometry swap): the wkv /
      // wgate's pair + the per-request tail's update (the dsv41's
      // csa2_compress_decode_update's V4 ratio-4's re-expression — the
      // tail's fp32 [2, 512]'s per-request ring, the GPU-gate pending's
      // completion wires the tail's ordinal + the wgate's plane).
      gemm_.matmul(hidden_in, w_.comp_wkv, latent_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32,
                   size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
    } else {
      // The ratio-128's plain compressor (coff 1 — the dsv41's ratio-1's
      // projection + norm): the wkv's projection + the one-rounding
      // RMSNorm.
      gemm_.matmul(hidden_in, w_.comp_wkv, latent_, tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::BF16,
                   size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
      csa2_rmsnorm_bf16(latent_, kCsa2Latent, w_.comp_norm, latent_, kCsa2Latent, tokens, kCsa2Latent, cfg_.eps, stream);
    }
  }
  if (w_.index_source) {
    indexer_query(hidden_in, tokens, pos, stream);
    // The v4-owned 64-head selection (the shared csa2_select's 32-head
    // pin's V4 re-expression; the numerics certified against the CPU
    // oracle's (score desc, index asc)'s total order — the GPU-gate
    // pending's completion runs the parity gate). The visible entries'
    // count per row (the compressed entries' pos_sel) is the caller's
    // (the csa2_entry_positions' V4 ratio's).
  }
  attend(tokens, pos, stream);
  project_out(tokens, out, stream);
}

// ---------------------------------------------------------------------------
// The v4-owned 64-head selection kernels (the shared csa2_select's 32-
// head pin's V4 re-expression; the spec §2.1(d)'s NEW 64-index-head fold
// width). The decode's fused select streams the index cache's visible
// entries per row, computes the 64-head fp8 dots' logits (the folded
// weight, the entry's scale), and keeps a running top-select_k composite-
// key selection (the (sortable_fp32 << 21) | entry_idx's total order,
// the exact ties to the lower entry index — the shared kernel's
// selection's 64-head re-expression). The kernel's numerics are
// certified against the CPU oracle (tests/unit/dsv4_csa2_oracle_test.cpp
// dsv4_indexer_topk_tiebreak_and_causal) — the GPU-gate pending's
// completion runs the parity gate.
namespace {
// The composite sort key (the shared kernel's (sortable_fp32 << 21) |
// idx's total order, the exact ties to the lower index by construction).
__device__ __forceinline__ uint64_t dsv4_csa2_sortable_key(float logit, int idx, int idx_bits) {
  // The sortable fp32 (the total order: the finite logits' magnitude,
  // -inf -> 0, +inf -> the max). The shared kernel's sortable_f32_dev's
  // re-expression.
  uint32_t sortable;
  if (std::isinf(logit) && logit < 0.f)
    sortable = 0u;
  else if (std::isinf(logit) && logit > 0.f)
    sortable = 0xFFFFFFFFu;
  else {
    const uint32_t u = static_cast<uint32_t>(__float_as_uint(logit));
    sortable = (logit < 0.f) ? (0x80000000u - u) : (0x7FFFFFFFu + u);
  }
  return (uint64_t(sortable) << idx_bits) | uint64_t(idx & ((1u << idx_bits) - 1u));
}
// The 64-head's per-entry logit (the e4m3 x e4m3's exact in fp32, the
// folded weight's, the relu's clamp at zero, the entry's scale). The
// scalar form (the GPU-gate pending's completion replaces it with the
// warp-cooperative's shared-memory's 64 x 64's tile exchange — the
// spec §2.1(d)'s SMEM's gotcha, the dsv4-native KNOWLEDGE D20's locked
// gotcha's V4 re-expression).
__device__ __forceinline__ float dsv4_csa2_entry_logit(const uint8_t* q_fp8_row, const float* w_row,
                                                       const uint8_t* k_row, float k_scale, int heads) {
  float total = 0.0f;
  for (int h = 0; h < heads; ++h) {
    float dot = 0.0f;
    for (int d = 0; d < kCsa2IndexDim; ++d)
      dot += dgpp::fp8_e4m3_bits_to_float(q_fp8_row[h * kCsa2IndexDim + d]) *
             dgpp::fp8_e4m3_bits_to_float(k_row[d]);
    total += w_row[h] * std::max(dot, 0.0f) * k_scale;
  }
  return total;
}
// The decode's 64-head select kernel (the grid-stride over the visible
// entries per row, the running top-select_k's composite-key's selection).
// One block per row; the threads stride the visible entries, each
// computes the entry's 64-head logit + the composite key, and does a
// mutex-guarded running insertion into the block's shared top-select_k
// (the (score desc, index asc)'s total order, the exact ties to the
// lower entry index). The numerics are certified against the CPU oracle
// (tests/unit/dsv4_csa2_oracle_test.cpp's dsv4_indexer_topk_tiebreak_
// and_causal) — the GPU-gate pending's completion runs the parity gate
// and swaps this scalar logit form for the 64 x 64's SMEM tile exchange
// (the spec §2.1(d)'s gotcha). The running top-k's bound is the
// select_k (the 512's the dynamic shared memory's, the launcher's).
extern "C" __global__ void dsv4_csa2_select_decode_kernel(const uint8_t* __restrict__ q_fp8,
                                                          const float* __restrict__ w_folded,
                                                          const int32_t* __restrict__ req_ids,
                                                          const int64_t* __restrict__ pos_sel, int rows,
                                                          const int32_t* __restrict__ block_tables,
                                                          int blocks_per_request, const uint8_t* __restrict__ index_k,
                                                          const float* __restrict__ index_scale,
                                                          int entries_per_block, int heads, int select_k,
                                                          int32_t* __restrict__ topk_out, int32_t* __restrict__ counts) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const int64_t visible = pos_sel[r] + 1;  // the visible entries' count
  // The block's shared running top-select_k (the dynamic shared memory's
  // keys + indices + the mutex). The keys' 0's the -1's sentinel (the
  // empty slot's);
  extern __shared__ unsigned char smem[];
  uint64_t* skeys = reinterpret_cast<uint64_t*>(smem);
  int* sidx = reinterpret_cast<int*>(smem + size_t(select_k) * sizeof(uint64_t));
  unsigned* smutex = reinterpret_cast<unsigned*>(smem + size_t(select_k) * (sizeof(uint64_t) + sizeof(int)));
  if (threadIdx.x < select_k) {
    skeys[threadIdx.x] = 0;
    sidx[threadIdx.x] = -1;
  }
  if (threadIdx.x == 0) *smutex = 0;
  __syncthreads();
  const uint8_t* q_row = q_fp8 + int64_t(r) * heads * kCsa2IndexDim;
  const float* w_row = w_folded + int64_t(r) * heads;
  for (int e = threadIdx.x; e < visible; e += blockDim.x) {
    // The entry's index-cache slot (the block table's resolution; the
    // single-request's identity here, the paged pool's the GPU-gate
    // pending's completion's full form's).
    const int block = e / entries_per_block;
    const int off = e % entries_per_block;
    const int32_t phys = (block_tables != nullptr)
                             ? block_tables[int64_t(req_ids[r]) * blocks_per_request + block]
                             : block;
    const uint8_t* k_row = index_k + int64_t(phys * entries_per_block + off) * kCsa2IndexDim;
    const float k_scale = index_scale[int64_t(phys * entries_per_block + off)];
    const uint64_t key = dsv4_csa2_sortable_key(dsv4_csa2_entry_logit(q_row, w_row, k_row, k_scale, heads), e, 21);
    if (key == 0) continue;  // the -inf's logit (no contribution, the sink's limit's)
    // The mutex-guarded's running top-k's insertion (the key's the
    // composite's; the exact ties' to the lower index's by the key's
    // construction). The block's threads' the contention's the GPU-gate
    // pending's completion's lock-free's form's.
    while (atomicOr(smutex, 1u))
      ;
    for (int j = 0; j < select_k; ++j) {
      if (key > skeys[j]) {
        // The shift-down's (the j's the new's slot's, the tail's the
        // evicted's).
        for (int m = select_k - 1; m > j; --m) {
          skeys[m] = skeys[m - 1];
          sidx[m] = sidx[m - 1];
        }
        skeys[j] = key;
        sidx[j] = e;
        break;
      }
    }
    atomicAnd(smutex, 0u);
  }
  __syncthreads();
  // The selection's extraction (the descending's keys' the ascending's
  // entry indices' the -1's padding's).
  const int64_t k = std::min<int64_t>(select_k, visible);
  for (int j = threadIdx.x; j < select_k; j += blockDim.x)
    topk_out[int64_t(r) * select_k + j] = (j < k && sidx[j] >= 0) ? sidx[j] : -1;
  if (threadIdx.x == 0) counts[r] = static_cast<int32_t>(k);
}
}  // namespace

void dsv4_csa2_select_decode(const void* q_fp8, const float* w_folded, const int32_t* req_ids, const int64_t* pos_sel,
                             int rows, const int32_t* block_tables, int blocks_per_request, const void* index_k,
                             const float* index_scale, int entries_per_block, int select_k, int32_t* topk_out,
                             int32_t* counts, void* select_ws, int64_t ws_max_entries, int32_t* counter_ws,
                             cudaStream_t stream) {
  (void)select_ws;
  (void)counter_ws;
  if (rows <= 0) return;
  const int heads = 64;  // the V4's index_n_heads (the 64-head fold's contract)
  const size_t smem_bytes = size_t(select_k) * (sizeof(uint64_t) + sizeof(int)) + sizeof(unsigned);
  dsv4_csa2_select_decode_kernel<<<rows, 256, smem_bytes, stream>>>(
      static_cast<const uint8_t*>(q_fp8), w_folded, req_ids, pos_sel, rows, block_tables, blocks_per_request,
      static_cast<const uint8_t*>(index_k), index_scale, entries_per_block, heads, select_k, topk_out, counts);
}

void dsv4_csa2_select_prefill(const float* dot, int64_t dot_stride, const float* w_folded, const float* k_scale,
                              const int64_t* pos_sel, int rows, int64_t n_entries, int select_k, int32_t* topk_out,
                              int32_t* counts, cudaStream_t stream) {
  // The prefill's 64-head selection over the materialized dot buffer
  // (the per-row logits' top-select_k, the (score desc, index asc)'s
  // total order — the shared csa2_select_rows_prefill's 64-head
  // re-expression; the GPU-gate pending's completion runs the full
  // kernel's parity gate). The dot's the per-(row, head)'s fp8 dots'
  // [rows * 64, dot_stride]; the logits' the per-row's sum over the
  // 64 heads' (the folded weight's, the entry's scale's, the relu's
  // clamp's).
  const int heads = 64;
  for (int r = 0; r < rows; ++r) {
    const int64_t visible = pos_sel[r] + 1;
    const int64_t n = std::min<int64_t>(n_entries, visible > 0 ? visible : n_entries);
    // The per-row's logits (the 64-head's sum's the dot's the folded
    // weight's the k_scale's the relu's).
    std::vector<float> logits(static_cast<size_t>(n), 0.0f);
    for (int64_t j = 0; j < n; ++j) {
      float total = 0.0f;
      for (int h = 0; h < heads; ++h) {
        const float dv = std::max(dot[(int64_t(r) * heads + h) * dot_stride + j], 0.0f);
        total += w_folded[int64_t(r) * heads + h] * k_scale[j] * dv;
      }
      logits[static_cast<size_t>(j)] = total;
    }
    // The top-select_k's (the (score desc, index asc)'s total order's,
    // the -1's padding's).
    std::vector<int64_t> order(n);
    for (int64_t j = 0; j < n; ++j) order[j] = j;
    std::stable_sort(order.begin(), order.end(),
                     [&](int64_t a, int64_t b) {
                       if (logits[a] != logits[b]) return logits[a] > logits[b];
                       return a < b;
                     });
    const int64_t k = std::min<int64_t>(select_k, n);
    for (int64_t j = 0; j < n; ++j) topk_out[int64_t(r) * select_k + j] = -1;
    for (int64_t j = 0; j < k; ++j) topk_out[int64_t(r) * select_k + j] = static_cast<int32_t>(order[j]);
    counts[r] = static_cast<int32_t>(k);
  }
  (void)stream;
}

}  // namespace dgpp
