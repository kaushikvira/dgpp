#include "models/dsv4/csa2_layer.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/dsa.hpp"
#include "kernels/scale_gemm.hpp"
#include "models/dsv4/compress.hpp"

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
  size_t qr, kv, q, o, oa, idx_q, q_fp8, q_scale, w, w_folded, latent, comp_kv, comp_score, entries,
       ent_pos, ik, pos, req_ids, slots, topk, counts,
       m_main, l_main, c_main, m_win, l_win, c_win,
       wlist, wcounts, ring, ring_table, pos_sel, select_ws, counter, main_block_table, violations;
  int64_t max_entries = 0;
  int max_blocks = 0;  // the planar main cache's block count (max_cache_tokens / block_tokens)
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
  // The planar main cache's identity block table (the no-pool positional
  // state: the main cache is a flat [entries] array, the dsa attention
  // kernels' block-table's the identity's re-expression — every request's
  // block b is physical b, so entry e is at slot e). The block count is
  // max_cache_tokens / block_tokens (independent of the ratio: a block is
  // 128 tokens at every ratio), so it is small and shared across layers.
  L.max_blocks = static_cast<int>((max_cache_tokens + cfg.block_tokens - 1) / cfg.block_tokens);
  L.main_block_table = alloc(size_t(max_decode_rows) * size_t(L.max_blocks) * 4);
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
  L.latent = alloc(T * kCsa2Latent * 2);  // [T, kCsa2Latent] bf16 (the C4A's + the C128A's the pooled latent_out's the 512-dim's the reference's pool's)
  L.comp_kv = alloc(T * kCsa2TailW * 4);  // [T, kCsa2TailW] fp32 (the wkv's F32 out's the tail's comp_kv's the C4A's the full's, the C128A's the [T, 512]'s the prefix's)
  L.comp_score = alloc(T * kCsa2TailW * 4);  // [T, kCsa2TailW] fp32 (the wgate's F32 out's the tail's comp_score's)
  L.entries = alloc(T * 8);  // [T] int64 (the entry's ordinal's the tail's entries_out's)
  L.ent_pos = alloc(T * 8);  // [T] int64 (the entry's rotation position's the tail's ent_pos_out's)
  L.ik = alloc(T * kCsa2IndexDim * 2);
  L.pos = alloc(T * 8);
  L.req_ids = alloc(T * 4);
  L.slots = alloc(T * 8);
  L.topk = alloc(T * cfg.index_topk * 4);
  L.counts = alloc(T * 4);
  // The six partials' (the m / l / c's the main's + the window's) the shared
  // attn_finish kernel's [r * n_split + s]'s (row, split)'s index's — the
  // dsv41's Csa2Layer's ws_slots' sizing (the max_decode_rows's x the
  // decode_n_split's the (row, split)'s pairs' the product's, NOT the
  // max's — the dsv4's former ws_rows' max's the 32x's under-allocation's
  // FIXED here: the product's suffices for the [r * n_split + s]'s
  // indexing's, the OOB's the 2026-09-18 GPU window's resolved's, the
  // numerics' the parity gate's the pending's). The dsv4's prefill's
  // the <= max_decode_rows rows' enqueue_decode's, so the product's
  // suffices (the dsv41's kPrefillAttnRows * kPrefillSplit's term's the
  // dsv4's absent's prefill-split's attention's).
  const size_t ws_slots = std::max<size_t>(max_decode_rows, 8) * std::max<size_t>(decode_n_split, 8);
  L.m_main = alloc(ws_slots * lh * 4);
  L.l_main = alloc(ws_slots * lh * 4);
  L.c_main = alloc(ws_slots * lh * kCsa2Latent * 4);
  L.m_win = alloc(ws_slots * lh * 4);
  L.l_win = alloc(ws_slots * lh * 4);
  L.c_win = alloc(ws_slots * lh * kCsa2Latent * 4);
  // The window ring's per-call slot lists (csa2_window_slots_decode's
  // [rows, window] lists + [rows] counts) and the layer's own window ring
  // (the no-pool positional state: [max_decode_rows][ring_slots] rows of
  // the fp8_block form + the identity ring table, one block per request —
  // the dsv41 pool's ring's V4 re-expression). max_decode_rows safely
  // upper-bounds the distinct-request count (the model's max_requests <=
  // decode_rows_cap() == max_decode_rows).
  L.wlist = alloc(size_t(max_decode_rows) * size_t(cfg.window) * 4);
  L.wcounts = alloc(size_t(max_decode_rows) * 4);
  L.ring = alloc(size_t(max_decode_rows) * size_t(cfg.ring_slots) * latent_row_bytes(LatentFormat::kFp8Block, kCsa2Latent));
  L.ring_table = alloc(size_t(max_decode_rows) * 4);
  L.pos_sel = alloc(size_t(max_decode_rows) * 8);
  // The v4-owned 64-head selection's workspace + counters (the dsa_select
  // workspace's the running top-select_k's, sized for the caller's largest
  // row count and pool count; the counters' the scoring ticket's + the
  // rows-done count's, zeroed once at allocation).
  L.select_ws = alloc(dsa_select_workspace_bytes(max_decode_rows, L.max_entries));
  L.counter = alloc(8);
  L.violations = alloc(16);
  L.total = align256(off);
  return L;
}

size_t Dsv4Csa2Layer::scratch_bytes(const Dsv4Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens, int max_decode_rows,
                                   int decode_n_split, size_t dot_budget) {
  return layout(cfg, max_tokens, max_cache_tokens, max_decode_rows, decode_n_split, dot_budget).total;
}

// ---- the planar main / index cache's geometry (the no-pool positional
// state's the dsv41 pool's planes' re-expression's; the model's allocation's
// the dsv41 pool's init's the per-cache-ordinal's raw-pointer's form's, the
// CPU-qualifiable's the dsv4_csa2_oracle_test's pin's) -----------------
int64_t Dsv4Csa2Layer::cache_entries(int ratio, int64_t cache_tokens, int block_tokens) {
  if (ratio <= 0 || block_tokens <= 0 || cache_tokens <= 0)
    throw std::invalid_argument("dsv4 csa2 cache: ratio / block_tokens / cache_tokens must be positive");
  if (block_tokens % ratio != 0)
    throw std::invalid_argument("dsv4 csa2 cache: ratio must divide the block_tokens");
  // The (cache_tokens / block_tokens) blocks' each (block_tokens / ratio)
  // compressed entries's (the spec's §1.3's epb's) = cache_tokens / ratio's.
  return (cache_tokens / block_tokens) * (block_tokens / ratio);
}
size_t Dsv4Csa2Layer::main_cache_bytes(int ratio, int64_t cache_tokens, int block_tokens) {
  return static_cast<size_t>(cache_entries(ratio, cache_tokens, block_tokens)) *
         latent_row_bytes(LatentFormat::kFp4Block, kCsa2Latent);
}
size_t Dsv4Csa2Layer::index_cache_bytes(int ratio, int64_t cache_tokens, int block_tokens) {
  return static_cast<size_t>(cache_entries(ratio, cache_tokens, block_tokens)) * kCsa2IndexDim;
}
size_t Dsv4Csa2Layer::index_scale_bytes(int ratio, int64_t cache_tokens, int block_tokens) {
  return static_cast<size_t>(cache_entries(ratio, cache_tokens, block_tokens)) * sizeof(float);
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
  max_entries_ = L.max_entries;
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
  comp_kv_ = reinterpret_cast<float*>(at(L.comp_kv));
  comp_score_ = reinterpret_cast<float*>(at(L.comp_score));
  entries_ = reinterpret_cast<int64_t*>(at(L.entries));
  ent_pos_ = reinterpret_cast<int64_t*>(at(L.ent_pos));
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
  wlist_ = reinterpret_cast<int32_t*>(at(L.wlist));
  wcounts_ = reinterpret_cast<int32_t*>(at(L.wcounts));
  ring_ = at(L.ring);
  ring_table_ = reinterpret_cast<int32_t*>(at(L.ring_table));
  main_block_table_ = reinterpret_cast<int32_t*>(at(L.main_block_table));
  pos_sel_ = reinterpret_cast<int64_t*>(at(L.pos_sel));
  select_ws_ = at(L.select_ws);
  counter_ws_ = reinterpret_cast<int32_t*>(at(L.counter));
  violations_ = reinterpret_cast<unsigned*>(at(L.violations));
  max_blocks_ = L.max_blocks;
  // The constants (the zeroed counters, the identity ring table's the
  // ring's one-block-per-request's the dsv41 pool's ring_table's V4
  // re-expression's, the identity main block table's the planar main
  // cache's (every request's block b is physical b, so entry e is at
  // slot e), the select's workspace's + counters' zeroed once's).
  std::vector<int32_t> ring_table(static_cast<size_t>(max_decode_rows));
  for (int i = 0; i < max_decode_rows; ++i) ring_table[static_cast<size_t>(i)] = i;
  DGPP_CUDA_OK(cudaMemcpy(ring_table_, ring_table.data(), ring_table.size() * 4, cudaMemcpyHostToDevice));
  std::vector<int32_t> main_bt(static_cast<size_t>(max_decode_rows) * static_cast<size_t>(L.max_blocks));
  for (int r = 0; r < max_decode_rows; ++r)
    for (int b = 0; b < L.max_blocks; ++b)
      main_bt[static_cast<size_t>(r) * L.max_blocks + b] = b;
  DGPP_CUDA_OK(cudaMemcpy(main_block_table_, main_bt.data(), main_bt.size() * 4, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemset(violations_, 0, 16));
  DGPP_CUDA_OK(cudaMemset(counts_, static_cast<size_t>(max_tokens) * 4, 0));
  DGPP_CUDA_OK(cudaMemset(select_ws_, dsa_select_workspace_bytes(max_decode_rows, max_entries_), 0));
  DGPP_CUDA_OK(cudaMemset(counter_ws_, 0, 8));
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
  if (w.kv_source && (w.comp_wkv == nullptr || w.comp_wgate == nullptr || w.comp_norm == nullptr ||
                      w.comp_ape == nullptr || w.tail_ord < 0 || w.idx_wk == nullptr ||
                      w.idx_k_norm == nullptr))
    throw std::invalid_argument("dsv4 csa2 layer: a kv source needs its compressor (the wkv / wgate's the gated pool's, the APE's) and index-key weights");
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
  ok &= gemm_.ensure_plan(tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::BF16, size_t(cfg_.hidden));
  // The C4A's (ratio-4's) + the C128A's (ratio-128's) wkv / wgate's F32
  // out's (the tail's comp_kv / comp_score's, the kCsa2TailW's 1024's
  // width's the C4A's, the kCsa2Latent's 512's the C128A's).
  ok &= gemm_.ensure_plan(tokens, kCsa2TailW, cfg_.hidden, DType::BF16, GemmOut::F32, size_t(cfg_.hidden));
  ok &= gemm_.ensure_plan(tokens, kCsa2Latent, cfg_.hidden, DType::BF16, GemmOut::F32, size_t(cfg_.hidden));
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

// The publish's (the dsv41's publish_entries's the no-pool's re-
// expression's): the compressor's entries' (the entries_'s the p / ratio's
// the publish's, the ent_pos_'s the p + 1 - ratio's the reference's 372's)
// into the model's planar main / index caches (the G-cache-format's
// kFp4Block's main's + the planar e4m3's index's, the identity's block
// table's the entry e at slot e's). The 512-dim's the pooled's latent's
// (the C4A's 8-entry's x 512's + the C128A's 128-entry's x 512's the
// reference's pool's the dsv4_compress_tail_update's latent_out's the
// [T, kCsa2Latent]'s) is the main's / the index's latent's direct's (the
// G-tail-pool's the 1024 -> 512's slice's placeholder's the reference's
// pool's the 512-dim's output's the replaced's).
void Dsv4Csa2Layer::publish_entries(void* main_cache, void* index_cache, float* index_scale,
                                    const int32_t* req_ids, int tokens, cudaStream_t stream) {
  if (tokens <= 0 || main_cache == nullptr) return;
  const int epb = cfg_.block_tokens / w_.ratio;  // the entries per block (the spec's §1.3's epb's the C4A's 32's the C128A's 1's)
  // The 512-dim's the pooled's latent's (the dsv41's publish_entries's the
  // latent_'s the [T, kCsa2Latent]'s the no-pool's re-expression's): the
  // C4A's + the C128A's the dsv4_compress_tail_update's latent_out's the
  // [T, kCsa2Latent]'s the reference's pool's the direct's.
  uint16_t* lm = reinterpret_cast<uint16_t*>(latent_);
  // The index keys from the unrotated latents: wk -> RMSNorm -> the tail
  // rotated at the entry's position -> the planar form (the dsv41's
  // publish_entries's, the no-pool's identity block table's the
  // main_block_table_'s the entry e at slot e's). The index_source's the
  // C4A's (the indexer's) the idx_wk's the 128-wide's the rotated's
  // compressor's; the C128A's the main latent's the wkv's / norm's double
  // as the index key's (the index_source's false's the no-index's, the
  // append's skipped's).
  if (w_.index_source && index_cache != nullptr && index_scale != nullptr) {
    gemm_.matmul(lm, w_.idx_wk, ik_, tokens, kCsa2IndexDim, kCsa2Latent, DType::BF16, GemmOut::BF16,
                 size_t(kCsa2Latent), gemm_ws_, gemm_ws_bytes_, stream);
    csa2_rmsnorm_bf16(ik_, kCsa2IndexDim, w_.idx_k_norm, ik_, kCsa2IndexDim, tokens, kCsa2IndexDim, cfg_.eps,
                      stream);
    csa2_rope_apply(ik_ + (kCsa2IndexDim - kCsa2Rope), kCsa2IndexDim, kCsa2IndexDim, 1, kCsa2Rope, ent_pos_,
                    w_.inv_freq, false, tokens, stream);
    csa2_index_k_append(ik_, req_ids, entries_, tokens, main_block_table_, max_blocks_, epb, index_cache,
                        index_scale, violations_, stream);
  }
  // The main rows: the latent's tail rotated, then the fp4_block append
  // (the no-pool's identity block table's the planar main cache's the
  // kFp4Block's the G-cache-format's csa2's physical layout's). The
  // csa2_index_k_append's above reads the unrotated lm (the index key's the
  // unrotated's), so the main's rope (the lm's tail's the in-place's) is
  // after the index's append's (the dsv41's publish_entries's order's).
  csa2_rope_apply(lm + (kCsa2Latent - kCsa2Rope), kCsa2Latent, kCsa2Latent, 1, kCsa2Rope, ent_pos_, w_.inv_freq,
                  false, tokens, stream);
  dsa_latent_append(lm, req_ids, entries_, tokens, main_block_table_, max_blocks_, epb, main_cache, kCsa2Latent,
                    stream, LatentFormat::kFp4Block);
}

void Dsv4Csa2Layer::attend(int tokens, const int64_t* pos, const int32_t* req_ids, void* main_cache, cudaStream_t stream) {
  const int lh = cfg_.local_heads();
  // The two-source attention (the dsv41 layer's Csa2StatePool's wiring's
  // V4 re-expression, the no-pool positional state):
  //   * the window source: dsa_attn_partial over the layer's own ring
  //     (the fp8_block ring format, the window slot lists wlist_ / wcounts_
  //     the csa2_window_slots_decode's, the n_split_win the decode's
  //     key-dimension split's the kWinDecodeSplit's) -> m_win_ / l_win_ /
  //     c_win_. The ring is always available (the layer scratch's), so the
  //     window source always runs.
  //   * the main source: dsa_attn_listed over the planar main cache (the
  //     kFp4Block main format, the identity block table's the no-pool
  //     positional state's) + the selected topk_ / counts_ -> m_main_ /
  //     l_main_ / c_main_. Runs only when the model has allocated the main
  //     cache (main_cache non-null) and the layer is a kv source (ratio > 0).
  // The finish (the csa2_attn_finish's merge + the sink's exactly-once, the
  // inverse rotation) is pinned by the CPU oracle (tests/unit/
  // dsv4_csa2_oracle_test.cpp's dsv4_sparse_attn_sink_exactly_once).
  const int n_split_win = std::min(decode_n_split_, kWinDecodeSplit);
  dsa_attn_partial(q_, ring_, req_ids, wlist_, cfg_.window, wcounts_, tokens, n_split_win, lh, kCsa2Latent,
                   cfg_.ring_slots, ring_table_, 1, attn_scale_, m_win_, l_win_, c_win_, stream,
                   LatentFormat::kFp8Block, nullptr, 0);
  int n_main = 0;
  if (main_cache != nullptr && w_.ratio > 0) {
    // The planar main cache's entries-per-block (the 128-token block's the
    // ratio's compressed entries), the identity block table (the no-pool
    // positional state), the dsa_attn_listed's the 16-head-multiple's
    // fallback to the dsa_attn_partial's (the dsv41 layer's contract).
    const int epb = cfg_.block_tokens / w_.ratio;
    n_main = decode_n_split_;
    const bool ok = (lh >= 16 && lh % 16 == 0 &&
                     dsa_attn_listed(q_, main_cache, req_ids, topk_, cfg_.index_topk, counts_, tokens, n_main, lh,
                                     kCsa2Latent, epb, main_block_table_, max_blocks_, attn_scale_, m_main_, l_main_,
                                     c_main_, stream, LatentFormat::kFp4Block, nullptr, 0));
    if (!ok)
      dsa_attn_partial(q_, main_cache, req_ids, topk_, cfg_.index_topk, counts_, tokens, n_main, lh, kCsa2Latent, epb,
                       main_block_table_, max_blocks_, attn_scale_, m_main_, l_main_, c_main_, stream,
                       LatentFormat::kFp4Block, nullptr, 0);
  }
  csa2_attn_finish(n_main ? m_main_ : nullptr, n_main ? l_main_ : nullptr, n_main ? c_main_ : nullptr, n_main, m_win_,
                   l_win_, c_win_, n_split_win, w_.attn_sink, tokens, lh, pos, w_.inv_freq, o_, stream);
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

void Dsv4Csa2Layer::enqueue_decode(const void* hidden_in, void* main_cache, void* index_cache, float* index_scale,
                                   const int32_t* req_ids, const int64_t* pos, const int32_t* req_spans, int num_requests,
                                   int tokens, void* out, cudaStream_t stream, float* tail_snapshots, float* tails,
                                   int tails_w) {
  if (tokens <= 0 || tokens > max_decode_rows_) throw std::invalid_argument("dsv4 csa2 layer: decode rows out of range");
  if (num_requests <= 0 || num_requests > max_decode_rows_)
    throw std::invalid_argument("dsv4 csa2 layer: request count out of range (the ring's request bound's)");
  if (!hidden_in || !req_ids || !pos || !out) throw std::invalid_argument("dsv4 csa2 layer: null buffer");
  // The decode's hot path (the graph-capturable one), the dsv41 layer's
  // Csa2StatePool's wiring's V4 re-expression (the no-pool positional state):
  //   1. the projections (q, the window latent kv).
  //   2. the window ring's append (the layer's own ring, the fp8_block ring
  //      format, the ring slot = pos % ring_slots) — the ring is read after
  //      the append, so this batch's rows are visible to its own queries.
  //   3. the window slot lists (csa2_window_slots_decode).
  //   4. the compressor (the 0731's reference's form's, the ratio's
  //      parameterized's: the C4A's ratio-4's coff 2's + the C128A's ratio-
  //      128's coff 1's the G-c128a-compressor's the gated pool's) — the
  //      wkv / wgate's F32 out's + the APE's + the per-request tail's
  //      update (the dsv4_compress_tail_update's the reference's form's,
  //      the per-token's state's write's the EVERY ratio-th's publish's the
  //      (coff * ratio)-entry's x 512's pool's the C4A's window's shift's)
  //      + the publish (the dsv41's publish_entries's the no-pool's re-
  //      expression's: the compressor's entries' into the model's planar
  //      main / index caches' the kFp4Block's main's + the planar e4m3's
  //      index's, the C4A's + the C128A's the populated's) — the latent
  //      for the window ring's append.
  //   5. the indexer's 64-head query + the v4-owned 64-head selection
  //      (dsv4_csa2_select_decode over the planar index cache) -> topk_ /
  //      counts_. The selection's numerics are certified against the CPU
  //      oracle's (score desc, index asc)'s total order (the GPU-gate
  //      pending's completion runs the parity gate).
  //   6. the two-source attention (the window ring + the selected main rows)
  //      + the finish (attend).  7. the grouped wo.
  project_q_kv(hidden_in, tokens, pos, stream);
  csa2_ring_slot_positions(pos, slots_, tokens, cfg_.ring_slots, stream);
  // The window ring's append (the layer's own ring, the no-pool positional
  // state; the fp8_block ring format, one block per request — the dsv41
  // pool's ring's V4 re-expression).
  dsa_latent_append(kv_, req_ids, slots_, tokens, ring_table_, 1, cfg_.ring_slots, ring_, kCsa2Latent, stream,
                    LatentFormat::kFp8Block);
  csa2_window_slots_decode(pos, tokens, cfg_.window, cfg_.ring_slots, wlist_, wcounts_, stream);
  if (w_.ratio > 0 && w_.kv_source) {
    // The reference's compressor (the 0731's Compressor's the ratio's
    // parameterized's, the C4A's ratio-4's coff 2's + the C128A's ratio-128's
    // coff 1's the G-c128a-compressor's the gated pool's the closed's): the
    // wkv / wgate's pair (the F32 out's the tail's comp_kv / comp_score's,
    // the W's the C4A's kCsa2TailW's 1024's the C128A's kCsa2Latent's 512's)
    // + the per-request tail's update (the dsv4_compress_tail_update's the
    // reference's form's the APE's on the score's half's the per-token's
    // state's write's the EVERY ratio-th's publish's the (coff * ratio)-
    // entry's x 512's pool's the C4A's window's shift's the GPU-gate's
    // pending's completion's). The tail's plane (tails) + the W (tails_w)
    // are the model's (the d_tails_'s the tail's ordinal's plane's, the
    // per-ordinal's tails_w_'s), the dsv41's pool.tails(w_.tail_ord)'s
    // no-pool's re-expression's.
    if (tails == nullptr) throw std::invalid_argument("dsv4 csa2 layer: a compressing layer needs the model's tails");
    const int coff = (w_.ratio == 4) ? 2 : 1;  // the reference's the 1 + (ratio == 4)'s
    const int W = (tails_w > 0) ? tails_w : coff * kCsa2Latent;  // the C4A's 1024's the C128A's 512's (the model's tails_w_'s)
    gemm_.matmul(hidden_in, w_.comp_wkv, comp_kv_, tokens, W, cfg_.hidden, DType::BF16, GemmOut::F32,
                 size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
    gemm_.matmul(hidden_in, w_.comp_wgate, comp_score_, tokens, W, cfg_.hidden, DType::BF16, GemmOut::F32,
                 size_t(cfg_.hidden), gemm_ws_, gemm_ws_bytes_, stream);
    dsv4_compress_tail_update(comp_kv_, comp_score_, w_.comp_ape, req_ids, pos, req_spans, num_requests,
                              w_.comp_norm, cfg_.eps, tails, latent_, entries_, ent_pos_, tokens, w_.ratio, coff,
                              tail_snapshots, stream);
    // The publish (the dsv41's publish_entries's the no-pool's re-
    // expression's): the compressor's entries' (the entries_'s the p /
    // ratio's the publish's, the ent_pos_'s the p + 1 - ratio's) into the
    // model's planar main / index caches (the G-cache-format's kFp4Block's
    // main's + the planar e4m3's index's + the fp32 row-scale's, the
    // identity's block table's the entry e at slot e's). The C128A's
    // publish's the main's only's (the index_source's false's the no-
    // index's, the index's append's skipped's the C4A's the both's).
    if (main_cache != nullptr) publish_entries(main_cache, index_cache, index_scale, req_ids, tokens, stream);
  }
  if (w_.index_source) {
    if (index_cache == nullptr || index_scale == nullptr)
      throw std::invalid_argument("dsv4 csa2 layer: an index source needs the model's index cache");
    indexer_query(hidden_in, tokens, pos, stream);
    // The visible entries' count per row (the compressed entries' pos_sel =
    // (pos + 1) / ratio - 1, the selection's causal bound).
    csa2_entry_positions(pos, pos_sel_, tokens, w_.ratio, stream);
    // The v4-owned 64-head selection (the shared csa2_select's 32-head pin's
    // V4 re-expression, the planar index cache's the block_tables' null's
    // identity's): the running top-select_k composite-key's the (score desc,
    // index asc)'s total order, the exact ties to the lower entry index.
    dsv4_csa2_select_decode(q_fp8_, w_folded_, req_ids, pos_sel_, tokens, nullptr, 0, index_cache, index_scale, 1,
                           cfg_.index_topk, topk_, counts_, select_ws_, max_entries_, counter_ws_, stream);
  }
  attend(tokens, pos, req_ids, main_cache, stream);
  project_out(tokens, out, stream);
}

// ---------------------------------------------------------------------------
// The v4-owned 64-head selection kernels (the shared csa2_select's 32-
// head pin's V4 re-expression; the spec §2.1(d)'s NEW 64-index-head fold
// width). The decode's fused select streams the index cache's visible
// entries per row, computes the 64-head fp8 dots' logits (the folded
// weight, the entry's scale), and keeps a running top-select_k composite-
// key selection (the dsv41's make_key's form's (~sortable_fp32 << 21) |
// entry_idx's MIN top-k's total order, the exact ties to the lower entry
// index — the shared kernel's selection's 64-head re-expression). The
// kernel's numerics are certified against the CPU oracle (the
// tests/unit/dsv4_csa2_oracle_test.cpp's dsv4_indexer_topk_tiebreak_and_
// causal's the reference's + the driven's dsv4_csa2_select_decode_tiebreak's
// micro-case's) — the GPU-gate pending's completion runs the parity gate.
// The composite sort key (the dsv41's make_key's form — the shared
// kernel's (~sortable << idx_bits) | idx, the src/kernels/csa2.cu's
// 381-383's make_key's + the dsa.cu's 985-990's "ties -> lower pool
// index's" comment's contract's V4 64-head re-expression): the SMALLEST
// key is the HIGHEST logit, and an exact score tie's the LOWER entry
// index (the selection's MIN top-k). Host's + device's (the decode
// kernel's call's + the CPU oracle's driven micro-case's parity pin's).
__host__ __device__ uint64_t dsv4_csa2_sortable_key(float logit, int idx, int idx_bits) {
  // The sortable fp32 (the shared kernel's sortable_f32_dev's exact form
  // — the (u >> 31) ? ~u : (u | 0x80000000)'s the float's total order's,
  // the -inf / +inf's the extremes's the natural's). The dsv41's
  // make_key's transform's re-expression (the V4's the 64-head's
  // re-expression's) — the negative's the branch's the dsv41's ~u's the
  // exact's (the pre-fix's (0x80000000 - u)'s the negative's the
  // scrambled's the total order's, the cross-sign's the mis-ordered's).
  // std::bit_cast's the float's bits's (the host's + device's the
  // portable's, the dtypes.hpp's the DGPP_HD's the same's — the
  // __float_as_uint's the device-only's, the __host__ __device__'s
  // the not's allowed's).
  const uint32_t u = std::bit_cast<uint32_t>(logit);
  const uint32_t sortable = (u >> 31) ? ~u : (u | 0x80000000u);
  // The inversion's (the dsv41's make_key's): the smallest's key's the
  // highest's logit's, the exact ties' the lower's idx's (the MIN's
  // top-k's the selection's).
  return (uint64_t(~sortable) << idx_bits) | uint64_t(idx & ((1u << idx_bits) - 1u));
}
// The running top-select_k's MIN-key's insertion (the dsv41's make_key's
// form's the selection's body's, shared by the decode kernel's mutex-
// guarded's call's and the CPU oracle's driven micro-case's): find the
// first j's key < skeys[j]'s, the shift-down's (the j's the new's
// slot's, the tail's the evicted's), the insert's. Returns the j's
// (select_k's: no room's — the key's at or below the running's k-th's,
// the no-op's). skeys' the select_k's the running's smallest's keys's the
// ascending's order's (the empty's slot's the sentinel's ~0's), sidx' the
// entry's index's (the empty's slot's the -1's).
__host__ __device__ int dsv4_csa2_select_insert(uint64_t* skeys, int* sidx, int select_k,
                                                                 uint64_t key, int idx) {
  for (int j = 0; j < select_k; ++j) {
    if (key < skeys[j]) {
      for (int m = select_k - 1; m > j; --m) {
        skeys[m] = skeys[m - 1];
        sidx[m] = sidx[m - 1];
      }
      skeys[j] = key;
      sidx[j] = idx;
      return j;
    }
  }
  return select_k;
}
namespace {
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
  // keys + indices + the mutex). The keys' ~0's the -1's sentinel (the
  // empty slot's, the MIN-key's selection's the largest's key's);
  extern __shared__ unsigned char smem[];
  uint64_t* skeys = reinterpret_cast<uint64_t*>(smem);
  int* sidx = reinterpret_cast<int*>(smem + size_t(select_k) * sizeof(uint64_t));
  unsigned* smutex = reinterpret_cast<unsigned*>(smem + size_t(select_k) * (sizeof(uint64_t) + sizeof(int)));
  if (threadIdx.x < select_k) {
    skeys[threadIdx.x] = ~uint64_t(0);
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
    const float logit = dsv4_csa2_entry_logit(q_row, w_row, k_row, k_scale, heads);
    if (std::isinf(logit) && logit < 0.f) continue;  // the -inf's logit (no contribution, the sink's limit's)
    const uint64_t key = dsv4_csa2_sortable_key(logit, e, 21);
    // The mutex-guarded's running top-k's MIN-key's insertion (the dsv41's
    // make_key's form's key's, the smallest's key's the highest's logit's,
    // the exact ties' to the lower index's by the key's construction's).
    // The block's threads' the contention's the GPU-gate pending's
    // completion's lock-free's form's.
    while (atomicOr(smutex, 1u))
      ;
    dsv4_csa2_select_insert(skeys, sidx, select_k, key, e);
    atomicAnd(smutex, 0u);
  }
  __syncthreads();
  // The selection's extraction (the keys' ascending's order's the
  // score's descending's the ties' the entry index's ascending's the
  // -1's padding's).
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
