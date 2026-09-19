#pragma once
// The CSA2 attention layer of DeepSeek-V4-Flash (deepseek_v4)
// (2026-09-17, docs/dsv4_kernel_port_spec.md §2.1): the low-rank q, the
// window latent ring, the C4A/C128A compressor, the 64-head indexer
// (the top-k sparse selection) and the two-source attention (the window
// ring + the selected main rows, the sink in the denominator), the
// grouped wo. One object serves every backbone layer of a model: rebind()
// points it at a layer's weights and role; the three-class dispatch is
// the V4 layout's (SWA-only ratio 0 / C4A ratio 4 with the indexer /
// C128A ratio 128 — src/models/dsv4/config.hpp's is_index_layer /
// compressor_coff encode it).
//
// The V4 geometry deltas vs the dsv41 base (docs/dsv4_kernel_port_spec.md
// §1, the spec's work order): hidden 5120 -> 4096, q_lora 1280 -> 1024,
// the index heads 32 -> 64 (dim 128, topk 512 — the 64-head fold width
// the spec §2.1(d) flags as NEW: the shared csa2_select kernels pin 32
// index heads, so the V4 selection runs the v4-owned dsv4_csa2_select_*
// kernels below, the shared kernels' 64 x 64 SMEM tile exchange
// re-expressed for 64 heads), the ratio set {0,1,2} -> {0,4,128} (the
// ratio-4 compressor the overlapping variant coff 2, the ratio-128 plain
// coff 1 — the dsv41 ratio-2/ratio-1 pair, one geometry swap). The
// window 128, the latent 512, the rope tail 64 and the attention finish
// are unchanged from the dsv41 base (the shared csa2 / dsa kernels).
//
// The decode path is the hot path (the graph-capturable one); the prefill
// and the DSpark draft block follow the dsv41 layer's contract and are
// GPU-gate pending (the morning's completion — the decode path + the
// 64-head select + the compressor are the surface this pass lands).
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/csa2.hpp"
#include "kernels/gemm.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

// The V4 CSA2 geometry (the dsv4 config's attention block, the world-2
// deployment: the local_heads / local_groups are the per-rank slices).
struct Dsv4Csa2Config {
  bool dense_mma = true;
  int hidden = 4096;
  int q_lora = 1024;
  int o_lora = 1024;
  int num_heads = 64;
  int o_groups = 8;
  int index_heads = 64;  // the V4's 64 (the dsv41's 32 — the NEW fold width)
  int index_topk = 512;
  int window = 128;
  int ring_slots = 160;
  int block_tokens = 128;
  float eps = 1e-6f;
  int tp = 2;  // the world-2 deployment (the dsv41 family's world-4)
  int local_heads() const { return num_heads / tp; }
  int local_groups() const { return o_groups / tp; }
  int heads_per_group() const { return num_heads / o_groups; }
  static void validate(const Dsv4Csa2Config& c);
};

// The V4 layer's weight view (the dsv4 binding's Dsv4WeightClass's
// Attention / Indexer / Compressor planes, the 128 x 128 fp8 grid's
// F32-decoded scales — the spec §3.1's loader-side decode).
struct Dsv4Csa2LayerWeights {
  GlmQuantMatrix wq_a, wkv, wq_b, wo_a, wo_b;  // fp8, 128 x 128 grids
  const uint16_t* q_norm = nullptr;
  const uint16_t* kv_norm = nullptr;
  const float* attn_sink = nullptr;  // f32 [local_heads]
  GlmQuantMatrix idx_wq_b;  // the index sources (indexer.wq_b [8192, 1024])
  const uint16_t* idx_wp = nullptr;
  const uint16_t* idx_wk = nullptr;  // the kv sources
  const uint16_t* idx_k_norm = nullptr;
  const uint16_t* comp_wkv = nullptr;  // the kv sources (bf16 [W, hidden])
  const uint16_t* comp_wgate = nullptr;  // the ratio-4 (C4A) overlapping
  const uint16_t* comp_norm = nullptr;
  const float* inv_freq = nullptr;  // device [64]: the layer's rotary table
  int ratio = 0;  // 0: SWA-only, 4: C4A (the indexer), 128: C128A
  int cache_ord = -1;
  int tail_ord = -1;  // the compressor tail (a ratio-4 kv source)
  bool kv_source = false;
  bool index_source = false;
};

// The V4 CSA2 layer: composes the shared csa2 / dsa elementwise +
// attention kernels with the V4 geometry (the Dsv4Csa2Config) + the
// v4-owned 64-head selection (the shared csa2_select's 32-head pin's V4
// re-expression). The decode path (the hot path) + the compressor + the
// indexer are this pass's surface; the prefill / the DSpark draft block
// are GPU-gate pending (the morning's completion, the dsv41 layer's
// contract).
class Dsv4Csa2Layer {
 public:
  Dsv4Csa2Layer(IGemm& gemm, const Dsv4Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens,
                void* scratch, size_t scratch_capacity, void* gemm_workspace, size_t gemm_ws_bytes,
                int max_decode_rows = 16, int decode_n_split = 32, size_t dot_budget = 64ull << 20);
  static size_t scratch_bytes(const Dsv4Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens,
                              int max_decode_rows = 16, int decode_n_split = 32,
                              size_t dot_budget = 64ull << 20);

  // Points the object at a layer's weights and role (the three-class
  // dispatch's rebind: the ratio's validation, the projection geometry's
  // check against the config, the indexer / compressor's weight presence
  // per role).
  void rebind(const Dsv4Csa2LayerWeights& w, int layer);
  // The GEMM plans for `tokens` rows (outside capture).
  bool prepare(int tokens);
  // The decode batch (the hot path, graph-capturable): the projections,
  // the window ring's append (the layer's own ring, the no-pool positional
  // state), the compressor (the ratio-4's overlapping / the ratio-128's
  // plain), the indexer's 64-head query + the v4-owned 64-head selection,
  // the two-source attention (the window ring + the selected main rows, the
  // sink in the denominator), the grouped wo. `out`: bf16 [tokens, hidden].
  //
  // The caches (the model's, the no-pool positional state — the dsv41
  // pool's planes' raw-pointer re-expression):
  //   main_cache  the compressed main KV (the kv sources' entries), the
  //                kFp4Block main format (the scales' inside the row's,
  //                self-describing); null: the main source is skipped
  //                (the window-only attention, the SWA layers' form).
  //   index_cache / index_scale  the planar index cache (the e4m3 codes'
  //                [entries, 128] + the fp32 row-scale's [entries]), the
  //                the 64-head selection's stream; null: the selection's
  //                the main source's skipped (the window-only attention).
  // The window ring is the layer's own scratch (the no-pool model's
  // positional state, the model's "the window ring's the layer scratch's")
  // — it is always available, so the window source always runs.
  void enqueue_decode(const void* hidden_in, void* main_cache, void* index_cache, const float* index_scale,
                      const int32_t* req_ids, const int64_t* pos, const int32_t* req_spans,
                      int num_requests, int tokens, void* out, cudaStream_t stream,
                      float* tail_snapshots = nullptr);

  const Dsv4Csa2Config& config() const { return cfg_; }
  int layer() const { return layer_; }
  // The index-key / q exactness violations so far (a host read; synchronizes).
  unsigned index_violations() const;

 private:
  struct Layout;
  static Layout layout(const Dsv4Csa2Config& cfg, int max_tokens, int64_t max_cache_tokens, int max_decode_rows,
                       int decode_n_split, size_t dot_budget);
  // The projections and the window row of `tokens` rows at `pos` (device).
  void project_q_kv(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream);
  // The indexer's 64-head query: the wq_b's 64-head projection -> the
  // tail's rotation -> the fp8 quant -> the folded weights (the 64-head
  // fold width the spec §2.1(d) flags as NEW).
  void indexer_query(const void* hidden_in, int tokens, const int64_t* pos, cudaStream_t stream);
  // The decode window's key-dimension split (the dsv41 kWinDecodeSplit's
  // V4 re-expression): a one-row window is not a single latency-bound
  // block, so the window's 128 keys are split across the key dimension to
  // fill the SMs. A FIXED count, independent of the row count, so a token's
  // window is computed identically whether it is a one-row decode step or
  // row 0 of a multi-row MTP verify (the speculative transcript must equal
  // the plain greedy one).
  static constexpr int kWinDecodeSplit = 8;

  // The attention of the current call's rows: the window ring's partials
  // (always, the ring is the layer scratch's) and — when the model has
  // allocated the main cache (main_cache non-null, the kFp4Block main
  // format, self-describing) — the selected main rows' partials, the sink
  // in the denominator, the finish. `req_ids` the caller's (the decode
  // batch's, the member req_ids_'s the prefill staging's only).
  void attend(int tokens, const int64_t* pos, const int32_t* req_ids, void* main_cache, cudaStream_t stream);
  // The grouped wo_a / wo_b (the block-diagonal over the groups).
  void project_out(int tokens, void* out, cudaStream_t stream);

  IGemm& gemm_;
  Dsv4Csa2Config cfg_;
  Dsv4Csa2LayerWeights w_;
  int layer_ = -1;
  int max_tokens_ = 0;
  int64_t max_cache_tokens_ = 0;
  int64_t max_entries_ = 0;  // the round-up-to-256 entry capacity (the select's ws_max_entries's)
  int max_decode_rows_ = 16;
  int decode_n_split_ = 32;
  float attn_scale_ = 0.f;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  // The scratch's regions (the layout's offsets).
  uint8_t* scratch_ = nullptr;
  uint16_t* qr_ = nullptr;  // [T, q_lora]
  uint16_t* kv_ = nullptr;  // [T, 512]
  uint16_t* q_ = nullptr;   // [T, lh * 512]
  uint16_t* o_ = nullptr;   // [T, lh * 512]
  uint16_t* oa_ = nullptr;  // [T, lg * o_lora]
  uint16_t* idx_q_ = nullptr;  // [T, 64 * 128]
  uint8_t* q_fp8_ = nullptr;   // [T * 64, 128]
  float* q_scale_ = nullptr;   // [T * 64]
  uint16_t* iw_ = nullptr;     // [T, 64] the indexer's weights
  float* w_folded_ = nullptr;  // [T * 64]
  uint16_t* latent_ = nullptr; // [T, 512]
  uint16_t* ik_ = nullptr;     // [T, 128]
  int64_t* pos_ = nullptr;     // [T]
  int32_t* req_ids_ = nullptr; // [T]
  int64_t* slots_ = nullptr;   // [T] the ring's slots
  int32_t* topk_ = nullptr;    // [T, index_topk]
  int32_t* counts_ = nullptr;  // [T]
  float* m_main_ = nullptr;
  float* l_main_ = nullptr;
  float* c_main_ = nullptr;
  float* m_win_ = nullptr;
  float* l_win_ = nullptr;
  float* c_win_ = nullptr;
  // The window ring's per-call slot lists (csa2_window_slots_decode's
  // output) and the layer's own window ring (the no-pool positional state:
  // [max_decode_rows][ring_slots] rows of the fp8_block form, one block per
  // request — the dsv41 pool's ring's V4 re-expression, the model's
  // "the window ring's the layer scratch's"). max_decode_rows safely
  // upper-bounds the distinct-request count (the model's max_requests <=
  // decode_rows_cap() == max_decode_rows).
  int32_t* wlist_ = nullptr;  // [max_decode_rows, window] the slot lists
  int32_t* wcounts_ = nullptr;  // [max_decode_rows]
  uint8_t* ring_ = nullptr;  // [max_decode_rows, ring_slots] fp8_block rows
  int32_t* ring_table_ = nullptr;  // [max_decode_rows] identity (one block per request)
  int32_t* main_block_table_ = nullptr;  // [max_decode_rows, max_blocks] identity (the planar main cache's)
  int64_t* pos_sel_ = nullptr;  // [max_decode_rows] the compressed entries' visible position
  void* select_ws_ = nullptr;  // the v4 select's workspace (the dsa_select_workspace_bytes's)
  int32_t* counter_ws_ = nullptr;  // [2] the select's counters (zeroed once)
  int max_blocks_ = 0;  // the planar main cache's block count (max_cache_tokens / block_tokens)
  unsigned* violations_ = nullptr;
};

// ---- the v4-owned 64-head selection (the shared csa2_select's 32-head
// pin's V4 re-expression; the spec §2.1(d)'s NEW 64-index-head fold
// width) ---------------------------------------------------------------
// The decode's 64-head selection: streams the index cache's visible
// entries per row, computes the 64-head fp8 dots' logits (the folded
// weight, the entry's scale), and keeps the running top-select_k
// composite-key selection (the (sortable_fp32 << 21) | entry_idx's total
// order, the exact ties to the lower entry index — the shared kernel's
// selection's 64-head re-expression). `heads` MUST be 64 (the V4's
// index_n_heads; the shared kernel's 32). `q_fp8` [rows, 64, 128] e4m3,
// `w_folded` [rows, 64] fp32, `index_k` / `index_scale` the planar index
// cache, `topk_out` [rows, select_k] (-1 padded), `counts` [rows].
void dsv4_csa2_select_decode(const void* q_fp8, const float* w_folded, const int32_t* req_ids,
                             const int64_t* pos_sel, int rows, const int32_t* block_tables,
                             int blocks_per_request, const void* index_k, const float* index_scale,
                             int entries_per_block, int select_k, int32_t* topk_out, int32_t* counts,
                             void* select_ws, int64_t ws_max_entries, int32_t* counter_ws, cudaStream_t stream);
// The prefill's 64-head selection over the materialized dot buffer (the
// per-row logits' top-select_k, the same composite-key's total order).
// `dot` fp32 [rows * 64, dot_stride], `logits` the per-row's [rows,
// n_entries] (the caller's), `topk_out` / `counts` as the decode's.
void dsv4_csa2_select_prefill(const float* dot, int64_t dot_stride, const float* w_folded,
                              const float* k_scale, const int64_t* pos_sel, int rows, int64_t n_entries,
                              int select_k, int32_t* topk_out, int32_t* counts, cudaStream_t stream);

}  // namespace dgpp
