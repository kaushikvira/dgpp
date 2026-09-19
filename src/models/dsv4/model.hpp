#pragma once
// DeepSeek-V4-Flash (DeepseekV4ForCausalLM) on the shared session core
// (2026-09-17, the serve integration; docs/dsv4_kernel_port_spec.md): the
// dsv41 model's structure (src/models/dsv41/model.hpp) re-expressed on the
// committed V4 layer surfaces — the CSA2 attention's decode path (the
// window ring, the C4A/C128A compressor, the 64-head indexer, the
// two-source attention's finish, the grouped wo; csa2_layer.hpp), the
// hash-routing MoE (the tid2eid table on the first num_hash_layers, the
// fused top-k router + the MXFP4 expert elsewhere; hash_layer.hpp) and
// the DSpark draft/verify glue (the target layers' stream mean, the block
// rows, the Markov-biased head, the confidence logit; dspark_layer.hpp) —
// at the V4 geometry (hidden 4096, 43 backbone layers, 256 experts,
// moe intermediate 2048, q_lora 1024, 64 index heads (dim 128, topk
// 512), the 3 `mtp.N.` draft stages, the fp8 128 x 128 dense grid (the
// e8m0 scales' loader-side F32 decode), the MXFP4 experts).
//
// One row walk (run_rows) serves every entry point; the session core owns
// everything around it. One layer: the mHC attention site (the
// coefficients, the collapse, the norm), the CSA2 decode enqueue (the
// projections, the window ring's append, the compressor, the 64-head
// indexer, the attention's finish, the grouped wo), the mHC ffn site,
// the hash router's gate scores + the fused top-k + the MXFP4 expert, the
// two folds and the stream update. The head: the weighted collapse, the
// final norm, the lm head on every walked row.
//
// Prefill chunks (prefill_chunk_tokens, the CSA2 decode enqueue's row
// bound) ride the decode path: the committed CSA2 surface's prefill
// enqueue and the DSpark draft stages' union attention (the
// [compressed pool | raw ring | block]'s 3-phase's single softmax's,
// dsv4_dspark_union_attn) — the model now calls the union attention on
// the decode/draft path (the 2026-09-18's dsv4 dspark + compressor
// wiring's the enqueue_layer's the draft stages' the SWA-only's the no-
// compressed-phase's, the no-raw-ring's, the in-memory block's), the q
// latent's + the block kv's stand in the csa2 seam's (the parallel
// agent's the csa2 partial fills's) until the wiring's complete's — the
// model passes the planar cache pointers
// (nullptr for now, the layer's the caller's contract's) and runs the
// draft stages' walk on the CSA2 decode enqueue. The numerics' the
// committed surface's (the parity gate's the GPU's the morning's
// completion's).
//
// No paged pool: the attention's state is positional (the window ring's
// the layer scratch's, the slot's the position's the ring's slot's), so
// the positional rings need no snapshot (a rejected draft's slot is
// never read by a later query). The ratio-4 (C4A) overlapping
// compressor's per-request tails (the 2026-09-18's dsv4 dspark +
// compressor wiring's, the dsv41 csa2_compress_decode_update's
// re-expression's, the fp32 [2, 1024]'s the pending even's kv + the
// score's) are the only per-request state (the snapshot_state_bytes'
// the tails' bytes's, the spec segments' the tails' the rollback's table
// the 2026-09-18's, the write / read's the snapshot's the copy's) and
// the prefix cache's off's (the arena's 0's slots' the state's 0's
// bytes's).
//
// TP: `tp_world` > 1 loads this rank's slices (64/W heads and 8/W output
// groups, every expert's and the shared expert's I/W intermediate slice,
// the lm-head vocab slice, the embedding's vocab sharding) and folds the
// attention's wo_b partial and the MoE partial per site through
// `boundary`; the indexer, the compressor, the router, the mHC
// coefficients and the norms are replicated — every rank's streams are
// bitwise the others'. The hash gate tables (the num_hash_layers'
// tid2eid's) are whole on every rank (the rows' token ids, no head
// dimension, so no per-rank split at any world): the loader keeps them
// mmap'ed in their shards, the model copies each whole table to the
// device at construction (the router kernel's the full table's the
// device pointer's read's, the token id's the row's).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "engine/boundary_reducer.hpp"
#include "engine/decode_outputs.hpp"
#include "engine/memory_plan.hpp"
#include "engine/session_model.hpp"
#include "kernels/gemm.hpp"
#include "kernels/glm_mhc_launch.hpp"
#include "kernels/glm_spec.hpp"
#include "models/dsv4/config.hpp"
#include "models/dsv4/csa2_layer.hpp"
#include "models/dsv4/dspark_layer.hpp"
#include "models/dsv4/hash_layer.hpp"
#include "models/dsv4/loader.hpp"
#include "models/glm/mhc.hpp"

namespace dgpp {

// The no-pool model's pool stand-in (the session core's paged-pool protocol
// — PagedBlockTable's methods + copy_block_contents — as no-ops): the
// attention's state is positional (the window ring's the layer scratch's),
// so no per-request blocks are ever reserved or copied; `has_pool()` stays
// false and every method is unreachable (the core's short-circuit's).
struct Dsv4NoopPool {
  bool ensure_request_blocks(int, int64_t, cudaStream_t) { return false; }
  void copy_block_contents(int32_t, int32_t, cudaStream_t) {}
  int64_t total_blocks() const { return 0; }
  int64_t blocks_in_use() const { return 0; }
  int64_t block_count_for_tokens(int64_t) const { return 0; }
  const int32_t* request_table_row(int) const { return nullptr; }
  bool share_blocks_into(int, const int32_t*, int64_t, cudaStream_t) { return false; }
  void pin_blocks(const int32_t*, int64_t) {}
  void unpin_blocks(const int32_t*, int64_t) {}
  int32_t acquire_pinned_block() { return -1; }
};

class Dsv4Model : public SessionModel<Dsv4Model> {
 public:
  using Base = SessionModel<Dsv4Model>;
  using Outputs = Base::Outputs;
  using SessionSnapshotMeta = Base::SessionSnapshotMeta;
  using SnapshotRequest = Base::SnapshotRequest;
  using RowRun = Base::RowRun;

  Dsv4Model(const Dsv4TextConfig& cfg, const std::string& checkpoint_dir, int max_tokens, int64_t max_cache_tokens,
            Dsv4Residency residency = Dsv4Residency::Streaming, BoundaryReducer* boundary = nullptr, int tp_rank = 0,
            int tp_world = 1, int max_requests = 1, bool mtp = false, int decode_rows = 0);
  ~Dsv4Model();
  Dsv4Model(const Dsv4Model&) = delete;
  Dsv4Model& operator=(const Dsv4Model&) = delete;

  using MemoryPlan = dgpp::MemoryPlan;
  static MemoryPlan plan_memory(const Dsv4TextConfig& cfg, int max_tokens, int64_t max_cache_tokens, int tp_rank = 0,
                                int tp_world = 1, Dsv4Residency residency = Dsv4Residency::Streaming,
                                int max_requests = 1, bool mtp = false, int decode_rows = 0);

  // The prefill chunk (the CSA2 decode enqueue's row bound — the committed
  // surface's prefill enqueue's the GPU-gate pending's completion's):
  // every prompt chunk's the decode path's (the window ring's the
  // positional's, the main cache's wiring's pending's).
  static constexpr int prefill_chunk_tokens() { return kPrefillChunkTokens; }
  static constexpr int kv_block_tokens_static() { return kBlockTokens; }
  // The decode batch's ceiling (the CSA2 layer's decode row bound's, the
  // kDecodeRowsMax's mirror's): six request slots at DSpark depth 4
  // (30 rows) in one batched replay.
  static constexpr int decode_rows_cap() { return kDecodeRowsCap; }
  static size_t session_snapshot_bytes(const Dsv4TextConfig& cfg, int tp_world, bool mtp);
  using Base::session_snapshot_bytes;
  static Dsv4Csa2Config csa2_config(const Dsv4TextConfig& cfg, int tp_world);
  static Dsv4HashConfig hash_config(const Dsv4TextConfig& cfg, int tp_world);
  static Dsv4DsparkConfig dspark_config(const Dsv4TextConfig& cfg, int targets, int lm_vocab_begin, int lm_vocab_count);

  const Dsv4TextConfig& config() const { return cfg_; }

  // The cold diagnostic forward: one request on slot 0, fresh state, the
  // prompt's chunks at the prefill chunk's (the last row's logits, the
  // parity gate's entry point's).
  Outputs forward(const std::vector<int64_t>& token_ids);

  // ---- the session core's hooks ----------------------------------------------
  Outputs run_rows(const RowRun& run);
  void reset_slot_state(int req);
  GlmSpecSegments spec_segments(int req, int snapshot_row0 = 0) const;
  size_t snapshot_state_bytes() const;
  size_t draft_state_bytes() const { return 0; }
  void write_state_snapshot(int req, uint8_t* dst, int spec_row);
  void write_draft_snapshot(int, uint8_t*, bool, int64_t) {}
  void read_state_snapshot(int req, const uint8_t* src);
  void read_draft_snapshot(int, const uint8_t*) {}
  bool has_pool() const { return false; }
  Dsv4NoopPool& pool() { return pool_; }
  const Dsv4NoopPool& pool() const { return pool_; }
  void graph_prepare();
  void mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row, bool capture,
                    int head_rows, int batch_requests);
  void snapshot_draft_state(int) {}
  void restore_draft_state(int) { draft_row_ = 0; }
  static constexpr bool kDraftChain = true;
  static constexpr bool kBatchedDraftChain = true;
  // The DSpark confidence head (engine/verify_schedule.hpp): the block's
  // per-position acceptance logits, [max_requests][block] on the device,
  // written by the step's draft (row 0 by the block, row c by chain row c).
  static constexpr bool kVerifyConfidence = true;
  int confidence_rows() const { return cfg_.dspark_block_size; }
  const float* device_confidence() const { return conf_; }
  // The block's output rows [T, draft_width] of its last run (the chain
  // rows' the hidden's the window's the store's).
  const uint16_t* draft_hidden_rows() const { return main_gather_; }
  void snapshot_chain_state(int) {}
  void restore_chain_state(int) {}

 private:
  static constexpr int kBlockTokens = 128;
  static constexpr int kPrefillChunkTokens = 32;  // the decode enqueue's bound
  static constexpr int kDecodeRowsCap = 32;
  static constexpr int kDecodeSplit = 32;
  static constexpr size_t kDotBudget = 64ull << 20;

  struct WalkRows {
    const int64_t* tokens = nullptr;
    const int32_t* req_ids = nullptr;
    const int64_t* pos = nullptr;
    const int32_t* spans = nullptr;
    int num_requests = 1;
    int req = 0;
    int64_t pos0 = 0;
    bool decode = false;
    bool capture = false;
  };
  void build_layer_objects(const Dsv4LayerResident& r, int layer);
  Dsv4Csa2LayerWeights csa2_view(const Dsv4LayerResident& r, int layer) const;
  Dsv4HashWeights hash_view(const Dsv4LayerResident& r, int layer) const;
  // The DSpark stage's weight view (the main projection's the stage 0's,
  // the head tensors' the last stage's — the layer's rebind's both's):
  // assembled at construction (resident's the draft stages' the stable's).
  Dsv4DsparkWeights dspark_view(const Dsv4LayerResident& first, const Dsv4LayerResident& last, int stage) const;
  int target_ordinal(int layer) const;
  const Dsv4DraftResident& draft_stage(int stage);
  // The layer over the streams (cur -> next, swapped), the two folds.
  void enqueue_layer(const Dsv4LayerResident& r, int layer, int T, const WalkRows& rows);
  void mhc_site(const uint16_t* streams, const GlmMhcWeights& w, const uint16_t* ln, int T, bool decode);
  void stream_update(const uint16_t* sublayer_out, int T);
  uint16_t* stage(uint16_t* fallback, int T, int width, bool capture);
  void fold(uint16_t* buf, int T, int width, bool capture);
  void gather_embedding(const int64_t* tokens, int T, bool capture);
  // The DSpark pieces (mtp): the first draft call of a step (main_x, the
  // block, base_logits_, row 0) and a chain row (block row draft_row_).
  void draft_first(int req, const int64_t* tokens, const int64_t* d_pos, const int32_t* d_req, int T, bool decode_row,
                   bool capture, int head_rows, int batch_requests);
  void draft_chain_row(int req, const int64_t* tokens, bool capture, int head_rows, int batch_requests);
  // The scratch byte formulas (the layer surfaces' layout's mirror's, the
  // memory plan's and the constructor's share them).
  static size_t hash_scratch_bytes(const Dsv4TextConfig& cfg, int tp_world, int max_tokens);
  static size_t dspark_scratch_bytes(const Dsv4TextConfig& cfg, int lm_vocab_count, int max_rows);

  Dsv4TextConfig cfg_;
  Dsv4LayerStream loader_;
  CublasLtGemm gemm_;
  bool dense_mma_ = true;  // the dense projections' decode form (DGPP_DSV4_DENSE_GEMV=1: the GEMV chunks)
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  Dsv4GlobalsResident globals_;
  GlmMhcConfig mhc_cfg_;
  Dsv4Csa2Config csa2_cfg_;
  Dsv4HashConfig hash_cfg_;
  Dsv4DsparkConfig dspark_cfg_;
  bool embed_sharded_ = false;
  int world_ = 1;

  std::unique_ptr<Dsv4Csa2Layer> csa2_;
  std::unique_ptr<Dsv4HashLayer> moe_;
  std::unique_ptr<Dsv4DsparkLayer> dspark_;
  Dsv4NoopPool pool_;  // the no-pool stand-in (the session core's protocol's)
  Dsv4DsparkWeights dspark_w_;  // the DSpark stage's view (rebound at construction, mtp)
  void* csa2_scratch_ = nullptr;
  size_t csa2_scratch_bytes_ = 0;
  void* hash_scratch_ = nullptr;
  size_t hash_scratch_bytes_ = 0;
  void* dspark_scratch_ = nullptr;
  size_t dspark_scratch_bytes_ = 0;
  // The DSpark union attention's staging (2026-09-18, the dsv4 dspark +
  // compressor wiring; the dsv4_dspark_union_attn's the 3-phase's single
  // softmax's the q latent's + the in-memory block's kv's + the out latent's,
  // the 512-dim's the dsv4 DSpark's kHeadDim's): the csa2 projection's
  // outputs (the q latent's, the block kv's) stand in the csa2 seam's
  // (the parallel agent's the csa2 partial fills's) until the wiring's
  // complete's.
  void* union_attn_scratch_ = nullptr;
  size_t union_attn_scratch_bytes_ = 0;
  float* inv_freq_window_ = nullptr;      // device [32]
  float* inv_freq_compressed_ = nullptr;  // device [32]
  std::vector<int> cache_ord_;            // per layer (-1: window only)
  std::vector<int> tail_ord_;             // per ratio-4 kv source (-1: none)
  int tails_ = 0;
  // The ratio-4 (C4A) overlapping compressor's per-request tails (2026-09-18,
  // the dsv4 dspark + compressor wiring; docs/dsv4_kernel_port_spec.md §2.1(b)):
  // fp32 [tails_][max_requests][2][tails_w_] (tails_w_ = kCsa2TailW = 1024,
  // the C4A's coff x kCsa2Latent's — the pending even's kv (the first W) + the
  // score (the second W), the dsv41 csa2_compress_decode_update's re-expression
  // on the C4A's width). The spec rows'
  // tails (the rollback's the table's the spec_rows_').
  float* d_tails_ = nullptr;
  float* spec_tails_ = nullptr;
  int tails_w_ = kCsa2TailW;  // the W's (the C4A's compressor's output width's, the 1024's)
  int targets_ = 0;                       // dspark target layers (the draft width is targets_ * H)
  int draft_row_ = 0;                     // the next block row a chain call emits (0: no block stands)
  // The hash gate tables' device copies (the num_hash_layers' tid2eid's
  // whole tables, the router kernel's the full table's read's):
  // [num_hash_layers][vocab, top_k] I64.
  std::vector<int64_t*> d_tid2eid_;

  // Activations [M rows].
  uint16_t* streams_a_ = nullptr;   // [M, 4, H]
  uint16_t* streams_b_ = nullptr;
  uint16_t* cur_ = nullptr;         // the live streams (one of the two)
  uint16_t* nxt_ = nullptr;
  uint16_t* x_ = nullptr;           // [M, H] the normed sublayer input
  uint16_t* y_ = nullptr;           // [M, H] the block output (the fold's fallback)
  uint16_t* collapsed_ = nullptr;   // [M, H]
  uint16_t* post_bf16_ = nullptr;   // [M, 4]
  uint16_t* comb_bf16_ = nullptr;   // [M, 16]
  float* post_f32_ = nullptr;       // [M, 4]
  float* comb_f32_ = nullptr;       // [M, 16]
  float* pre_a_ = nullptr;          // [M, 4] the collapse coefficients in use
  float* pre_b_ = nullptr;          // [M, 4] the next site's
  float* pre_cur_ = nullptr;
  float* pre_nxt_ = nullptr;
  float* one_hot_ = nullptr;        // [M, 4] = (1, 0, 0, 0): layer 0's collapse
  float* mhc_logits_ = nullptr;     // [M, 24]
  uint16_t* gate_logits_ = nullptr;  // [M, n_experts] the router's scores
  int32_t* topk_ids_ = nullptr;     // [M, top_k]
  float* topk_w_ = nullptr;         // [M, top_k]
  // The DSpark pieces (mtp). `h_` / `logits_` / `h_tail_logits_` are the
  // session core's (the head's outputs); the draft's own buffers below.
  uint16_t* main_hidden_ = nullptr;  // [M, targets * H] the target layers' stream means of the walk's rows
  uint16_t* main_gather_ = nullptr;  // [rows, targets * H] the draft rows' gathered main hidden
  uint16_t* main_x_ = nullptr;       // [max(M, rows), H] main_norm(main_proj(main hidden))
  int64_t* blk_pos_ = nullptr;       // [rows] the block rows' positions
  int64_t* blk_tok_ = nullptr;       // [rows] the block rows' tokens
  int32_t* blk_req_ = nullptr;       // [rows]
  int32_t* blk_spans_ = nullptr;     // [rows + 1]
  float* base_logits_ = nullptr;     // [rows, vocab slice] the block's head rows
  float* conf_ = nullptr;            // [max_requests, block] the confidence logits
  int64_t* d_draft_pos_ = nullptr;   // [M] the prefill draft's staged positions
  int32_t* d_draft_req_ = nullptr;   // [M] the prefill draft's staged req ids
};

}  // namespace dgpp
