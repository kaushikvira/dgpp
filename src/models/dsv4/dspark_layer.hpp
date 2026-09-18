#pragma once
// The DSpark draft/verify glue of DeepSeek-V4-Flash (deepseek_v4)
// (2026-09-17, docs/dsv4_kernel_port_spec.md §2.2): the target layers'
// stream mean, the block rows off the accepted verify rows, the
// Markov-biased head row and the confidence logit — the dsv41 DSpark
// glue (src/kernels/dsv41_dspark.hpp) re-expressed for the V4 geometry
// (hidden 4096, the markov rank 256, the 3 targets [40, 41, 42], the
// draft block 5) — plus the V4's DSpark union attention (the 3-phase
// single-softmax-over-the-union's [compressed | raw ring | block]'s,
// the dsv4-native dspark_attn op's V4 re-expression; the shared dgpp
// kernels have no union-attention yet, so the v4-owned
// dsv4_dspark_union_attn kernel below is the surface). One object
// serves the model's DSpark draft stages; rebind() points it at a
// stage's weights. The decode path is graph-capturable; the numerics
// of every op are certified against the CPU oracles (tests/unit/
// dsv4_dspark_oracle_test.cpp) — the GPU-gate pending runs the parity
// gate.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/dsv41_dspark.hpp"
#include "kernels/gemm.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

// The V4 DSpark geometry (the dsv4 config's DSpark block, the world-2
// deployment). The targets' the stream-mean's layers (the backbone's
// [40, 41, 42]); the markov rank's 256 (the dsv41's 128's NEW fold);
// the draft block's 5 (the verify's m = 6's, the M = 8's future's).
struct Dsv4DsparkConfig {
  int hidden = 4096;
  int num_targets = 3;  // the target layers' count (the stream mean's hc_mult)
  std::vector<int> target_layer_ids = {40, 41, 42};  // the backbone's targets
  int markov_rank = 256;  // the V4's markov rank (the dsv41's 128's NEW fold)
  int block_size = 5;  // the draft block (the verify's m = 6)
  int noise_token_id = 0;
  int lm_vocab_begin = 0;  // this rank's vocab slice
  int lm_vocab_count = 0;
  int window = 128;  // the raw ring's window (the dspark_attn's kSwaWindow)
  int max_block = 8;  // the dspark_attn's kMaxBlock (the m <= 8's decode's)
  int max_tokens = 4096;  // the dspark_attn's kMaxTokens (the compressed's raw's)
  float eps = 1e-6f;
  // The 64 heads' the 512-dim latent's (the dsv4-native dspark_attn's
  // kHeads / kHeadDim, the MLA's geometry).
  static constexpr int kHeads = 64;
  static constexpr int kHeadDim = 512;  // 448 NoPE + 64 RoPE
  int local_heads() const { return kHeads; }  // the MLA's single latent KV head (the replicated's)
  static void validate(const Dsv4DsparkConfig& c);
};

// The V4 DSpark stage's weight view (the dsv4 binding's DSparkBlock's
// planes: the main projection's fp8 128 x 128 grid, the markov's
// bf16 embed / head / confidence, the lm head's fp8 grid, the attn
// sink's f32 [64] — the spec §3.1's loader-side decode).
struct Dsv4DsparkWeights {
  GlmQuantMatrix main_proj;  // the main projection (fp8, 128 x 128 grid)
  const uint16_t* main_norm = nullptr;  // the main norm's bf16 [hidden]
  const uint16_t* norm = nullptr;  // the draft's norm's bf16 [hidden]
  const uint16_t* markov_embed = nullptr;  // the markov's embedding's bf16 [vocab, rank]
  const uint16_t* markov_head = nullptr;  // the markov's head's bf16 [vocab, rank]
  const float* confidence = nullptr;  // the confidence's f32 [hidden + rank]
  GlmQuantMatrix lm_head;  // the shared lm head's fp8 grid
  const float* attn_sink = nullptr;  // the dspark_attn's sink's f32 [64] (nullable)
  int stage = 0;  // the draft stage's ordinal (0..num_nextn - 1)
};

// The DSpark draft/verify glue (the dsv41 model's draft_first's V4
// re-expression, the world-2 deployment). The stream mean's the target
// layers' attention input's; the block rows' the [next, noise, ...]'s
// layout's; the markov bias's the head row's; the confidence's the
// block row's; the union attention's the 3-phase's single softmax's.
class Dsv4DsparkLayer {
 public:
  Dsv4DsparkLayer(IGemm& gemm, const Dsv4DsparkConfig& cfg, int max_rows, void* scratch, size_t scratch_capacity,
                  void* gemm_workspace, size_t gemm_ws_bytes);

  void rebind(const Dsv4DsparkWeights& w, int layer);
  bool prepare(int rows);

  // The target layers' stream mean (the dsv41_stream_mean_bf16's): the
  // hc_mult = num_targets' streams' the fp32 sum in stream order's, the
  // one-rounding bf16 mean's.
  void stream_mean(const void* streams, int rows, void* out, cudaStream_t stream);
  // The block rows (the dsv41_dspark_block_rows's): the [next, noise,
  // ...]'s layout off the accepted verify rows' (the P = 1 + max's, the
  // -1's padding's).
  void block_rows(const int64_t* step_pos, const int64_t* tokens, const int32_t* req_ids, int groups,
                  int rows_per_group, int64_t* pos_out, int64_t* tok_out, int32_t* req_out, int32_t* spans_out,
                  cudaStream_t stream);
  // The Markov-biased head row (the dsv41_dspark_markov_bias's) + the
  // confidence logit (the dsv41_dspark_confidence's).
  void markov_bias(const float* base, int block_row, const int64_t* tok, int tok_stride, int groups, float* out,
                    int64_t out_group_stride, int rows_out, cudaStream_t stream);
  void confidence(const void* x, int block_row, const int64_t* tok, int tok_stride, int groups, float* conf_out,
                  int conf_stride, cudaStream_t stream);
  // The DSpark union attention (the v4-owned dsv4_dspark_union_attn's):
  // the 3-phase's single softmax over [compressed | raw ring | block]'s,
  // the shared online-softmax state's, the sink's exactly-once's.
  void union_attn(const void* q_latent, const void* pool, const int32_t* kv_slots, int n_comp, const void* raw_ring,
                  int raw_n, const void* block_kv, int n_block, void* out_latent, int rows, cudaStream_t stream);

  const Dsv4DsparkConfig& config() const { return cfg_; }
  int layer() const { return layer_; }

 private:
  struct Layout;
  static Layout layout(const Dsv4DsparkConfig& cfg, int max_rows);
  IGemm& gemm_;
  Dsv4DsparkConfig cfg_;
  int max_rows_;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  Dsv4DsparkWeights w_;
  int layer_ = 0;
  // The scratch's (the stream mean's out's, the block rows's, the
  // union attention's q staging's).
  uint8_t* scratch_ = nullptr;
  uint16_t* stream_mean_ = nullptr;
  int64_t* blk_pos_ = nullptr;
  int64_t* blk_tok_ = nullptr;
  int32_t* blk_req_ = nullptr;
  int32_t* blk_spans_ = nullptr;
  float* base_logits_ = nullptr;
  float* conf_ = nullptr;
};

// The v4-owned DSpark union attention kernel (the dsv4-native
// dspark_attn op's V4 re-expression; the 3-phase's single softmax's).
// The one CTA per row's 512 threads' the thread t's owns (head t/8,
// dim-chunk t%8) -> 64 output dims' (the C5.5's CTA geometry's); the
// q's staged in SMEM once's (read once across all phases's, the
// bandwidth-first's rule's); the dequantized KV chunk's staged in
// SMEM's (the fp8's e4m3 x ue8m0's exact in fp32's, D19's; the
// block's bf16 -> fp32 upcast's exact's). The shared online-softmax
// state's (the running max m + normalizer l + the 512-dim fp32
// accumulator's) spans all 3 phases' (the single softmax's, no
// per-phase rescale boundary's). The sink's (nullable's, the [64]
// fp32's) initializes the state before phase 1's (m = sink[h], l = 1's,
// the sink's mass's the merged denominator's exactly-once's; null's =
// the pre-sink numerics's m = -inf, l = 0's). The numerics are
// certified against the CPU oracle (tests/unit/dsv4_dspark_oracle_test
// .cpp's dsv4_dspark_union_attn) — the GPU-gate pending's parity gate.
void dsv4_dspark_union_attn(const void* q_latent, const void* pool, const int32_t* kv_slots, int n_comp,
                            const void* raw_ring, int raw_n, const void* block_kv, int n_block, void* out_latent,
                            int rows, const float* attn_sink, cudaStream_t stream);

}  // namespace dgpp
