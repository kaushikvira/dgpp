#pragma once
// DeepSeek-V4-Flash weight loader (Phase B step 1, 2026-09-17) using
// ResidentLayerStream — the dsv41 loader's structure (src/models/dsv41/
// loader.{hpp,cpp}) adapted to the flat V4 family: the same shared stream
// (allocations, staging, resident images, byte accounting, digests) and
// the same per-class slice formulas (plan §2 of docs/deepseek_v41_flash_
// plan.md, re-derived for V4's geometry — T=2 is the only deployment
// world). V4 deltas vs dsv41:
// - the fp8 dense projections ride the ORIGINAL 128 x 128 grid (dsv41
//   added the 32 x 32 one): the checkpoint's F8_E8M0 scales are decoded
//   to the F32 grid the 128 x 128 fp8 kernels read (launch_scale_gemm_
//   f32, rs = cs = 7) — the scale-format gap of
//   docs/dsv4_kernel_port_spec.md §3, closed loader-side, no kernel edit.
// - MXFP4 routed experts (e2m1 pairs + e8m0 per 32) kept as they are,
//   sliced 2-way on the intermediate dim; the draft stages reuse the
//   backbone's expert set (no draft variant).
// - no Engram: V4's "hash" surface is the num_hash_layers gate tables
//   ffn.gate.tid2eid I64 [vocab, top-k] — token-id routing, NO head
//   dimension, so (unlike dsv41's head-sharded Engram tables) there is
//   no per-rank row split: every rank maps the whole table. The tables
//   stay in their shards, mapped read-only like dsv41's Engram tables.
//
// Placement (every slice a formula in the world size W, dsv41 plan §2,
// re-derived for V4):
//   attention: wq_a and wkv replicated; wq_b rows per head block (64/W
//              heads); wo_a rows per output group (o_groups/W groups);
//              wo_b packed columns of those groups; attn_sink per local
//              head; the norms replicated.
//   indexer:   replicated (wq_b, weights_proj, the indexer's own
//              compressor's ape / wkv / wgate / norm), ratio-4 layers.
//   compressor: replicated (ape, wkv, wgate, norm), ratio > 0 layers.
//   MoE:       router and its bias (or the mapped tid2eid table on the
//              first num_hash_layers) replicated; every routed expert
//              and the shared expert sliced on the intermediate dim at
//              I/W (w1/w3 rows, w2 columns on a 32-block boundary).
//   mHC:       replicated (fn rounded fp32 -> bf16 at load, base /
//              scale fp32; the model-level hc_head in the globals).
//   hash:      the tid2eid tables stay in their shards, mapped.
//   draft:     stage 0's main_proj / main_norm, the last stage's norm,
//              Markov head, confidence head and hc_head replicated (a
//              layer's bytes cannot depend on the head sharding — the
//              stream's counting pass has none — so the [vocab, rank]
//              head is held whole and its lm-head rows are addressed at
//              use).
//   globals:   embed replicated (a row gather) or vocab-sharded, the
//              final norm, head vocab-sharded under VocabSharded.
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/dtypes.hpp"
#include "loaders/resident_stream.hpp"
#include "loaders/safetensors.hpp"
#include "loaders/weight_build.hpp"
#include "models/dsv4/binding.hpp"
#include "models/dsv4/config.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

using Dsv4Residency = LoaderResidency;
using Dsv4HeadSharding = LoaderHeadSharding;
using Dsv4ReplicatedDigest = ReplicatedDigest;

constexpr int kDsv4Fp8Block = 128;  // the release's fp8 grid (both axes, rs = cs = 7)

struct Dsv4AttnResident {
  GlmQuantMatrix wq_a;   // fp8 [q_lora, hidden], replicated
  GlmQuantMatrix wkv;    // fp8 [head_dim, hidden], replicated
  GlmQuantMatrix wq_b;   // fp8 [local_heads * head_dim, q_lora]
  GlmQuantMatrix wo_a;   // fp8 [local_groups * o_lora, heads * head_dim / o_groups]
  GlmQuantMatrix wo_b;   // fp8 [hidden, local_groups * o_lora] (packed columns)
  const uint16_t* q_norm = nullptr;   // bf16 [q_lora]
  const uint16_t* kv_norm = nullptr;  // bf16 [head_dim]
  const float* attn_sink = nullptr;   // f32 [local_heads]
  // The indexer (ratio-4 layers): the fp8 query projection, the BF16
  // weights projection and the indexer's own 128-wide rotated compressor.
  GlmQuantMatrix idx_wq_b;               // fp8 [index_heads * 128, q_lora]
  const uint16_t* idx_wp = nullptr;      // bf16 [index_heads, hidden]
  const float* idx_comp_ape = nullptr;   // f32 [4, 2 * index_head_dim]
  const uint16_t* idx_comp_wkv = nullptr;   // bf16 [2 * index_head_dim, hidden]
  const uint16_t* idx_comp_wgate = nullptr; // bf16 [2 * index_head_dim, hidden]
  const uint16_t* idx_comp_norm = nullptr;  // bf16 [index_head_dim]
  // The main KV compressor (ratio > 0): coff 2 at ratio 4 (overlapping),
  // coff 1 at ratio 128.
  const float* comp_ape = nullptr;       // f32 [ratio, coff * head_dim]
  const uint16_t* comp_wkv = nullptr;    // bf16 [coff * head_dim, hidden]
  const uint16_t* comp_wgate = nullptr;  // bf16 [coff * head_dim, hidden]
  const uint16_t* comp_norm = nullptr;   // bf16 [head_dim]
  int local_heads = 0;
  int head_begin = 0;
  int local_groups = 0;
  int group_begin = 0;
  bool index_source() const { return idx_wq_b.payload != nullptr; }
  bool kv_source() const { return comp_wkv != nullptr; }
};

struct Dsv4MoeResident {
  const uint16_t* router = nullptr;     // bf16 [E, hidden]
  const float* router_bias = nullptr;   // f32 [E] (null on the hash layers,
                                        // which route through the mapped tid2eid table)
  GlmQuantMatrix shared[3];             // fp8: w1, w3 [S/W, hidden]; w2 [hidden, S/W]
  std::vector<GlmFp4Matrix> experts;    // [E * 3]: w1, w3, w2 per expert (MXFP4, inter-sliced)
  int n_experts = 0;
  int64_t local_inter = 0;              // I/W
  int64_t local_shared_inter = 0;       // S/W
  // The view-table order the MoE layer expects: gate (w1), up (w3), down (w2).
  const GlmFp4Matrix& expert(int e, int i) const { return experts[static_cast<size_t>(e) * 3 + i]; }
};

struct Dsv4MhcResident {
  const uint16_t* attn_fn = nullptr;  // bf16 [24, 4 * hidden] (fp32 in the file, rounded at load)
  const float* attn_base = nullptr;   // f32 [24]
  const float* attn_scale = nullptr;  // f32 [3]
  const uint16_t* ffn_fn = nullptr;
  const float* ffn_base = nullptr;
  const float* ffn_scale = nullptr;
};

struct Dsv4DraftResident {
  GlmQuantMatrix main_proj;                 // stage 0: fp8 [hidden, targets * hidden]
  const uint16_t* main_norm = nullptr;      // stage 0: bf16 [hidden]
  const uint16_t* norm = nullptr;           // last stage: bf16 [hidden]
  const uint16_t* markov_w1 = nullptr;      // last stage: bf16 [vocab, rank], replicated
  const uint16_t* markov_w2 = nullptr;      // last stage: bf16 [vocab, rank], replicated
  const float* confidence = nullptr;        // last stage: f32 [hidden + rank] (bf16 in the file)
  const uint16_t* hc_head_fn = nullptr;     // last stage: bf16 [hc_mult, hc_dim]
  const float* hc_head_base = nullptr;      // last stage: f32 [hc_mult]
  const float* hc_head_scale = nullptr;     // last stage: f32 [1]
};

struct Dsv4LayerResident {
  int layer = -1;
  const uint16_t* attn_norm = nullptr;  // bf16 [hidden]
  const uint16_t* ffn_norm = nullptr;   // bf16 [hidden]
  Dsv4AttnResident attn;
  Dsv4MoeResident moe;
  Dsv4MhcResident mhc;
  Dsv4DraftResident draft;
  size_t bytes = 0;
};

struct Dsv4GlobalsResident {
  const uint16_t* embed = nullptr;       // bf16 [embed_vocab_count, hidden]
  int embed_vocab_begin = 0;
  int embed_vocab_count = 0;
  const uint16_t* final_norm = nullptr;  // bf16 [hidden]
  const uint16_t* lm_head = nullptr;     // bf16 [lm_vocab_count, hidden]
  int lm_vocab_begin = 0;
  int lm_vocab_count = 0;
  const uint16_t* hc_head_fn = nullptr;  // bf16 [hc_mult, hc_dim] (the model-level mHC head)
  const float* hc_head_base = nullptr;   // f32 [hc_mult]
  const float* hc_head_scale = nullptr;  // f32 [1]
  size_t bytes = 0;
};

// One tid2eid hash table kept in its shard (the num_hash_layers gate
// tables): the I64 [vocab, top-k] tensor, mapped read-only with
// random-access advice, never copied to the device. The host gathers the
// rows a step needs (a token's top-k expert ids) into staging; the router
// reads the staged ids. Unlike dsv41's Engram tables the rows are token
// ids, not (n-gram, head) slots — there is no head dimension, so the
// table is whole on every rank (no per-rank row split at any world).
class Dsv4HashTableMmap {
 public:
  Dsv4HashTableMmap(const std::string& path, uint64_t data_begin, int64_t rows, int topk);
  ~Dsv4HashTableMmap();
  Dsv4HashTableMmap(const Dsv4HashTableMmap&) = delete;
  Dsv4HashTableMmap& operator=(const Dsv4HashTableMmap&) = delete;
  // The token's top-k expert ids (a row of the mapped table).
  const int64_t* row(int64_t token) const;
  // dst[(t * topk + j)] = the expert ids of token ids[t], for t < n.
  // Faults the pages in parallel (a thread per 256 rows past the first
  // 256), the dsv41 Engram gather's pattern.
  void gather(const int32_t* ids, int n, int64_t* dst) const;
  int64_t rows() const { return rows_; }
  int topk() const { return topk_; }
  size_t mapped_bytes() const { return len_; }

 private:
  int fd_ = -1;
  uint8_t* base_ = nullptr;
  size_t len_ = 0;
  uint64_t data_begin_ = 0;
  int64_t rows_ = 0;
  int topk_ = 0;
};

// The tid2eid tables of a rank: one mapping per hash layer (in layer
// order), the whole table on every rank.
struct Dsv4HashTables {
  std::vector<std::unique_ptr<Dsv4HashTableMmap>> tables;  // per hash layer
  int vocab = 0;
  int topk = 0;
  size_t mapped_bytes() const {
    size_t b = 0;
    for (const auto& t : tables) b += t ? t->mapped_bytes() : 0;
    return b;
  }
};

// The local TP geometry at (rank, world): every slice bound the builders
// and the views use, in one place.
struct Dsv4LocalGeometry {
  int world = 1, rank = 0;
  int local_heads = 0, head_begin = 0;      // attention heads
  int local_groups = 0, group_begin = 0;    // wo_a / wo_b output groups
  int64_t local_inter = 0;                   // moe_intermediate_size / W
  int64_t local_shared_inter = 0;            // shared_expert_inter() / W
  int lm_vocab_begin = 0, lm_vocab_count = 0;
  int embed_vocab_begin = 0, embed_vocab_count = 0;
  static Dsv4LocalGeometry from_config(const Dsv4TextConfig& cfg, int rank, int world,
                                       Dsv4HeadSharding head);
};

// The family behind the shared stream (loaders/resident_stream.hpp).
struct Dsv4LoaderFamily {
  using Config = Dsv4TextConfig;
  using Expected = Dsv4ExpectedTensor;
  using LayerResident = Dsv4LayerResident;
  using GlobalsResident = Dsv4GlobalsResident;
  using Geometry = Dsv4LocalGeometry;
  using PresentMap = std::unordered_map<std::string, Dsv4TensorDesc>;
  struct Builder;  // models/dsv4/loader.cpp
  static const char* who() { return "dsv4 loader"; }
  static uint64_t loader_format() { return 1; }
  static int max_layer(const Config& c) { return c.max_layer(); }
  static int main_layers(const Config& c) { return c.num_hidden_layers; }
  static std::vector<Expected> layer_table(const Config& c, int layer) {
    return dsv4_expected_layer_tensors(c, layer);
  }
  static std::vector<Expected> global_table(const Config& c) { return dsv4_expected_global_tensors(c); }
  static void validate_binding(const Config& c, const PresentMap& present);
  static void check_sources(const Config&, const LoaderTensorMap&) {}
  static bool digest_included(const Expected& e);
  static bool discard_after_pack(const Expected&) { return false; }
  // The replicated set (rank-invariant reads at world > 1), exposed so
  // the host-side byte reconcile can build its per-class closed form.
  static bool is_replicated(const Expected& e);
  // The e8m0 decode the fp8 builder applies to the checkpoint's F8_E8M0
  // scale bytes (the dsv41 formula; 255 is NaN): 2^(byte - 127).
  static float e8m0_to_float(uint8_t b);
  // The MXFP4 slice contract (the builder's guard, exposed for the host
  // test): a column slice starts on a 32-element block and spans whole
  // blocks; row slices are free. Throws std::invalid_argument.
  static void check_mxfp4_slice(int64_t col_start, int64_t cols);
  // The counting-pass source plan of one layer's tensors by weight class
  // (rank, world): the host-side byte reconcile's per-class input.
  static std::map<Dsv4WeightClass, uint64_t> class_source_plan(const Config& c, int layer, int rank,
                                                               int world);
  static void build_globals(const Config& c, const Geometry& geo, const LoaderTensorMap& tensors,
                            LayerBump& bump, GlobalsResident& out, uint64_t& source_bytes,
                            uint64_t& verbatim_bytes, LoaderHeadSharding head);
  static size_t globals_bytes(const Config& c, int rank, int world, LoaderHeadSharding head);
  static size_t extra_resident_bytes(const Config&, int, int) { return 0; }
  static size_t min_staging_bytes() { return 0; }
  static void after_restore(const Config&, int, const LoaderTensorMap&, LayerResident&) {}
};

extern template class ResidentLayerStream<Dsv4LoaderFamily>;

class Dsv4LayerStream : public ResidentLayerStream<Dsv4LoaderFamily> {
 public:
  Dsv4LayerStream(const Dsv4TextConfig& cfg, const std::string& checkpoint_dir, int rank = 0,
                  int world = 1, Dsv4Residency residency = Dsv4Residency::Streaming,
                  Dsv4HeadSharding head = Dsv4HeadSharding::Full, bool resident_mtp = false);
  ~Dsv4LayerStream() override = default;

  static void set_resident_image_dir(const std::string& dir);
  static const std::string& resident_image_dir();
  // The deployment's `engine.embed_sharding` ("vocab": each rank holds its
  // lm-head slice of the embedding; "replicated": the whole table).
  static void set_embed_vocab_sharded(bool on);
  static bool embed_vocab_sharded();

  // The tid2eid hash tables mapped from their shards: built on the first
  // call from the shard mappings, so it must precede release_sources();
  // the mappings outlive the release.
  const Dsv4HashTables& load_hash_tables();

 protected:
  const std::string& image_dir() const override { return resident_image_dir(); }

 private:
  Dsv4HashTables hash_{};
  bool hash_loaded_ = false;
};

}  // namespace dgpp
