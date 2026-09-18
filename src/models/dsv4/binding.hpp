#pragma once
// Expected-tensor table for DeepSeek-V4-Flash (DeepseekV4ForCausalLM,
// 2026-09-17, /data/models/DeepSeek-V4-Flash-0731): every tensor the
// checkpoint must contain — names, dtypes, exact shapes — derived from the
// parsed config, not observed from one file. The table drives the offline
// validator and the resident loader, as the other tables do.
//
// Naming is checkpoint truth (no `model.` prefix): backbone layers under
// `layers.L.` (L in 0..42), the DSpark draft stages under `mtp.S.` (S in
// 0..2), the globals `embed.weight`, `norm.weight`, `head.weight` and the
// model-level `hc_head_fn/base/scale`.
//
// Format contract (verified against the 48 shard headers): an fp8 matrix X
// [N, K] is the pair X.weight F8_E4M3 [N, K] + X.scale F8_E8M0
// [ceil(N/128), ceil(K/128)]; an MXFP4 matrix is X.weight I8 [N, K/2]
// (two e2m1 codes per byte, the low nibble the even element) + X.scale
// F8_E8M0 [N, K/32]; a hash layer's gate table is ffn.gate.tid2eid I64
// [vocab, top-k]. Everything else is BF16 / F32 as listed.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "models/dsv4/config.hpp"

namespace dgpp {

enum class Dsv4WeightClass : int {
  Embed,
  LmHead,
  FinalNorm,
  LayerNorm,      // attn_norm, ffn_norm, q_norm, kv_norm, compressor.norm, indexer.compressor.norm, draft norms
  Attention,      // wq_a, wq_b, wkv, wo_a, wo_b, attn_sink
  Indexer,        // indexer.wq_b, weights_proj, indexer.compressor.*
  Compressor,     // compressor.wkv, wgate, ape, norm
  Router,         // ffn.gate.weight, .bias, .tid2eid
  SharedExpert,
  RoutedExpert,
  Mhc,            // hc_attn/ffn fn/base/scale, hc_head fn/base/scale
  Draft,          // main_proj, main_norm, markov_head, confidence_head (the draft's own extras)
};

enum class Dsv4TensorRole : uint8_t {
  Plain,        // BF16 / F32 as stored
  Fp8Payload,   // F8_E4M3 [N, K]
  Fp8Scale,     // F8_E8M0 [ceil(N/128), ceil(K/128)]
  Fp4Payload,   // I8 [N, K/2]
  Fp4Scale,     // F8_E8M0 [N, K/32]
  HashTable,    // I64 [vocab, top-k] (the hash layers' gate table, never resident)
};

struct Dsv4ExpectedTensor {
  std::string name;
  DType dtype{};
  std::vector<int64_t> shape;
  Dsv4WeightClass cls = Dsv4WeightClass::Attention;
  int layer = -1;   // layer index (draft stages at num_hidden_layers + S); -1 for globals
  int expert = -1;  // routed-expert id, -1 otherwise
  Dsv4TensorRole role = Dsv4TensorRole::Plain;

  size_t numel() const {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
  }
  size_t nbytes() const { return numel() * dtype_size(dtype); }
};

// The checkpoint name prefix of a layer: "layers.L." or "mtp.S.".
std::string dsv4_layer_prefix(const Dsv4TextConfig& cfg, int layer);

std::vector<Dsv4ExpectedTensor> dsv4_expected_text_tensors(const Dsv4TextConfig& cfg);
std::vector<Dsv4ExpectedTensor> dsv4_expected_layer_tensors(const Dsv4TextConfig& cfg, int layer);
std::vector<Dsv4ExpectedTensor> dsv4_expected_global_tensors(const Dsv4TextConfig& cfg);

struct Dsv4TensorDesc {
  DType dtype{};
  std::vector<int64_t> shape;
};

struct Dsv4BindReport {
  size_t expected = 0;
  size_t matched = 0;
  size_t missing = 0;
  size_t dtype_mismatch = 0;
  size_t shape_mismatch = 0;
  size_t unexpected = 0;
  size_t fp4_matrices = 0;      // routed expert matrices
  size_t fp8_matrices = 0;      // dense projections
  size_t hash_tables = 0;
  std::vector<std::string> errors;
  // Tensors of layers beyond the config's stack (a truncated diagnostic
  // forward over the first layers): counted, not errors.
  int beyond_stack = 0;
  bool ok() const {
    return missing == 0 && dtype_mismatch == 0 && shape_mismatch == 0 && unexpected == 0;
  }
};

Dsv4BindReport dsv4_validate_text_binding(
    const Dsv4TextConfig& cfg,
    const std::unordered_map<std::string, Dsv4TensorDesc>& present,
    size_t max_errors = 32);

// Tensor-parallel geometry acceptance (T=2 deployment): throws
// std::invalid_argument naming the dim that does not divide. world must
// divide the attention heads (whole output groups per rank), the index
// heads, the vocabulary, the routed experts and the expert / shared
// intermediate size, with every intermediate slice a multiple of 32 (the
// MXFP4 block) and the dense slices a multiple of the fp8 block.
void dsv4_tp_validate_geometry(const Dsv4TextConfig& cfg, int rank, int world);

}  // namespace dgpp
