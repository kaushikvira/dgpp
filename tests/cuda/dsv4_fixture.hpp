#pragma once
// Synthetic mini-checkpoint writer for the DeepSeek-V4-Flash tests (the
// dsv41_fixture.hpp pattern, adapted to the flat V4 layout): enumerates
// the binding table for a small config and writes config.json + one
// safetensors shard, so fixture and table cannot disagree. Values are
// deterministic per tensor NAME (glm_rng's scheme) in the release's own
// formats: fp8 e4m3 payloads with e8m0 scales on 128 x 128 blocks,
// MXFP4 experts (random e2m1 nibbles, e8m0 per 32), I64 tid2eid hash
// tables (deterministic token -> expert ids), bf16 / fp32 for the rest —
// magnitudes that keep a forward's nonlinearities informative.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/dtypes.hpp"
#include "glm_rng.hpp"
#include "loaders/minijson.hpp"
#include "models/dsv4/binding.hpp"
#include "models/dsv4/config.hpp"

namespace dsv4fx {

namespace fs = std::filesystem;
using dgpp::Dsv4ExpectedTensor;
using dgpp::Dsv4TensorRole;
using dgpp::Dsv4TextConfig;
using dgpp::Dsv4WeightClass;
using dgpp::float_to_bf16_bits;
using glmrng::Rng;
using glmrng::seed_for;

// The tiny release: hidden 256, 4 heads on the 512-wide latent (2 heads
// per rank at world 2), 2 output groups of 2 heads, q_lora / o_lora 128,
// a 4 x 128 indexer (topk 4) on the ratio-4 layer, four backbone layers
// with the release's ratio pattern [window, window, C4A, C128A], three
// hash (tid2eid) layers, eight backbone experts of 256 (4 per rank at
// world 2) top-2 sqrtsoftplus, one shared expert, vocab 256, three
// shipped DSpark stages (the generator runs one: block 5, targets 2-3,
// Markov rank 32, noise token 250).
inline const char* tiny_config_json() {
  return R"json({
 "architectures": ["DeepseekV4ForCausalLM"],
 "model_type": "deepseek_v4",
 "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
 "expert_dtype": "fp4",
 "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3",
                         "scale_fmt": "ue8m0", "weight_block_size": [128, 128]},
 "vocab_size": 256, "hidden_size": 256, "num_hidden_layers": 4,
 "num_attention_heads": 4, "num_key_value_heads": 1,
 "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 128, "o_lora_rank": 128, "o_groups": 2,
 "hidden_act": "silu", "swiglu_limit": 10.0, "rms_norm_eps": 1e-20,
 "max_position_embeddings": 4096, "sliding_window": 16,
 "rope_theta": 10000.0, "compress_rope_theta": 160000.0,
 "rope_scaling": {"type": "yarn", "factor": 16.0, "original_max_position_embeddings": 256,
                  "beta_fast": 32, "beta_slow": 1},
 "compress_ratios": [0, 0, 4, 128, 0, 0, 0],
 "num_hash_layers": 3,
 "index_n_heads": 4, "index_head_dim": 128, "index_topk": 4,
 "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
 "moe_intermediate_size": 256, "n_routed_experts": 8, "n_shared_experts": 1,
 "num_experts_per_tok": 2, "norm_topk_prob": true,
 "routed_scaling_factor": 1.5, "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
 "num_nextn_predict_layers": 1,
 "dspark_block_size": 5, "dspark_noise_token_id": 250, "dspark_target_layer_ids": [2, 3],
 "dspark_markov_rank": 32
})json";
}

inline Dsv4TextConfig tiny_config() {
  const auto t = dgpp::minijson::parse(tiny_config_json());
  return Dsv4TextConfig::parse(t.root);
}

inline bool has(const std::string& name, const char* needle) { return name.find(needle) != std::string::npos; }

// The bytes of one tensor.
inline std::vector<uint8_t> tensor_bytes(const Dsv4TextConfig& cfg, const Dsv4ExpectedTensor& e) {
  std::vector<uint8_t> out(e.nbytes());
  const std::string& name = e.name;
  Rng rng(seed_for(name));
  const size_t n = e.numel();
  switch (e.role) {
    case Dsv4TensorRole::Fp8Payload:
      // e4m3 codes of ~N(0, 0.67): the scales below put the weights near
      // 0.03.
      for (size_t i = 0; i < n; ++i) out[i] = dgpp::float_to_fp8_e4m3_bits(2.0f * rng.normal3());
      return out;
    case Dsv4TensorRole::Fp8Scale:
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(121 + rng.next() % 3);  // 2^-6 .. 2^-4
      return out;
    case Dsv4TensorRole::Fp4Payload:
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(rng.next());  // every e2m1 code is finite
      return out;
    case Dsv4TensorRole::Fp4Scale:
      for (size_t i = 0; i < n; ++i) out[i] = static_cast<uint8_t>(118 + rng.next() % 3);  // 2^-9 .. 2^-7
      return out;
    case Dsv4TensorRole::HashTable:
      // The tid2eid table: a deterministic token -> expert-id map (a
      // valid expert for the fixture's E), the table's own semantics —
      // no scoring.
      for (size_t i = 0; i < n; ++i) {
        const int64_t token = static_cast<int64_t>(i / e.shape.back());
        const int64_t col = static_cast<int64_t>(i % e.shape.back());
        const int64_t id = (token * 7 + col * 3) % cfg.n_routed_experts;
        std::memcpy(out.data() + i * 8, &id, 8);
      }
      return out;
    case Dsv4TensorRole::Plain:
      break;
  }
  const bool is_norm = has(name, "norm") && e.shape.size() == 1;  // gains near 1
  const bool is_bias = has(name, ".bias");
  const bool is_hc_scale = e.cls == Dsv4WeightClass::Mhc && has(name, "_scale");
  const bool is_small = e.cls == Dsv4WeightClass::Mhc || has(name, "attn_sink") || is_bias ||
                        has(name, "ape") || has(name, "confidence");
  const bool is_embed = e.cls == Dsv4WeightClass::Embed || e.cls == Dsv4WeightClass::LmHead ||
                        has(name, "markov_head");
  for (size_t i = 0; i < n; ++i) {
    float v;
    if (is_norm) v = 0.9f + 0.2f * (0.5f * (rng.unit() + 1.0f));
    else if (is_hc_scale) v = 1.0f + 0.1f * rng.unit();
    else if (is_small) v = 0.02f * rng.normal3();
    else if (is_embed) v = 0.3f * rng.normal3();
    else v = 0.05f * rng.normal3();  // projections, routers, indexers, compressors, gates
    if (e.dtype == dgpp::DType::BF16) {
      const uint16_t bits = float_to_bf16_bits(v);
      std::memcpy(&out[i * 2], &bits, 2);
    } else if (e.dtype == dgpp::DType::F32) {
      std::memcpy(&out[i * 4], &v, 4);
    } else {
      throw std::runtime_error("fixture dtype not handled: " + name);
    }
  }
  return out;
}

inline std::vector<uint8_t> fixture_bytes(const Dsv4TextConfig& cfg, const Dsv4ExpectedTensor& e) {
  return tensor_bytes(cfg, e);
}

// Writes `dir` (config.json + one safetensors shard holding every text
// tensor, the hash tables included) for `cfg`.
inline void write_fixture(const Dsv4TextConfig& cfg, const std::string& dir, const char* config_json) {
  fs::path root(dir);
  fs::remove_all(root);
  fs::create_directories(root);
  {
    const fs::path p = root / "config.json";
    std::FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write config.json");
    std::fwrite(config_json, 1, std::strlen(config_json), f);
    std::fclose(f);
  }
  const auto table = dgpp::dsv4_expected_text_tensors(cfg);
  std::string header = "{";
  std::vector<uint8_t> data;
  size_t off = 0;
  bool first = true;
  for (const auto& e : table) {
    const auto b = fixture_bytes(cfg, e);
    std::string shape = "[";
    for (size_t i = 0; i < e.shape.size(); ++i) {
      if (i) shape += ",";
      shape += std::to_string(e.shape[i]);
    }
    shape += "]";
    if (!first) header += ",";
    first = false;
    header += "\"" + e.name + "\":{\"dtype\":\"" + std::string(dgpp::dtype_name(e.dtype)) +
              "\",\"shape\":" + shape + ",\"data_offsets\":[" + std::to_string(off) + "," +
              std::to_string(off + b.size()) + "]}";
    data.insert(data.end(), b.begin(), b.end());
    off += b.size();
  }
  header += "}";
  const fs::path shard = root / "model.safetensors";
  std::FILE* f = std::fopen(shard.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write shard");
  const uint64_t hlen = header.size();
  std::fwrite(&hlen, 8, 1, f);
  std::fwrite(header.data(), 1, hlen, f);
  std::fwrite(data.data(), 1, data.size(), f);
  std::fclose(f);
  std::printf("dsv4 fixture written: %zu tensors, %.2f MB payload\n", table.size(),
              static_cast<double>(data.size()) / 1048576.0);
}

inline void write_fixture(const Dsv4TextConfig& cfg, const std::string& dir) {
  write_fixture(cfg, dir, tiny_config_json());
}

}  // namespace dsv4fx
