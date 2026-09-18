// The DeepSeek-V4-Flash config parser (the FLAT layout, unlike V4.1's
// text_config nest): the release's values parse, the compression schedule
// (ratios, the hash layers, the ratio-4 indexer layers, the draft stages
// from the compress_ratios tail) and the DSpark / MoE facts derive from the
// file, the architecture registry dispatches on it, and the unsupported
// shapes are refused by name.
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "dsv4_config_json.hpp"
#include "loaders/architecture.hpp"
#include "loaders/minijson.hpp"
#include "models/dsv4/config.hpp"

namespace {

using dsv4_test::config_json;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Dsv4TextConfig parse(const std::string& text) {
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Dsv4TextConfig::parse(t.root);
}

std::string refusal(const std::string& text) {
  try {
    (void)parse(text);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

bool has(const std::string& msg, const char* needle) { return msg.find(needle) != std::string::npos; }

}  // namespace

DGPP_TEST(dsv4_config_parses_the_release) {
  const dgpp::Dsv4TextConfig c = parse(config_json());
  require(c.hidden_size == 4096 && c.vocab_size == 129280 && c.num_hidden_layers == 43, "shape");
  require(c.rms_norm_eps == 1e-6f && c.max_position_embeddings == 1048576, "eps / positions");
  require(c.bos_token_id == 0 && c.eos_token_id == 1 && c.pad_token_id == -1, "tokens");
  require(c.num_attention_heads == 64 && c.num_key_value_heads == 1 && c.head_dim == 512 && c.qk_rope_head_dim == 64,
          "attention dims");
  require(c.q_lora_rank == 1024 && c.o_lora_rank == 1024 && c.o_groups == 8 && c.heads_per_group() == 8, "lora ranks");
  require(c.sliding_window == 128 && c.qk_nope_head_dim() == 448, "window / nope");
  require(c.rope_theta == 10000.0 && c.compress_rope_theta == 160000.0, "thetas");
  require(c.rope_factor == 16.0 && c.original_max_position_embeddings == 65536 && c.beta_fast == 32.0 && c.beta_slow == 1.0,
          "yarn");
  // The compression schedule: 46 entries = 43 backbone + 3 draft stages.
  require(c.compress_ratios.size() == 46 && c.num_draft_stages() == 3 && c.max_layer() == 46, "schedule length");
  require(c.compress_ratio(0) == 0 && c.compress_ratio(1) == 0 && c.compress_ratio(2) == 4 && c.compress_ratio(3) == 128 &&
              c.compress_ratio(42) == 4 && c.compress_ratio(43) == 0 && c.compress_ratio(45) == 0,
          "ratios");
  require(c.num_hash_layers == 3 && c.is_hash_layer(0) && c.is_hash_layer(2) && !c.is_hash_layer(3), "hash layers");
  require(c.is_index_layer(2) && !c.is_index_layer(3) && c.is_index_layer(42) && !c.is_index_layer(43), "indexer layers");
  require(c.compressor_coff(2) == 2 && c.compressor_coff(3) == 1, "compressor coff");
  require(c.index_n_heads == 64 && c.index_head_dim == 128 && c.index_topk == 512, "indexer");
  require(c.hc_mult == 4 && c.hc_sinkhorn_iters == 20 && c.hc_eps == 1e-6f && c.hc_coeff_rows() == 24 && c.hc_dim() == 16384,
          "mhc");
  require(c.moe_intermediate_size == 2048 && c.n_routed_experts == 256 && c.num_experts_per_tok == 6 &&
              c.n_shared_experts == 1 && c.shared_expert_inter() == 2048,
          "moe");
  require(c.routed_scaling_factor == 1.5f && c.norm_topk_prob && c.swiglu_limit == 10.0f &&
              c.scoring_func == "sqrtsoftplus" && c.topk_method == "noaux_tc",
          "router");
  require(c.num_nextn_predict_layers == 1 && c.num_nextn_predict_layers <= c.num_draft_stages(), "nextn vs shipped");
  require(c.dspark_block_size == 5 && c.dspark_noise_token_id == 128799 && c.dspark_markov_rank == 256, "dspark");
  require(c.dspark_target_layer_ids.size() == 3 && c.is_dspark_target(40) && c.is_dspark_target(41) &&
              c.is_dspark_target(42) && !c.is_dspark_target(39),
          "dspark targets");
  require(c.fp8_block_size == 128 && c.fp4_block_size == 32, "blocks");
  const dgpp::GlmMoeConfig m = c.moe_config(1024);
  require(m.inter == 1024 && m.n_experts == 256 && m.top_k == 6 && m.n_shared_experts == 1 && m.swiglu_limit == 10.0f,
          "moe_config");
  require(m.router_mode == dgpp::MoeRouterMode::SqrtSoftplusBias && m.routed_scaling_factor == 1.5f, "moe_config router");
}

DGPP_TEST(dsv4_config_refuses_unsupported_shapes) {
  require(has(refusal(config_json("\"model_type\": \"deepseek_v4\"", "\"model_type\": \"deepseek_v41\"")), "model_type"),
          "v41 type");
  require(has(refusal(config_json("\"architectures\": [", "\"text_config\": {}, \"architectures\": [")), "text_config"),
          "v41 layout nest");
  require(has(refusal(config_json("\"architectures\": [", "\"vision_config\": {}, \"architectures\": [")), "vision_config"),
          "vision tower");
  require(has(refusal(config_json("\"architectures\": [", "\"engram_layer_ids\": [0, 1, 2], \"architectures\": [")),
              "engram_layer_ids"),
          "v41 hash layout");
  require(has(refusal(config_json("\"num_key_value_heads\": 1", "\"num_key_value_heads\": 2")), "num_key_value_heads"),
          "kv heads");
  require(has(refusal(config_json("\"head_dim\": 512", "\"head_dim\": 256")), "head_dim"), "head dim");
  require(has(refusal(config_json("\"qk_rope_head_dim\": 64", "\"qk_rope_head_dim\": 32")), "qk_rope_head_dim"),
          "rope dim");
  require(has(refusal(config_json("\"o_groups\": 8", "\"o_groups\": 7")), "o_groups"), "groups");
  require(has(refusal(config_json("\"type\": \"yarn\"", "\"type\": \"default\"")), "rope_scaling.type"), "yarn");
  require(has(refusal(config_json("\"hc_mult\": 4", "\"hc_mult\": 2")), "hc_mult"), "hc_mult");
  require(has(refusal(config_json("\"scoring_func\": \"sqrtsoftplus\"", "\"scoring_func\": \"softmax\"")), "scoring_func"),
          "scoring");
  require(has(refusal(config_json("\"topk_method\": \"noaux_tc\"", "\"topk_method\": \"greedy\"")), "topk_method"),
          "topk method");
  require(has(refusal(config_json("\"n_shared_experts\": 1", "\"n_shared_experts\": 2")), "n_shared_experts"), "shared");
  require(has(refusal(config_json("\"swiglu_limit\": 10.0", "\"swiglu_limit\": 0.0")), "swiglu_limit"), "clamp");
  require(has(refusal(config_json("\"index_topk\": 512", "\"index_topk\": 500")), "index_topk"), "topk pow2");
  require(has(refusal(config_json("\"index_head_dim\": 128", "\"index_head_dim\": 64")), "index_head_dim"), "index dim");
  require(has(refusal(config_json("\"tie_word_embeddings\": false", "\"tie_word_embeddings\": true")),
              "tie_word_embeddings"),
          "tie");
  require(has(refusal(config_json("\"num_hash_layers\": 3", "\"num_hash_layers\": 44")), "num_hash_layers"), "hash range");
  require(has(refusal(config_json("\"num_nextn_predict_layers\": 1", "\"num_nextn_predict_layers\": 4")),
              "num_nextn_predict_layers"),
          "nextn beyond shipped stages");
  // The schedule's consistency rules.
  {
    const std::string anchor =
        "\"compress_ratios\": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0, 0, 0]";
    require(config_json().find(anchor) != std::string::npos, "ratio anchor");
    require(has(refusal(config_json(anchor, "\"compress_ratios\": [0, 0, 4, 128]")), "compress_ratios"), "ratio length");
    require(has(refusal(config_json(anchor,
                                  "\"compress_ratios\": [0, 0, 8, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, "
                                  "4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, "
                                  "4, 128, 4, 128, 4, 128, 4, 0, 0, 0]")),
                "compress_ratios"),
            "ratio 8");
    require(has(refusal(config_json(anchor,
                                  "\"compress_ratios\": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, "
                                  "4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, "
                                  "4, 128, 4, 128, 4, 128, 4, 0, 0, 4]")),
                "compress_ratios"),
            "draft ratio");
  }
  require(has(refusal(config_json("\"dspark_target_layer_ids\": [40, 41, 42]", "\"dspark_target_layer_ids\": [40, 41, 43]")),
              "dspark_target_layer_ids"),
          "target outside the backbone");
  // The quantization contract: the release's own format only.
  require(has(refusal(config_json("\"quant_method\": \"fp8\"", "\"quant_method\": \"modelopt\"")), "quant_method"),
          "method");
  require(has(refusal(config_json("\"weight_block_size\": [128, 128]", "\"weight_block_size\": [32, 32]")),
              "weight_block_size"),
          "block");
  require(has(refusal(config_json("\"scale_fmt\": \"ue8m0\"", "\"scale_fmt\": null")), "scale_fmt"), "scale fmt");
  require(has(refusal(config_json("\"expert_dtype\": \"fp4\"", "\"expert_dtype\": \"nvfp4\"")), "expert_dtype"),
          "nvfp4 cast");
  require(has(refusal(config_json("\"weight_block_size\": [128, 128]",
                                  "\"weight_block_size\": [128, 128], \"engram_dtype\": \"fp4\"")),
              "engram_dtype"),
          "engram re-pack");
  require(has(refusal(config_json("\"weight_block_size\": [128, 128]",
                                  "\"weight_block_size\": [128, 128], \"moe_quant_algo\": \"NVFP4\"")),
              "moe_quant_algo"),
          "modelopt cast");
  require(has(refusal(config_json("\"quantization_config\": {", "\"quantization_config\": null, \"x\": {")),
              "quantization_config"),
          "missing quantization");
}

DGPP_TEST(dsv4_architecture_registry_dispatches) {
  const auto d = dgpp::minijson::parse(R"({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4"})");
  require(dgpp::detect_architecture(d.root) == dgpp::ModelArchitecture::DeepseekV4, "deepseek_v4");
  require(std::string(dgpp::model_architecture_name(dgpp::ModelArchitecture::DeepseekV4)) == "deepseek_v4", "name");
  const auto t = dgpp::minijson::parse(R"({"model_type": "deepseek_v4"})");
  require(dgpp::detect_architecture(t.root) == dgpp::ModelArchitecture::DeepseekV4, "by model_type");
  // The V4.1 class name starts with the V4 prefix: it must still dispatch
  // to the V4.1 family, not fall through to V4.
  const auto v41 = dgpp::minijson::parse(R"({"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41"})");
  require(dgpp::detect_architecture(v41.root) == dgpp::ModelArchitecture::DeepseekV41, "v41 unchanged");
  bool refused = false;
  try {
    const auto v3 = dgpp::minijson::parse(R"({"architectures": ["DeepseekV3ForCausalLM"], "model_type": "deepseek_v3"})");
    (void)dgpp::detect_architecture(v3.root);
  } catch (const std::runtime_error&) {
    refused = true;
  }
  require(refused, "DeepSeek-V3 is not this family");
}

DGPP_TEST(dsv4_config_reads_the_landed_checkpoint) {
  namespace fs = std::filesystem;
  if (const char* dir = std::getenv("DGPP_DSV4_CHECKPOINT_DIR"); dir && *dir) {
    const fs::path cfg = fs::path(dir) / "config.json";
    if (!fs::exists(cfg)) return;
    require(dgpp::detect_architecture_file(cfg.string()) == dgpp::ModelArchitecture::DeepseekV4, "arch");
    const dgpp::Dsv4TextConfig c = dgpp::Dsv4TextConfig::from_json_file(cfg.string());
    require(c.num_hidden_layers == 43 && c.max_layer() == 46 && c.n_routed_experts == 256, "landed values");
    require(c.num_hash_layers == 3 && c.num_draft_stages() == 3, "landed schedule");
    return;
  }
  // The release lives on local disk, not in the hub cache.
}
