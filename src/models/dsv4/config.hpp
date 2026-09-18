#pragma once
// DeepSeek-V4-Flash (DeepseekV4ForCausalLM, model_type deepseek_v4)
// configuration, parsed from the checkpoint's FLAT config.json (2026-09-17,
// /data/models/DeepSeek-V4-Flash-0731). Unlike DeepSeek-V4.1 the layout has
// no `text_config` nest and no vision tower: every field the assembly
// consumes sits at the root. The same policy as the other parsers: every
// field is parsed into a known-supported value or rejected with a message
// naming the field, at load time. The reference is the checkpoint's own
// `inference/model.py` (ModelArgs / Block / DSparkBlock) and its
// `inference/config.json`.
//
// Layout facts (verified against the 48 shard headers):
// - `num_hidden_layers` 43 is the backbone only; the DSpark draft stages
//   (checkpoint prefix `mtp.S.`) are the extra `compress_ratios` entries
//   past the backbone — the release's list has 46 entries (43 + 3 stages),
//   while `num_nextn_predict_layers` (1) is the number of stages the
//   generator actually runs, not the number shipped in the checkpoint.
// - `num_hash_layers` 3 names the first 3 layers: their gate routes by
//   token id through the `ffn.gate.tid2eid` [vocab, top-k] table instead
//   of a learned bias (inference/model.py Gate: `hash = layer_id <
//   n_hash_layers`).
// - A layer compresses its KV when its compress_ratio > 0; the indexer
//   (top-k sparse selection) exists only at ratio 4 (inference/model.py
//   Attention: `if self.compress_ratio == 4: self.indexer = ...`). The
//   ratio-4 compressor is the overlapping variant (coff 2), the ratio-128
//   one is plain (coff 1).
//
// The quantization contract (the release as shipped): dense projections
// `X` are the pair X.weight F8_E4M3 [N, K] + X.scale F8_E8M0
// [N/128, K/128] (weight_block_size [128, 128], scale_fmt ue8m0); the
// routed experts are MXFP4 (X.weight I8 [N, K/2] — two e2m1 codes per
// byte, the low nibble first — + X.scale F8_E8M0 [N, K/32]) via the root
// `expert_dtype` "fp4". Everything else is BF16 / F32 / I64 as listed in
// the binding table. Nothing is requantized: the engine consumes these
// bytes as they are.
#include <cstdint>
#include <string>
#include <vector>

#include "loaders/minijson.hpp"
#include "models/glm/moe.hpp"

namespace dgpp {

struct Dsv4TextConfig {
  // --- model shape -------------------------------------------------------
  int hidden_size = 4096;
  int vocab_size = 129280;
  int num_hidden_layers = 43;
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;
  std::string hidden_act = "silu";
  int max_position_embeddings = 1048576;
  int64_t bos_token_id = 0;
  int64_t eos_token_id = 1;
  int64_t pad_token_id = -1;

  // --- attention ----------------------------------------------------------
  int num_attention_heads = 64;
  int num_key_value_heads = 1;   // one 512-wide latent per token, K == V
  int head_dim = 512;            // the latent width, rope tail included
  int qk_rope_head_dim = 64;
  int q_lora_rank = 1024;
  int o_lora_rank = 1024;
  int o_groups = 8;
  bool attention_bias = false;
  int sliding_window = 128;
  double rope_theta = 10000.0;          // window-only layers, no scaling
  double compress_rope_theta = 160000.0; // every layer with compress_ratio > 0, with YaRN
  double rope_factor = 16.0;
  int original_max_position_embeddings = 65536;
  double beta_fast = 32.0;
  double beta_slow = 1.0;

  // --- compression schedule -------------------------------------------------
  // One entry per layer and draft stage: 0 = sliding window only, r = the
  // main KV compressed r-to-1 (the release's ratios are 4, overlapping,
  // and 128).
  std::vector<int> compress_ratios;
  int num_hash_layers = 0;    // the first N layers hash-route via tid2eid
  int index_n_heads = 64;
  int index_head_dim = 128;
  int index_topk = 512;

  // --- mHC (single-pass) ----------------------------------------------
  int hc_mult = 4;
  int hc_sinkhorn_iters = 20;
  float hc_eps = 1e-6f;

  // --- MoE ------------------------------------------------------------------
  int moe_intermediate_size = 2048;
  int n_routed_experts = 256;
  int n_shared_experts = 1;
  int num_experts_per_tok = 6;
  bool norm_topk_prob = true;
  float routed_scaling_factor = 1.5f;
  float swiglu_limit = 10.0f;
  std::string scoring_func = "sqrtsoftplus";  // or sigmoid
  std::string topk_method = "noaux_tc";

  // --- DSpark / MTP ------------------------------------------------------
  int num_nextn_predict_layers = 0;  // stages the generator runs (<= shipped)
  int dspark_block_size = 5;
  int64_t dspark_noise_token_id = 128799;
  std::vector<int> dspark_target_layer_ids;
  int dspark_markov_rank = 256;

  // --- weight formats ----------------------------------------------------
  int fp8_block_size = 128;  // one e8m0 scale per 128x128 block of a dense fp8 matrix
  int fp4_block_size = 32;   // one e8m0 scale per 32 codes along K of an fp4 matrix

  // Parses config.json's root object (the flat V4 layout). Throws
  // std::runtime_error naming the offending field on anything unsupported.
  static Dsv4TextConfig parse(const minijson::Value& root);
  static Dsv4TextConfig from_json_file(const std::string& path);

  // --- derived layer facts ------------------------------------------------
  // Layers are indexed 0..num_hidden_layers-1 (the backbone) and
  // num_hidden_layers..max_layer()-1 (the DSpark draft stages, checkpoint
  // prefix `mtp.S.`). The stage count is the compress_ratios tail past
  // the backbone (the release ships 3 stages; num_nextn_predict_layers is
  // the subset the generator runs).
  int num_draft_stages() const { return static_cast<int>(compress_ratios.size()) - num_hidden_layers; }
  int max_layer() const { return num_hidden_layers + num_draft_stages(); }
  bool is_draft(int l) const { return l >= num_hidden_layers && l < max_layer(); }
  int draft_stage(int l) const { return is_draft(l) ? l - num_hidden_layers : -1; }
  int compress_ratio(int l) const { return compress_ratios[static_cast<size_t>(l)]; }
  // Hash (engram) layers: 0..num_hash_layers-1, the gate's tid2eid table
  // instead of a learned bias.
  bool is_hash_layer(int l) const { return l >= 0 && l < num_hash_layers; }
  // The indexer (top-k sparse selection) exists only at ratio 4.
  bool is_index_layer(int l) const { return compress_ratio(l) == 4; }
  // The compressor's coefficient: 2 for the overlapping ratio-4 window,
  // 1 otherwise (inference/model.py Compressor: coff = 1 + (ratio == 4)).
  int compressor_coff(int l) const { return is_index_layer(l) ? 2 : 1; }
  bool is_dspark_target(int l) const;
  int shared_expert_inter() const { return n_shared_experts * moe_intermediate_size; }
  int qk_nope_head_dim() const { return head_dim - qk_rope_head_dim; }
  int heads_per_group() const { return num_attention_heads / o_groups; }
  int hc_coeff_rows() const { return (2 + hc_mult) * hc_mult; }
  int hc_dim() const { return hc_mult * hidden_size; }

  // The routed chain's configuration (models/glm/moe.hpp): the
  // sqrtsoftplus router with its bias (or the hash table on the first
  // num_hash_layers), the shared expert in the chain, the clamped SwiGLU.
  // `local_inter` is this rank's slice of moe_intermediate_size; the
  // draft stages reuse the backbone's expert set, so there is no draft
  // variant (unlike V4.1).
  GlmMoeConfig moe_config(int local_inter) const;
};

}  // namespace dgpp
