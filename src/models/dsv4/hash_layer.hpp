#pragma once
// The hash-routing MoE layer of DeepSeek-V4-Flash (deepseek_v4)
// (2026-09-17, docs/dsv4_kernel_port_spec.md §2.3-§2.4): the V4's
// "Engram" reframe — NOT a hashed n-gram memory (the dsv41 Engram's),
// but the MoE gate's hash-ROUTING mode. The first num_hash_layers = 3
// layers' gate routes by token id through the ffn.gate.tid2eid
// [vocab, top-k] INT64 table (inference/model.py Gate: `hash = layer_id
// < n_hash_layers`) instead of the learned gate + bias; the rest use
// the learned gate (the sqrtsoftplus scoring + the noaux_tc biased
// top-k selection + the renormalize + the routed_scaling_factor 1.5).
// The routed experts are MXFP4 (the e2m1 x e8m0/32 decode, the clamped
// SwiGLU, the router weight folded into the down epilogue — the
// dsv4-native ops/moe/experts' V4 re-expression). One object serves a
// layer's MoE block; rebind() points it at the layer's weights + role.
// The numerics of the router (the fused top-k + the tid2eid hash mode)
// and the expert (the MXFP4 decode) are certified against the CPU
// oracles (tests/unit/dsv4_moe_oracle_test.cpp) — the GPU-gate pending
// runs the parity gate.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/gemm.hpp"
#include "models/quant_matrix.hpp"

namespace dgpp {

// The V4 MoE geometry (the dsv4 config's MoE block, the world-2
// deployment). 256 experts, top-k 6 (the V4's 6-per-token, the dsv41's
// 8's), inter 2048, the routed_scaling_factor 1.5 (the dsv41's 2.5's
// V4 swap), the hash_layers 3 (the first 3 route by the tid2eid
// table's).
struct Dsv4HashConfig {
  int hidden = 4096;
  int inter = 2048;  // the per-expert's intermediate's
  int n_experts = 256;
  int top_k = 6;  // the V4's 6-per-token (the dsv41's 8's)
  int num_hash_layers = 3;  // the first 3 layers' the tid2eid's routing's
  int vocab_size = 129280;  // the tid2eid table's rows'
  float routed_scaling_factor = 1.5f;  // the V4's (the dsv41's 2.5's swap)
  bool norm_topk_prob = true;
  float swiglu_limit = 10.0f;
  // The world-2's expert slice (the intermediate's the rank's half's —
  // the glm MoE's the expert slicing's the I/world's).
  int tp = 2;
  int local_inter() const { return inter / tp; }
  static void validate(const Dsv4HashConfig& c);
};

// The V4 MoE layer's weight view (the dsv4 binding's MoE's planes: the
// router's gate / bias' the bf16 / f32's, the tid2eid table's I64's,
// the MXFP4 experts' the e2m1 x e8m0/32's — the spec §3.1's loader-side
// decode). The experts' the [n_experts * 3]'s gate / up / down's the
// I8 payload's (two e2m1 per byte's) + the e8m0/32 scales' the F8_E8M0's.
struct Dsv4HashWeights {
  const uint16_t* router_gate = nullptr;  // bf16 [n_experts, hidden]
  const float* router_bias = nullptr;  // f32 [n_experts] (null: the hash layers' no bias's)
  const int64_t* tid2eid = nullptr;  // I64 [vocab, top_k] (the hash layers')
  // The MXFP4 experts' (the e2m1 pairs' the I8's, the e8m0/32 scales'
  // the F8_E8M0's). The [n_experts * 3]'s gate, up, down's.
  const uint8_t* expert_payload = nullptr;  // I8 [n_experts * 3, ...]
  const uint8_t* expert_scales = nullptr;  // F8_E8M0 [n_experts * 3, ...]
  // The slot path's expert views (the GLM MoE's expert table's re-
  // expression's): the routed's MXFP4's triples's (the w1's w3's w2's
  // per expert's, the inter's sliced at I/world's) + the fp8 shared
  // expert's triple's + the slice's dims's (the world's I's / S's). The
  // rebind's builds the device's view table's (the MoeExpertView's) from
  // these's.
  const GlmFp4Matrix* experts = nullptr;  // [n_experts * 3] (w1, w3, w2 per expert)
  const GlmQuantMatrix* shared = nullptr;  // [3] (w1, w3, w2)
  int n_experts = 0;
  int64_t local_inter = 0;  // I/world (the routed expert's inter slice)
  int64_t local_shared_inter = 0;  // S/world (the shared expert's inter slice)
  int layer = 0;  // the layer's ordinal (the hash check's: layer < num_hash_layers')
  bool hash_layer = false;  // the tid2eid's routing's (the rebind's the set's)
};

// The hash-routing MoE layer (the dsv4-native ops/moe's V4 re-expression,
// the world-2 deployment). The router's the fused top-k's (the
// learned gate's the non-hash layers' the tid2eid table's the hash
// layers' — the v4-owned dsv4_moe_router kernel's, the GPU-gate
// pending's); the expert's the MXFP4's (the glm fp4 expert kernels'
// composition's).
class Dsv4HashLayer {
 public:
  Dsv4HashLayer(IGemm& gemm, const Dsv4HashConfig& cfg, int max_tokens, void* scratch, size_t scratch_capacity,
                void* gemm_workspace, size_t gemm_ws_bytes);

  void rebind(const Dsv4HashWeights& w, int layer);
  bool prepare(int tokens);

  // The fused top-k router (the v4-owned dsv4_moe_router's): the
  // learned gate's the non-hash layers' (the sqrtsoftplus scoring's +
  // the noaux_tc biased selection's + the renormalize's + the
  // routed_scaling_factor's); the tid2eid table's the hash layers'
  // (the table order's, the duplicate-id's the independent's, the no
  // bias's). `out_ids` [tokens, top_k], `out_w` [tokens, top_k].
  void route(const void* logits, const int64_t* tokens, int tokens_n, int32_t* out_ids, float* out_w,
             cudaStream_t stream);
  // The MXFP4 expert (the glm fp4 expert kernels' composition's): the
  // e2m1 x e8m0/32 decode's, the clamped SwiGLU's, the router weight's
  // the down epilogue's fold's. `x` [tokens, hidden] bf16, `out`
  // [tokens, hidden] bf16 (the routed sum's, the all-reduce's the
  // caller's).
  void expert(const void* x, int tokens, const int32_t* topk_ids, const float* topk_w, void* out,
              cudaStream_t stream);

  const Dsv4HashConfig& config() const { return cfg_; }
  int layer() const { return layer_; }
  bool is_hash_layer() const { return w_.hash_layer; }

 private:
  struct Layout;
  static Layout layout(const Dsv4HashConfig& cfg, int max_tokens);
  IGemm& gemm_;
  Dsv4HashConfig cfg_;
  int max_tokens_;
  void* gemm_ws_ = nullptr;
  size_t gemm_ws_bytes_ = 0;
  Dsv4HashWeights w_;
  int layer_ = 0;
  uint8_t* scratch_ = nullptr;
  int32_t* topk_ids_ = nullptr;
  float* topk_w_ = nullptr;
  uint16_t* gate_ = nullptr;
  uint16_t* up_ = nullptr;
  uint16_t* down_ = nullptr;
};

// The v4-owned fused top-k router kernel (the dsv4-native
// ops/moe/router's V4 re-expression; the sqrtsoftplus scoring's + the
// noaux_tc biased selection's + the renormalize's + the
// routed_scaling_factor's, AND the V4-only hash-layer's tid2eid
// table-lookup's mode's). The one CTA per token's; the learned's the
// gate's the n_experts' scores' the top-k's (the biased desc's, the
// idx asc's the ties's); the hash's the tid2eid table's the table
// order's (the duplicate-id's the independent's, the no bias's). The
// numerics are certified against the CPU oracle (tests/unit/
// dsv4_moe_oracle_test.cpp's dsv4_moe_hash_router_tid2eid +
// dsv4_moe_learned_router_tiebreak_and_order) — the GPU-gate pending's
// parity gate.
void dsv4_moe_router(const uint16_t* logits, const float* bias, const int64_t* tokens, const int64_t* tid2eid,
                     bool hash, int m, int e, int topk, float scale, bool renormalize, int32_t* out_ids,
                     float* out_w, cudaStream_t stream);

}  // namespace dgpp
