---
license: mit
base_model: zai-org/GLM-5.3-BF16
base_model_relation: quantized
pipeline_tag: text-generation
library_name: transformers
tags: [glm, moe, rtn, int4, int8, compressed-tensors, dgpp]
---

# GLM-5.3 Int4/Int8 mixed-precision round-to-nearest, group 64

A mixed-precision quantization of **GLM-5.3** (754B-parameter MoE, model type `glm_moe_dsa`) made from the
**original BF16 release** (`zai-org/GLM-5.3-BF16`, not the derived FP8 checkpoint). Built for the
DGPP inference engine on four NVIDIA DGX Sparks (TP=4), where the whole model has to fit
in ~100 GiB per rank with room left for context.

| | |
|---|---|
| Total size | 398 GiB (≈ 99.8 GiB per rank at TP=4) |
| Routed experts, layers 3–77 (`mlp.experts.*.{gate,up,down}_proj`) | **int4** symmetric, group 64, bf16 scales |
| Attention (`q_a`, `q_b`, `kv_a_proj_with_mqa`, `kv_b`, `o_proj`) and shared experts, layers 3–77 | **int8** symmetric, group 64, bf16 scales |
| Dense layers 0–2, DSA indexers, MoE routers, norms, embeddings, `lm_head`, the whole MTP block (layer 78) | **bf16**, byte-exact from the source |
| Format | compressed-tensors `pack-quantized` (`weight_packed` int32, `weight_scale` bf16, `weight_shape`), declared in `config.json` |

## Method

Plain **round-to-nearest** with per-group min-max scales (llm-compressor's `QuantizationModifier`, no calibration
data, no activation-aware smoothing), one decoder layer at a time. Every layer is independent of every other, and
because nothing is folded into the norms there is no interaction with the router or the DSA indexer. This is the
simplest possible checkpoint in this format and the baseline the AWQ companion is measured against.

## Measured error (per layer)

Relative RMS error against the bf16 layer on 32 × 2048 real tokens. "Contribution" is the error of what the layer
adds (output minus input); "residual" is the error of the full residual stream leaving the layer, which is the
quantity that propagates.

| layer | variant | contribution | residual |
|---|---|---|---|
| 3 | **this checkpoint** (RTN int4 g64) | 2.40 % | 0.54 % |
| 3 | AWQ int4 g64, fold-corrected (companion) | 2.38 % | 0.53 % |
| 3 | fp8 e4m3 g128 on every projection, for reference | 2.46 % | 0.55 % |
| 3 | int8 g64 on every projection, for reference | 1.49 % | 0.33 % |
| 6 (owns an indexer) | **this checkpoint** | 3.44 % | 1.58 % |
| 6 | AWQ, fold-corrected (companion) | 3.27 % | 1.50 % |
| 21 | **this checkpoint** | 11.09 % | 0.59 % |
| 21 | AWQ, fold-corrected (companion) | 10.95 % | 0.58 % |
| 42 (owns an indexer) | **this checkpoint** | 9.89 % | 2.01 % |
| 42 | AWQ, fold-corrected (companion) | 9.85 % | 2.00 % |
| 63 | **this checkpoint** | 12.50 % | 2.34 % |
| 63 | AWQ, fold-corrected (companion) | 12.50 % | 2.34 % |

The contribution metric grows with depth because each layer's addition shrinks relative to the residual stream;
the residual error rises from 0.5 % (layer 3) to 2.3 % (layer 63). On this model the int4/int8 mix lands at the error of an all-fp8 recipe at
half the bytes. The AWQ companion (`HawkBearPig/GLM-5.3-Int4-Int8Mix-AWQ-g64`) improves on this checkpoint by
0.9 / 4.8 / 1.2 / 0.4 / 0.02 % of the error at layers 3 / 6 / 21 / 42 / 63: real in the shallow indexer layer, negligible at depth. Per layer the two checkpoints are equivalent; this one is the simpler artifact (no calibration, no smoothing, no correction step).

**No end-to-end evaluation yet** (perplexity, benchmarks): the numbers above are per-layer.

## Loading

Standard compressed-tensors layout: `model.safetensors.index.json`, per-layer shards `layer-NNN.safetensors`,
`passthrough.safetensors` (embeddings, norms, `lm_head`, MTP block), `config.json` with `quantization_config`,
tokenizer files. Any loader that understands `glm_moe_dsa` and compressed-tensors `pack-quantized` W4A16/W8A16
should read it; only DGPP has been exercised. vLLM / transformers loading is untested.

Tooling: llm-compressor 0.13.0, compressed-tensors 0.18.0, transformers 5.14.1, torch 2.13.0, CPU-only build.

License: MIT, inherited from GLM-5.3.

---

## DGPP serving notes (2026-09-12)

The repository copy of the card above, with what the engine measured. The
port is `docs/glm53_plan.md`; the family is `glm_moe_dsa` in `dgpp-serve`.

**Memory plan, world 4 (rank 0, the release as shipped):** model weights
resident 99.30 GiB per rank (the draft layer's BF16 experts requantized to
int4/int8 at load; 102.4 GiB with them kept BF16), then per context token
91.8 KiB with the bf16 latent cache and 52.6 KiB with the fp8 one (the 512
latent + the 64-wide rope key on all 79 layers, the fp8 index entry on the
22 indexed layers, the draft's included). The templates
carry a 144K-token bf16 cache without MTP (`plain`, plan 110.75 GiB), 120K
bf16 with MTP (`mtp1`, 110.41) and 208K fp8 with MTP (`mtp1_large-cache`,
110.35), the embedding vocab-sharded (`engine.embed_sharding: vocab`, −1.33
GiB per rank, bitwise the replicated lookup) — the ceilings under the
engine's 4 GiB headroom on a 121.6 GiB node, after the loader's 2.2 GiB
staging mirror was measured, made a plan item and freed before the caches
are allocated and a one-hour soak at the 120K shape showed flat memory and
no reclaim on any node (2026-09-12/13, `docs/measurements.md`). Since
2026-09-19 the shipped template keeps the BF16 matrices decode streams in
their lossless 12-bit form (`engine.bf16_weights: "bf12"`: +5–6.5 %
single-stream decode, transcripts identical) — that form ALONE: each matrix's
BF16 bytes go back to the node as its layer loads and prefill expands what it
reads (within 1 % on this model), so the footprint is 0.3 GiB UNDER the BF16
plan and the template carries the 120K shape above. (With both forms resident
— `"bf12+bf16"`, the first form of this feature — the node held 100K.)

**What the engine multiplies:** the packed codes times the bf16 group
scale, exactly (fp32), in the GEMV core the decode and the grouped prefill
run; only `kv_b` is dequantized to bf16 at load (the absorbed-attention
kernels are bf16). The draft layer's routed experts are requantized with
the checkpoint's own recipe (int4 g64) and its shared expert with int8.

**Gates (all on 2026-09-12):** the DSA layer against its host oracle at
the full geometry (rope 64, kpool 1, relu, select_k 2048); the fixture
forward against a pure-python reference, the decode / prefix / speculator
session gates, TP worlds 2 and 4, the graph engine at world 2; the
four-layer real-weight forward (27 and 2,112 tokens, streaming) against
transformers' own `GlmMoeDsa` layers on the real weights (per layer on the
engine's own input: relative l2 0.002–0.0035, one bf16 ulp of max |d|; the
engine against a fp32 reference of the dense layers 0.0025–0.0031, the bf16
floor; the MoE layer's 1–2 % rows are routing near-ties at margins under
5e-5). The checkpoint itself: `--verify-only` green on all four nodes,
`docs/checkpoint_budget_glm53.md` from the headers (every packed triple
checked against the config's quantization groups). The four nodes (2026-09-12,
`docs/benchmarks.md`, `benchmarks/results/2026-09-12-glm53-full.md`): boot
30 s from the resident image (405 s the first time), T=1 51 ms/step, MTP
depth 1 68–76 ms/pass at 1.77–1.97 tokens/pass (36–42 ms/token; acceptance
77–97 % by class), four live requests 180–185 ms per eight-row step,
prefill 7.3–9.5 ms per prompt token from 520 to 16.8K tokens (the deferred
packed tile kernel), gsm8k 59/60, HumanEval 40/40, schema extraction 30/30
with thinking on; MTP == T=1 transcripts and identical op streams across
the ranks.
