# DeepSeek-V4-Flash (0731) kernel port spec

> Phase B de-risking (2026-09-17, night pass, code-only). File-level port
> map for the V4 loader + kernel-geometry work: which DeepSeek-V4.1 (dsv41)
> DGPP code is the base to adapt, the exact geometry deltas, which
> `~/work/dsv4-native` op + test to borrow numerics/oracles from (file:line),
> and what is genuinely new vs adapted. The one genuinely new numerical
> surface — the fp8 128x128 scale-contract area — is answered in §3 with
> file:line evidence, and pinned by the CPU oracle in §4.
>
> Sources of truth (all read-only):
> - **dgpp dsv41** — `src/models/dsv41/`, `src/kernels/{csa2,dsv41_dspark,
>   dsv41_engram,glm_moe,fp4_gemv,scale_gemm,fp8_gemv,fp8_dequant,mma_gemv}*`
>   (the base tree; `docs/deepseek_v41_flash_plan.md` §3 "Reuse, adapt, new"
>   is the house format this spec follows).
> - **dsv4-native** (`~/work/dsv4-native`) — the separate native C++ V4
>   engine (Vision-Exp variant), ops oracle-qualified + token-verified
>   (`docs/STATUS.md` §2-§5, `docs/KNOWLEDGE.md`).
> - **The checkpoint** `/data/models/DeepSeek-V4-Flash-0731` (config.json +
>   the 48 safetensors shard headers, verified 2026-09-17).

## 1. Geometry delta (dsv41 → V4-0731)

Every value verified against the checkpoint's `config.json` + the shard
headers (dtype/shape census 2026-09-17; 72,317 tensors, 166.9 GB).

| key | dsv41 (V4.1-Flash) | V4-0731 | evidence (0731) |
|---|---|---|---|
| hidden | 5120 | **4096** | `config.json hidden_size` |
| layers / nextn | 40 / 3 | **43 / 1** | `num_hidden_layers 43`, `num_nextn_predict_layers 1`; checkpoint: `layers.0..42` + `mtp.0.*` |
| compress_ratios | `[0,0,(4,128)×19,0,0,0]` | **46 entries: `[0,0,(4,128)×21,0,0,0]`** — C4A (ratio 4, indexer) = even layers 2..42 (21), C128A (ratio 128) = odd 3..41 (20), SWA-only 0,1; tail 43..45 = the 3 nextn slots | `compress_ratios[42] == 4` (layer 42 IS a C4A indexer layer) |
| dspark targets | [37,38,39] | **[40,41,42]** (C4A/C128A/C4A — all compressed) | `dspark_target_layer_ids` |
| dspark block / markov | 5 / 256 | **5 / 256** (same), noise 128799 | `dspark_block_size` etc. |
| dspark experts | 128 routed / top-3 (draft set) | **reuses the backbone 256-expert set** (no separate drafter experts) | `mtp.0.ffn.experts.0..255` = 256 experts, same geometry as `layers.*` |
| routed experts / topk | 384 / 6 | **256 / 6** | `n_routed_experts`, `num_experts_per_tok` |
| moe inter | 2304 | **2048** | `moe_intermediate_size`; expert `w1 [2048, 2048]` I8 (4096 e2m1 codes) |
| q_lora | 1280 | **1024** | `wq_a [1024, 4096]` F8_E4M3 |
| index heads | 32 | **64** (dim 128, topk 512) | `index_n_heads 64`; `indexer.wq_b [8192, 1024]` = 64×128 |
| fp8 dense grid | 32×32 (e8m0/32) | **128×128 (e8m0/128)** | `quantization_config.weight_block_size [128,128]`, `scale_fmt ue8m0`; `wq_a.scale [8, 32]` |
| fp4 expert group | 32 (e8m0/32, MXFP4) | **32 (unchanged)** | `experts.*.w1.scale [2048, 128]` F8_E8M0 = 4096 codes/32 |
| activation scheme | (bf16 math pinned) | **`dynamic`** — the reference `inference/kernel.py` act_quant + fp8_gemm (per-128 e4m3 + ue8m0 on BOTH operands, §3.2) | `quantization_config.activation_scheme` |
| mHC | 4-copy single-pass (hc_mult 4, sinkhorn 20, eps 1e-6) | **same constants** | `hc_mult`, `hc_sinkhorn_iters`, `hc_eps`; 9-tensor binding `hc_{attn,ffn}_{fn,base,scale}` + `hc_head_{fn,base,scale}` |
| engram | present (layers 1, 14) | **ABSENT — 0 engram tensors** | tensor census (2026-09-17) |
| hash layers | — | **NEW: `num_hash_layers 3`** — layers 0-2 route by `ffn.gate.tid2eid` I64 [129280, 6] (token → 6 expert ids) instead of a learned `gate.bias` (layers 3..42 ship the F32 bias; layers 0-2 do not) | tensor census |
| lm_head / embed | BF16 (the dsv41 loader's sliced `head.weight` bf16 copy) | **`head.weight` BF16 [129280, 4096], `embed.weight` BF16 [129280, 4096]** (unquantized, same as dsv41) | tensor census |
| compressor dtype | BF16 (`load_bf16`, `dsv41/loader.cpp:336-337`) | **BF16 (same)**: C4A `compressor.{wkv,wgate} [1024, 4096]` + `ape [4, 1024]`; C128A `[512, 4096]` + `ape [128, 512]`; indexer compressor `wkv [256, 4096]` + `ape [4, 256]` | tensor census |
| mtp.0 extras | — | `main_proj` F8_E4M3 **[4096, 12288]** (3 target-layer 4096-dim features concatenated) + e8m0 scale [32, 96]; `confidence_head.proj` | tensor census |
| rope | YaRN factor 16, θ 10000 / compress 160000 | **identical** | `rope_scaling`, `compress_rope_theta` |

## 2. Component port map

Format per component: **(a)** the dsv41 base to adapt, **(b)** the geometry
deltas, **(c)** the dsv4-native op + test to borrow numerics/oracles from
(file:line), **(d)** genuinely new vs adapted.

### 2.1 csa2.cu — compressor / indexer / sparse attention

**(a)** `src/models/dsv41/csa2_layer.cu` + `src/kernels/csa2.cu` (the whole
CSA2 layer object: low-rank q, window latent, compressor, index keys,
selection, two-source attention, grouped wo). Rebind the `Csa2Config`
(`src/models/dsv41/csa2_layer.hpp:39-46`) to the V4 geometry; the
layer-class dispatch (Window/Full/Reindex/Reuse, `src/models/dsv41/
config.hpp Dsv41LayerMode`) becomes the V4 three-class dispatch
(SWA-only / C4A / C128A — `src/models/dsv4/config.hpp`
`is_index_layer`/`compressor_coff` already encode it).

**(b)** hidden 5120→4096, q_lora 1280→1024, index heads 32→64
(`indexer.wq_b [8192, 1024]`), index_topk 512 unchanged, SWA window 128
unchanged. Compressor projections are BF16 in BOTH releases (the dsv41
loader already takes `load_bf16` for `compressor.{wkv,wgate}`,
`src/models/dsv41/loader.cpp:336-337`) — V4 keeps the BF16 GEMM path
(`src/kernels/gemm.hpp` / `bf16_gemv`) + the F32 `ape` adds
(`compressor.ape [4, 1024]` C4A / `[128, 512]` C128A; the dsv41 ape
shapes differ: ratio-2/1 pairs). The ratio-4 compressor stays the
overlapping variant (coff 2), ratio-128 plain (coff 1) — same as
dsv41's ratio-2/ratio-1 pair, one geometry swap.

**(c)** dsv4-native `src/ops/compressor/compressor.h:1-60` (the two-step
`state_store` + `compress` contract: the per-dim softmax pooling, the
fp32 RMSNorm, the GPT-J interleaved-pair RoPE on the last 64 dims, the
bf16 round-trip-then-quant of the NoPE 448 dims, the 584 B record layout)
+ `src/ops/compressor/compressor.cu`; oracle `tests/test_op_compressor.cpp`
(the independent double reference, the FLIP-RATE record guard at
`tests/test_op_compressor.cpp:366-381`, the C3.1 bf16-ulp budget at
:196-197). The indexer q-side fold + per-(compressed-)token topk-512:
dsv4-native `src/ops/indexer_token/indexer_token.h:1-150` (the 132 B/token
V3.2 cache, the q-side `fused_indexer_q_rope_quant` numerics, the
topk (score desc, index asc) tie-break) + oracle
`tests/test_op_sparse_attn.cpp:177-186` / `tests/test_op_indexer_token.cpp`
(`indexer_token_ref`, `ops/common/op_test.h:1601`). The sparse-MLA
attention over the 512×4 raw records: dsv4-native
`src/ops/sparse_attn/sparse_attn.h:1-30` (one CTA per row, 512 threads,
the e4m3×ue8m0 in-register dequant, the online-softmax in the 512-dim
latent) + oracle `tests/test_op_sparse_attn.cpp` (`sparse_attn_ref`,
`ops/common/op_test.h:1900`).

**(d)** Adapted: the csa2 layer skeleton, the two-source attention, the
selection, the BF16 compressor GEMM. New: the 64-index-head fold width
(the dsv41 indexer is 32 heads — the SMEM tile exchange of
`fused_indexer_q_rope_quant` is sized for 64×64, dsv4-native KNOWLEDGE
D20's locked gotcha), the 584 B record's C128A variant (block 8, coff
1).

### 2.2 dsv41_dspark.cu — DSpark draft glue + union attention

**(a)** `src/kernels/dsv41_dspark.{cu,hpp}` (stream mean, block rows,
Markov-biased head row, confidence logit) + the DSpark draft transaction
in `src/models/dsv41/model.cpp` (the `glm_spec` feed) +
`src/kernels/glm_spec.cu`.

**(b)** block 5 / markov_rank 256 / noise 128799 unchanged; targets move
[37,38,39]→[40,41,42]; **the draft reuses the backbone's 256-expert MoE**
(0731's `mtp.0` ships the full 256-expert set + `main_proj [4096, 12288]`
+ `confidence_head`) instead of dsv41's separate 128-expert drafter —
the draft stage's MoE config drops the `draft` variant entirely
(`src/models/dsv4/config.hpp moe_config`: "the draft stages reuse the
backbone's expert set"). One nextn layer (mtp.0) instead of three.

**(c)** dsv4-native `src/ops/dspark_attn/dspark_attn.h:1-120` (the
single-softmax-over-the-UNION-of-three-KV-sources contract: the C4A
compressed pool + the 128-slot projected-main-hidden ring + the
in-memory bf16 draft/verify block KVs, non-causal, the `attn_sink`
exactly-once) + the double oracle `tests/test_op_dspark_attn.cpp:205-214`
(`dspark_attn_ref`, `ops/common/op_test.h:1990`; the +inf degenerate
limit + the finite vs-oracle cases at `tests/test_op_dspark_attn.cpp:20-25`).
The drafter's acceptance numerics: `tests/test_draft_diff_oracle.cpp`
(draft-vs-verify diff, the p1 acceptance gate) + `docs/KNOWLEDGE.md` D30
(the locked fact #3 the union semantics).

**(d)** Adapted: the block-rows / Markov / confidence glue (same shapes),
the draft transaction. New: the `main_proj [4096, 12288]` fp8 GEMM (3×
target-layer hidden concatenated — a new dense-fp8 shape, §3.1), the
backbone-expert draft MoE (no draft expert set to bind).

### 2.3 dsv41_engram.cu — REFRAMED: V4-0731 has no Engram

The 0731 checkpoint ships **zero** engram tensors (tensor census
2026-09-17: `engram*` / `hash*` name hits = 0; the `num_hash_layers`
field is the only "hash" surface). The dsv41 Engram module
(`src/kernels/dsv41_engram.{cu,hpp}`, the hashed n-gram memory at layers
1/14) is **not ported** — there is nothing in the 0731 checkpoint to
bind. What V4 has instead is the **hash-routing layer** (§1):
`num_hash_layers 3`, the `ffn.gate.tid2eid` I64 [129280, 6] table —
layers 0-2 select their 6 experts by token id (a deterministic
table lookup) with **no gate bias** (the bias exists only on layers
3..42). That is a small NEW router mode: the dsv41 router
(`src/models/glm/moe.hpp` `GlmMoeConfig` + the router kernel in
`src/kernels/glm_moe.cu`) gains a `tid2eid` input (int64 table, top-k
column slice) in place of the bias-add + score path.

**(c)** dsv4-native `src/ops/moe/router/router.{h,cu}` (the router op)
+ `tests/test_op_router.cpp` for the top-k selection numerics
(score-desc, index-asc tie-break — the same convention the indexer's
topk uses, `tests/test_op_indexer_token.cpp`); the hash-table routing
itself has no dsv4-native counterpart (the Vision-Exp model has no
hash layers) — the oracle for it is the table's own semantics
(`tid2eid[token, :6]` → the 6 expert ids, no scoring), pinned by a
trivial host test in the Phase B loader.

**(d)** New: the tid2eid router mode. Not ported: the Engram kernels
(hash / gather / gate) — dead code for V4.

### 2.4 The fp4 (MXFP4) MoE core

**(a)** `src/kernels/fp4_gemv.{cu,cuh}` (`kMxGroup` = 32, e8m0/32 scale
mode, `src/kernels/fp4_gemv.cuh:77-78` + the MXFP4 dispatch at
`src/kernels/fp4_gemv.cu:45`) + the grouped/tile MXFP4 launches in
`src/kernels/glm_moe.cu` (`kMxfp4Group` paths, `glm_moe.cu:3803-3824`
`check_fp4_mma_shape` + the grouped-mma at :3063) + `GlmMoeLayer`
(`src/models/glm/moe_layer.hpp`, `experts_fp4`,
`src/models/glm/moe.hpp:126-173`). This is dsv41's D2 decision
(`docs/deepseek_v41_flash_plan.md` §4 D2: "MXFP4 is a scale variant of
the fp4 core, not a new core") — reused as-is.

**(b)** 256 experts (vs 384), inter 2048 (vs 2304), K = 4096 (vs 5120):
the compiled-K set of `fp4_gemv` gains 4096 (the K=5120/576 set's
pattern); per-rank TP shards of `n = 2048` (world 2) or the full 2048
(single-node). The shared expert stays the fp8 triple
(`shared_experts.w{1,2,3}` F8_E4M3 + e8m0/128). The hash layers'
`tid2eid` feeds the router (§2.3); `sqrtsoftplus` scoring + `noaux_tc`
topk + ×1.5 routed factor + SwiGLU clamp 10 — all unchanged from dsv41.

**(c)** dsv4-native `src/ops/moe/experts/experts.h:1-60` (the fused
W4A16 MXFP4 decode contract: e2m1×2^(byte−127) exact in fp32, the
per-32 block partial scaled once, the swiglu-clamp, the router weight
folded into the down epilogue, the per-pair scratch + combine) + the
naive per-expert DOUBLE oracle `tests/test_op_moe_experts.cpp:1-4, 31-99`
(the term-based tolerance `moe_tol`); the shared-expert fp8 half:
`tests/test_op_moe_shared_expert.cpp` (`shared_expert_fp8_ref`,
`ops/common/op_test.h:334`); the gate/router: `tests/test_op_moe_gate.cpp`.

**(d)** Adapted: everything (the MXFP4 core is dsv41-landed). New: the
K=4096 compile entry + the 256-expert geometry + the tid2eid-fed
router on layers 0-2.

### 2.5 mHC single-pass

**(a)** `src/kernels/glm_mhc.cu` + `src/kernels/glm_mhc_launch.hpp`
(`launch_mhc_compute_normed` with the `MhcSinglePass` struct — the
dsv41 single-pass form: `pre_in`/`pre_out`, the fp32 `post_f32`/
`comb_f32` exports, `launch_mhc_stream_update_f32` the one-rounding
update, `launch_mhc_collapse_normed` the head collapse) +
`src/models/glm/mhc.hpp` (`GlmMhcConfig`/`GlmMhcWeights`). The dsv41
plan's row: `docs/deepseek_v41_flash_plan.md:605` ("adapt:
single-pass (pre_in/pre_out), fp32 fn, eps, one-rounding update,
weighted final collapse").

**(b)** `fn` width 4×5120 → 4×4096: per-layer `hc_{attn,ffn}_fn`
F32 [24, 16384] (24 = (2+4)×4 = `hc_coeff_rows()`), `_base` [24],
`_scale` [3]; top-level `hc_head_fn` [4, 16384], `hc_head_base` [4],
`hc_head_scale` [1]. 43 backbone layers + mtp.0 (the nextn layer
carries its own 9-tensor set). Sinkhorn 20 iters, eps 1e-6, hc_mult 4
— unchanged.

**(c)** dsv4-native `src/ops/mhc/mhc.h:1-100` (the 4-copy reference
math: the `hc_pre` flatten-RMS + `[24, 4h]` GEMM + the 20-iter Sinkhorn
+ the y combine; `hc_post` the 4→4 expansion; `hc_head` the 4→1
collapse BEFORE the final norm — the exact sequence pinned, the
K-split bit-exactness note) + the naive FP64 oracle
`tests/test_op_mhc.cpp:106-112` (the independent double loops, the
FP64 sigmoid), the doubly-stochastic invariant probe
`tests/test_op_mhc.cpp:339-386` (the 5e-2 algorithm tolerance — the
FP64 ref's own 20-iter Sinkhorn deviates 0.0228, KNOWLEDGE D28) and the
16-lane `shfl_xor` mask gotcha (D28 verification finding 1).

**(d)** Adapted: the single-pass launcher set (the dsv41 D4 form is the
DGPP re-expression of this same 4-copy state machine). New: nothing
numerical — only the [24, 16384] width and the mtp.0 site.

### 2.6 lm_head / embed

**(a)** The dsv41 loader's `Embed`/`LmHead` weight-class builders
(`src/models/dsv41/loader.cpp` `is_replicated`, the `load_bf16` /
`load_bf16_col_ranges` paths — the replicated lm head, the sliced
embedding rows) + `src/kernels/bf16_gemv.cu` (the lm_head GEMV) + the
embedding gather in the model layer.

**(b)** Both are **BF16 [129280, 4096] in 0731** (dsv41's `head.weight`
was BF16 too — the dsv41 loader's plain sliced bf16 copy, `src/
models/dsv41/loader.cpp:545-555`; only the dense projections are fp8 in
that release). vocab 129280 unchanged; the `dspark_noise_token_id`
128799 and the image-free token space stay inside it. The Markov head
(`dspark_markov_rank 256`) rows come from the embedding (the
`markov_embed`/`markov_head` the dsv41 dspark glue reads —
`src/kernels/dsv41_dspark.hpp` `dsv41_dspark_markov_bias`), sliced per
rank over the 256-dim rank.

**(c)** dsv4-native `tests/test_op_linear.cpp` covers the
`lm_head/embed` dense math only in its bf16-input follow-up (the spec's
`linear_bf16`, `src/ops/linear/linear.h:40-41` "a bf16 GEMM only
serves MTP draft inputs (Phase 6+)") — for the BF16 lm head the
oracle is the plain bf16 GEMM double reference (the same class as
`tests/test_op_rmsnorm.cpp`'s naive double loops); the sampler
(greedy over 129280) is `tests/test_op_sampler.cpp`.

**(d)** Adapted: the whole path (bf16 load + bf16 GEMV). New: nothing.

### 2.7 The fp8 dense GEMM surface — see §3 (the scale-format verdict)
and §3.2 (the dynamic activation scheme, the one genuinely new
numerical surface).

## 3. The fp8 128x128 scale-format question

> Question (task §2): does the existing 128x128 path consume E8M0 (not
> F32) scales the way the V4 checkpoint ships them, or is there a
> scale-format gap?

**Verdict: there is a scale-format gap at the kernel boundary — the
existing 128x128 path consumes F32 scales ONLY. The checkpoint's
F8_E8M0 bytes must be decoded to F32 at load time (the dsv41 pattern).
No kernel change is required; the minimal change is loader-side.**

### 3.1 The evidence (file:line)

- The tile GEMM's scale load is `const float*` end to end:
  `src/kernels/scale_gemm.cu:67` — `scales[(size_t)(gn >> rs) *
  scale_cols + scale_col]` with `scales` a `const float*`
  (`scale_gemm.cu:39`), and `scale_cols = (k + (1 << cs) - 1) >> cs`
  (`scale_gemm.cu:43`). The GEMV core: `src/kernels/fp8_gemv.cuh:221`
  (`scale_row = scales + (row >> rs) * scale_cols`) and
  `src/kernels/fp8_gemv.cuh:148, 333` (`scale_row[c0 >> cs]`). The
  dequant bridge: `src/kernels/fp8_dequant.cu` (F32 scale grid,
  `launch_fp8_dequant_blocks`). The streaming tensor-core form:
  `src/kernels/mma_gemv.hpp:45-49` (`launch_mma_gemv_fp8_{bf16,f32}`,
  `const float* scales`). **No fp8 kernel in the tree decodes e8m0.**
- The MXFP4 (fp4) core, by contrast, DOES consume e8m0 bytes natively:
  `src/kernels/fp4_gemv.cu:318` — `scales[row * scale_cols_of(K,
  kMxGroup) + boff / 16]` over a `const uint8_t*`
  (`src/models/glm/moe.hpp:147-150`, `scale_group 32`, e8m0, no
  global). So "e8m0 consumption" exists in the tree — only on the fp4
  axis.
- The dsv41 loader closes exactly this gap for the 32×32 grid:
  `src/models/dsv41/loader.cpp:24-32` (`e8m0_to_float`: "2^(byte - 127);
  255 is NaN") + `:116-119` (`convert_scales` materializes the F32
  grid) + `:137, 184` (the call sites); `loader.hpp:5-6` states the
  contract ("converted to the fp32 grid the fp8 core reads, plan D2").
  The dsv41 plan row: `docs/deepseek_v41_flash_plan.md` §3 ("e8m0
  scales converted to fp32 at load").
- The 0731 checkpoint ships the scales as `F8_E8M0`
  [ceil(N/128), ceil(K/128)] (e.g. `layers.0.attn.wq_a.scale [8, 32]`
  against `wq_a [1024, 4096]`; `mtp.0.main_proj.scale [32, 96]`
  against `[4096, 12288]`). Verified 2026-09-17: **0 of the 8,832 MiB
  of F8_E8M0 scale bytes in the 35,718 scale tensors is 0xFF** — the
  255→NaN policy of `e8m0_to_float` is unobservable in this
  checkpoint, so the V4 loader may keep it (or map 255→0.0) without
  numerical effect; observed byte range 62..161.

**The minimal change for the Phase B loader (owner: the dsv4
config+binding agent's loader half) — NO kernel edits:**

1. In the V4 fp8 builder (the dsv41 `load_fp8`/`load_fp8_rows`/
   `load_fp8_col_ranges` pattern, `src/models/dsv41/loader.cpp:123-196`
   as the template), set `b = kDsv4Fp8Block = 128` (vs dsv41's 32) and
   run the SAME `e8m0_to_float` decode (`ldexp(1.0f, byte - 127)`,
   255→NaN) over the source scale rows into the materialized F32
   `q.scales` grid — the decode is byte-for-byte the dsv41 one; only
   the grid dimensions change (ceil(N/128)×ceil(K/128)).
2. Column/row slices: the dsv41 "whole 32-column blocks" constraint
   (`load_fp8_col_ranges`, `loader.cpp:154-156`) becomes "whole 128
   blocks" for the V4 dense matrices — check every planned V4 slice
   (the `wo_a`/`wo_b` group slices, the TP re-blocks) against it; a
   TP-sliced axis re-blocks at `gcd(128, slice)` and passes
   `rs`/`cs` < 7 to the existing `launch_scale_gemm_grid_*`
   (`src/kernels/scale_gemm.hpp` — the grid launchers already take
   `rs`/`cs` 5..7).
3. The MXFP4 expert builders copy dsv41's `load_mxfp4_rows/cols`
   (`loader.cpp:230-278`) unchanged — e8m0 bytes stay e8m0 (the fp4
   core consumes them natively, §3.1).
4. The CPU oracle for all of this already exists:
   `tests/unit/dsv4_fp8_scale_test.cpp` (§4) — the decode table and
   the index math are pinned BEFORE the loader lands; the loader's
   decode must pass that test's contract (add the loader's own
   `e8m0_to_float` to the test's target set once it exists, or
   keep the mirror + a digest check of the converted grid).

### 3.2 The genuinely new numerical surface: the dynamic activation scheme

The checkpoint's `inference/kernel.py` (the pinned reference) does NOT
run bf16-math GEMMs on the dense fp8 matrices:

- `act_quant` (`kernel.py:41-130`): per-row, per-128-block
  `amax` (1e-4 floor) → `scale = round_pow2(amax / 448)`
  (`fast_round_scale`, `kernel.py:36-37`) → e4m3 codes
  `clamp(x / scale, ±448)`; the scale stored as **ue8m0**
  (`scale_fmt`), i.e. a power of two.
- `fp8_gemm` (`kernel.py:204-281`): `C = A @ B^T` with per-128-block
  scaling on **both** operands — each 128-wide k-block partial
  (fp8×fp8 MMA, fp32 accumulate) is scaled by
  `a_scale[m, k/128] × b_scale[n/128, k/128]` (cast to fp32), block
  partials summed, ONE bf16 rounding at the output.

That is the dsv4-native linear op's contract, verbatim:
`src/ops/linear/linear.h:1-40` —
`out[m,n] = bf16( Σ_kb 2^(xs[m,kb] + ws[nb,kb] − 254) · Σ_{k∈kb}
x[m,k]·w[n,k] )` (the "no fp32 intermediate in global memory"
bandwidth rule), qualified by `tests/test_op_linear.cpp`
(shapes (1/64/512, 4096, 4096) + the M=65/192 dispatch boundaries,
`tests/test_op_linear.cpp:1-30`; the naive double reference
`linear_fp8_ref` / `linear_fp8_ref_precomputed`,
`ops/common/op_test.h:515, 543`; the zero-input bit-exact 0x0000
probe at `tests/test_op_linear.cpp:25-30`).

**DGPP's existing dense path is NOT this**: `src/kernels/scale_gemm.hpp:1-15`
pins "the reference numerics (dequantized weights, bf16 math), NOT
DeepSeek-style dynamic activation quantization" — bf16 activations ×
dequantized-fp8-weights (bf16×bf16 mma, F32 scales applied in the
weight-tile load). For dsv41 that was the pinned reference's math; for
V4 the pinned reference (`inference/kernel.py`) IS the dynamic scheme,
so **the fp8×fp8 block-scaled GEMM + the per-128 activation
quantizer is the one genuinely new numerical surface in this port.**

Options for tomorrow (decision, flagged — the spec does not fix it):

- **A (parity): implement the dynamic scheme.** New kernels: an
  `act_quant` (bf16 → e4m3 + ue8m0/128, the 1e-4 amax floor + the
  power-of-two scale) and an fp8×fp8 block-scaled GEMM/GEMV (the
  per-128-k-block partial × 2^(xs+ws−254), one bf16 rounding).
  Oracles: dsv4-native `ops/linear` + `tests/test_op_linear.cpp`
  (borrow the double reference + the term-based tolerance model);
  the CPU pin of the block-rescale math already exists in
  `tests/unit/dsv4_fp8_scale_test.cpp` (`dsv4_block128_partial_rescale_math`).
  The §3.1 F32-scale decode is then still needed for the WEIGHT side
  only if a bf16-act fallback path is kept.
- **B (dsv41-style): keep the bf16-act path** (`launch_scale_gemm_*`
  with F32-decoded e8m0 weight scales). Numerics differ from the
  reference by the activation-quant error class (e4m3×ue8m0 on the
  activations); token gates may still pass (the dsv41 precedent: the
  engine's quant-class delta vs the reference is a documented
  tolerance class, `docs/KNOWLEDGE.md`-style "QUANT/KERNEL CLASS"
  finding). Cheaper, but it is a conscious deviation from the pinned
  reference — record the delta against the `linear_fp8_ref` oracle.

Either way the WEIGHT-side scale-format gap (§3.1) is closed the same
way: loader-side e8m0→F32 decode, no kernel edit.

## 4. The CPU oracle contract (this pass)

`tests/unit/dsv4_fp8_scale_test.cpp` (host, no GPU; registered in the
root `CMakeLists.txt` as `dsv4_fp8_scale_test`, label `host`):

| test | pins |
|---|---|
| `dsv4_e8m0_decode_full_table` | the loader's decode formula (`ldexp(1.0f, b−127)`, 255→NaN) against the naive bit-level oracle (IEEE bits `b<<23` for 1..254, the denormal 2⁻¹²⁷ bits for 0) — all 256 bytes |
| `dsv4_e8m0_decode_finite_range` | no overflow for a finite byte (0→2⁻¹²⁷ denormal, 254→2¹²⁷), strict monotonicity |
| `dsv4_block128_scale_index_math` | the 128×128 scale-index arithmetic for [N,K] with N,K not multiples of 128: kernel formulas (`scale_gemm.cu:43,67`, `fp8_gemv.cuh:148,221,333`) vs the naive `(n/128)·ceil(K/128) + k/128` reference, over the V4 dense shapes + ragged tails; the stage-single-column (cs≥5) and 16-byte-chunk-single-column (cs≥4) invariants |
| `dsv4_block128_partial_rescale_math` | the per-128-k-block rescale of the §3.2 dynamic-scheme GEMM (`2^(xs+ws−254)` exact, the fp32 accumulation bounded by fp32 rounding) vs an exact double reference — the contract pin for the new fp8×fp8 kernel before it exists |

## 5. Open items for tomorrow

1. **The §3.2 decision (A parity vs B deviation)** — owner: the Phase B
   kernel-geometry agent. If A: new `act_quant` + fp8×fp8 block-scaled
   GEMM (borrow `ops/linear` numerics, `tests/test_op_linear.cpp`
   oracle); if B: document the quant-class delta.
2. **The tid2eid router mode** (§2.3) — the one V4-only router surface;
   trivial host oracle, the selection semantics are the table's.
3. **The 584 B record's C128A variant** (§2.1) — dsv4-native's C128A is
   the ratio-128 plain-pooling instantiation; the 0731 C128A ships a
   `wgate` ([512, 4096] BF16) — confirm against
   `inference/model.py`'s `Compressor` whether the 0731 C128A pooling
   is gated (the Vision-Exp reference's C128A was plain).
4. **TP=2 placement** — the dsv41 family is world-4; V4's geometry
   (64 heads / 8 groups / 256 experts / 4096 hidden) divides at world 2
   (n=2048, k=4096 per rank) — the `launch_scale_gemm_grid_*` re-block
   (`rs`/`cs` 6/5) is the dsv41 D2 pattern; the dsv4-native TP=2 MoE
   (`src/ops/moe/layer/layer_tp2.cu` + `tests/test_op_moe_layer_tp2.cpp`)
   is the oracle for the sharded expert layout.
5. **The mtp.0 `main_proj [4096, 12288]` fp8 shape** — a new dense-fp8
   GEMM shape (K = 12288, not a multiple of the 128-block? 12288 = 96×128
   ✓ — it divides; the scale grid [32, 96] confirms it).
