# Qwen3.8-Flash-Next — FP8 attention, Phase C plan (block scales + incoherent rotation + decode)

Branch: `feat/qwen-fp8-mma-attn` (stacked on Phase A `docs/qwen_fp8_kv_plan.md` and
Phase B.1 `docs/qwen_fp8_mma_plan.md`). Scope: **Qwen3.8-Flash-Next only** (the QSA
kernels + the Qwen serve family).

Reference techniques, adapted (not ported) — see §1 for why neither library is usable:
* FlashAttention-3 (arXiv:2407.08608): async overlap of softmax + matmul, and
  **block quantization + incoherent processing** for FP8.
* Transformer Engine `MXFP8BlockScaling` / `fp8_dpa` recipes: E8M0 power-of-two scale
  per 32 values, applied by the tensor core.

## 0. What is already done (verified 2026-10-10, HEAD `3c6f41d6`)

| piece | state | where |
|---|---|---|
| fp8 E4M3 KV pool + per-(slot, kv-head) fp32 scales | shipped (A) | `src/models/qwen/kv_pool.hpp:97-99`, `layers.hpp:235-238` |
| fused norm+RoPE+quantize+scatter append | shipped (A) | `qsa_norm_rope_append_fp8` (`src/kernels/qsa.cu:967`) |
| in-kernel fp8 dequant on all 3 attention paths | shipped (A) | `qsa.cu`, `qsa_prefill.cu`, `qsa_warp.cu` |
| hw e4m3→f16 **pair** decode in the decode gather | shipped | `qsa.cu:32-42`, `:818-832` |
| **native E4M3 MMA QKᵀ + PV, prefill-warp (B.1)** | shipped, default ON | `qsa_warp.cu:86-107` (mma/quantize), toggle `qsa_warp.cu:532` |
| reachability gate | `!decode && T >= 128 && qsa_warp_supported` | `src/models/qwen/layers.cpp:822-830` |
| e5m2 for Q / P | **NO-GO** (0.0491 → 0.0665 / 0.0685 / 0.0826) | `docs/qwen_fp8_mma_plan.md` §"e5m2 operand study" |
| **B.2 decode / short prefill** | **not started** | `qsa.cu:737` scores are still SIMT |
| **block (MXFP8) scales** | **not started** | scale plane is one fp32 per (slot, kv-head) |
| **Hadamard / incoherent rotation** | **not in the Qwen path**; asset exists | `src/kernels/dsv4_attn.cu:206` `hadamard128_smem`, `:224` kernel, `:619` wrapper |

**Two evidence problems that gate everything else**

1. *B.1 has no valid perf evidence.* `results/bench-fp8-mma-b1-*.json` (22.13 ms/chunk)
   and `bench-fp8-mma-clean-*.json` (18.99 ms/chunk) are single-shot probes with an
   **87-token prompt** → `T < 128` → the E4M3 warp kernel is **never entered**. Both
   records measure the bf16 path twice. No `make ab` record exists.
2. *The version banner is not trustworthy.* `make status` printed
   `git a91cceca95f4` (Phase A) for a process started 13 s after the binary was
   relinked with B.1 in it — the stamp is configure-time metadata. Identify deployed
   code by build timestamp + `DGPP_QSA_FP8_MMA` behaviour, not the banner.

## 1. Why we adapt FA3 / TE instead of using them

* GB10 = **cc 12.1**, built as **`sm_121a`** (`CMakeLists.txt:9`). FA3 is
  `sm_90a`-specific (TMA + warp-specialized producer/consumer + async `wgmma`); `sm_90a`
  PTX cannot JIT onto sm_121, and `src/` contains **no** `cp.async.bulk`/TMA today.
* TE's `fp8_dpa` / `fp8_mha` are **Beta** and run *only* in the cuDNN FusedAttention
  backend — dense attention, no paged block-sparse top-k gather, which is what QSA is.
* What **is** available, and verified on this box (appendix A): the
  **block-scaled FP8 MMA** `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0`
  — i.e. TE's MXFP8 recipe, in hardware. Same instruction family the NVFP4 experts
  already use (`src/kernels/moe_w4a4.cu:85` `kind::mxf4nvf4.block_scale`).
  It **requires the `a` arch suffix**: plain `sm_121` rejects it
  (`Instruction 'mma with block scale' not supported on .target 'sm_121'`) — do not
  "simplify" the arch flag.

**Honest ceiling (unchanged from Phase A/B):** at short context the K/V is < 0.1 % of the
bytes a decode step streams, so none of this buys prose/decode tok/s. The payoff is
(a) long-prefill time and (b) **tolerance headroom** — fp8 attention that lands under the
dequant path's error instead of 1.5× above it.

## 1.5 Sequencing decisions (agreed 2026-10-10)

* **C.0 defines what "success" means for the whole phase.** If prefill is neutral at
  every length, the phase's goal is **accuracy headroom**, not speed, and C.1/C.2 are
  justified by the `l2_rel` band alone. Locked because the Phase A byte-budget argument
  (K/V < 0.1 % of a short decode step) makes a decode tok/s claim unfalsifiable; challenge
  it if C.0 shows a real ≥ 10 % prefill win at ≥ 16k.
* **C.1 ships as C.1a (QKᵀ only) first.** `kTileTok` stays **16**; PV keeps the rank-1 `γ`
  epilogue. Locked because 32-token tiles + 2 stages = 32 KiB/block would cut resident
  blocks/SM ~12 → ~3; challenge it only if the C.1a error split shows PV dominating.
* **C.3b is gated on tile fill, not on effort.** If no real serving shape fills ≥ 50 % of
  the m16 A-tile, decode stays SIMT and C.3a (cp.async restore + warp stages) is the whole
  of B.2. Challenge it if the scheduler starts batching ≥ 16 query rows per block.

## 2. C.0 — Measurement gate (do this first; ~1 h, no kernel work)

Outcome: a real prefill number for B.1 on/off, and a harness that stays useful for C.1-C.3.

* Harness (built, in `dgpp-gateway`): `make ab-env VAR=DGPP_QSA_FP8_MMA A=1 B=0
  LENGTHS="4096 16384 65536 262144" REPEAT=5 BENCH=1` — `bench/ab-env.sh` boots the same
  lane once per arm (the launcher forwards any `DGPP_*` from the head's environment to
  every rank, `dgpp-cluster` `spawn_peer`), runs `bench/ttft-probe.py` (unique nonce
  prompts, server-side `timings.prompt_ms`), optionally `make bench` for the prose guard,
  restores the knob to unset, and writes `results/ab-env-<VAR>-<ts>/summary.json`.
  It **refuses to run against a live stack** (it restarts it) unless `FORCE=1`.
  `make ttft LENGTHS="..."` probes the running stack without any restart.
* Record `results/ab-fp8-mma-b1.json` with `DGPP_QSA_FP8_MMA=1` vs `0` at
  **4k / 16k / 64k / 256k** prompt tokens × ≥5 repeats, same binary, plus `make bench`
  C1/C2/C4 to prove no prose regression.
* Pre-registered expectation: **neutral at 4k** (K/V still a rounding error in the byte
  budget), **≥ 10 % faster prefill at ≥ 16k**, C1/C2/C4 within ±2 %.
* Decision: if prefill is neutral everywhere, B.1 stays default-ON (it is strictly less
  work) but C.1's motivation becomes **accuracy**, not speed — say so in the record.

## 3. C.1 — MXFP8 block scales in the tensor core (the accuracy lever)

Replace "one fp32 absmax scale per 256-dim row + a rank-1 `α·βᵏ` epilogue" with **8 E8M0
scales per 256-dim row (one per 32 values)**, applied by `kind::mxf8f6f4` inside the mma.

* **Numerics**: `score = Σ_{b=0..7} 2^(sfa_b + sfb_b) · <qa_b, kb_b>`; the hardware
  applies the pair product per k-block, so the scales no longer have to factor out of the
  sum. That is the whole point: the current per-row scale lets one outlier in a 256-dim
  row cost 3 effective bits for the entire row. Expect `l2_rel` **0.049 → ~0.02**.
* **Pool layout** (`kv_pool.hpp/.cpp`): scale plane becomes **8 bytes of E8M0 per
  (slot, kv-head)** instead of 1 float per (slot, kv-head) — 32 B/token/layer vs 16 B
  today. Cost: +1.5 % on the KV rows → pool **7.41 → ~7.52 GiB/rank**, ratio to bf16
  **1.86× → 1.83×**. Update `layer_scale_bytes()`, `init/view/reset_all`,
  `copy_block_contents` (prefix-cache block copy — `kv_pool.cpp:142`) and the
  memory-plan label.
* **Write path**: `kv_append_kernel`'s fp8 branch and `qsa_norm_rope_append_fp8` must
  reduce **8 per-block absmaxes** instead of one row absmax. The fused kernel already runs
  one warp per head for the absmax; the block form is a segmented warp reduce (8 × 32-lane
  segments = the natural `__reduce_max_sync`-shaped tree). Quantize uses the existing
  helpers `e8m0_ceil_log2_byte` / `e8m0_byte_to_float` (`latent_format.hpp:117-125`),
  so scales are powers of two by construction — no fp32 scale left in the cache.
* **Read path (warp kernel)**: drop the `α·βᵏ` epilogue; feed per-k-step scale registers
  (8 blocks → 2 × b32 per operand, byte selectors alternating `0/1` per `ks`) to
  `mma_e4m3_sf`. The bf16-dequant fallback (`DGPP_QSA_FP8_MMA=0`) must dequant from the 8
  block scales, not a row scale.
* **Spike first (C.1.0, blocking)**: an **SF-fragment oracle** — same shape as the existing
  A-fragment oracle (`qwen_fp8_mma_attn_test.cu:450`, "max |mma − host dot| = 0") that
  pins which lane/byte holds which (row, k-block) scale for `scale_vec::1X`. Do not write
  the pool change before that oracle passes. **This is the one unknown that can kill C.1.**
* **Split it**: C.1a = block scales for **QKᵀ only** (Q/K reduce along `d=256` → 8 clean
  32-wide blocks; no tile change). C.1b = PV. PV's reduction dim is *tokens*, so a 32-value
  block spans **two 16-token tiles** (`kTileTok = 16`, `qsa_warp.cu:49`): either fuse to a
  32-token k-step (`kTileTok = 32` → 16 KiB fp8/stage, and 2 stages = 32 KiB/block, which
  cuts resident blocks/SM from ~12 to ~3 — likely a bad trade) or keep PV on the existing
  rank-1 `γ` epilogue. Default: **ship C.1a, measure, decide C.1b on the error split**
  (the host study already reports Q/K vs P/V error contributions separately).
* Toggle `DGPP_QSA_FP8_MX` (default ON when the pool is fp8), independent of
  `DGPP_QSA_FP8_MMA`, so pool format and math dtype A/B separately.
* **Acceptance**: `l2_rel` band **≤ 0.035** (i.e. at or below the Phase A real-path
  0.0360 — currently 0.0551 warp / 0.0491 projected), pinned by
  `qwen_fp8_mma_attn_test` real-path + a new block-scale study; bf16 lane bitwise
  unchanged (`qsa_test` 12/12); pool `make gate` green; C.0 harness shows prefill ≤ −0 %
  (no regression) at 4k and the same ≥ 10 % win at ≥ 16k.

## 4. C.2 — Incoherent processing (Hadamard) before the fp8 quant

FA3's second FP8 technique; cheapest accuracy per engineering dollar.

* Orthogonality makes it free mathematically: quantize `H·q̃`, `H·k̃` where `q̃/k̃` are the
  **post-RMSNorm, post-RoPE** rows → `Σ (H q̃)(H k̃) = q̃·k̃` exactly, so only the
  *quantization* changes. Ordering constraint: **H after RoPE** (RoPE's rotation does not
  commute with a full-dim H), on **both** the Q side (`layers.cpp` before the warp call,
  or fused into the Q norm) and the K side (`qsa_norm_rope_append_fp8`).
* `d = 256 = 16²`: use a Kronecker rotation `H₂ ⊗ H₁₂₈` (hadamard128 on each half + one
  2-point butterfly across halves) so the port is the existing
  `hadamard128_smem` (`dsv4_attn.cu:206`, 7 xor stages, in-register/in-smem) **unchanged**,
  plus 16 fmas per row. V must **not** be rotated (it is not paired with a rotation on the
  output side); P·V stays as is.
* Study first, host-only, in the existing harness: **C.2.0** =
  `qwen_fp8_mma_incoherent_study` — `l2_rel` for {e4m3 row scale, block scale, block scale
  + H} × {QKᵀ, PV} on the standard distribution. FA3 reports **2.6× lower FP8 error**; if
  the study shows < 1.3× on our distributions, **drop C.2** and keep C.1 alone (H costs a
  real pass in the append hot path).
* Interaction: H + block scales is where the win is (H flattens the row, block scales stop
  paying for the residual outliers). If C.1 already reaches ≤ 0.02, C.2 is optional.
* Acceptance: append-path cost measured (`make bench` prose C1 within 1 %), `l2_rel`
  improvement ≥ 1.3× over C.1 alone, pinned by a real-path test (append + warp vs bf16).

## 5. C.3 — B.2: decode / short prefill onto the same footing

Two separate, independent defects:

* **C.3a (cheap, do first)**: the fp8 branch of `attn_partial_kernel` gathers with plain
  global `uint4` loads + in-register decode (`qsa.cu:818-832`), which **bypasses the
  `issue_rows` cp.async pipeline** (`qsa.cu:720`) the bf16 branch uses — the known Phase A
  penalty. Fix: cp.async the raw 16 B of codes into the tile, decode on the consumer side
  (or better, keep codes raw and let C.3b's mma eat them). Also measure
  `DGPP_QSA_WARP_STAGES=1 vs 2` (`qsa_warp.cu:53`, the 2-stage path already exists at
  `:286-289`, default 1) — free latency hiding if smem allows.
* **C.3b (the real work)**: scores are SIMT — `acc += qreg[j][i] * bf16_bits_to_float(krow[i])`
  plus a 5-level shuffle tree per head over 32 tokens (`qsa.cu:737-760`). Move QKᵀ/PV onto
  the mma like B.1. Design constraints: the decode tile is 32 tokens/block with
  `kHpw` heads per warp and one **row** of Q per head (the mma is m16 — 15/16 rows of the
  A tile are wasted unless the block batches ≥ 16 query rows, i.e. this pays off for
  **prefill-partial** (`T ≥ 128`, the `qsa_attn_prefill_partial` shape) and for
  **verify/decode with 16+ tokens**, i.e. MTP-4 × 4 slots only if the scheduler batches
  rows). Pre-registered go/no-go: **only implement where the m16 tile is ≥ 50 % full**;
  otherwise keep SIMT + fix C.3a and record B.2 as NO-GO for pure decode.
* Acceptance: prose C1/C2/C4 within ±2 %, long-prefill TTFT better than the C.0 baseline,
  `l2_rel` unchanged (C.3 is a plumbing change — same math as B.1/C.1).

## 6. Sequencing + decision gates

| # | work | gate to proceed | effort |
|---|---|---|---|
| C.0 | prefill measurement harness + B.1 A/B record | prefill win ≥ 10 % at ≥ 16k, else C.1 motivation = accuracy only | ~1 h |
| C.1.0 | SF-fragment oracle (`scale_vec::1X` lane/byte map) | oracle max diff 0 — **hard blocker for C.1** | ~2 h |
| C.1a | block scales for QKᵀ (pool + append + warp read) | `l2_rel ≤ 0.035`, gate green | ~1 day |
| C.1b | block scales for PV (32-token k-step or keep γ) | error split says PV is worth the tile change | ~0.5 day, likely skipped |
| C.2.0 | host incoherent-processing study | ≥ 1.3× error reduction, else drop | ~2 h |
| C.2 | H after RoPE on Q + K append | prose C1 within 1 % | ~0.5 day |
| C.3a | cp.async restore in the fp8 decode gather + warp stages 1 vs 2 | C1/C2/C4 within ±2 % | ~2 h |
| C.3b | decode/prefill-partial mma | ≥ 50 % m16 tile fill on the real shapes | ~1-2 days |

Housekeeping on this branch regardless: **push `feat/qwen-fp8-mma-attn`** (it is local-only
through `3c6f41d6`), and keep `121a` in the arch flag (§1). The C.0 harness is already in
`dgpp-gateway` (`bench/ab-env.sh` + the `ttft` / `ab-env` targets).

Non-goals: no FA3/TE/cuDNN dependency, no TMA rewrite, no e5m2 (NO-GO), no YaRN/1M-KV
changes, no Qwen3.5 full-attn changes, no pool dtype change beyond the scale planes.

## Appendix A — the hardware facts, reproducible

On this box (GB10, driver 580.173.02, CUDA 13.0), all-ones e4m3 operands (0x38),
`ue8m0` scales, C = 1 per accumulator; expected `Σd = 4 × 32 + 4 = 128`:

```
$ nvcc -O0 -gencode arch=compute_121a,code=sm_121a probe.cu && ./probe
mxf8f6f4 block_scale: sum=132.000   # 4*(32*1)+4 -> works
$ # scales 2^1 on both operands (sfa=sfb=0x80):
mxf8f6f4 block_scale: sum=516.000   # 4*(32*2*2)+4 -> sfa*sfb applied multiplicatively
$ nvcc -O0 -arch=sm_121 probe.cu
ptxas error: Instruction 'mma with block scale' not supported on .target 'sm_121'
ptxas error: Feature '.kind::mxf8f6f4' not supported on .target 'sm_121'
```

Beware: `-arch=sm_121a` through some nvcc paths silently emits a **stub** `.cubin`
(BRA-self-loop, no mma) instead of failing. Verify with `cuobjdump -sass | grep -i mma`
or a run, not with a build exit code.
