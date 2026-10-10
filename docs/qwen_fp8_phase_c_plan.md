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

Owner-signed (2026-10-10, before the night runs):

* **Verifier model** = `vllm-rtx5090/Qwen3.8-27B` for both the author and the adversarial
  pass. Chosen for speed / a single free endpoint; accepted cost: the checker shares the
  author's blind spots.
* **`DGPP_QSA_FP8_MX` rollout**: enabled by default **if the accuracy bar is met and prose
  C1/C2/C4 stay within +2 %** of the `DGPP_QSA_FP8_MMA=1` baseline (a ≤ 2 % perf cost is
  acceptable in exchange for a real accuracy gain). Never enabled without a measured
  record, so the code lands default-OFF first.
* **Accuracy bar** = **`l2_rel ≤ 0.035`** — at or under the Phase A dequant path's 0.0360
  (today's E4M3 path: 0.0551 warp / 0.0491 projected). With this bar C.2 (Hadamard) stays
  optional: dropped unless it buys ≥ 1.3×.
* **Push policy**: both repos — the `dgpp` branch as each work item lands, and
  `dgpp-gateway` master including the owner's two earlier commits.
* **Lane**: restarts are authorized overnight; the lane is left **up, default config, knob
  unset, `make smoke` green** between work items and at the end of the night.

## 2. C.0 — Measurement gate (do this first; ~1 h, no kernel work)

Outcome: a real prefill number for B.1 on/off, and a harness that stays useful for C.1-C.3.

* Harness (built, in `dgpp-gateway`): `make ab-env VAR=DGPP_QSA_FP8_MMA A=1 B=0
  LENGTHS="4096 16384 65536 131072" REPEAT=5 BENCH=1` — `bench/ab-env.sh` boots the same
  lane once per arm (the launcher forwards any `DGPP_*` from the head's environment to
  every rank, `dgpp-cluster` `spawn_peer`), runs `bench/ttft-probe.py` (unique nonce
  prompts, server-side `timings.prompt_ms`), optionally `make bench` for the prose guard,
  restores the knob to unset, and writes `results/ab-env-<VAR>-<ts>/summary.json`.
  It **refuses to run against a live stack** (it restarts it) unless `FORCE=1`.
  `make ttft LENGTHS="..."` probes the running stack without any restart.
* Record `results/ab-env-DGPP_QSA_FP8_MMA-*/summary.json` with `DGPP_QSA_FP8_MMA=1` vs
  `0` at **4k / 16k / 64k / 128k** prompt tokens × ≥5 repeats, same binary, plus the
  `BENCH=1` prose guard (C1/C2/C4) to prove no regression. Stay ≤ 128k: the lane is
  standard-context (native 256k, `engine.kv_capacity` 1048576), so a 256k prompt leaves
  nothing for the reply.
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
* **Pool layout** (`kv_pool.hpp/.cpp`), **decision (1): the block-scale plane is
  additive — the fp32 row scale stays.** Per (slot, kv-head) row the pool keeps
  today's fp32 row scale **and** gains **8 bytes of E8M0 block scales** (one per 32 of
  the 256-dim row): **260 B → 268 B/row** (256 codes + 4 fp32 scale + 8 e8m0).
  Rationale: keeps `DGPP_QSA_FP8_MX=0` bit-identical to the current binary (the A/B is
  *same-binary*, the only trustworthy shape) and leaves the `DGPP_QSA_FP8_MMA=0` dequant
  path untouched; it costs only 8 B on a 260 B row → pool **~7.41 → ~7.64 GiB/rank**,
  ratio to bf16 **1.86× → ~1.81×** (still the fp8 win). Unlock: if the A/B shows the MX
  path wins, a later change may drop the fp32 plane. Update `layer_block_scale_bytes()`,
  `init/view/reset_all`, `copy_block_contents` (prefix-cache block copy — `kv_pool.cpp`;
  a missed copy silently corrupts cached prefixes) and the memory-plan label.
  **Implemented (C.1a), pending GPU.**
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
* **Read path (all attention paths, C.1a follow-up)**: in `MX=1` every path consumes the
  block plane, so the toggle is path-independent (the fail-loud guard in `layers.cpp` is
  gone): the warp kernel runs the block-scaled QK^T (MMA=1) or the block-aware dequant
  (MMA=0); `attn_partial_kernel` (decode / short prefill) and
  `attn_prefill_partial_kernel` (long prefill, `DGPP_QSA_WARP=0`) dequant a K code at dim
  `c` with the e8m0 scale of block `c/32` (the 16-code 16-byte gather chunk is half a
  32-dim block, so the block index is constant per load). **V stays row-quantized** in
  C.1a (the PV `γ` epilogue is untouched), so `v_bscale` is **not written** — the plane
  stays zero (the pool memsets it) and is reserved for C.1b; a block plane computed from
  the bf16 source would misdescribe the row-quantized V codes. `qsa_fp8_mx()`
  (`kernels/qsa.hpp`) is the single per-call env read.
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

### 3.1 C.1a error attribution (host study, 2026-10-11) — verdict: 0.035 is NOT reachable on the tensor-core path

`tests/cuda/qwen_fp8_error_attribution.cpp` (host-only, no GPU; exact e4m3/e8m0
recipes from `latent_format.hpp`, fp32 accumulation, the DSA-pin bf16 reference;
mean of 3 seeds, stable under a seed offset) attributes the C.1a baseline
(Q blk, K blk, P e4m3 pow2-γ, V row) across four K distributions. **Host-simulated
numbers** (GPU-measured ones are marked):

* **Where the 0.051 comes from** (unit-normal, per-factor isolation, all others bf16):
  Q **0.0261**, K **0.0261**, P **0.0219**, V **0.0264** — four roughly equal
  contributors, combining in quadrature to 0.0504 ≈ the measured 0.0517 baseline
  (GPU: 0.0510). No single factor dominates; there is no cheap single lever.
* **Block scales buy nothing in L2, on any distribution.** Unit-normal:
  base 0.0517 vs Q-row 0.0512 / K-row 0.0520 (a wash; GPU 0.0510 vs 0.0551 row).
  Outlier regimes: K-row **beats** K-blk by 0.0215 (RMSNorm-pinned + 3 outlier
  dims 5–20×: 0.0567 vs 0.0782), 0.0302 (sink, one dim at 32× row rms: 0.0512 vs
  0.0814), 0.0234 (lognormal-row + 10–100× spikes: 0.1335 vs 0.1568). The e8m0
  power-of-two scale spends the granularity gain: a block's max lands in
  (224, 448] of the e4m3 range instead of pinned at 448, and when a block holds
  the outlier its other 31 elements (or the 7 cold blocks) lose the precision
  the finer granularity was supposed to buy. At the quantizer level the truth is
  sharper: e4m3's 448 max + subnormals already cover ~10⁵:1 of within-row range,
  so the elements a row scale underflows carry negligible L2 energy — block
  scales' only win is a **bounded worst-element error** (max_rel ≤ 2⁻⁴ vs 1.0
  for row at 1000× spread; pinned by the fixed
  `qwen_fp8_kv_block_scale_quantizer`), not an L2 gain.
* **Hadamard (C.2) does not earn 1.3× on our distributions**: 1.00× unit-normal
  (0.0516), 1.20× RMSNorm-pinned + outliers (0.0782 → 0.0651), 1.08× sink
  (0.0814 → 0.0753), and **harmful** (0.86×) on the lognormal-row stress regime
  (H preserves row norms, so it cannot remove row-level outliers). FA3's 2.6×
  claim does not reproduce here.
* **The 0.035 bar is met only by the dequant path** (Q bf16, K row, P bf16, V
  row — the Phase A path, GPU-measured 0.0360): host 0.0299 (RMSNorm-pinned +
  outliers), 0.0368 (sink), 0.0377 (unit-normal). Best tensor-core path:
  0.0505 (unit-normal, V blk), 0.0567 (RMSNorm-pinned + outliers, K row),
  0.0512 (sink, K row), 0.1335 (stress, K row).

**Verdict: NO-GO for 0.035 on the fp8-everywhere (tensor-core) path.** Best
achievable there is ~0.05 (unit-normal) / ~0.057 (realistic outlier dims).
**Recommendations:** (1) keep C.1a as shipped — correct, neutral-to-positive on
unit-normal (GPU 0.0510 vs 0.0551), repositioned as a worst-element bound, not
an L2 lever; (2) **C.1b (PV block scales): NO-GO** — V-blk is −0.0013…−0.0038
on every distribution for the expensive 32-token tile change; (3) **C.2
(Hadamard): NO-GO** at the 1.3× gate (best 1.20×); (4) revised bar: ~0.05 for
the tensor-core path; if 0.035 accuracy is a hard requirement, the dequant path
(`DGPP_QSA_FP8_MMA=0`) is the accuracy mode (meets 0.035 on realistic
outlier-dim data) and the tensor-core path is the speed mode — a two-mode story,
not one fp8-everywhere path. **Next work item:** sample a *real* post-RoPE
Q/K distribution from a live forward pass (dump one layer of a 16k prefill) to
validate the synthetic regimes before any further C.x investment.

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

* **C.3a (cheap, do first) — implemented, pending A/B.** The fp8 branch of
  `attn_partial_kernel` gathered with plain global `uint4` loads + in-register decode
  (`qsa.cu:818-832`), which **bypassed the `issue_rows` cp.async pipeline** (`qsa.cu:720`)
  the bf16 branch uses — the known Phase A penalty (the ~9 % C4 prose gap vs bf16).
  **Fix (this change):** cp.async the raw e4m3 codes (256 B/token at D=256, half the bf16
  tile's 528 B/row) into the raw-code smem tiles `kt8`/`vt8` (the same smem the bf16 tiles
  use, so the block's smem budget and occupancy are unchanged), and decode on the consumer
  side (`scores_phase_fp8`/`pv_phase_fp8`, the hardware e4m3→f16 pair decode + the per-
  (row, kv-head) scale, row for `MX=0` and the per-32-dim-block e8m0 for `MX=1`). Tile t+1's
  codes are issued under tile t's scores / PV, exactly the bf16 `issue_rows` shape, so the
  load latency is no longer serialized behind the decode. Bitwise the serial form (same
  codes, same scales, same op order); the bf16 branch is untouched. The prefill-partial
  kernel (`qsa_prefill.cu`, the `DGPP_QSA_WARP=0` long-prefill fallback) is left on its
  serial gather: the deployed long-prefill default is the warp kernel (already cp.async),
  so that fallback is not on the measured path. `DGPP_QSA_WARP_STAGES` stays 1: the warp
  kernel's fp8 stage is 8 KiB (raw codes) vs 16.5 KiB (bf16, padded), but the smem
  allocation is the bf16 size (`kSmem = kStages·16896` B), so 2 stages = 33 KiB → ~3 blocks/SM
  vs 1 stage's ~6; residency already hides the gather, so 2 stages is a bad trade. Not
  flipped.
* **C.3b (the real work) — NO-GO for decode, verified.** The m16 A-tile gate is ≥ 50 %
  full. The decode kernel (`attn_partial_kernel`) is one **query row** per block per head
  (grid `(rows, n_split, local_heads/hpb)`; `rows` is 1 for standard decode, ≤ 4 for
  MTP-4 × 4 slots), so the m16 tile is **1/16 = 6.25 %** full (standard decode) or
  **4/16 = 25 %** (MTP-4 × 4) — both far below the 50 % gate; 15/16 (or 12/16) rows are
  wasted. The only shape that fills the tile is **prefill-partial** (`T ≥ 128`, the
  `qsa_attn_prefill_partial` shape, ≥ 16 query rows per block), which is already on the
  warp kernel's mma (B.1) by default. **Verdict: keep decode SIMT + the C.3a cp.async
  restore; record B.2 as NO-GO for pure decode.** Do not implement C.3b.
* Acceptance: prose C1/C2/C4 within ±2 %, long-prefill TTFT better than the C.0 baseline,
  `l2_rel` unchanged (C.3 is a plumbing change — same math as B.1/C.1).

## 6. Sequencing + decision gates

| # | work | gate to proceed | effort |
|---|---|---|---|
| C.0 | prefill measurement harness + B.1 A/B record | prefill win ≥ 10 % at ≥ 16k, else C.1 motivation = accuracy only | ~1 h |
| C.1.0 | SF-fragment oracle (`scale_vec::1X` lane/byte map) | oracle max diff 0 — **hard blocker for C.1** | ~2 h |
| C.1a | block scales for QKᵀ (pool + append + warp read) — **implemented, pending GPU** | `l2_rel ≤ 0.035`, gate green | ~1 day |
| C.1b | block scales for PV (32-token k-step or keep γ) | error split says PV is worth the tile change | ~0.5 day, likely skipped |
| C.2.0 | host incoherent-processing study | ≥ 1.3× error reduction, else drop | ~2 h |
| C.2 | H after RoPE on Q + K append | prose C1 within 1 % | ~0.5 day |
| C.3a | cp.async restore in the fp8 decode gather + warp stages 1 vs 2 — **implemented, pending A/B** | C1/C2/C4 within ±2 % | ~2 h |
| C.3b | decode/prefill-partial mma — **NO-GO for decode** (m16 tile 6.25 % / 25 % full, < 50 % gate) | — | — |

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
