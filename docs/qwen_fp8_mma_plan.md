# Qwen3.8-Flash-Next — native FP8-MMA attention (Phase B plan)

Branch: `feat/qwen-fp8-mma-attn` (stacked on the shipped Phase A,
`docs/qwen_fp8_kv_plan.md`). Scope: **Qwen3.8-Flash-Next only** (the QSA
kernels + the Qwen serve family). GLM / MiMo / DSV4 are out of scope.

**Status (2026-10-10): plan only. No kernel, test, or `CMakeLists.txt` change.**
Phase A is done and is the base: the QSA K/V cache is stored as FP8 E4M3
(1 B/elem) + a per-(slot, kv-head) fp32 scale, and the three attention kernels
**dequantize to bf16 in-kernel** before the existing math
(`qsa_warp.cu:122`, `qsa_prefill.cu` fp8 gather, `qsa.cu` decode). Phase B
replaces only the attention *read* math: it feeds the stored E4M3 codes
straight to the tensor cores for `Q·Kᵀ` and `P·V`, applying the per-row /
per-(slot, kv-head) fp32 scales in the epilogue — **no bf16 dequant step**.

## 0. Goal + the honest win

Move the QSA attention's `Q·Kᵀ` and `P·V` products onto the E4M3 tensor cores
(`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`, the asset at
`fp8_gemm.cu:184` and — inside an attention-adjacent kernel — `dsa.cu:1650`),
so the stored codes are consumed *without* dequantizing to bf16.

- **Q** quantized per-(query-head-row) to E4M3 + an fp32 scale `α`.
- **K** already stored E4M3 + per-(slot, kv-head) scale `βᵏ`.
- **`Q·Kᵀ`** on E4M3, epilogue applies `α·βᵏ`.
- **P** (post-softmax) quantized per-row to E4M3 + an fp32 scale `γ`, with the
  per-token V scale `βᵛ` **absorbed into the P code** (§2.3).
- **V** already stored E4M3 + per-(slot, kv-head) scale `βᵛ`.
- **`P·V`** on E4M3, epilogue applies `γ`.
- The split-KV partials + `dsa_attn_combine` merge stay; only the inner
  products change dtype.

**The win is at long context, and only there.** Phase A showed the K/V is
**< 0.1 %** of the bytes a short-context decode step streams (weights dominate
~6.2 GB/token/rank), so at short context fp8 can at best be *neutral* — and
Phase B *adds* Q + P quantization work on top, so a naive always-on E4M3 path
is a **net loss at short context**. The E4M3 path is a *speed* win only where
the K/V is a real fraction of the read traffic (long prefill), which is exactly
where `qsa_attn_prefill_warp` is the serving default. The E4M3 path is therefore
**gated on the fp8 pool** (`k_scale != nullptr`) and should be **measured at
short context to confirm neutral** before being trusted at long context (§10).

## 1. The bf16 reference (exact math)

One QSA query-head row `m` (one of the ≤16 query heads per KV head in the warp
kernel), over its selected-token list, head dim `d = 256`, attention scale `s`:

```
S_mn   = s · Σ_{j=0}^{d-1} q_m[j] · k_n[j]          (score)
M      = max_n S_mn                                 (running max, online)
P_mn   = exp(S_mn − M)                              (unnormalized probability)
l_m    = Σ_n P_mn                                   (denominator, UNROUNDED)
out_m  = (Σ_n P_mn · v_n) / l_m                     (attention output)
```

The kernel pins this exactly: `P` rounds to bf16 for the V accumulation and
`l` stays unrounded (`qsa.hpp:140-141`, "the DSA pin"); the prefill/decode
rounding orders match (`qsa_prefill.cu:3`); every gather form is bitwise every
other (`qsa.hpp:155`). Phase B breaks all three — §4 states the replacement.

## 2. The E4M3 projections + scale factorization

E4M3 (E4M3, RNE) has a 3-bit mantissa: a *representable* value `x` quantizes to
`e4m3(x)` with relative error `≤ 2⁻⁴` (call it `η`); values far below the
row's `absmax` underflow to 0 (rel err → 1) but are a negligible fraction of the
row norm (Phase A measured 0.001 % of elements, `|x|/absmax ≤ 2.1e-6`).

### 2.1 The four projections

```
Q:  α_m   = (max_j |q_m[j]|) / 448        q̃_m[j] = e4m3(q_m[j] / α_m)      ⇒ q_m[j] ≈ α_m·q̃_m[j]
K:  βᵏ_n  = existing per-(slot, kv-head)  k̃_n[j] = e4m3(k_n[j] / βᵏ_n)     ⇒ k_n[j] ≈ βᵏ_n·k̃_n[j]
V:  βᵛ_n  = existing per-(slot, kv-head)  ṽ_n[j] = e4m3(v_n[j] / βᵛ_n)     ⇒ v_n[j] ≈ βᵛ_n·ṽ_n[j]
P:  w_mn  = P_mn · βᵛ_n  (V-weighted)     γ_m = (max_n w_mn) / 448
    p̃_mn = e4m3(w_mn / γ_m)                                                     ⇒ w_mn ≈ γ_m·p̃_mn
```

`α_m` is per-(row, query-head) — computed once in the warp prologue (§5).
`βᵏ_n`, `βᵛ_n` are the Phase A scales already in the pool. `γ_m` is per-(row,
query-head, tile) — computed in-warp from the tile's `P` (§5).

### 2.2 Lemma 1 — the score factorizes exactly

The E4M3 MMA computes `T_mn = Σ_j q̃_m[j]·k̃_n[j]` (E4M3×E4M3 → fp32); the
epilogue sets `S̃_mn = s · α_m · βᵏ_n · T_mn`. In the limit of exact E4M3 codes:

```
S̃_mn = s · α_m βᵏ_n Σ_j (q_m[j]/α_m)(k_n[j]/βᵏ_n) = s Σ_j q_m[j] k_n[j] = S_mn   ✓
```

`α_m` is per-row (the M dim of the score tile) and `βᵏ_n` is per-token (the N
dim), so the epilogue is a **rank-1 outer product** over the score tile — no
per-element work. This is the same structure `fp8_gemm.cu:184` uses
(`a_scales[m] · w_scales[n]` promoted in the epilogue) and `dsa.cu:1650`
(per-pool `ks` folded in registers).

### 2.3 Lemma 2 — the PV factorizes exactly (the key design decision)

The per-token V scale `βᵛ_n` is the one scale that does **not** factor out of a
K-reducing MMA for free: the PV MMA reduces over tokens, and `βᵛ_n` varies per
token. The resolution is to **absorb `βᵛ_n` into the P code** (the `w_mn = P_mn·βᵛ_n`
projection of §2.1). The E4M3 MMA computes `U_md = Σ_n p̃_mn·ṽ_n[d]`; the
epilogue sets `out̃_m[d] = γ_m · U_md / l_m`. In the limit of exact E4M3:

```
γ_m·U_md = Σ_n (γ_m p̃_mn)(ṽ_n[d]) = Σ_n (P_mn βᵛ_n)(v_n[d]/βᵛ_n) = Σ_n P_mn v_n[d]
out̃_m[d] = (Σ_n P_mn v_n[d]) / l_m = out_m[d]   ✓
```

`βᵛ_n` cancels (one copy in the P code, one in the V code), and `γ_m` is a clean
per-row scale. **This is why P is quantized as the V-weighted probability, not
the raw probability.** Fallback if the study (§7) shows `w_mn` underflows
excessively: rescale the V tile to a tile-common scale (one extra E4M3 rounding
on V) and use the raw `P` code — strictly more error, so prefer the absorb.

### 2.4 Theorem — exactness

In the limit of exact E4M3 codes, the E4M3-MMA attention with the epilogue
scales `(α_m βᵏ_n)` on the score and `γ_m` on the PV (with `βᵛ_n` absorbed into
P) is **exactly** the bf16 reference of §1. The *only* divergence source is the
E4M3 quantization of the four operands — the same isolation property that made
Phase A bitwise-the-bf16-kernel-over-the-dequantized-rows.

## 3. The numerical proof (the tolerance band)

Let `η = 2⁻⁴`. Per-element: `|q_m[j] − α_m q̃_m[j]| ≤ η|q_m[j]|`, and likewise for
`k`, `v`, and `|w_mn − γ_m p̃_mn| ≤ η|w_mn|`. Hence `‖q_m − α_m q̃_m‖₂ ≤ η‖q_m‖₂`, etc.

**Score.** Write `q_m = α_m q̃_m + e^q`, `k_n = βᵏ_n k̃_n + e^k` with
`‖e^q‖₂ ≤ η‖q_m‖₂`, `‖e^k‖₂ ≤ η‖k_n‖₂`. Then

```
|S_mn − S̃_mn| ≤ s(‖e^q‖₂‖k_n‖₂ + ‖q_m‖₂‖e^k‖₂) ≤ 2 s η ‖q_m‖₂‖k_n‖₂,
|S_mn|         ≤ s ‖q_m‖₂‖k_n‖₂,
⇒  |S_mn − S̃_mn| / |S_mn| ≤ 2η   (worst case; ~√2·η with independent √d averaging)
```

**Softmax.** 1-Lipschitz in the logit differences: a score perturbation `Δ`
perturbs each `P_mn` by `≈ P_mn(e^Δ − 1) ≈ P_mn·Δ`, so the probability vector is
perturbed by `O(Δ) = O(η)` (this is the *new* term vs Phase A, which had exact
bf16 `P`).

**Output.** With `γ_m p̃_mn = w_mn(1+δʷ)`, `ṽ_n[d] = (v_n[d]/βᵛ_n)(1+δᵛ)`,
`|δʷ|,|δᵛ| ≤ η`:

```
γ_m·U_md − Σ_n P_mn v_n[d]  =  Σ_n P_mn v_n[d] (δʷ + δᵛ + O(η²))
|out̃_m[d] − out_m[d]|       ≤ (2η / l_m) Σ_n P_mn |v_n[d]|  ≤  2η · max_n |v_n[d]|
```

The output is a convex combination (`Σ_n P_mn / l_m = 1`) of the V rows, each
within `η` relative, so the **L2 relative output error is `O(η)`** — the same
necessary-condition bound as Phase A, with a constant a little larger because
the Q and P quantization perturb the *weights* (not just the V rows). Phase A
measured `l2_rel ≈ 0.036` (K/V only, under the `2⁻⁴` bound); Phase B is expected
at `l2_rel ≈ 0.04–0.08`.

**Scope (stated, not hand-waved).** This bounds the *attention layer* — a
**necessary condition**. It does **not** bound the 48-layer compounding or the
final **argmax** (a **sufficient condition**); those are validated empirically
by the gate + bench, exactly as Phase A's §0b. Nobody should read "within
tolerance" as a theorem about the final token.

**Pre-registered band:** attention `l2_rel ≤ 0.10`. Analytic basis: Q+K each
contribute `2⁻⁴` to the scores (two quantized operands → `2·2⁻⁴` worst case),
P contributes `2⁻⁴` to the PV; Phase A measured `0.036` with K/V only, so adding
Q+P lands `~0.04–0.08`. Confirmed by the §7 study (which prints the measured
value and sets the real band). The transcript gate uses this band, not a
bit-exact match (the `prefill_fp8_gemm` "NOT bitwise the dequantized bf16 chain"
precedent).

## 4. The numerics contract

**B.1 (the warp kernel) breaks *none* of the three pins the brief names.**
Those pins — `qsa.hpp:140-141` ("probabilities round to bf16 for the V
accumulation, `l` stays unrounded"), `qsa.hpp:155` ("every gather form is
bitwise every other"), and `qsa_prefill.cu:3` (prefill/decode rounding orders
match) — all live on the **partial / decode kernels**
(`qsa_attn_partial`, `qsa_attn_partial_gather`, `qsa_attn_prefill_partial`).
B.1 touches only `qsa_attn_prefill_warp`, so those pins stand untouched and the
partial kernels keep their Phase A dequant-to-bf16 path and their bitwise
relationships. (The brief's line refs were from an earlier revision; the pins
are at the locations above.)

The **only** statement B.1 replaces is the warp kernel's own
**`qsa_warp.cu:8-9`** ("probabilities rounded to bf16 for P V"): → *in the
E4M3 warp path, the V-weighted probability `P·βᵛ` quantizes to E4M3 per-row
(scale `γ`) for the V accumulation; `l` stays fp32-unrounded (it sums the
pre-quantization `P`); the path is tolerance-equal (band §3) to the bf16 warp
path, not bitwise.*

Consequently **`tests/cuda/qsa_test.cu` and `tests/cuda/qwen_full_attn_test.cpp`
need no B.1 change** — they exercise the bf16 partial / full-attn paths only
(`qwen_full_attn_test.cpp` is the model35 `QwenFullAttnLayer` family, out of
scope). The only test changes are the new `qwen_fp8_mma_attn_test.cu` (§7) and
widening the warp assertion in `qwen_fp8_kv_attn_test.cu` (0.0625 → 0.10).

## 5. The kernel design — `qsa_attn_prefill_warp` first

`qsa_warp.cu` already has the MMA + `ldmatrix` scaffold (bf16 `m16n8k16`,
`qsa_warp.cu:55-57`; `ldmatrix` `:45-53`; P A-fragment reuse `:230-231`; the
Q·Kᵀ loop `:192-194`; the P·V loop `:243-245`). Phase B is a dtype swap of that
scaffold to E4M3, **not** a new lowering. The bf16 path (scales null) is
byte-for-byte unchanged; the E4M3 path is a new branch gated on
`k_scale != nullptr` **and** the `DGPP_QSA_FP8_MMA` toggle (default ON, mirroring
`DGPP_QSA_WARP` at `layers.cpp:815`). The toggle decouples the E4M3 *attention*
from the fp8 *pool*: `DGPP_QSA_FP8_MMA=0` keeps the Phase A dequant path (and
the pool win) while skipping Phase B, so short- and long-context can be A/B'd
independently and the short-context loss can be switched off without giving up
the fp8 pool (§0, §10).

### 5.1 Tile / register layout (dim 256, 16-token tile, one warp)

- **Q A-fragment (E4M3):** `qa[8][4]` — 8 k-steps of `m16n8k32` (256/32), 4
  regs each (16 E4M3/lane). Half the registers of the bf16 `qa[16][4]`.
  Quantized in the prologue: each warp reads its ≤16 query-head-rows (already
  read for `qa`), reduces `α_m = absmax/448` over the 256-dim row (warp
  `__shfl_xor` tree, the `fp8_quantize_rows_kernel` recipe at `fp8_gemm.cu:26-46`
  done in-warp), packs `e4m3(q/α)` into the A-fragment. `α_m` kept in 2 regs
  (rows `g`, `g+8`).
- **K tile (E4M3 in smem):** the Phase A path dequants K to bf16 in smem
  (`qsa_warp.cu:122`); Phase B keeps the **E4M3 codes** in smem (1 B/elem — half
  the smem), `ldmatrix` reads the codes (2 E4M3 per 16-bit element, the
  `fp8_gemm.cu` / `dsa.cu:1650` layout). `βᵏ_n` read per-token for the epilogue.
- **`Q·Kᵀ`:** `mma m16n8k32 e4m3` → fp32 `sc[2][4]` (2 n8 tiles = 16 tokens).
  Epilogue: `sc[m,n] *= α_m · βᵏ_n` (rank-1, §2.2), then the existing
  `exp2`-domain masking + row-max + online-softmax (unchanged).
- **P A-fragment (E4M3):** from the tile's fp32 `P` (the `p[j][e]` regs), form
  `w_mn = P_mn · βᵛ_n`, reduce a per-row scale `γ_m`, pack `e4m3(w/γ_m)` into the
  4-reg A-fragment (replacing `pack_bf16` at `qsa_warp.cu:230-231`). Use a
  **power-of-two `γ_m`** (`2^⌈log2(absmax(w)/448)⌉`, the `fp8_block` recipe at
  `latent_format.hpp`): RNE relative error is then scale-invariant (no
  mantissa/scale interaction), which is cleaner than `absmax/448`. **`l` keeps
  summing the unrounded `P`** (the DSA pin).
- **smem / MMA budget:** the E4M3 K/V tiles are 256 B/row (**8 KiB/stage** vs the
  bf16 33.8 KiB) and need no `+8` ldmatrix row padding; `qa[8][4]` is 32 regs
  (vs 64 bf16); ~**48 E4M3 MMAs per tile vs 96 bf16**.
- **V tile (E4M3 in smem):** codes, `βᵛ_n` per-token.
- **`P·V`:** `mma m16n8k32 e4m3` → fp32 `acc[32][4]` (32 n8 tiles = 256 dim).
  **The PV k-dim is 32 but the tile has 16 tokens → pad to 32 k-lanes** (16 real
  + 16 zero; the zero P/V codes contribute 0). Epilogue: `acc[m,d] *= γ_m`
  (per-row, §2.3). The online `acc *= a0/a1` rescale and the final `/ l_m` are
  unchanged (they act on the fp32 accumulator).

### 5.2 What is *not* changed

The gather (lanes 0-15 resolve the tile's 16 physical rows via the block table),
the `cp.async` staging, the online-softmax rescale, the split-KV +
`dsa_attn_combine` merge, and the `qsa_gate_out` epilogue. Only the two MMAs and
their epilogue scales change dtype.

## 6. The deferred hard case — decode / short-prefill

`qsa_attn_partial` (`qsa.cu:579`, the FMA + `__shfl_xor_sync` tree at
`qsa.cu:738,745,778`) and `qsa_attn_prefill_partial` (`qsa_prefill.cu`) have
**no tensor cores today** — they are warp-shuffle FMA dot products. An E4M3-MMA
version there is a **net-new tensor-core lowering over `topk`-gathered
(non-contiguous) rows** (smem staging + `ldmatrix` + the gather), not a dtype
swap. **This plan defers it.** Rationale: (1) decode/short-prefill is where the
agentic workload lives and Phase A already made it *neutral* (the dequant path
is fine there); (2) the win is at long prefill, where `qsa_attn_prefill_warp` is
the serving default; (3) it is a separate, larger project. Decode/short-prefill
stay on the Phase A dequant path. Revisit only if a long-context *decode* run
shows the dequant gather is a bottleneck (Phase A §7.2 already notes the fp8
gather is not pipelined — a separate, smaller win).

## 7. The pre-implementation evidence (measure before you build)

Phase A stood up `tests/cuda/qwen_fp8_kv_attn_test.cu` *before* writing any fp8
kernel, and it paid off. The Phase B analogue is a **new**
`tests/cuda/qwen_fp8_mma_attn_test.cu` (GPU, `ctest -R qwen_fp8_mma_attn`):

1. **`qwen_fp8_mma_projected_study`** (host fp32, *no kernel*): take realistic
   bf16 Q/K/V (the test's standard unit-normal distribution). Quantize all four
   operands exactly as §2.1 (Q per-row, K/V the existing per-(slot,head) scales,
   P V-weighted per-row). Compute the **projected MMA math in fp32** — i.e. the
   *exact* E4M3 result `s·α·βᵏ·Σ q̃·k̃` and `γ·Σ p̃·ṽ/l` (isolating the
   quantization from the tensor-core rounding) — and compare to the bf16
   reference (`qsa_attn_partial` + `dsa_attn_combine` + `qsa_gate_out` over bf16
   K/V, the `run_attn` helper in `qwen_fp8_kv_attn_test.cu`). Report `l2_rel`,
   `max_abs`, and the sample-match rate; **assert the pre-registered band
   `l2_rel ≤ 0.10`** and print the measured value to set the real band.
2. **`qwen_fp8_mma_warp_matches_band`** (GPU, after the kernel): the E4M3
   `qsa_attn_prefill_warp` output vs the bf16 reference, within the band the
   study set. This is the acceptance test for the kernel.

The study also decides the §2.3 fallback: if `w_mn = P·βᵛ` underflows
excessively (print the underflow fraction, as Phase A's codec study does),
switch to the tile-common V rescale.

## 8. File-by-file changes

| file | change |
|---|---|
| `src/kernels/qsa_warp.cu` | the E4M3 branch (gated on `k_scale != nullptr`): Q A-fragment quantize + `α`; K/V E4M3 smem tiles; `mma m16n8k32 e4m3` for both MMAs; the `α·βᵏ` score + `γ` PV epilogues; P V-weighted quantize + `γ`; PV 32-k padding. Update the `:8-9` doc (the only pin B.1 replaces, §4). bf16 path unchanged. |
| `src/kernels/qsa.hpp` | **no change** — the `:140-141,155` pins are on the partial kernels and stand (§4); the warp declaration's doc already says tolerance-equal. (`k_scale`/`v_scale` params already exist on `qsa_attn_prefill_warp`.) |
| `tests/cuda/qwen_fp8_mma_attn_test.cu` | **new**: the §7 study + the kernel band test. |
| `src/models/qwen/layers.cpp` | read the `DGPP_QSA_FP8_MMA` env (default ON, the `DGPP_QSA_WARP` pattern at `:815`); the E4M3 warp path runs only when the fp8 pool is active **and** the toggle is on. |
| `tests/cuda/qwen_fp8_kv_attn_test.cu` | widen the warp-path assertion `0.0625 → 0.10` (the §3 band). |
| `CMakeLists.txt` | register the new test (implementation step, noted here). |

**No change** to `tests/cuda/qsa_test.cu` or `tests/cuda/qwen_full_attn_test.cpp`
(bf16-only; the latter is the model35 full-attn family, out of scope — §4).

No change to the pool, the append quantize, `qsa_kv_append`, the config knob, or
the partial kernels — Phase A's storage is the unchanged input.

## 9. Sequencing (first shippable step)

1. **Step 0 — evidence first:** write `tests/cuda/qwen_fp8_mma_attn_test.cu`
   (the §7.1 host study). Run it; read the measured `l2_rel`; set the real band.
   *Nothing is built until the band is known* — the Phase A precedent.
2. **Step 1 — first shippable:** the E4M3 `qsa_attn_prefill_warp` (§5). Gated on
   the fp8 pool; bf16 path byte-for-byte unchanged. Verify against the Step-0
   band (`qwen_fp8_mma_warp_matches_band`).
3. **Step 2:** widen the fp8-kv warp assertion (`qwen_fp8_kv_attn_test.cu`,
   0.0625 → 0.10). **No change** to `qsa_test.cu` / `qwen_full_attn_test.cpp`
   (bf16-only, §4/§8).
4. **Step 3 (deferred):** the decode/short-prefill E4M3 lowering (§6) — only if
   a long-context decode run demands it.
5. **Step 4:** `make gate` (transcript, band §3) + `make bench` / `make ab` on
   the fp8 lane vs the bf16 lane, **at long context** (the win) and at short
   context (confirm neutral, §10). Record to `results/`.

## 10. Risks

- **Short context is a net loss / neutral.** K/V is < 0.1 % of streamed bytes
  there; Phase B *adds* Q + P quantization. Gate the E4M3 path on long context
  (or measure short-context neutral before trusting it). The honest framing:
  **Phase B's win is long-context prefill; short context is neutral at best.**
- **Quality is wider than Phase A.** Q and P are now quantized too (not just
  K/V) → the band is wider (§3). Validate on the agentic + prose benches, not
  just the gate.
- **The V-weighted P underflow.** If `w_mn = P·βᵛ` underflows E4M3
  excessively, use the §2.3 fallback (tile-common V rescale). The §7 study
  decides; it is a strict error increase, so prefer the absorb.
- **The PV 32-k padding.** The 16-token tile must pad to 32 k-lanes for the
  E4M3 MMA (§5.1); the zero lanes contribute 0 but cost smem (the V tile is 32
  rows). Confirm the smem fits the one-stage residency the kernel relies on
  (`qsa_warp.cu` header: one stage → 5 warps/SM).
- **Graph capture.** The E4M3 path must stay deterministic/capturable (fixed
  grids, no host reads) — same contract the existing QSA kernels honor. `α_m`
  and `γ_m` are computed in-warp (no host reads), so this holds.
- **TP=2 fold.** `α_m`/`γ_m` are per-rank (per query-head-row); `βᵏ`/`βᵛ` shard
  exactly like the K/V rows (`local_kv_heads`). No new collective — the
  boundary reducer is untouched.

## 11. Open implementation caveat

The E4M3 **quantize + pack recipe has in-repo precedent**: `fp8_gemm.cu:26-46`
(`fp8_quantize_rows_kernel`, cited in §7.1) already converts rows to E4M3 and
packs 4 codes per 32-bit register (`lo | hi<<16`). What has **no** in-repo
precedent is the specific `m16n8k32` **A-fragment lane / k→register mapping**
(the PTX ISA `.f8f6f4` layout) — the repo's E4M3 MMAs (`fp8_gemm.cu`,
`dsa.cu:1650`) load fragments via `ldmatrix`, so the in-kernel Q-quantize
*packing into the A-fragment* is the one piece to verify. The implementer must
sanity-check that mapping against a small host oracle (the §7.1 study's test 3)
before trusting it.

## 12. Verification ladder

unit (codecs/pool, Phase A) → **§7 study (pre-registered band)** → kernel band
test → widen the fp8-kv warp assertion → `make gate` (transcript, band §3) → memory plan (unchanged,
Phase A) → `make bench` / `make ab` (long-context win, short-context neutral) →
`results/` record. The bf16 lane stays bitwise-unchanged (the E4M3 path is
gated on non-null scales, so a bf16 run touches none of it).
