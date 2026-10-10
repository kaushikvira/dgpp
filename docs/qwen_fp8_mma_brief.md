# Phase B brief — native FP8-MMA attention for Qwen3.8-Flash-Next

You are a research + design agent. **Do not implement kernels.** Your single
deliverable is `docs/qwen_fp8_mma_plan.md` — a plan specific enough that the
parent can implement it without re-deriving anything.

Repo: `~/work/dgpp` (branch `feat/qwen-fp8-mma-attn`, stacked on Phase A).
Read `docs/qwen_fp8_kv_plan.md` first — it is the Phase A record and the source
of the direction sketch (its §3.7 and §7.2/§7.3 are your starting point).

## Context you must absorb

- **Phase A is DONE and shipped** (same branch, committed): the Qwen QSA K/V
  cache can be stored as FP8 E4M3 (per-(slot, kv-head) fp32 scale), the attention
  kernels **dequantize to bf16 in-kernel** and run the existing bf16 math. Prose
  C1 is neutral vs bf16 (53.2 vs 53.3) and the pool is 1.86x smaller
  (7.41 GiB vs 13.81 GiB/rank). See `docs/qwen_fp8_kv_plan.md` §0b, §3.6, §3.7,
  §6, §7.
- **Phase A is bitwise** the bf16 kernel over the *dequantized* rows — the only
  divergence is the quantizer. **Phase B is NOT bitwise**: it changes the inner
  products' dtype, so it is *tolerance-equal* (the `prefill_fp8_gemm` precedent:
  `src/kernels/fp8_gemm.cu` header, "NOT bitwise the dequantized bf16 chain").
- The three attention entry points and where they are called:
  `qsa_attn_partial` (decode + short prefill; `layers.cpp:803,807`),
  `qsa_attn_prefill_partial` (long prefill, `qsa_prefill.cu`),
  `qsa_attn_prefill_warp` (long prefill, tensor cores, `qsa_warp.cu`).

## The goal for the plan

Move the QSA attention's **`Q·Kᵀ` and `P·V` products onto the E4M3 tensor cores**
so the stored codes are consumed *without* dequantizing to bf16 — the
"native, no-dequant" end state. Concretely:

- Q quantized per-row to E4M3 + an fp32 scale; K is already stored E4M3 +
  per-(slot, kv-head) scale; the `Q·Kᵀ` MMA runs on E4M3 with the
  `q_scale * k_scale` product applied in the epilogue.
- The probabilities `P` (post-softmax) quantized per-row to E4M3; V already
  E4M3; the `P·V` MMA on E4M3 with `p_scale * v_scale` in the epilogue.
- The split-KV partials + `dsa_attn_combine` merge structure stays; only the
  inner products change dtype.

## Frontier (you OWN deciding these)

1. **The numerics contract and tolerance band.** Phase B breaks three pins:
   `qsa.hpp:126-129` ("probabilities round to bf16 for the V accumulation, `l`
   stays unrounded"), `qsa_prefill.cu:3` (prefill/decode rounding order), and
   `qsa.hpp:141-143` ("every gather form is bitwise every other"). Decide what
   replaces them, and how the band is *established* (derive it, or measure it —
   propose the measurement). Name the tests to update (`tests/cuda/qsa_test.cu`,
   `tests/cuda/qwen_full_attn_test.cpp`).
2. **The kernel design per path.** At minimum the *tractable* first step:
   `qsa_attn_prefill_warp` already has the MMA + `ldmatrix` scaffold (bf16
   `m16n8k16`); moving it to `m16n8k32` E4M3, quantizing Q, quantizing the reused
   P accumulator, and applying the epilogue scales. The *hard* case is
   decode/short-prefill: today they have **no tensor cores at all**
   (warp-shuffle FMA, `qsa.cu:274-279,627`, `qsa_prefill.cu:109`), so Phase B
   there needs a **new tensor-core lowering over `topk`-gathered (non-contiguous)
   rows**. Decide whether the plan covers it and how, or explicitly defers it
   (leaving decode on the Phase A dequant path is acceptable and already neutral).
3. **The pre-implementation evidence.** Phase A stood up
   `tests/cuda/qwen_fp8_kv_attn_test.cu` *before* writing any fp8 kernel, and it
   paid off. Propose the Phase B analogue: a host/GPU numeric study that measures
   the FP8-MMA divergence (Q/K/P/V E4M3 through the existing bf16 attention vs
   the projected MMA math) and **pre-registers the acceptance band** before any
   kernel is written.
4. **Sequencing, file-by-file changes, risks, and the verification ladder.**

## Locked decisions (each with its reason; challenge only with evidence)

- **Reuse the existing E4M3 MMA assets, don't invent.** `fp8_gemm.cu:184`
  (`m16n8k32 e4m3`, `ldmatrix`, 80-byte stride) and **`dsa.cu:1650`** (the same
  MMA *inside the DSA attention* — the closest precedent). Locked because these
  are already validated on this hardware; unlock if you find a structural reason
  they don't fit QSA.
- **Prefill-warp first.** Locked because its MMA/`ldmatrix` scaffold already
  exists, so it is the smallest real step. Unlock if the design work shows the
  decode path is actually cheaper to land first.
- **Phase A stays the base.** The fp8 pool, the append quantize, and the
  dequant read path are unchanged inputs. Phase B replaces only the attention
  *read* math.
- **Research + plan only.** Produce `docs/qwen_fp8_mma_plan.md`. Do not edit
  `src/`, `tests/`, or `CMakeLists.txt`.

## Acceptance (what convinces the parent the plan is real)

The plan must be specific enough to implement without re-deriving:
- the exact numerics contract + how the band is established (with a concrete
  proposed measurement, not "measure it later");
- the kernel design with real structure (which kernel, the tile/register layout
  sketch, the quantize points, the epilogue scale application), citing
  `file:line` for every asset it reuses;
- the file-by-file change list;
- the evidence + verification ladder, naming the tests to add/update;
- the sequencing with a clear "first shippable step";
- the risks, especially anything that would make Phase B a net loss at short
  context (Phase A already showed the KV is <0.1% of the streamed bytes there —
  be honest that Phase B's win is at long context).

## Non-goals

- No kernel or test implementation. No changes under `src/` or `tests/`.
- No re-litigating Phase A (it is done and measured).
- No other model families (GLM / MiMo / DeepSeek) — Qwen3.8-Flash-Next only.
- No speculative-decoding / MTP / MoE work.

## Output

Write `docs/qwen_fp8_mma_plan.md` (markdown), then return a short summary:
what you decided, the first shippable step, and any open question you could not
settle. Keep prose tight — this repo's docs are dense and factual, no filler.
