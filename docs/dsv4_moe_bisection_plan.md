# dsv4 layer-0 MoE bisection — work plan (2026-10-21)

Read `docs/dsv4_HANDOFF.md` first. This doc is the *plan of record* for the
round that follows it: it fixes the evidence contract, the two workstreams,
the lead list, the rig rules, and what "done" means for each stream.

## Objective

Layer 0's stream 3 (the MoE's lane) is wrong — cos 0.7195 vs the checkpoint's
own math, L2 7.953 vs 8.102 — while streams 0, 1, 2 are bit-exact. Every
sub-step of the MoE path is a suspect; the job of this round is to **turn that
into a named, evidenced divergence** and, if the evidence is decisive, fix it.

Success is observable, not structural:

1. the harness names the first divergent sub-step with numbers (round 1), or
2. the engine's layer-0 dump matches the reference: streams 0-3 cos ≈ 1.0,
   and then `scripts/dsv4_coherence_check.sh` reproduces the reference's
   opening (round 2), and `scripts/dsv4_gates.sh smoke` passes.

## What is already ruled out (do not re-derive)

The handoff's "Verified NOT the cause" list: the slot kernel composition vs the
CPU oracle, the routing formulas, the SwiGLU clamp direction, the loader's
expert order, the shapes/dtypes, the shared expert's weight, `wo_a`,
`ffn_norm`/`hc_ffn_*` bindings, the mHC's site mechanism (streams 0-2 exact).
**Also do not compare `moe_out_l0.f32` against the world-1 reference** — ours is
that rank's pre-all-reduce partial (that probe is what sent round 1 down a
3.6×-error wild goose chase).

## The gap that this plan exists to close

The only MXFP4 (group-32) slot-path parity that exists is
`tests/cuda/dsv4_moe_slot_test.cu`, and it runs at **toy geometry: hidden 64,
inter 32, k = 64 / 32**. The dsv4 deployment runs **k = 4096 and 1024** with
`I_r = I_s = 1024` slices, MXFP4 at 32 codes/scale, 35 slots. Nothing in the
suite has ever compared the *slot* path against any oracle at the real K, at
the real slice, with the real weights. That is the first thing to build.

## Evidence contract (fixed up front — both streams code against this)

Dump root `/tmp/dsv4moe/`, one subdirectory per producer:

```
/tmp/dsv4moe/engine/     # what the engine's layer-0 MoE sees and emits (rank 0, world 2)
/tmp/dsv4moe/oracle/     # what the checkpoint's math says (rank-0 slice emulated)
/tmp/dsv4moe/ref/        # the checkpoint's own python at world 1 (sub-step hooks)
```

Common files (little-endian, raw, no headers; `meta.txt` carries the dims):

| file | shape | dtype | meaning |
|---|---|---|---|
| `meta.txt` | text | — | `T H K I_r I_s fp4_group sh_rs sh_cs world rank tokens` |
| `x.bin` | [T, H] | bf16 | the MoE site's input (post `ffn_norm`) |
| `ids.bin` | [T, K] | i32 | `topk_ids_` / the reference's `indices` |
| `w.bin` | [T, K] | f32 | `topk_w_` / the reference's `weights` |
| `slot_act.bin` | [T*(K+1), I] | bf16 | post-SwiGLU activations, slot-major (`slot = t*(K+1)+j`) |
| `slot_down.bin` | [T*(K+1), H] | f32 | the down outputs, unrounded, slot-major |
| `out.bin` | [T, H] | bf16 | the site's output, **pre-all-reduce**, matching `moe_out_l0.f32`'s semantics |

`slot = t*(K+1) + j` with `j < K` = routed expert `ids[t][j]`, `j == K` = the
shared expert's row. `T` = 5 (the probe prompt `The capital of France is`,
token ids `671,6102,294,8760,344`), `H` = 4096, `K` = 6, `I_r = I_s = 1024`,
`fp4_group = 32`.

Everything is compared after the *same* slice convention: a comparison is
meaningful only when both sides agreed on it, so a mismatch in the slice rule
is itself a finding, not a nuisance (L3).

## Workstream 1 — engine/GPU: the layer-0 MoE parity harness (owner: coder, main checkout)

Build and run a CUDA test that puts the engine's **actual** MoE code at layer 0
against an **oracle over the same real host matrices**:

1. Load layer 0's real weights through `src/models/dsv4/loader.cpp` (the host
   `GlmFp4Matrix` experts, the fp8 shared triple, `tid2eid`, the gate), plus the
   probe prompt's `x` (from `--x-from` a dumped `x.bin`, else a deterministic
   synthetic row).
2. Compute the whole MoE for those 5 tokens in **fp64 on the CPU** over those
   same host matrices: routing (hash path), gate/up, the clamped SwiGLU, down,
   the router-weight fold, the shared expert's fp8 path — at the **rank-0
   slice** (and rank 1, as a second case: two ranks must sum to the world-1
   result).
3. Run `Dsv4HashLayer::route()` + `Dsv4HashLayer::expert()` on the real
   weights/geometry in the engine's own build, and dump every sub-step in the
   contract's format (`/tmp/dsv4moe/engine/`).
4. Emit per-sub-step `cos`, `max-abs`, and the first index that diverges, for
   ids / w / slot_act (routed rows and the shared row separately) / slot_down
   (likewise) / out.

Then bisect with the harness as the instrument: the wiring knobs to flip, one
at a time, are `fp4_group`, `I_r`/`I_s` and the slice rule, the view-table
fields the kernels actually read, `sh_rs`/`sh_cs`, and the `topk_w` path into
`launch_moe_slot_accum`.

**Deliverable:** the harness committed on `dsv4-flash`, a PASS/FAIL table in
`docs/`, and a *named* first divergence (or a green harness). If the harness
goes green, the fix must additionally move the **end-to-end** number: re-run the
dump config and report layer 0's four stream cosines against `/tmp/dsv4ref`.

**Non-goals for this round:** prefill/grouped-path performance, MTP, the decode
graph, the tokenizer/chat template, the rest of the layer stack.

## Workstream 2 — checkpoint/host truth + wiring audit (owner: coder-flash, worktree)

No engine boot, no exclusive GPU in round 1. Three jobs:

1. **The spec table.** Read the checkpoint's own `inference/model.py`
   (`.../snapshots/local/inference/model.py`) — `Gate`, the hash path, the
   routed-expert MXFP4 decode, the shared expert, the SwiGLU clamp, the down
   epilogue — and write down, as citable line-level facts, exactly what the
   reference does at each of L5/L6/L7/L9. Where the reference is world-1-only,
   say so explicitly and state what a TP slice *must* mean for the sum to be
   invariant.
2. **The wiring audit** against that spec, in `src/models/dsv4/loader.cpp`,
   `binding.cpp`, `hash_layer.{hpp,cu}`, `model.cpp`'s MoE site: the expert
   table order, the inter slice rule (contiguous vs interleaved, and whether
   the gate/up slice and the down column slice agree), the MXFP4 scale layout
   (`[rows, k/32]` e8m0 row-major) as the kernels index it, the shared
   expert's F8_E8M0 decode and 128×128 grid (`sh_rs = sh_cs = 7`), the
   `tid2eid` dtype/layout, the gate/bias bindings, and that
   `routed_scaling_factor`/`swiglu_limit`/`norm_topk_prob` reach
   `Dsv4HashConfig` from `config.json` (1.5 / 10.0 / true — a stale default
   here is a one-line structural error with exactly this signature).
   Each item: PASS/FAIL + the file:line + the value observed.
3. **The inverse test (CPU-only, uses the dumps on this box).** From
   `/tmp/dsv4dump/req_0000/rank_0/` and `/tmp/dsv4ref/`: solve for the MoE
   output vector our engine would have to have produced for layer 0's stream 3
   to come out as observed, given the reference's mHC coefficients and the
   (bit-exact) streams 0-2. If that inferred vector is plausible (right
   magnitude, right-ish direction) → the MoE is the culprit and mHC is
   exonerated; if it is not → the FFN-site comb/post coefficient path is a
   live suspect (L8) and the harness's target changes. This is the cheapest
   single test in the whole plan.

**Deliverable:** `docs/dsv4_moe_reference_spec.md` (the spec table + the audit
+ the inverse test's numbers), and, if the audit finds a smoking gun, a
*proposed* one-line diff with its rationale — not applied blind.

## Lead list (each with what kills it)

| # | lead | killed by |
|---|---|---|
| L0 | a wrong layer-0 MoE output is *sufficient* to explain "only stream 3" | WS2's inverse test |
| L1 | the fp4 slot path's view fields at real K/MXFP4 geometry (never exercised) | WS1 harness at real K |
| L2 | MXFP4 scale addressing/stride at k = 4096 / 1024 | WS1 harness + `load_chunk_scale` audit |
| L3 | the slice rule (contiguous halves; gate/up slice vs down column slice) | WS2 audit + a rank-0 + rank-1 sum test |
| L4 | the shared expert: sliced vs replicated, and whether the all-reduce doubles/drops it | WS2 spec + WS1's `shared_only` case |
| L5 | the hash-layer routing (ids order, weights = unbiased sqrtsoftplus, renormalize, ×1.5) | WS2 spec + WS1's `ids`/`w` comparison on real weights vs the checkpoint's `Gate` |
| L6 | the SwiGLU clamp: gate upper-only, up both sides, order vs the bf16 rounding | WS2 spec + a two-point clamp case in the harness |
| L7 | the down epilogue's fold (routed × w, shared × 1) and the single rounding point | WS1 harness (`out` case) |
| L8 | the mHC FFN-site coefficients (only stream 3 moves) | WS2's inverse test |
| L9 | the MoE site's input (`ffn_norm` semantics) | WS1 harness with a dumped `x.bin` — removes the input class entirely |
| L10 | the world-2 partial vs world-1 comparison | only compare post-collective states, or a rank-0+rank-1 sum |

## Rig rules (hard)

- **One stack at a time.** Take the Qwen lane down before any engine boot
  (`cd ~/work/q-dgx-gateway && make down-dgpp-512k`). Do **not** boot the vLLM
  reference lane (`v-dgx-gateway make up-base`) — it is mutually exclusive with
  our engine and round 1 does not need it.
- **The GPU rig is WS1's, exclusively.** WS2 must not boot `dgpp-serve` /
  `dgpp-cluster` and must not run a GPU allocation while WS1's engine is up;
  WS2's round-1 work is CPU-only by construction.
- Dev build only (`build-dsv4-merge`), never `build-release`. The dump needs
  `decode_graph: false`; `DGPP_DSV4_DUMP_LAYERS` stays out of `.env.dgpp` and
  travels in the dsv4 config's `node_env` (both allowlists).
- The reference lane configs are `q-dgx-gateway`'s `config/dgpp-dsv4-*.json`;
  the recipe-style rule from AGENTS.md holds: don't edit `.env` files that
  another lane owns.
- Leave the box as found: engine down, `/tmp/dsv4moe/` and `/tmp/dsv4ref/` in
  place for the next round.

## Sequencing

Round 1 (parallel, both streams): WS1 builds the harness and runs it; WS2
builds the spec table, the audit, and the inverse test. Both end by handing me
(a) the dump directory, (b) the evidence table, (c) a proposed next action.
Round 2 (I decide after reading the evidence): the fix landed by exactly one
stream, verified by the other; then the coherence check, then the gates.
