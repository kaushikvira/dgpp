# Benchmarks

This page is the current benchmark snapshot for the production paths in the
tree. It reports the newest applicable serving measurements, through
2026-09-16, for each model, world size, concurrency and prompt class.
Superseded baselines and optimization chronology live in `benchmarks/results/`;
they are not mixed into the headline tables here. Modeled bandwidth floors and
unmeasured cases are identified separately. Section 8 lists missing
measurements, and section 9 gives the procedures.

- §1 how to read a number
- §2 the environment and the deployments
- §3 decode, one request
- §4 decode, per prompt class
- §5 decode under concurrency
- §6 prefill and time to first token
- §7 quality, so the throughput numbers are comparable
- §8 what is not measured yet
- §9 reproducing all of it

## 1. How to read a number

**The unit of decode work is a pass, not a token.** One pass is one replay of
the decode graph: the engine's `ms/step` in the stats line. Without
speculative decoding a pass yields one token per request. With MTP a pass
yields `tok/pass` tokens, between 1 and 1 + `mtp_depth`, according to how many
drafts the verify accepted. So

    ms/token = ms/pass / tok/pass

and a change that raises acceptance moves `ms/token` while leaving `ms/pass`
alone. Both are quoted throughout, because only their ratio is the user's
experience and only `ms/pass` is the engine's cost.

**Aggregate tokens/s is a client-side reading over a phase**, summing every
concurrent request's tokens across the phase's decode span. It is not
`1000 / ms/token`: at concurrency c the requests share each pass, so each one
sees roughly c times its solo pace while the world delivers more in total.
Per-request pace is given beside the aggregate wherever both were recorded.

The decode class tables use the phase aggregate emitted by the benchmark
campaign that produced them. The newest campaigns use request-wall throughput;
older still-current class sweeps use the explicitly named output-span scope,
from first visible output to last visible output. Each section identifies the
scope when it matters. Per-request latency is **ms per client update**; an SSE
update can contain several tokens. Token rates use the server's completion
usage.

For a C1 regression check, capture both builds with
`--classes all --concurrency 1 --repeat 3 --json-out RUN.json`, using the
same model, deployment and `--max-tokens`. Then run
`python3 scripts/bench_compare.py BASELINE.json CANDIDATE.json`. It requires
matching greedy C1 transcripts, checks medians in both timing scopes, and
prints the min/max spread. Its default allows no median slowdown; resolve
small noisy differences with repeated A/B runs, rather than selecting a
favorable sample. The corpus retains the five C1 anchors and now provides
sixteen distinct prompts per class for wider request bursts.

The [September 15 Qwen batching and continuation campaign](../benchmarks/results/2026-09-15-qwen-batching-prefill.md)
records the matched baseline, configuration tradeoffs and regression gates
for the opt-in paths. The shipped Qwen templates remain at four slots with
monolithic admission; the wider eight-slot graphs and prefill budgets are
available but are not represented as defaults below.

**Greedy and sampled throughput are equivalent on the current proposal
rule.** The 2026-09-10 sweeps measure them within run-to-run drift on every
family. Superseded results from the older deterministic-draft rule remain only
in the dated records.

**Prefill is quoted per token** (`prefill ms / prompt tokens`), because it is
close to linear in the prompt and the per-token rate is what transfers between
lengths. Absolute milliseconds are given too, since time to first token is
what a caller waits.

**Run-to-run drift on this hardware is about 2 % across a day.** Single runs
are therefore quoted as ranges where a range was recorded, and every A/B in
the source documents is a pair of runs on one binary with one knob between
them. Do not read a 1 % difference between two rows of this page as a result,
especially when their dates differ.

**Dates identify provenance, not a progression of alternatives.** A current
path can legitimately cite an older measurement when later changes do not
touch that path and its regression gate passed. Before/after comparisons are
kept in the linked dated record rather than in this snapshot.

## 2. The environment and the deployments

Four NVIDIA DGX Spark (GB10) nodes, 121 GB of unified memory each, one
200 Gb/s RoCE port per node behind two PCIe x4 physical functions, CUDA 13.
Ranks are resident: each holds its slice of the model on the GPU for the life
of the world and boots from a per-rank image cache. All decode numbers are
from a warm world.

A deployment is a cluster config. These are the ones with a committed
template, which is what "supported" means on this page:

| model | world | template | modes measured |
|---|---|---|---|
| `unsloth/GLM-5.3-Flash-FP8` | 4 | Copy `cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json` to a separate FP8 config and set `model` to the FP8 repository | T=1 (`--no-mtp`), MTP depth 1 |
| `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8` | 4 | `cluster_glm-5.3-flash_nvfp4-fp8_w4.example.json` | T=1, MTP depth 1 |
| `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8` | 2 | `cluster_glm-5.3-flash_nvfp4-fp8_w2.example.json` (the 256K-context two-slot shape: `--knobs "--max-concurrency 2 --kv-capacity 262144 --prefix-cache-gib 2"`) | T=1, MTP depth 1 |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 4 | `cluster_qwen-3.8-flash-next_fp8_w4.example.json` | T=1 (`--no-mtp`), MTP depth 1, depth 2 (`--mtp-depth 2`) |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 2 | `cluster_qwen-3.8-flash-next_fp8_w2.example.json` | T=1, MTP depth 1, depth 2 |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 1 | `cluster_qwen-3.8-flash-next_nvfp4_w1.example.json` (the FP8 dense stack; `--dense-weights checkpoint` for BF16) | T=1, MTP depth 1, depth 2; BF16 or FP8 dense stack |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 2 | `cluster_qwen-3.8-flash-next_nvfp4_w2.example.json` (FP8 dense stack, mmap n-gram table, 262K context) | MTP depth 1; mmap is shipped, resident is the measured placement baseline |
| `nvidia/GLM-4.7-NVFP4` | 4 | `cluster_glm-4.7_nvfp4_w4.example.json` | T=1, MTP depth 1, depth 2 |
| `HawkBearPig/GLM-5.3-Int4-Int8Mix-RTN-g64` (the full GLM-5.3) | 4 | `cluster_glm-5.3_int4-int8_w4.example.json` (eight slots; the fp8 latent cache at 208K: `--knobs "--kv-dtype fp8 --kv-capacity 212992 --prefix-cache-gib 1.5"`) | T=1, MTP depth 1, depth 2 (two slots); bf16 or fp8 latent cache |
| `deepseek-ai/DeepSeek-V4.1-Flash` | 4 | `cluster_deepseek-v4.1-flash_mxfp4-fp8_w4.example.json` (six slots at DSpark depth 4; the two-slot depth-5 shape: `--knobs "--max-concurrency 2 --mtp-depth 5"`) | T=1, DSpark depths 4 and 5 with the scheduled verify depth |

The single-Spark world has a second axis, `engine.dense_weights`: the
checkpoint's own BF16 dense projections, or the same projections encoded to
block FP8 as they load. Both are measured below; FP8 is the faster one and the
BF16 one is the reference the transcripts are compared against.

GLM-4.7 has been measured only at world 4. Qwen3.8-Flash-Next-NVFP4 is measured
at worlds 1 and 2. The engine takes the world from the config. The GLM-5.3 hybrid is measured
at worlds 4 and 2; the two-node deployment carries the same weights on half
the ranks, so each rank holds 94.71 GiB of them instead of 50.74 GiB, and its
latent cache is FP8 rather than BF16 to buy back context (160K tokens at four
request slots).

## 3. Decode, one request

### GLM-5.3-Flash-FP8, world 4

| mode | ms/pass | tok/pass | ms/token | tokens/s | date |
|---|---|---|---|---|---|
| T=1, one graph | 29.94–29.97 | 1.0 | 29.94–29.97 | 33.4 | 2026-09-10 |
| MTP, greedy, through the service | 34–41 | 1.68–1.97 | 20.62–23.79 | 42.2–48.7 | 2026-09-10 |

The T=1 step reads p50 30.0 ms and p99 31.0 ms. The MTP range is the current
five-class C1 sweep in §5; sampled rates land within the same run-to-run drift.

Context costs little: the step is about 41 ms under 2K tokens of context and
42–43 ms from 8K to 32K, after the 2026-09-06 select rewrite that took the long
end down from 52–56 ms.

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 4

| mode | ms/pass | tok/pass | ms/token | date |
|---|---|---|---|---|
| T=1, through the service (`--no-mtp`) | 24.9 | 1.0 | 24.9 | 2026-09-19 |
| MTP, greedy | 31.5–33.2 | 1.72–1.98 | 16.1–19.3 | 2026-09-19 |

The routed experts are NVFP4 and everything else is the FP8 release's own
bytes. The current MTP range comes from the five-class service sweep in §4.
Both rows are with the template's `bf16_weights` on (`"bf12+bf16"` on four
nodes, `"bf12"` on two: decode is the same) — the lossless
12-bit form of the BF16 matrices decode streams (bit-identical transcripts):
the same binary with the key off measures 27.0 ms at T=1 and 33.7–34.4 ms per
MTP pass, the 2026-09-16 figures
([record](../benchmarks/results/2026-09-19-glm-flash-line-rate/README.md)).

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 2

| mode | ms/pass | tok/pass | ms/token | date |
|---|---|---|---|---|
| T=1 | 45.68 | 1.0 | 45.68 | 2026-09-12 |
| MTP, greedy | 59.40 | 1.734 | 34.25 | 2026-09-12 |
| MTP, sampled, through the service | 60 | 1.70–1.81 | 34.1–35.3 | 2026-09-12 |

The same weights on half the ranks: 1.74x the four-node pass at T=1, 1.75x under
MTP. The T=1 step is 45.68 ms mean with p99 47, so the world is as steady as the
four-node one. The MTP and T=1 runs produced the same rank-consistency digest,
which is the speculative-identity gate (§9.6) passing on this world.

### Qwen3.8-Flash-Next-FP8, world 4 and world 2

| world | mode | ms/pass | tok/pass | ms/token | date |
|---|---|---|---|---|---|
| 4 | T=1 | 21.40–21.50 | 1.0 | 21.40–21.50 | 2026-09-10 |
| 4 | MTP, greedy | 24–25 | 1.6–2.0 | 12.0–15.6 | 2026-09-14 |
| 2 | T=1 | 32.5 | 1.0 | 32.5 | 2026-09-09 |
| 2 | MTP | 41 | 1.56–1.98 | 20.8–26.3 | 2026-09-09 |

Depth 1 is the shipped setting. The QSA prefill optimization added on
2026-09-15 does not change decode arithmetic; 80 paired C1 responses retained
identical text, usage and decode-step counts (§6).

Since 2026-09-19 the templates' `bf16_weights: "bf12+bf16"` streams the GDN and
QSA projections, the draft block's and the head from their lossless 12-bit
form (1.65 → 1.24 GiB per rank at world 4, 3.20 → 2.41 at world 2; 81 % of
what a world-4 rank reads per token is BF16, 1.7 GB of it packed). The same
binary through the service, MTP greedy, engine decode tok/s at one to four
live requests: world 4 73.3 / 118.8 / 141.0 / 162.2 → 78.0 / 123.3 / 144.3 /
164.3 (+6.4 / +3.8 / +2.4 / +1.3 %); world 2 47.1 / 73.3 / 85.4 / 96.2 → 51.4 /
78.2 / 91.0 / 100.6 (+9.0 / +6.7 / +6.6 / +4.6 %); transcripts byte-identical,
prefill unchanged
([record](../benchmarks/results/2026-09-19-glm-flash-line-rate/README.md) §7).
The rows above predate it.

### Qwen3.8-Flash-Next-NVFP4, world 1 (one Spark)

| dense stack | mode | ms/pass | tok/pass | ms/token | tokens/s | date |
|---|---|---|---|---|---|---|
| BF16 | T=1 | 47 | 1.0 | 46.7–47.2 | 21.4 | 2026-09-10 |
| BF16 | MTP, greedy | 59 | 1.56–1.88 | 31.4–37.7 | 27–32 | 2026-09-10 |
| BF16 | MTP, sampled | 59 | 1.62 | 36.6 | 27 | 2026-09-10 |
| FP8 | T=1 | 30.2–30.9 | 1.0 | 30.2–30.9 | 32.3 | 2026-09-10 |
| FP8 | MTP, greedy | 39–40 | 1.54–1.93 | 20.5–25.8 | 39–49 | 2026-09-10 |
| FP8 | MTP, sampled | 39–40 | 1.6 | 24.7 | 40 | 2026-09-10 |
| FP8 | MTP depth 2, greedy | 48 | 1.77–2.74 | 17.4–27.1 | 37–57 | 2026-09-10 |

The FP8 dense stack is the same checkpoint with every dense projection encoded
to block FP8 at load. It costs about 4 GiB less resident, runs T=1 at 30.3 ms
against a 25.5 ms bandwidth floor, and its greedy transcripts diverge from the
BF16 stack's after 170–316 characters at equal eval scores (§7).

Depth 2 on the FP8 world is the only place in this project where depth 2 is
worth setting, and only for some classes: see §4.

### Qwen3.8-Flash-Next-NVFP4, world 2, FP8 dense, mmap n-gram (2026-09-16)

| class | ms/pass | tok/pass | ms/token | tokens/s |
|---|---:|---:|---:|---:|
| prose | 25 | 1.83 | 13.6 | 73.7 |
| code | 25 | 1.73 | 14.7 | 67.9 |
| json | 25 | 1.88 | 13.4 | 74.9 |
| math | 25 | 1.83 | 13.6 | 73.8 |
| chat | 26 | 1.59 | 16.1 | 62.1 |

The current recipe maps the n-gram table from NVMe. It uses 36.50 GiB of
planned model memory per rank, 23.84 GiB less than the otherwise identical
resident placement. The matched resident baseline has the same draft
acceptance and greedy transcripts. Mmap changes request-wall throughput by at
most 3.6% and prefill time by at most 2.9%. The
[two-Spark result](../benchmarks/results/2026-09-16-qwen-nvfp4-w2.md) contains
the repeated decode, prefill, memory and quality measurements.

### GLM-4.7-NVFP4, world 4

| mode | ms/pass | tok/pass | ms/token | date |
|---|---|---|---|---|
| T=1 | 49.0 | 1.0 | 49.0 | 2026-09-10 |
| MTP depth 1 | 56–59 | 1.84–1.97 | 28.4–32.1 | 2026-09-14 |
| MTP depth 2 | 72–73 | 2.32–2.60 | 28.0–31.2 | 2026-09-10 |

The T=1 floor here is about 40 ms, and 6.3 GB per rank per step of it is the
BF16 attention projections that modelopt left unquantized. That is also what
bounds this family under concurrency (§5).

Since 2026-09-19 the template's `bf16_weights: "bf12+bf16"` streams those
projections from their lossless 12-bit form (6.36 → 4.78 GiB per rank, +4.9
GiB resident beside the BF16 bytes; `"bf12"` alone is 1.4 GiB UNDER the BF16
plan for +2–4 % of prefill): the same binary through the service measures 31.4–33.0 → 35.0–
36.5 tok/s single stream (+11–12 %) and 66.8–70.2 → 69.9–73.8 at four live
requests (+5–7 %), transcripts byte-identical
([record](../benchmarks/results/2026-09-19-glm-flash-line-rate/README.md)).
The rows above predate it.

### The full GLM-5.3 (int4/int8 g64), world 4

| mode | ms/pass | tok/pass | ms/token | date |
|---|---|---|---|---|
| T=1 | 51.1 | 1.0 | 51.1 | 2026-09-12 |
| MTP depth 1 | 61–66 | 1.69–1.97 | 31–39 | 2026-09-14 |

Since 2026-09-19 the template's `bf16_weights: "bf12"` packs what this
checkpoint leaves BF16 on the matmul seam (the dense layers' and the draft's
projections, the indexers, the head: 1.65 → 1.24 GiB per rank): 28.3–28.8 →
29.8–30.6 tok/s single stream (+5–6.5 %), 45.4–46.6 → 48.1–48.4 at four
(+3–7 %), transcripts identical. The template is sized to the node's
ceiling: with both forms resident it held 100K of context; with the 12-bit
form alone (`"bf12"`, the BF16 bytes returned as each layer loads — the
template since the same day) it is 0.3 GiB under the BF16 plan and carries
the 120K shape again, decode level at one, four and eight live requests and
prefill within 1 %. The rows
above predate it.

The 754B model at 99.3 GiB of int4/int8 weights per rank: the T=1 step
streams about 10.2 GB per rank (the audit's floor 44.5 ms at 230 GB/s) plus
158 collectives. MTP at depth 1 gives 1.4x per token. Transcripts: MTP == T=1
on all four prompts; op streams identical across the four ranks at every
shutdown (§9.6). Boot from the resident image 30 s; the first boot, which
captures it, 405 s. The campaign ran at 48K tokens of bf16 latent cache
because rank 2's node then had 119.67 GiB (its launch firmware's 4 GB
display reservation; fixed by the SoC firmware update the same night); the
templates carry 120K bf16 at four slots with MTP (110.41 GiB per rank), 144K
plain (110.75) and 208K with the fp8 latent cache (110.35), the embedding
vocab-sharded (−1.33 GiB, exact), the ceilings under the 4 GiB headroom that
replaced the 8 GiB one on 2026-09-12/13 once the loader's 2.2 GiB staging
mirror was measured, itemized and freed before the caches and a one-hour soak
at the 120K shape stayed flat with no reclaim on any node.
The fp8 latent cache runs at the same pace (68–69 ms/pass, 74–98 %
acceptance by class) and its greedy transcripts diverge from the bf16
cache's after 97–464 characters, the two-node Flash finding again.

### DeepSeek-V4.1-Flash MXFP4/FP8, world 4

The current six-slot template uses DSpark depth 4, scheduled verify depth,
adaptive lambda, grouped prefill, tensor-core decode/prefill projections and
the stream-ordered eager fold. The latest like-for-like client run reports:

| load | aggregate decode | per stream | mean TTFT | date |
|---|---:|---:|---:|---|
| C1 | 49.64 tok/s | 54.8 tok/s | 0.269 s | 2026-09-14 |
| C6 | 108.49 tok/s | 20.41 tok/s | 0.727 s | 2026-09-14 |

The aggregate and per-stream rates use the external recipe client's two
different timing scopes, so they should not be divided into each other. On the
fixed five-class DGPP corpus, the scheduled-depth path spans 18.9–24.6
ms/token before the later tensor-core and grouped-prefill changes; the newer
like-for-like result above is the current end-to-end reading. The full
[dated campaign](../benchmarks/results/2026-09-14-dsv41-flash.md) retains the
optimization sequence.

## 4. Decode, per prompt class

Classes are five fixed prompts, greedy, 300 tokens each unless the table says
otherwise: **chat** a technical explanation, **code** a Python module with
tests, **prose** a long history, **json** twenty-five records, **math** a
worked problem. §9.2 has the exact texts and the procedure.

The step time is nearly flat across classes. What moves is draft acceptance,
and therefore tokens per pass, and therefore the pace.

### GLM-5.3-Flash-FP8, world 4, MTP greedy (2026-09-10)

Current service-path pace at concurrency 1:

| class | ms/token | tokens/s |
|---|---:|---:|
| prose | 21.85 | 45.9 |
| code | 21.01 | 47.7 |
| json | 20.62 | 48.7 |
| math | 22.02 | 45.5 |
| chat | 23.79 | 42.2 |

The graph pass spans 34–41 ms and commits 1.68–1.97 tokens per pass by class.

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 4, MTP greedy (2026-09-19)

HTTP service, 256 output tokens, MTP depth one, default thinking mode, the
template as shipped (`bf16_weights` on: `"bf12+bf16"`). Means of two repetitions.
Engine timing excludes admission/prefill and the first token emitted by
prefill. Tokens/pass counts committed decode work.

| class | engine ms/pass | committed tok/pass | engine ms/token | engine tok/s | same binary, `bf16_weights: checkpoint` |
|---|---:|---:|---:|---:|---:|
| prose | 31.48 | 1.861 | 16.91 | 59.12 | 54.8 |
| code | 32.17 | 1.977 | 16.28 | 61.44 | 57.1 |
| json | 31.63 | 1.962 | 16.12 | 62.02 | 57.6 |
| math | 32.03 | 1.917 | 16.70 | 59.86 | 55.6 |
| chat | 33.20 | 1.723 | 19.27 | 51.90 | 50.4 |

Tokens per pass are unchanged and all five C1 transcripts are byte-identical
to the control's: the 12-bit form is a storage format for the same BF16
values. The chat class drifts between boots (31.7–33.2 ms/pass across this
campaign's runs of the same build). See the
[campaign](../benchmarks/results/2026-09-19-glm-flash-line-rate/README.md).

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 2, MTP greedy (2026-09-12)

| class | drafts accepted | tok/pass | ms/pass | ms/token |
|---|---|---|---|---|
| chat | 73.4 % | 1.734 | 59.40 | 34.25 |
| code | 98.0 % | 1.974 | 60.17 | 30.49 |
| prose | 83.5 % | 1.829 | 58.71 | 32.09 |
| json | 97.4 % | 1.974 | 59.16 | 29.98 |
| math | 88.7 % | 1.887 | 59.96 | 31.78 |

Both ranks' transcripts identical on every class. Acceptance matches the
four-node run class for class, so the whole difference is the pass: 58.7–60.2 ms
against 33.7–34.3, a factor of 1.75 that follows the doubled per-rank weight
read. Per token the two-node world lands at 1.69x to 1.76x of the four-node
figures.

### Qwen3.8-Flash-Next-NVFP4, world 1, MTP greedy (2026-09-10)

| class | BF16 dense: tok/pass, accept, ms/token | FP8 dense: ms/token | FP8 depth 2: ms/token |
|---|---|---|---|
| prose | 1.56, 56 %, 37.7 | 25.8 | 27.1 (+5 %) |
| code | 1.88, 88 %, 31.4 | 22.3 | 20.1 (−10 %) |
| math | 1.83, 83 %, 31.9 | 20.7 | 18.5 (−11 %) |
| json | 1.88, 88 %, 31.4 | 20.5 | 17.4 (−15 %) |

The pass is 59 ms on the BF16 stack, 39–40 on FP8, and 48 at depth 2. The
second draft stands 60–84 % of the time on code, math and JSON and only 26–32 %
on prose, which is why depth 2 pays on the first three and loses on the last.
A code-heavy deployment is the case for setting `mtp_depth: 2` here.

### GLM-4.7-NVFP4, world 4, MTP greedy, 256-token prompts (2026-09-10)

| class | depth 1: ms/pass, tok/pass, ms/token | depth 2: ms/pass, tok/pass, ms/token, p1 / p2 | tokens/s |
|---|---|---|---|
| chat | 61, 1.86, 32.5 | 72, 2.32, 31.2, 85 % / 48 % | +4 % |
| code | 61, 1.93, 33.0 | 72, 2.52, 28.7, 92 % / 60 % | +13 % |
| math | 62, 1.95, 31.3 | 72, 2.55, 28.4, 94 % / 62 % | +10 % |
| json | 61, 1.99, 31.1 | 73, 2.60, 28.0, 96 % / 65 % | +11 % |
| 6,525-token prompt | 66, 1.86, 35.4 | 79, 2.35, 33.7, 85 % / 52 % | +5 % |

Depth 2 gains 4–13 % single-stream here and loses under concurrency (§5), so
the shipped template keeps depth 1. Use `--mtp-depth 2` for that single-stream
tradeoff.

### Qwen3.8-Flash-Next-FP8, MTP greedy, through the service

Client-side pace of one request, ms/token, at concurrency 1:

| class | world 4 | world 2 |
|---|---|---|
| prose | 14.88 | 22.62 |
| code | 13.28 | 20.82 |
| json | 12.87 | 20.23 |
| math | 13.48 | 21.37 |
| chat | 15.82 | 24.19 |

World 4 is the 2026-09-14 row-aware-lowering run; world 2 is the 2026-09-10
sweep. These come from the service rather than `glm_gen_check`, which is a
GLM-only evidence app and refuses this checkpoint. §9.2 says what that changes.

### GLM-4.7-NVFP4, world 4, MTP depth 1, through the service (2026-09-14)

| class | ms/token at c=1 |
|---|---|
| prose | 31.85 |
| code | 31.45 |
| json | 30.03 |
| math | 30.67 |
| chat | 33.90 |

### The full GLM-5.3 (int4/int8), world 4, MTP depth 1, through the service (2026-09-14)

The current row-aware-lowering run reports:

| class | ms/pass | ms/token at c=1 | c=1 tok/s |
|---|---:|---:|---:|
| prose | 61 | 35.46 | 28.2 |
| code | 62 | 34.25 | 29.2 |
| json | 62 | 34.25 | 29.2 |
| math | 61 | 34.84 | 28.7 |
| chat | 66 | 39.37 | 25.4 |

At T=1 every class remains about 51 ms/step. Depth 1 remains the shipped
setting; the older depth-2/3 experiments are retained in the dated full-GLM
record rather than presented as current-path numbers here.

## 5. Decode under concurrency

Unless a subsection says otherwise, each table here is **aggregate tokens/s per
prompt class**, measured 2026-09-10 on one binary with
`serve_load.py --classes all` (§9.3): for each class and each concurrency, that
many streaming requests at once drawn from that class alone.
Prompt 0 of each class is the same text the single-stream corpus uses, so the
c=1 column is comparable to §4; prompts 1 to 3 exist because a phase at
concurrency c needs c distinct requests.

Read down a column for how a class costs, and across a row for how the world
scales. Aggregate tokens/s is the phase total, not per request.

### GLM-5.3-Flash-FP8, world 4

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 45.9 | 55.8 | 62.8 | 21.85 |
| code | 47.7 | 57.2 | 65.5 | 21.01 |
| json | 48.7 | 55.6 | 67.5 | 20.62 |
| math | 45.5 | 59.5 | 67.3 | 22.02 |
| chat | 42.2 | 55.7 | 62.2 | 23.79 |

Sampled at the card's defaults the same sweep reads prose 45.6 / 58.0 / 62.4,
code 46.0 / 58.3 / 65.1, json 47.8 / 54.9 / 66.6, math 42.5 / 59.0 / 66.7 and
chat 39.9 / 55.1 / 60.4: within the drift of the greedy run.

The engine's own batched decode under MTP, from the 2026-09-07 batch families.
The last column is what the same load cost before those families existed, when
the engine ran one scalar replay per live request in sequence:

| live requests | batch family | ms per pass | scalar replays in sequence |
|---|---|---|---|
| 1 | scalar | ~41 | ~41 |
| 2 | 4-row | 60 | 83 |
| 3 | 6-row | 90 | ~124 |
| 4 | 8-row | 107 | not recorded |

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 4 (2026-09-19)

Same workload as §4, with distinct prompts per loaded phase. Means of two
repetitions, 256 output tokens per request, engine rates (admission and
prefill excluded), the template as shipped. The control column is the same
binary with `--bf16-weights checkpoint` and `DGPP_PREFILL_OVERLAP=off`.

| class | C1 | C2 | C3 | C4 | control C1 / C2 / C3 / C4 |
|---|---:|---:|---:|---:|---|
| prose | 59.12 | 80.00 | 97.27 | 114.25 | 54.8 / 75.2 / 91.0 / 105.2 |
| code | 61.44 | 84.86 | 98.49 | 109.72 | 57.1 / 80.3 / 91.1 / 102.6 |
| json | 62.02 | 80.47 | 106.11 | 119.84 | 57.6 / 76.0 / 98.0 / 112.3 |
| math | 59.86 | 87.21 | 98.31 | 110.20 | 55.6 / 82.7 / 93.7 / 104.6 |
| chat | 51.90 | 77.51 | 92.99 | 104.81 | 50.4 / 75.5 / 88.9 / 100.4 |

Geometric means: +6.8 / +5.2 / +6.5 / +6.4 % at one to four live requests.
Two live requests run four-row steps on the narrow 12-bit kernel, three and
four run six- and eight-row steps on its windowed form (which replaced a
cuBLASLt call over the BF16 bytes); the decode collective now claims every
ready peer per scan round. A seven-minute mixed soak on this build served
562 requests with none failed and identical operation streams on all ranks.

### GLM-5.3-Flash NVFP4/FP8 hybrid, world 2 (2026-09-12)

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 29.5 | 39.7 | 39.5 | 34.01 |
| code | 32.6 | 42.1 | 41.9 | 30.73 |
| json | 33.0 | 35.5 | 44.9 | 30.43 |
| math | 29.5 | 36.8 | 47.3 | 34.06 |
| chat | 22.7 | 40.5 | 38.9 | 44.25 |

Half the ranks hold twice the weights each, and the sweep reads 53 to 57 % of
the four-node aggregate in every class and at every concurrency. The four-node
world also keeps scaling to c=4 in each class, while two nodes flatten between
c=2 and c=4 on prose, code and chat: the per-rank weight read is already the
step's floor, so a wider batch buys less. Server-side, from the retire lines,
the pass is 60–75 ms at c=1, 82–104 at c=2 and 151–179 at c=4, with acceptance
69–98 %.

**Sampling does not cost throughput on the current proposal rule.** The sampled
and greedy sweeps agree within run-to-run drift on both GLM-5.3 checkpoints and
on Qwen.

### Qwen3.8-Flash-Next-FP8, world 4 (2026-09-14)

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 67.2 | 101.4 | 142.1 | 14.88 |
| code | 75.3 | 114.0 | 158.8 | 13.28 |
| json | 77.7 | 120.7 | 167.3 | 12.87 |
| math | 74.2 | 117.4 | 159.9 | 13.48 |
| chat | 63.2 | 103.1 | 144.1 | 15.82 |

The row-aware dense lowering leaves C1/C2 at parity and raises C4 by 12–15%.
The current eight-row step is 34.2 ms. The September 15 QSA change affects
prefill only, so these remain the current decode values.

This is the fastest world in the project at every concurrency.

### Qwen3.8-Flash-Next-FP8, world 2

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 44.4 | 62.4 | 71.0 | 22.62 |
| code | 48.2 | 72.3 | 80.9 | 20.82 |
| json | 49.6 | 74.4 | 83.2 | 20.23 |
| math | 46.9 | 73.4 | 80.4 | 21.37 |
| chat | 41.7 | 63.8 | 69.3 | 24.19 |

Sampled: prose 43.1 / 59.1 / 75.2, code 47.9 / 65.6 / 81.7, json 49.2 / 68.9 /
86.0, math 48.1 / 68.0 / 82.5, chat 42.3 / 60.4 / 75.2. The pass is 48–58 ms at
c=1 and about 89 at c=4. This world previously had no concurrency-1 or -2 point
at all.

### Qwen3.8-Flash-Next-NVFP4, world 2, FP8 dense, mmap n-gram table (2026-09-16)

Median request-wall tokens/s from three repeated greedy runs. The repeated
prompts become prefix-cache hits, and every result below includes request wall
time. The separate output-span C4 range is 120.1–138.9 tok/s.

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---:|---:|---:|---:|
| prose | 67.9 | 92.9 | 120.3 | 13.6 |
| code | 71.6 | 109.0 | 130.2 | 14.7 |
| json | 73.3 | 110.7 | 136.9 | 13.4 |
| math | 70.9 | 107.8 | 132.7 | 13.6 |
| chat | 61.5 | 96.6 | 119.0 | 16.1 |

The current mmap placement is within 0.8–1.5% of the resident baseline at C1,
0.4–1.6% at C2 and 0.2–3.6% at C4. All five C1 transcripts match byte for
byte. Mmap saves 23.84 GiB per rank and is the deployment default.

### Qwen3.8-Flash-Next-NVFP4, world 1, FP8 dense

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 44.7 | 61.9 | 70.9 | 22.45 |
| code | 48.3 | 69.6 | 80.1 | 20.76 |
| json | 50.3 | 71.2 | 83.7 | 19.94 |
| math | 48.0 | 69.3 | 79.3 | 20.90 |
| chat | 42.6 | 63.0 | 69.4 | 23.66 |

**One Spark matches two.** Set this table beside the world-2 one above: every
class at every concurrency lands within a few percent. A single Spark serving
the NVFP4 checkpoint with its dense stack encoded to FP8 delivers what two
Sparks deliver on the FP8 checkpoint, at a quarter of the eval difference (§7).
The earlier BF16-dense reading on this world was 119–124 ms per step and 49–55
tokens/s at four live requests.

### GLM-4.7-NVFP4, world 4 (2026-09-14)

| class | c=1 | c=2 | c=4 | c=1 greedy, ms/token |
|---|---|---|---|---|
| prose | 31.4 | 45.2 | 62.7 | 31.85 |
| code | 31.8 | 47.2 | 65.6 | 31.45 |
| json | 33.3 | 48.0 | 68.7 | 30.03 |
| math | 32.6 | 46.6 | 65.3 | 30.67 |
| chat | 29.5 | 43.9 | 62.7 | 33.90 |

The row-aware dense lowering removes the second read of the 6.3 GB/rank BF16
attention projections at eight rows. C1/C2 remain at parity; C4 rises by
22–29%, and the current eight-row step is 109.4 ms. Depth 1 remains the shipped
setting; the depth-2 single-stream tradeoff is recorded in §4.

### The full GLM-5.3 (int4/int8), world 4 (2026-09-14)

`serve_load.py --think` (the template has no thinking switch), greedy, 320
tokens, MTP depth 1; aggregate tokens/s:

| class | c=1 | c=2 | c=4 |
|---|---|---|---|
| prose | 28.2 | 34.6 | 43.0 |
| code | 29.2 | 37.1 | 46.2 |
| json | 29.2 | 35.8 | 47.0 |
| math | 28.7 | 38.1 | 45.3 |
| chat | 25.4 | 35.2 | 42.3 |

The current eight-row step is 159.0 ms. C1/C2 remain within run noise of the
previous lowering; C4 gains 1–9%. The packed-prefill change on September 15
starts at 128 rows and leaves these decode kernels unchanged.

### DeepSeek-V4.1-Flash MXFP4/FP8, world 4 (2026-09-16)

The common five-class service corpus, three repetitions at a 256-token cap,
temperature zero, thinking disabled and six configured slots. Values are
medians in tokens/s; wall rates include admission and prefill, while engine
rates use the scheduler's decode time and exclude the first token.

| class | C1 wall | C6 wall | C1 engine decode | C6 engine decode |
|---|---:|---:|---:|---:|
| prose | 38.88 | 87.55 | 39.84 | 90.89 |
| code | 58.70 | 105.07 | 61.57 | 109.34 |
| json | 71.01 | 120.82 | 74.96 | 126.65 |
| math | 59.35 | 105.45 | 62.47 | 110.11 |
| chat | 43.98 | 86.83 | 45.30 | 90.27 |

The retained changes size the graph collective consumer by payload and use
4,096-token prefill chunks. Against the original binary, the five-class
geometric mean improved 1.3% for C6 engine decode and 1.6% for C6 client wall
throughput; C1 was effectively flat. C6 chat's median was lower, so this is a
workload-dependent gain. The [matched study](../benchmarks/results/2026-09-16-dsv41-perf/README.md)
contains every baseline, sample, correctness gate and rejected experiment.
Weight and KV precision are unchanged. Do not compare this corpus with the
historical external-recipe table below.

### DeepSeek-V4.1-Flash MXFP4/FP8, world 4 (2026-09-14)

The six-slot deployment at that recorded revision was measured with the vLLM DGX Spark recipe's
prompt set and client, temperature zero and thinking disabled. These are
aggregate decode rates after the first token:

| live requests | aggregate tok/s | per-stream tok/s | mean TTFT |
|---:|---:|---:|---:|
| 1 | 49.64 | 54.8 | 0.269 s |
| 6 | 108.49 | 20.41 | 0.727 s |

This corpus differs from the five-class matrices above, so compare only within
this table. Every four-rank operation stream matched at shutdown. The current
stream-ordered fold is the default; `DGPP_DSV41_EAGER_FOLD=1` selects the
superseded host-driven diagnostic path.

## 6. Prefill and time to first token

The headline figures use `serve_prefill_probe.py`: a fresh prompt with a
unique nonce goes through the HTTP service, cannot attach to the prefix cache,
and is timed from the engine's `prefill_ms` counter. This is the caller-facing
path. Warm direct-engine ritual results remain in the dated records for kernel
analysis and are not mixed into this table.

Median of three cold prompts. Prompt targets are approximate; the linked
records contain each actual tokenizer count.

| model and current configuration | ~2K | ~8K | ~32K | date |
|---|---:|---:|---:|---|
| GLM-5.3-Flash-FP8, world 4 | — | — | — | rerun pending |
| GLM-5.3-Flash NVFP4 hybrid, world 4 | **1.297 s** | **5.249 s** | **24.523 s** | 2026-09-19 |
| GLM-5.3 NVFP4 hybrid, world 2 | — | — | — | rerun pending |
| GLM-4.7-NVFP4, world 4 | 2.809 s | 16.865 s | — | 2026-09-10 |
| Qwen3.8-Flash-Next-FP8, world 2 | **1.299 s** | **5.108 s** | **21.168 s** | 2026-09-15 |
| Qwen3.8-Flash-Next-NVFP4, world 2, mmap n-gram | **1.241 s** | **4.870 s** | **20.286 s** | 2026-09-16 |
| full GLM-5.3 int4/int8, world 4 | — | — | — | rerun pending |
| DeepSeek-V4.1-Flash MXFP4/FP8, world 4 | **1.599 s** | **5.726 s** | — | 2026-09-16 |

Flash hybrid prefill uses context-aware DSA query tiles inside the existing
scratch allocation; since 2026-09-19 the draft block's prefill rows stop at
their cache state (−6.0 / −7.3 / −7.6 %) and the KDA layers' bulk folds fly
beside the next row block's compute (−5.2 / −4.7 / −3.9 %), both with
identical first tokens and long-context transcripts
([record](../benchmarks/results/2026-09-19-glm-flash-line-rate/README.md)).
Its actual prompt lengths are 1,944–1,989, 7,723–7,811 and 31,291–31,306
tokens. Median rates are 1,507, 1,486 and 1,277 prompt tok/s.
Matched cold-prefill times fell by 21.7%, 41.5% and 66.4%, respectively, with
no weight or KV precision change. Decode rates remain essentially flat.
The [campaign record](../benchmarks/results/2026-09-16-glm-flash-perf/README.md)
contains the control runs, profiles and output checks.

DeepSeek's 4,096-token chunk reduced matched ~8,400-token prefill time by 3.7%
(5.945 to 5.726 seconds), reserving 1.71 GiB more scratch per rank. The
[study](../benchmarks/results/2026-09-16-dsv41-perf/README.md) includes the
repeated chunk sweep and exact long-prompt output comparisons.

Qwen's current QSA kernel runs at 0.637, 0.634 and 0.656 ms per actual prompt
token at the three sizes. It starts at 128 prefill rows; decode, speculative
verification and short prefill keep the existing path. Across the matched
campaign it reduced cold service prefill by 7.75%, 13.73% and 14.43%, while
all paired outputs and usage remained identical. See the
[QSA record](../benchmarks/results/2026-09-15-qwen-qsa-prefill.md) for the A/B
measurements and profiles.

Full GLM's packed tensor-core path starts at 128 rows, and short C1 prompts
retain GEMV. Its previous prefill measurements predate context-aware DSA
query tiling, as do the Flash FP8 and two-node hybrid measurements. Those
configurations await a fresh service sweep. The
[packed-prefill record](../benchmarks/results/2026-09-15-glm-packed-prefill.md)
and [Flash investigation](../benchmarks/results/2026-09-16-glm-flash-perf/README.md)
retain the earlier measurements and numerical checks.

DeepSeek-V4.1-Flash's current six-slot path measured **1,383 prompt tok/s**
and 2.133 s TTFT for the benchmark's 2,950-token cold prompt. That result uses
tensor-core MXFP4 prefill, grouped admission and the stream-ordered eager fold.
The different prompt set prevents inserting it into the target-length table.

The Qwen continuation budget remains opt-in. With a 256-token busy budget its
current QSA path measures 2.259 / 9.004 / 36.983 s at 2K / 8K / 32K. Adding a
2,048-token idle budget recovers monolithic throughput while no request is
decoding; the [idle-budget record](../benchmarks/results/2026-09-15-qwen-idle-prefill.md)
documents the latency/throughput tradeoff.

**Time to first token**, GLM-5.3-Flash-FP8 through the service, is prefill plus
the pick plus the HTTP hop: a 25-token prompt 1,109 ms cold and 751 ms hot, a
1,592-token prompt 2,643 ms cold and 949 ms hot. Over a mixed conversation
workload the split was 179 ms on prefix-cache hits against 1,141 ms on misses.
On the single Spark, TTFT is 506 ms at T=1 and 532 ms under MTP for a short
prompt.

## 7. Quality, so the throughput numbers are comparable

A faster configuration is only interesting if it answers as well. Every world
above is checked with the same task-level runner through the same endpoint,
greedy (§9.5). Item counts differ between campaigns and are given, since a
score is not comparable across different denominators.

| model | world | HumanEval | GSM8K | schema extraction |
|---|---|---|---|---|
| GLM-5.3-Flash-FP8 | 4 | 155/164 (94.5 %) | 293/300 (97.7 %) | 100/100 |
| GLM-5.3 NVFP4 hybrid | 4 | 157/164 (95.7 %) | 293/300 (97.7 %) | 100/100 |
| GLM-5.3 NVFP4 hybrid | 2 | 158/164 (96.3 %) | 291/300 (97.0 %) | 100/100 |
| Qwen3.8-Flash-Next-FP8 | 4 | 39/40 | 59/60 | 30/30 |
| Qwen3.8-Flash-Next-FP8 | 2 | 38/40 | 59/60 | 30/30 |
| Qwen NVFP4, BF16 dense | 1 | 39/40 | 59/60 | 30/30 |
| Qwen NVFP4, FP8 dense | 1 | 38/40 | 59/60 | 30/30 |
| Qwen NVFP4, FP8 dense, mmap n-gram | 2 | 39/40 | 59/60 | 30/30 |
| GLM-4.7-NVFP4 | 4 | 39/40 | 60/60 | 30/30 |
| GLM-5.3 full int4/int8 | 4 | 40/40 | 59/60 | 30/30 |
| DeepSeek-V4.1-Flash MXFP4/FP8 | 4 | 40/40 | 60/60 | 30/30 |

The two-node row is the same checkpoint on the same items, with the latent
cache in FP8 rather than BF16, and it lands inside the run-to-run movement of
the four-node row: one more HumanEval problem, two fewer GSM8K, extraction
exact. Halving the world and quantizing the latent cache cost nothing
measurable at task level (2026-09-12).

The two GLM-5.3 checkpoints were run on identical items: 155 HumanEval problems
pass on both, 7 fail on both, and 2 pass only on the hybrid. The single-point
HumanEval differences in the smaller campaigns sit on a near tie that has moved
both ways between runs of the same build, so they are not a ranking.

Two further identities are checked on every world and are pass/fail rather than
scores: **MTP produces the plain greedy transcript token for token**, and
**every rank's op stream is identical**, verified by four matching md5 sums at
shutdown. Both hold on every configuration on this page.

## 8. What is not measured yet

The current snapshot still has these gaps:

1. **Per-class concurrency on the common corpus is incomplete.** DeepSeek's
   current C1/C6 result uses the recipe corpus, and the single-Spark Qwen BF16
   dense stack has only its older four-live point. Fill the common C1/C2/C4
   matrix with
   `scripts/serve_load.py HOST PORT --concurrency 1,2,4 --classes all`.
2. **Long-context service prefill remains incomplete.** Qwen FP8 world 2,
   Qwen NVFP4 world 2 and the four-node Flash hybrid have current 32K points.
   Flash FP8, the two-node Flash hybrid and full GLM need remeasurement after
   context-aware DSA query tiling. Qwen world 4, Qwen NVFP4 world 1 and
   GLM-4.7 still lack current 32K service measurements.
3. **Current per-class concurrency above four is incomplete.** DeepSeek has a
   current C6 result on the recipe corpus, Qwen's opt-in eight-slot path has a
   matched C6/C8 campaign, and full GLM ships eight slots. A common current
   five-class C1/C2/C4/C6/C8 sweep has not been run across them.
4. **The two-node GLM-5.3 world has greedy numbers only, and no endurance
   run.** Its §5 sweep was not repeated at temperature 1, and neither
   `serve_soak_run.sh` nor `serve_failure_drill.sh` has been run against it;
   both now take their rank count from the deployment, so either will run
   against the two-node config unchanged.
5. **Full GLM-5.3 long-context cost and operational coverage.** Packed
   int4/int8 tensor-core prefill is implemented from 128 rows; see the
   [matched campaign](../benchmarks/results/2026-09-15-glm-packed-prefill.md).
   Profile the remaining cost on the new path before selecting another
   kernel or collective change. The short-prompt cutoff protects measured
   C1/MTP behavior; lowering it requires service and numerical evidence.
   This campaign does not add a soak, failure drill or sampled sweep.

The current tables include two-Spark Qwen and four-Spark Flash hybrid service
prefill through 32K, and DeepSeek at six live requests. The
dated records provide the exact workloads and reproduction commands.

## 9. Reproducing all of it

Everything below runs against a booted world. Bring one up with the launcher
and the config for the deployment you are measuring:

```bash
cp deploy/cluster_qwen-3.8-flash-next_fp8_w4.example.json deploy/cluster_qwen-3.8-flash-next_fp8_w4.json   # shared nodes and SSH login come from .env
scripts/dgpp-cluster up --config deploy/cluster_qwen-3.8-flash-next_fp8_w4.json
```

Run one procedure at a time on the fabric. Two measurements at once share the
memory fabric and both readings are then wrong.

### 9.1 Single-stream decode

Through the service, which is what a user feels:

```bash
scripts/serve_bench.py HOST PORT 300 label
```

It reports time to first token, the decode pace between the first and last
content chunk, and the server's usage line. The engine's own view is rank 0's
stats line every 10 seconds, which carries `ms/step`, `tok/step/req` and the
measured acceptance per draft position. For the mode sweep without the HTTP
path, `glm_gen_check --decode-graph [--mtp]` runs the same recipe on the
fabric directly.

### 9.2 Per prompt class

```bash
scripts/fabric_mtp_classes.sh --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json OUT_DIR chat code prose json math
```

**That script holds the corpus.** Five fixed prompts, one per class, unchanged
since 2026-09-05: chat asks why a CUDA graph replay beats eager launching, code
asks for a Python SSE parser with pytest tests, prose asks for a long history of
the Roman Republic, json asks for twenty-five records, math asks for a worked
train problem. Every per-class number this project has published comes from
those five texts, so they are the corpus and not an example of one. It runs
greedy, 300 steps per class, on all four ranks, and compares transcripts across
ranks.

It uses `glm_gen_check`, which is a GLM-only evidence app: it refuses the Qwen
checkpoints, whose class numbers therefore come from the service instead (§9.3).
Where both methods have been run on one world they agree within 0.05–0.7
ms/token, so the two are interchangeable in practice; the page says which
produced each row.

`serve_load.py --classes` reuses the same five texts as prompt 0 of each class
and adds three more per class, because a phase at concurrency c needs c distinct
requests and repeating one would share a decode path and attach to the prefix
cache. A `check_corpus()` call re-reads the shell script on every run and aborts
if prompt 0 has drifted from it, so the concurrency-1 column stays comparable to
the single-stream tables.

Through the endpoint instead, and for comparing transcripts across MTP depths:

```bash
scripts/mtp_depth_check.py --out /tmp/depth1 --max-tokens 300
diff -r /tmp/depth1 /tmp/depth2      # must be byte-identical
```

### 9.3 Concurrency

```bash
scripts/serve_load.py HOST PORT --concurrency 1,2,4 --max-tokens 320
scripts/serve_load.py HOST PORT --concurrency 1,2,4 --classes all                  # the §5 matrices
scripts/serve_load.py HOST PORT --concurrency 1,2,4 --classes all --temperature 1  # sampled
```

For each concurrency it opens that many streaming requests at once, greedy and
with thinking off unless `--temperature` says otherwise, and reports per request
the token count, time to first token and client-side pace, and per phase the
wall time, the aggregate decode tokens/s and the phase's stamps so rank 0's
stats lines can be laid beside it. Without `--classes` the prompts are a mixed
set of long-answer technical essays; with it, each phase draws from one class
(§9.2). Add `--isolation 4` to assert that a prompt's greedy answer alone is
byte-identical to its answer beside three others.
`scripts/fabric_glm4_load.sh CONFIG OUT` wraps the mixed form for the GLM-4.7
worlds.

### 9.4 Prefill

Through the endpoint, with the prefix cache defeated by a unique nonce per
prompt and the cost read from `/v1/metrics` deltas:

```bash
scripts/serve_prefill_probe.py HOST PORT 512 2048 8192 --repeat 3
```

On the fabric directly, in steady state, where the timed prefill is the third
repeat and the four ranks' generated ids are compared:

```bash
scripts/fabric_prefill_repeat.sh --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json OUT_DIR 512 2048 8192
```

### 9.5 Quality

```bash
scripts/serve_eval.py HOST PORT --out OUT_DIR --tasks gsm8k,extract
```

Greedy, the served model taken from `GET /v1/models`, HumanEval completions
executed against the problem's own tests in a subprocess. Pass `--limit` to
run a subset, and record the denominator with the score. The data lives in
`DGPP_DATA_DIR` (default `data/`); prepare it with
`python3 scripts/prepare_data.py download`. To include HumanEval, use
`--tasks humaneval,gsm8k,extract --allow-code-execution` only on an isolated
evaluation machine. Generated code runs without a security sandbox.
See [fixture preparation](testing.md).

### 9.6 The determinism checks that make a benchmark meaningful

A throughput number from a world that is not deterministic is not worth
recording. Every campaign above ran with these:

- rank agreement: `scripts/dgpp-cluster down` prints one op-stream md5 per
  rank and the four must be identical;
- speculative decode identity: the MTP transcript must equal the plain greedy
  transcript, checked by `mtp_depth_check.py` and `diff -r`;
- batching isolation: `serve_load.py --isolation C`.

### 9.7 Where the raw records live

`benchmarks/results/2026-09-16-glm-flash-perf/` contains the current four-node
Flash hybrid campaign: matched decode and cold prefill, long-prompt output
checks, profiles and the investigation record.

`build-ci/fabric-runs/glm53_serve_2026-09-12/` (MTP depth 1: transcripts,
pace, API check, the class and concurrency sweeps, the prefill probe, the
eval's per-item JSONL), `glm53_plain_2026-09-12/` (T=1) and
`glm53_large_2026-09-12/` (the fp8 latent cache) on the head node hold the
full GLM-5.3's first campaign; `benchmarks/results/2026-09-12-glm53-full.md`
is its dated record.
`build-ci/fabric-runs/glm53_w2_2026-09-12/` on the head node holds the two-node
GLM-5.3 campaign: the class sweep, the isolation check, both prefill methods,
the eval's per-item JSONL, the rituals' logs, and the context-ceiling readings.
`build-ci/fabric-runs/bench_2026-09-10/` on the head node holds tonight's run:
every phase's raw output, the server logs, and `RESULTS.md` with the tables as
they came off the fabric. `benchmarks/results/` holds the dated engineering
record behind the earlier numbers. `docs/signoff_v1.md` is the v1 sign-off with
its own method statement, `docs/measurements.md` the platform and GLM-4.7
readings, `docs/qwen38_single_spark.md` the single-Spark campaign, and
`docs/qwen38_optimization_plan.md` and `docs/nvfp4_plan.md` the optimization
rounds with their A/B pairs.
