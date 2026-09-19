# GLM-5.3-Flash: line-rate analysis and two exact optimizations, 2026-09-19

Four GB10 DGX Sparks, `HawkBearPig/GLM-5.3-Flash-NVFP4-FP8` (rev `17b8d4d8`),
TP=4, four slots, MTP depth 1, BF16 KV, 786,432-token pool, 8 GiB prefix
cache — the deployed recipe. No weight, KV or activation precision changed.
Every A/B below is one binary with an environment switch, on an otherwise
idle fabric, the same prompts in both arms.

Retained:

| change | single-stream effect | concurrency | outputs |
|---|---|---|---|
| **bf12**: lossless 12-bit resident form of the decode GEMV's bf16 weights (`engine.bf16_weights: "bf12"`; every template) | C1 decode **+6.9 to +7.5 %** on all five classes; step 34.2 → 32.0 ms | C4 −0.65 to +0.17 % (noise) | 15/15 C1 transcripts byte-identical |
| **state-only MTP prefill** (default; `DGPP_MTP_PREFILL_FULL=1` restores) | cold prefill **−6.0 / −7.3 / −7.6 %** at ~2K / ~8K / ~32K | n/a | identical first tokens, usage, long-context text, MTP passes and acceptance |

Rounds two and three (§5, §6) added the five-to-eight-row packed kernel, the
other families, the prefill fold overlap and the collective's multi-claim —
then the 12-bit form as the ONLY resident one (`"bf12"` saves memory: the two
ceiling-bound templates have their 160K and 120K contexts back) and the
interleaved gate (the decode collective 40.9 → 35.7 us). Round four (§7)
gave the format a tail for rows that are not multiples of 1024 columns and
carried it to Qwen3.8-Flash-Next-FP8: +6.4 % single stream on four Sparks,
+9.0 % on two.

## 1. Where the physical limits are

### Decode

One MTP step reads, per rank (shapes from the checkpoint headers, slice rules
from the loader; routed experts at the two-distinct-routes worst case):

| class | bytes | share |
|---|---:|---:|
| native bf16: KDA in/out projections 2.40 GB, lm head 2 × 0.317, indexer/router/mHC/eh_proj 0.47 | 3.5 GB | 52 % |
| NVFP4 routed experts, 16 slots × 42 layers (+ FP8 draft experts 0.10) | 2.48 GB | 36 % |
| FP8 projections, dense MLPs, shared experts | 0.82 GB | 12 % |
| **total** | **6.79 GB** (5.55 if both rows route alike) | |

The decode cores stream at the part's rate: `micro_gemv_bw` today 238–250
GB/s on cold bf16 matrices; in the profiled step the bf16 GEMVs read 275 GB/s
effective (the L2-prefetched prefix included), the fp4 slot kernels 235–268,
the fp8 GEMVs 233. At 250 GB/s the bytes alone are **27.2 ms of the 34.4 ms
step**: the step runs at 79 % of pure line rate, and the remainder is the
serial protocol — 94 collectives 4.2 ms (floor ≈ 2.1), mHC 1.6, small
latency-bound kernels ≈ 2.5, launch seam 0.5. The kernels are not the
opportunity; only fewer bytes, fewer or faster collectives, or more tokens per
step move the number.

### Prefill

Per 2,048-token chunk every one of the 288 × 42 routed experts is touched
(57 rows each): 1.02 GB of NVFP4 per layer per rank, 4.08 ms at 250 GB/s. The
ldmatrix kernels spend 4.06 ms per chunk-layer (2 × 1.21 + 1.65) — **at the
read-once floor**, with the tensor-core work overlapped under it. Listed
attention runs at the tensor cores' bf16 rate for its 2,048-key lists. What is
not at a floor: the bulk all-reduces (15 % of GPU time, ≈ 45 % of the two-lane
wire rate, nothing overlapped with them), the sequential KDA recurrence (8 %),
and — until today — the draft block's prefill rows (6–7 %, see §3).

## 2. bf12: the bf16 weights' exponents are compressible, losslessly

52 % of the step's bytes are native bf16, and "no further quantization" rules
out the obvious lever. But a bf16 weight's exponent carries ≈ 2.6 bits of
entropy (measured on q/k/v/o and the lm head: 99.98 % of a tensor's weights
sit in fifteen consecutive exponents). `src/kernels/bf12_gemv.{hpp,cu}` stores
the sign+mantissa byte and a 4-bit exponent code against a **per-row** window
— 12 bits a weight — and rebuilds the exact bf16 bits in registers:

- code 15 escapes to a per-row, column-sorted side table (1.5e-4 of weights);
- a row past 64 escapes (outlier channels: 35 q/k rows in the whole model)
  stays bf16 in a side array and its warp runs `bf16_gemv::row_dots` itself —
  a warp is one weight row, so the branch is uniform across lanes;
- lane/step ownership and FMA order are the production kernel's, so every
  output is **bitwise** `launch_bf16_gemv`'s (`bf12_gemv_test`: 1–4 rows, both
  outputs, strided activations, zeros, subnormals, Inf/NaN, the escape bound,
  raw rows, mixed-scale fused rows).

The companions are built after the resident load (70 matrices, 2.55 → 1.92
GiB per rank, 1.3 s), registered with `CublasLtGemm`, and taken by decode
launches of up to four rows; the L2 prefetch windows follow them through
`IGemm::resident_view`. Wider batches and prefill keep the bf16 bytes, which
is why both forms stay resident (+1.92 GiB/rank, in the memory plan) and why
the switch is opt-in.

Microbench (`bf12_bench.cu`, real weights, cold copies, bitwise-checked):

| matrix | bf16 GEMV | bf12 | speedup |
|---|---:|---:|---:|
| KDA q slice 2048×4096, 2 rows | 70–74 µs | 56–58 µs | 1.22–1.28× |
| KDA o slice 4096×2048, 2 rows | 70–73 µs | 54–56 µs | 1.28–1.33× |
| lm head slice 38720×4096, 2 rows | 1,278–1,295 µs | 942–970 µs | 1.33–1.36× |
| q slice, 1 / 3 / 4 rows | 72–75 µs | 56–59 µs | 1.2–1.33× |

Fabric, engine decode tok/s, medians of three (C4: 8-row steps, which keep
the cuBLASLt path on the bf16 bytes by design):

| class | C1 off | C1 on | Δ | C4 off | C4 on | Δ |
|---|---:|---:|---:|---:|---:|---:|
| prose | 54.85 | 58.96 | +7.49 % | 100.84 | 100.19 | −0.65 % |
| code | 57.15 | 61.11 | +6.93 % | 102.53 | 102.51 | −0.01 % |
| json | 57.60 | 61.70 | +7.12 % | 105.05 | 105.08 | +0.02 % |
| math | 55.66 | 59.54 | +6.96 % | 106.27 | 106.44 | +0.17 % |
| chat | 50.38 | 53.92 | +7.02 % | 100.31 | 99.69 | −0.62 % |

All 15 C1 transcripts are byte-identical between the arms; tok/pass and p1
are unchanged.

Two live requests run four-row steps, which do take the packed form
(one repeat per class, same binary): 75.8 / 81.2 / 76.9 / 81.9 / 74.4 →
79.4 / 84.0 / 79.8 / 86.5 / 79.6 tok/s, **+3.4 to +7.0 %**. So concurrency
is never worse: C2 gains, C3/C4 are unchanged.

It is independent of MTP: a plain T=1 step's bytes are 62 % bf16 (3.14 of
5.05 GB). With `--no-mtp`, prose and code: 37.0 → **40.2 tok/s (+8.5 %)**,
27.0 → 24.9 ms/step.

A call the smem bound splits into GEMV chunks (the draft block's `eh_proj`,
k = 8192, from three rows) takes the companion per chunk: the bf16 chunks
re-read the weights just the same.

## 3. The draft block's prefill rows only need their state

`mtp_run_rows` ran the whole draft block over every prompt row — DSA
selection and listed attention, the output projection, a fold, then the MoE
through the host-segmented GEMV path (`moe_->enqueue`, 5 % of the profiled
prefill on its own) and a second fold — and then returned before the head:
nothing reads those rows' output. What later drafts read is the block's DSA
cache (latent rows, index pools, tail ring), a function of the site's input
alone. `DsaLayer::enqueue_prefill(..., state_only)` now stops after those
writes and the block returns (no staged handout is taken).

| target (actual tokens) | full block | state-only | Δ | prompt tok/s |
|---|---:|---:|---:|---:|
| ~2K (1,944–1,989) | 1.458 s | **1.371 s** | −6.0 % | 1,424 |
| ~8K (7,723–7,811) | 5.975 s | **5.539 s** | −7.3 % | 1,410 |
| ~32K (31,291–31,306) | 27.624 s | **25.535 s** | −7.6 % | 1,226 |

Equivalence: nine cold probes agree on prompt hash, first token, usage and
finish; 128-token generations after 2,066 / 7,568 / 30,182-token prompts are
identical to the control and to the 2026-09-16 record, with the same passes
(71 / 70 / 66), tok/pass (1.79 / 1.81 / 1.92) and p1 (79 / 83 / 94 %).
`glm_tp_test` (MTP, prefix and resumable-prefill gates), `dsa_test`,
`glm_moe_test`, `glm_dsa_*` pass. Full suite with both changes: 109 of 110
(`raw/ctest-summary.txt`); the one failure, `script_reports_test`, is its
inline-Python lint matching an untracked `scripts/setup.sh` that is not part
of this work.

## 4. Tried, not retained

| experiment | result | decision |
|---|---|---|
| bulk pool 16 × 256 KiB (4 rounds per 16 MiB fold instead of 8) | 2K 1,371 → 1,453 ms, 8K 5,539 → 5,752 | regresses (incast); removed |
| bulk pool 8 × 512 KiB | 1,354 / 5,453 ms (−1.2 / −1.6 %) | inside session drift; removed |
| bulk pool 16 × 512 KiB | 1,456 / 5,724 ms | regresses; removed |
| bf12 window per tensor instead of per row | 21 of 34 fused `in_proj` refused (f_a/g_a/b rows leave a shared window) | replaced by the per-row window + raw rows |

The bulk fold is therefore not round-count-bound; its cost needs the
per-phase timeline the latency path already has before another attempt.

## 5. Round two (same day): ship it, widen it, carry it to the other families

Requested after the first round: make the 12-bit form a deployment setting
enabled in every template, extend it to the wider batches and the other
families, look at the collective transport, and overlap the prefill folds
with compute. Every A/B is again one binary, one switch, the same prompts;
the control arm is `--bf16-weights checkpoint` with `DGPP_PREFILL_OVERLAP=off`.

### 5.1 `engine.bf16_weights` and the five-to-eight-row kernel

The switch is `engine.bf16_weights: "checkpoint" | "bf12"` (parser, CLI flag
`--bf16-weights`, the settings record to the peers, the config digest, the
memory plan; `DGPP_BF12=on|off` overrides for A/B). `Bf12Companions` is the
shared owner every family uses.

Three- and four-request batches run six- and eight-row steps, which the BF16
sites handed to cuBLASLt (the bytes once, ~210 GB/s). The wide kernel stages
the activations one 1024-column window at a time (16 KB of shared memory at
eight rows, a barrier on both sides of every restage) and carries the lane
accumulators across the windows, so a row's chain is still the scalar one —
bitwise the GEMV chunks at any row count (`bf12_wide_bench.cu`):

| matrix, 8 rows | cuBLASLt (BF16) | bf12 wide |
|---|---:|---:|
| KDA q slice 2048×4096 | 80–84 µs | 57–60 µs |
| KDA o slice 4096×2048 | 87 µs | 57–60 µs |
| lm head slice 38720×4096 | 1,373–1,417 µs | 932–960 µs |

One trap, caught by the transcript gate: widening the GEMV lowering to eight
rows also re-routed five-to-eight-row PREFILL chunks (a prefix-cache cut
leaves them) from their Lt algorithm to the GEMV chain, and C1 transcripts
moved. `CublasLtGemm::set_bf12_wide` now opens the wide launches to decode
batches only; with that, 10/10 C1 and 3/3 long-context transcripts are
identical to the control.

GLM-5.3-Flash, engine decode tok/s, final build from the shipped template:

| live requests | prose | code | json | math | chat | geomean vs control |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 59.1 | 61.4 | 62.0 | 59.9 | 51.9 | +6.8 % |
| 2 | 80.0 | 84.9 | 80.5 | 87.2 | 77.5 | +5.2 % |
| 3 | 97.3 | 98.5 | 106.1 | 98.3 | 93.0 | +6.5 % |
| 4 | 114.3 | 109.7 | 119.8 | 110.2 | 104.8 | +6.4 % |

### 5.2 Other families

| family (world 4) | packed | C1 | C4 | transcripts | resident cost |
|---|---|---:|---:|---|---:|
| GLM-4.7 NVFP4 | 374 matrices: q/k/v/o of 93 attention blocks, head, eh_proj (6.36 → 4.78 GiB) | 32.1 → 35.8 tok/s (**+11–12 %**, 60.0 → 53.8 ms/pass) | +4.6–6.6 % | 6/6 identical | +4.9 GiB (fits) |
| full GLM-5.3 int4/int8 | 89 matrices: dense layers' and draft's projections, 22 indexers, head, eh_proj (1.65 → 1.24 GiB) | 28.5 → 30.2 tok/s (**+5.3–6.5 %**) | +3.2–6.6 % | 4/4 identical | +1.27 GiB: the template was at its ceiling, context 120K → 100K |
| GLM-5.3-Flash, two nodes | the four-node set at twice the slice (4.98 → 3.74 GiB) | not re-measured | — | — | +3.83 GiB: context 160K → 132K |
| Qwen3.8-Flash-Next | nothing yet: hidden 2560 needs a 512-column tail block, and its GDN/GR sites are fused launches (1.3–5 GB/step reachable) | — | — | — | — |
| DeepSeek-V4.1-Flash | nothing: its BF16 sites ride the tensor-core lowering (< 1 % of the step) | — | — | — | — |

### 5.3 Prefill: the folds beside the compute

`BoundaryReducer::begin_async` is the bulk machine's submit without its wait.
A chunk of ≥ 1024 rows runs each KDA attention site in two row blocks: block
A's fold flies beside block B's site, and the FFN fold's block B beside the
next KDA layer's block A (the FFN site keeps the whole chunk — its experts
are read once per chunk). Fabric bisect of exactness (long-context
transcripts against the control):

| arm | identical |
|---|---|
| every site in blocks, asynchronous folds | 0/3 |
| every site in blocks, synchronous folds | 0/3 (not a race: the numerics of the split) |
| **KDA sites only** (shipped) | **3/3** |
| DSA sites only | 0/3 |

At production shapes the KDA layer, the mHC site and update, the FP8
projections and (with `set_plan_rows`) the Lt projections are bitwise under
the row split (`kda_split_check.cu`, `mhc_split_check.cu`,
`fp8_split_check.cu`, `lt_split_check.cu`); a DSA site is not, like any
change of the chunk cuts. Shipped form: 37 % of the fold bytes hidden.

| target | control | shipped | Δ |
|---|---:|---:|---:|
| ~2K | 1.374 s | **1.297 s** | −5.6 % |
| ~8K | 5.526 s | **5.249 s** | −5.0 % |
| ~32K | 25.502 s | **24.523 s** | −3.8 % |

(the control already has §3's state-only draft rows; against the 2026-09-16
record the three sizes are −11.5 / −12.6 / −11.6 %).

### 5.4 Collectives: the "skew" was half our own serialization

Per collective on the current build (16 KiB payloads): 45 µs = copy 3.7 +
handshake 15–18 + skew 19.5–20.6 + fold 5.6; engine post 3.1. A new timeline
line records the gaps between a generation's consecutive claims: **0 % under
5 µs, 80–85 % at 5–10 µs, on every rank** — the peers' doorbells were
co-resident and the kernel gated them one scan round at a time (scan of
system loads + gate pass + ack ≈ 8–9 µs), two of the three rounds landing in
`skew`. One election per peer per round (graph kernel only): a quarter to a
third of the gaps fall to 3–5 µs, rank 0's collective 43.8 → 41.6 µs
(−0.2 ms/step), results bitwise, a seven-minute mixed soak clean. What is
left inside a round is the gate pass itself and doorbells that land during
it.

GPU-posted sends: `benchmarks/micro/uar_probe.cpp` shows the GB10 **does**
map an mlx5 doorbell (UAR) page for device access
(`cudaHostRegisterIoMemory | Mapped` succeeds, unlike device-memory
registration). Whether device stores reach the NIC in order is the next
probe; the prize is the engine's notice + post, ≈ 3 µs of each handshake
(≈ 0.3 ms/step) — smaller than the gate serialization above.

## 6. Round three (same day): the 12-bit form alone, and the interleaved gate

Round two's list put two items first; the request was both, "we certainly
want our kv cache back". Same method: microbench, unit gate, same-binary
fabric A/B with the transcripts as the exactness check. Raw data:
`raw/round3-*` (the C1–C4 load runs as `round3-load-summary.json`: engine
decode tok/s per class and concurrency).

### 6.1 bf12-only residency (`engine.bf16_weights: "bf12"`)

With both forms resident the format COST 0.75 of the packed matrices, and the
two templates sized to their nodes' memory had paid it in context (two-node
Flash 160K → 132K, the full GLM-5.3 120K → 100K). Three things had to hold
for the bf16 bytes to go:

* **Something must be able to free them.** The matrices sat inside each
  layer's single `cudaMalloc`. Now a packable matrix is a *side grant* of the
  layer's bump: its own range of the CUDA virtual-memory API
  (`loaders/releasable_range.*`; the GB10 supports it, 2 MiB granularity,
  and a release returns the whole mapped size at once). The staging mirror,
  the byte formula and the resident image keep the grant order, so an image
  written in one mode restores in the other and nothing is rebuilt. The
  GLM-5.3-Flash loader allocates its caches BEFORE its weights, so the release
  cannot wait for the end of the load: each layer is packed as it lands and
  its ranges released at once — no more than one layer's bf16 bytes ever sit
  beside their companions. The address stays reserved: it is still the key
  every call site holds, nothing can alias it, and a stray read faults cleanly.
* **Prefill must get its bf16 bits back, exactly.** `launch_bf12_expand`
  rebuilds them into a scratch at the device memcpy's rate (16.8 MB: 130 us
  against 122; the 317 MB head: 2.3 ms) and cuBLASLt then runs the algorithm
  it always ran. The head is far larger than any scratch worth holding, and
  every family runs it over all T rows of a chunk — so the question was
  whether an Lt call can run in WEIGHT-row blocks. With the whole call's
  algorithm pinned it is bitwise the unsplit call on every shape tried (the
  heads at 5–2048 rows, in_proj, eh_proj; with the block's own algorithm
  several small-m shapes differ). One trap: an algorithm picked for an
  aligned n refuses a block that is less aligned (700 rows of 3000 at five
  activation rows) — blocks are multiples of 64 rows.
* **Decode batches past eight rows** (the full GLM-5.3's sixteen-row shape)
  read the bf16 bytes through Lt. Under `"bf12"` they take eight-row packed
  launches instead — tolerance-equal, as those batches already are to the
  narrow ones — so a captured graph never touches the scratch.

The first fabric run put the cost of the expansion at ~17 ms per prefill
chunk on four-node Flash — and showed that a short prompt is THREE chunks
(the prefix cache's header cut, the body, the snapshot tail), each a
bandwidth-bound call in which whole-matrix expansion triples the DRAM
traffic. In 8 MiB weight-row blocks the scratch stays inside the 24 MB L2
and only the packed bytes cross DRAM. Expansion cost on a KDA in_proj (52.6
MB, cold, through the GEMM seam):

| activation rows | whole matrix | 8 MiB blocks |
|---|---|---|
| 8 | +345 us | +132 us |
| 64 | +380 us | +154 us |
| 239 | +358 us | +190 us |
| 512 | +430 us | +488 us |
| 1024 | +489 us | +385 us |

Calls of up to 256 rows take the blocks; wide chunks keep whole matrices in
two slots (the fold overlap's row blocks call a site's in and o twice and
expand each once).

Fabric, GLM-5.3-Flash on four nodes, same binary, against both forms resident:

| | both forms | 12-bit alone |
|---|---|---|
| transcripts (5 classes + 3 long) | — | identical |
| decode, 1 / 2 / 3 / 4 live (tok/s, 3 reps) | 58.75 / 82.30 / 99.23 / 112.89 | 58.75 / 82.93 / 99.61 / 113.14 |
| cold prefill 256 / 2K / 8K / 32K tokens (ms) | 446 / 1283 / 5287 / 24600 | 484 / 1328 / 5358 / 24889 |
| a 37–68-token chat prompt's prefill (ms) | 262–308 | 296–338 |
| memory plan per rank (GiB; BF16-only: 93.88) | 95.85 | 93.42 |

About 10 ms per prefill chunk: +1.2–1.4 % at 8K–32K, +3.5 % at 2K, +25–40 ms
on a short prompt. The other deployments (memory plan per rank, GiB):

| recipe | BF16 only | both forms | 12-bit alone | what it bought |
|---|---|---|---|---|
| GLM-5.3-Flash, two nodes, 160K | 107.96 | 111.79 | 107.01 | 160K again with 4.8 GiB of the node left at boot (both forms: 0.05 at 160K, shipped 132K) |
| full GLM-5.3, four nodes, 120K | 110.44 | 111.71 | 110.11 | 120K again (both forms are refused there: shipped 100K) |
| GLM-4.7, four nodes | 79.21 | 84.10 | 77.84 | — (room to spare: stays `"bf12+bf16"`) |
| GLM-5.3-Flash, four nodes | 93.88 | 95.85 | 93.42 | — (room to spare: stays `"bf12+bf16"`) |

Each validated against both forms resident on the same binary, transcripts
identical (5 + 3 on every recipe): two-node Flash decode level, prefill 745 /
1923 / 8223 → 818 / 1969 / 8356 ms at 256 / 2K / 8K (+20 ms a chunk: twice
the bytes a rank); the full GLM-5.3 level at one, four and eight live
requests (48.42 → 48.43 and 46.39 → 46.06 tok/s at four; 56.17 → 56.01 and
55.62 → 56.80 at eight, the sixteen-row batches' packed launches), prefill
4546 → 4589 ms at 2K and within 0.5 % at 8K and 30K; GLM-4.7 decode level,
prefill +22–32 ms short, +3.6 % at 2K, +1.7 % at 8K. Hence three values:
`"bf12"` where memory bounds the context, `"bf12+bf16"` where there is room.

Gates: `bf12_gemv_test` (the released instance answers every call bitwise
with the bf16 bytes SCRIBBLED on the device — whole, in blocks, reused,
short-call blocks, decode at every width), `glm_loader_test` (side grants:
staging layout, release returns memory, one image across modes),
`glm_tp_test` (a 1024-wide fixture built in all three modes: prefill, draft
and decode bitwise; the plan's ordering; the build inside its plan).

### 6.2 The interleaved gate

The plan was one interleaved pass for a round's co-claimed peers. Built
(`block_fold_payloads`), it moved nothing: 40.9 → 40.6 us. The claim gaps
showed why — the co-claims fell under 3 us as intended, but at two loads per
peer in flight the joint pass was two round trips, as long as the two passes
it replaced, and the first ack moved later by what the second gained. The
real finding was what a round IS: a chain of system-load round trips, each of
which 255 threads wait out at a barrier. Taking round trips out of the chain,
step by step (rank 0's collective on the four-node Flash step, 16 KiB rows):

| step | collective (us) |
|---|---|
| sequential gate (round two) | 40.9 = copy 3.6 + handshake 15.3 + skew 16.4 + fold 5.5 |
| joint pass, two loads per peer in flight | 40.6 |
| four per peer (a 16 KiB row is one round trip for every peer) | 39.1 |
| the rest of the door — {len, ctl}, {hash}: one cache line — in ONE round trip behind the `seq` acquire (was four or five dependent loads, half of them on thread 0) | 36.7 |
| the engine's poison every eighth round; the last claim ends the wait | 36.0 |
| open gates derived on every thread: no barrier before the first pass | 35.7 = copy 3.5 + handshake 14.5 + skew 13.7 + fold 3.9 (the final build, re-measured: 35.5) |

A round is now 5–7 us where it was 7–10, and 27–34 % of the gaps are under
3 us. −5.2 us × 94 collectives = −0.49 ms of a 32.5 ms step. Same-binary
load A/B (three repetitions, five classes): +0.89 % at one live request,
−0.36 / −0.14 / +0.29 % at two to four (two same-gate arms differ by up to
0.56 %: level). Those rows are 32–64 KiB — past the 80 KiB staging budget for
three peers (shared memory is 99 KB a block), so the canonical fold re-reads
the NIC-placed rows: at 64 KiB the rounds gain 3.9 us and the fold gives 4.3
back (12.3 → 16.6 us, re-reading rows whose loads were only just issued); the
collective is 73.6 against 73.2 us. Taking long rows peer by peer inside the
pass was worse (77.4 us: the first ack waits for all three rows) and was
reverted. Results are bitwise (the canonical fold): transcripts identical;
`bus_test`, `glm_tp_test` (45/45), a seven-minute mixed soak with both
features on (541 requests, 0 failed, op streams identical).
`DGPP_BUS_GATE=sequential` keeps the per-peer gates.

## 7. Round four (same day): format v2 and Qwen3.8-Flash-Next

Round three's list had "Qwen: a 512-column tail block plus bf12 twins of the
fused launches". The request: "we should implement bf12 for qwen as well".
Raw data: `raw/round4-*`.

**Where Qwen's bytes are.** Of the 3.83 GB a world-4 rank of the FP8
checkpoint reads per token, 3.1 GB (81 %) is BF16: GR sites 1,272 MB
(replicated), GDN projections 1,038, QSA 354, the head 318 (twice under
MTP), routers 126, shared experts 118. None of it fitted the format: hidden
is 2560 and the world-4 slices are 1536 wide (k % 1024 = 512), the GR up
projection is k = 320.

**Format v2 (the tail).** Padding 2560 to 3072 would stream a fifth more
bytes — most of the gain — so the columns that do not fill a super-block
follow the row in units cut by the 256-column steps they hold: two full steps
as [sm 512 B | exp 256 B] (a lane's exp vector shared with its pair), a single
or partial step as [sm 8 B/lane | exp 4 B/lane] (vectors shared by two and by
four lanes). Twelve bits a weight, every load an aligned 16-byte vector, rows
without a tail unchanged; k = 2560 costs a lane eight loads where the bf16
core issues ten. Bitwise on thirteen shapes — tail-only rows down to k = 8,
every unit alone and combined — through the narrow, wide and expansion
kernels and the new multi-problem launch (`bf12_gemv_test`).

**The real weights pack** (`qwen_escapes.py`): 2–7 escapes per 10,000 weights
in every class (GDN, QSA, GR down / up, shared experts, routers, the head), no
row kept raw anywhere, widest row 16.

**The microbench decided the scope** (`qwen_gemv_bench.cu`: real weights,
production launchers, cold = enough copies to stay out of the 24 MB L2, warm
= one copy re-read; every output bitwise):

| shape (world 4) | bf16 → packed, cold | warm |
|---|---|---|
| GDN qkv [2560 × 2560], 13.1 MB | 53.8 → 43.5 us (−19 %); two rows −15 % | +6 %; two rows +37 % |
| GDN out [2560 × 1536], 7.9 MB | 35.3 → 28.0 us (−21 %); two rows −18 % | +2 %; two rows +19 % |
| lm head [62080 × 2560], 318 MB | 1212 → 900 us (−26 %) | — |
| GR down [320 × 10240], 6.6 MB | 30.4 → 29.2 us (−4 %); two rows +10 % | +109 %; two rows +70 % |
| GR up [10240 × 320], 6.6 MB | 28.6 → 28.7 us | +76 %; two rows +56 % |

A thread runs its ops in order here: rebuilding a weight's bits is ~12 integer
ops where the bf16 core spends ~3, so the packed chain is ~2.3× the ops per
FMA — invisible while DRAM is the limit, the whole cost once the bytes sit in
L2. The projections and the head are read cold or half-cold and win. The GR
sites, routers and shared experts are read WARM behind the prefetch windows
(which is how `gr_norm_down` runs at 347 GB/s and `gr_act_up` at 630), and the
two GR shapes are latency-bound even cold (40 blocks of 20 KB rows; 1,280
blocks of 640-byte rows): packed they would cost ~2 ms a step. They stay
BF16 — a third of Qwen's bytes this format cannot help.

**Fabric** (same binary, `DGPP_BF12=off|both`, MTP greedy, engine decode tok/s;
transcripts identical, 5 + 3 on each world; prefill unchanged — both forms
resident):

| live requests | world 4 (4 Sparks) | world 2 (2 Sparks) |
|---|---|---|
| 1 | 73.3 → 78.0 (+6.4 %) | 47.1 → 51.4 (+9.0 %) |
| 2 | 118.8 → 123.3 (+3.8 %) | 73.3 → 78.2 (+6.7 %) |
| 3 | 141.0 → 144.3 (+2.4 %) | 85.4 → 91.0 (+6.6 %) |
| 4 | 162.2 → 164.3 (+1.3 %) | 96.2 → 100.6 (+4.6 %) |

248 matrices a rank (1.65 → 1.24 GiB at world 4, 3.20 → 2.41 at world 2).
Every Qwen recipe has memory to spare (the tightest, two-node FP8, plans
100.8 GiB with both forms), so the family keeps both forms resident under
either value of the key and its templates take `"bf12+bf16"`; under
`dense_weights: "fp8"` (the NVFP4 templates) the projections and the head are
already block-FP8 and nothing is packed.

**Found on the way.** `WeightPrefetcher::add` bridges holes of up to 2 MB
between a window's adds; a companion is a separate allocation, and under
`"bf12"` a hole can be a released — unmapped — BF16 range (2 MB ranges exist:
the full GLM-5.3's `wk` and `weights_proj`). No fault was ever seen (a layer's
released ranges are reserved back to back, so a small one sits between other
released ranges, not between two companions), but it was luck of layout:
companions now enter a window through `add_isolated`.

## 8. Opportunities still open, by expected value

Decode (GLM-5.3-Flash step now ≈ 31 ms; bytes ≈ 24 ms of it at line rate):

1. **The collective at two to four live requests** (32–64 KiB rows): the
   interleaved gate's gain is returned by the unstaged fold. Either fold in
   the gate's own pass when a round claims every peer (the rows are already
   in registers; the staging row holds the copy a failed gate would need), or
   stage two of three rows within the 99 KB of shared memory: ≈ 4–8 us per
   collective at those widths (0.4–0.8 ms of a 60–70 ms step).
2. **GPU-published staging fold** (the engine hashes 16 KiB per generation
   on the CPU before posting; the bulk path already publishes it from the
   kernel): ≈ 1.5 us/collective. And the 64-byte doorbell as an inline send
   (`max_inline_data = 0` came from the M0 smoke tool, never measured). After
   round three the handshake (14.5 us: the peers' spread, the sender's notice
   and post, the wire) is the collective's largest part.
3. **A packed row chain that is cheap from L2** (~12 integer ops a weight
   today against the bf16 core's ~3): what keeps Qwen's GR sites (1.27 GB a
   token, a third of its bytes), routers and shared experts on BF16, and
   what costs the packed projections +19–37 % when they are read warm at two
   rows. A layout whose fields need fewer shifts and masks to reach the f32
   bit pattern, or 64-bit field extraction if the part's 64-bit integer ops
   are cheap (unmeasured). Worth ≈ +5 % on Qwen if it reaches parity.
   Format v2's tail also makes GLM-5.3-Flash's indexer `wq_b` (k = 1536)
   packable (0.15 ms; not in its pack list yet).
4. **Adaptive depth-2 MTP at C1** (deferred by request) and a truncated
   draft-head vocabulary (excluded: acceptance).
5. **Batches past eight rows** (the full GLM's 16, GLM-4.7's/DeepSeek's 32):
   a bf12 variant of the tensor-core `mma_gemv` — one read of 0.75 of the
   bytes where `"bf12"` now reads them once per eight rows and the other
   modes read all of them; also what DeepSeek needs.

Prefill (GLM-5.3-Flash ≈ 0.66 ms/token):

6. **`"bf12"`'s expansion beside the compute**: a wide chunk's GEMMs are
   compute-bound and its expansions bandwidth-bound — expanding the next
   matrix on a side stream while the current GEMM runs would hide most of the
   ~10 ms a chunk (−1 to −3 % of prefill in that mode; a third slot and an
   event per call).
7. **A prefill that computes the head on its last row only** (every family
   runs the lm head over all T rows of every chunk — 79 M dot products at
   2K — for the sake of the prefill == forward bitwise gate): a numerics
   decision, ≈ 1 % of prefill and the head's expansion under `"bf12"`.
8. **Short prompts are three chunks** (header cut, body, snapshot tail: 262 ms
   for 37 tokens), each a full walk of the weights: a request-level cost
   larger than anything a kernel has left.
9. **Find what in a DSA site moves under a row split** (the bisect says the
   site, not the fold): fixing it takes the hidden fold share from 37 % to
   50 % and beyond (≈ −2 %).
10. **Bulk fold wire rate** (≈ 45 % of the two-lane ceiling; 16-slot rounds
    regress on incast): needs the per-phase bulk timeline first.
11. **Chunk-parallel KDA recurrence** (8 % of prefill, a numerics project) and
    skipping the q-side projections in the state-only draft rows (< 1 %).
12. **`graph_replay_arm` spins on `walk_pub` while holding `coll_mu`**
    (`graph_replay_finish` releases it first): latent, never fired; hoist it.

## Reproduction

```bash
cmake --build build-ci -j -- -k && build-ci/bf12_gemv_test
# decode A/B (arms differ only in --bf16-weights)
scripts/dgpp-cluster up --config deploy/cluster_glm-5.3-flash_nvfp4-fp8_w4.json --bin build-ci/dgpp-serve   # control: --knobs="--bf16-weights checkpoint"
python3 benchmarks/results/2026-09-16-dsv41-perf/timed_load.py HOST 18080 \
  --concurrency 1,4 --classes all --max-tokens 256 --repeat 3 --warm 1 --json-out on.json
# prefill A/B (control arm: DGPP_MTP_PREFILL_FULL=1)
DGPP_DATA_DIR=build-ci/eval_data python3 scripts/serve_prefill_probe.py HOST 18080 \
  2048 8192 32768 --repeat 3 --seed 916 --tag glm-flash-perf --json-out prefill.json
python3 benchmarks/results/2026-09-16-glm-flash-perf/long_transcripts.py HOST 18080 long.json
```

Round three's arms are environment switches on one binary (the launcher
forwards `DGPP_*` to every rank): `DGPP_BF12=both|on|off` picks the resident
form over the config's, `DGPP_BUS_GATE=sequential` the per-peer gates, and
`DGPP_BUS_TIMELINE=1` + `python3 scripts/bus_window_skew.py LOGDIR` gives the
collective's decomposition and claim gaps (read rank 0; use a fresh
`--log-dir` per arm). `bf12_expand_bench.cu`, `expand_block_bench.cu` and
`lt_nsplit_check.cu` are the expansion's microbenches and the weight-row
block check.

`st_index.py`, `exponent_stats.py` and `row_escapes.py` reproduce the
exponent statistics from the checkpoint; `bf12_bench.cu` is the standalone
microbench (nvcc flags of `build-ci/compile_commands.json`, linked against
`libdgpp_kernels.a`).
