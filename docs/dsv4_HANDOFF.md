# DeepSeek-V4-Flash on the dgpp engine — handoff (2026-09-20)

**Read this first, then `docs/deepseek_v4_flash_plan.md` (the evidence trail) and
`docs/dsv4_attention_spec.md` (the numerics spec, 13 gaps, §5).**

## TL;DR

The engine **boots, serves and generates** DeepSeek-V4-Flash on the real
2× Spark world (MTP depth 5 + the decode graph, ~10.7 tok/s) — the plumbing is
complete. The **output is still degenerate**, and the cause is now *localized*,
not guessed: **layer 0's stream 3 (the MoE's lane) is wrong while streams 0, 1
and 2 are bit-exact against the checkpoint's own reference**. Everything else
(embedding, RoPE, window ring, mHC, attention, compressor, head) is confirmed
correct. The next session's job is that one lane.

## What works / what does not

| | state |
|---|---|
| boot, resident image, 46/46 layers | ✅ `rank 0 serving: ok (16-18 s)` |
| MTP depth 5 + captured decode graph | ✅ (needs `engine.mtp_depth: 5`; see invariants) |
| the MoE (was a **stub** — the model had no FFN) | ✅ implemented, GPU-verified vs the CPU oracle |
| serving + generating | ✅ `deepseek-v4-flash`, 10.7 tok/s (client-measured, 512 tok) |
| **coherent output** | ❌ `' Dors &#otoys precursor…'` where the reference says `'The DGX Spark is NVIDIA's compact, personal AI supercomputer…'` |
| performance target (40-60 tok/s) | ❌ 10.7; the vLLM lane on the same checkpoint does **36.9** |

## The one open bug, with its evidence

`tools/dsv4_reference_layers.py` (the checkpoint's own math) vs the engine's
`DGPP_DSV4_DUMP_LAYERS` dump, prompt `The capital of France is` (tokens
`671,6102,294,8760,344`), state = the RAW 4-stream hidden (`layer_NN_hc.f32`):

```
layer 0: stream 0 cos 0.9999 (L2 1.304 vs 1.303)
         stream 1 cos 1.0000 (L2 1.484 vs 1.484)
         stream 2 cos 1.0000 (L2 1.484 vs 1.484)
         stream 3 cos 0.7195 (L2 7.953 vs 8.102)   <-- the MoE's lane
```

So the divergence enters in the MoE's contribution. Verified NOT the cause:
the kernel composition itself (the slot path matches the CPU oracle, 9.8e-05 /
1.5e-02), the routing (ids from `tid2eid` I64 ✓, weights = the unbiased
sqrtsoftplus scores at those ids, renormalized then ×1.5 ✓ — the reference's
exact order), the SwiGLU clamp (the kernels already do "gate upper-only, up both
sides"), the loader's expert order (`experts[e*3+0/1/2]` = w1 gate / w3 up / w2
down ✓), the tensor shapes/dtypes (`tid2eid` I64; the routed MXFP4 scales
`[rows, k/32]`; the shared expert's `F8_E8M0 [16,32]` = a 128×128 grid → `sh_rs =
sh_cs = 7` ✓), the shared expert's weight (1, unweighted in the reference too),
`wo_a` (we keep it fp8 + scales; the reference's own trap is a bf16-build issue),
and `ffn_norm`/`hc_ffn_*` bindings (no swap).

**Ranked next suspects (each is one small, checkable step):**
1. **The `MoeExpertView` fields the slot kernels actually consume** for MXFP4
   (`fp4_group`, `fp4_scales`, `fp4_global`, the `scale_shift_*` values) vs what
   `launch_moe_slot_gate_up_swiglu_fp4` / `launch_moe_slot_down_fp4` expect —
   dump the 771-entry table for layer 0 and compare against
   `MoeExpertView::of(GlmFp4Matrix)` on the loader's views; the MXFP4 scale grid
   is per-row × group-32 while the struct's `scale_shift_rows/cols` are the FP8
   convention (7/7) — check the fp4 core reads `fp4_group`, not those fields.
2. **The routing weights' path into `launch_moe_slot_accum`**: dump `topk_ids_`
   and `topk_w_` for layer 0 and compare against the reference's `indices` /
   `weights` for the same tokens (the reference's `Gate.forward` is 40 lines).
3. **The shared expert's payload/scale arguments** (`sh_gate_p/s`, `sh_up_p/s`,
   `sh_down_p/s`) as the kernels receive them — one oracle case with the
   checkpoint's real shapes.
4. Only after 1-3: the mHC's `post`/`comb` handling of the FFN site (but note
   streams 0-2 bit-exactness makes the site mechanism itself proven).

**The caveat that stops a false lead:** `moe_out_l0.f32` (the sub-step probe in
`model.cpp`'s MoE site) is THIS rank's **pre-all-reduce partial**, while the
reference run is world 1. Its raw diff (ours 14.5 vs ref 52.9, cos 0.09) is NOT
evidence of a 3.6× error. Compare post-collective `layer_NN_hc` (what the table
above does), or run the reference at world 2, before believing an MoE-output
diff.

## The rig (copy-paste)

```bash
# 0. one stack at a time. The Qwen lane is the working one.
cd ~/work/q-dgx-gateway && make down-dgpp-512k        # if the Qwen lane is up
#   (the vLLM REFERENCE lane for the same checkpoint: cd ~/work/v-dgx-gateway && make up-base / make down-base)

# 1. our engine, world 2, decode graph OFF (the dump needs it)
export DGPP_ENV_FILE=/home/kv/work/q-dgx-gateway/.env.dgpp
cd ~/work/dgpp
cmake --build build-dsv4-merge -j20 --target dgpp_serve_app      # the dev build; never build-release
python3 scripts/dgpp-cluster --bin $PWD/build-dsv4-merge/dgpp-serve \
    up --config /home/kv/work/q-dgx-gateway/config/dgpp-dsv4-w2-eager.json

# 2. dump our per-layer states (the env comes from the config's node_env;
#    config/dgpp-dsv4-dump.json carries it, or set it in .env.dgpp — see invariants)
rm -rf /tmp/dsv4dump && mkdir -p /tmp/dsv4dump
python3 scripts/dgpp-cluster --bin $PWD/build-dsv4-merge/dgpp-serve \
    up --config /home/kv/work/q-dgx-gateway/config/dgpp-dsv4-dump.json
curl -s http://127.0.0.1:8888/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","prompt":"The capital of France is","max_tokens":2,"temperature":0}' >/dev/null
ls /tmp/dsv4dump/req_*/rank_0/ | head        # layer_NN.f32, layer_NN_hc.f32, moe_out_l0.f32, logits_top.txt, meta.txt

# 3. the reference (the checkpoint's own math) — see docs/dsv4_reference_runbook.md
rm -rf /tmp/dsv4ref && mkdir -p /tmp/dsv4ref
docker run --rm --gpus all --entrypoint python3 \
  -v /data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local:/ckpt:ro \
  -v /tmp/dsv4ref:/out -v $PWD/tools/dsv4_reference_layers.py:/ref.py:ro \
  eugr/spark-vllm:latest /ref.py --ckpt /ckpt --out /out \
  --token-ids 671,6102,294,8760,344 --layers 0-3 --state-layout full
#   (layers 0-3 ~14 s, 4.8 GiB; --state-layout full writes layer_NN.f32 = the 4 streams)

# 4. compare (this is the bisection)
python3 - <<'PY'
import struct, glob, math
L=lambda p: struct.unpack('<%df'%(len(open(p,'rb').read())//4), open(p,'rb').read())
ours=L(glob.glob('/tmp/dsv4dump/req_*/rank_0/layer_00_hc.f32')[0]); ref=L('/tmp/dsv4ref/layer_00.f32')
H=4096
cos=lambda u,v: sum(x*y for x,y in zip(u,v))/((math.sqrt(sum(x*x for x in u))*math.sqrt(sum(x*x for x in v))) or 1)
for s in range(4): print(f"stream {s}: cos={cos(ours[s*H:(s+1)*H], ref[s*H:(s+1)*H]):.4f}")
PY

# 5. coherence (the metric to move: currently 0)
bash scripts/dsv4_coherence_check.sh

# 6. when it is fixed: the gates + the perf target
DSV4_SMOKE_MODE=fabric DGPP_BUILD_DIR=$PWD/build-dsv4-merge \
  DSV4_FABRIC_CONFIG=/home/kv/work/v-dgx-gateway/config/dgpp-dsv4-flash.json \
  bash scripts/dsv4_gates.sh smoke      # then: parity / sanitizer / tps / reference
```

## Invariants and gotchas (all learned the hard way)

- **One stack at a time** (130 GB unified): the Qwen 512K lane, the vLLM
  reference lane and our dsv4 world are mutually exclusive; `make down-dgpp-512k`
  / `make down-base` first.
- **world 1 is impossible** for dsv4: all weights on one rank = 155 GiB vs ~116
  free — the engine refuses the plan. The smoke gate's world-1 shape is therefore
  a designed SKIP, and the resident image is **world-2 only**.
- **The resident image** (`~/.cache/dgpp/resident/<key>.img`, 83 GB) is keyed by
  the per-rank TENSOR SET: world size, rank, head sharding, the checkpoint's
  config + shard headers. **Not** by kv_capacity / max_concurrency / mtp / its
  depth — verified for kv 262144/C2, kv 1048576/C6, mtp 1/5/off. A *stale* image
  is a real hazard in principle (a loader change with the same key would be
  misread) — deleting both nodes' images and rebooting (~3 min, shards → image)
  ruled that out once and is cheap to repeat.
- **`engine.mtp_depth` must be 5** (= the checkpoint's `dspark_block_size`,
  "the DSpark block's 5 drafts"). At depth 1 the decode batch is 8 rows while the
  draft needs `groups × block = 2 × 6 = 12` → `mtp_run_rows: the draft blocks
  exceed the decode batch`. `--mtp` also requires `decode_graph: true` (the
  engine says so itself).
- **The decode graph must be kernels-only**: host syncs and copies inside a
  capture invalidate it. Two real instances were fixed: `push_position`-style
  per-step position copies (the draft's) and the MoE's per-call view-table
  upload (`cudaEventSynchronize` + an H2D memcpy) — the table is now one
  per-layer device slot prepared before the capture.
- **The dump needs `decode_graph: false`** and its env key must pass BOTH
  allowlists: `scripts/site_env.py`'s `NODE_KEYS` and the engine's
  `src/serve/cluster_config.cpp` `kNodeKeys`. A pinned **release** binary will
  reject a site env carrying a key it does not know — that is why
  `DGPP_DSV4_DUMP_LAYERS` is **not** in `.env.dgpp` (the Qwen lane's release
  refused to boot with it there); set it per-run via the dsv4 config's
  `node_env` (`config/dgpp-dsv4-dump.json`).
- **compute-sanitizer**: `--tool memcheck` (OOB) AND `--tool initcheck`
  (uninitialized) are both clean now — re-run initcheck after any new kernel
  wiring; it found the draft's uninitialized read. Attach it to **rank 0 only**
  with `dgpp-cluster --head-wrap "…/compute-sanitizer --tool initcheck"`.
- **A missing smem opt-in silently disables a launch**: `dsa_prepare_kernel_smem`
  enumerates every `LatentFormat` instantiation per kernel SYMBOL; a forgotten
  format keeps the 48 KiB default and any larger dynamic request is rejected with
  `invalid argument`. The window ring's new format was missing → its attention
  launches failed. Check that list whenever a format is added.
- **The reference lane is on this box**: `v-dgx-gateway`'s `.env.base` /
  `.env.miaai` serve the SAME checkpoint (`make up-base`), `docs/DSV4-DGPP.md`
  holds the knob mapping, and `bench/reasoning_check.py` is the quality gate.
  The vLLM lane answers the smoke prompt correctly and does 36.9 tok/s — use it
  as the arbiter and the perf target.

## Model facts worth not re-deriving

- 46 layers: 43 backbone (2 SWA-only, 21 C4A ratio-4, 20 C128A ratio-128) +
  3 DSpark draft stages; hidden 4096; 64 index heads; head_dim 512 with 64 RoPE
  dims; 256 routed experts, top-6, + 1 shared; `moe_inter` 2048; `swiglu_limit`
  10 (gate upper-only); `routed_scaling_factor` 1.5; `norm_topk_prob` true;
  `num_hash_layers` 3 (layers 0-2 route by the I64 `tid2eid` table, weights =
  the unbiased sqrtsoftplus scores at those ids); hc_mult 4, sinkhorn 20 iters,
  hc_eps 1e-6; native position ceiling 1M (no YaRN knob).
- KV: the window ring is the reference's mixed record (NoPE 448 e4m3 + per-64
  e8m0, **RoPE 64 raw bf16**, 584 B rows); the C4A main cache is our `kFp4Block`;
  the C128A compressor is the reference's 128-entry **gated** pool.
- `~/work/dsv4-native` is a parallel C++ engine for the sibling Vision-Exp model
  and `docs/dsv4_vs_native_diff.md` is the three-way diff; where it contradicts
  the Python reference, the **Python reference wins** (it drops the indexer's
  relu and folds heads — ours matches the reference).

## Definition of done

1. `scripts/dsv4_coherence_check.sh` reproduces the reference's opening
   (`The DGX Spark is NVIDIA's compact…`) — today: 0 chars.
2. `dsv4_gates.sh` fabric smoke + parity + sanitizer pass; `tps` lands in the
   40-60 tok/s band (the reference is 36.9, so this is a real target, not a wish).
3. MTP acceptance is non-zero (`accept p1` is 0% while the draft is wrong — a
   correct FFN should lift it to ~70% at depth 5, which is also where the speed
   comes from).
4. Then: `docs/deepseek_v4_flash_plan.md` updated, the config pinned in
   `q-dgx-gateway`, and the lane left serving (dsv4 if correct, otherwise Qwen).

## Repo state at handoff

- `dsv4-flash` = `1bba8a7` (96+ commits ahead of `master`), all committed,
  `dsv4-flash`'s own suites green (`dsv4_moe_slot_test`, the six oracles,
  `unit_tests`).
- Unmerged salvage on agent branches (kept deliberately):
  `pi-subagents/mu81vkc2-a0d5-6bc9c3d6` = the WIP 430-line
  `tests/cuda/dsv4_gpu_parity_test.cu` + the oracle-refactor header — the
  kernel-vs-oracle gate that has never existed; finish it if a suspect needs it.
- The **Qwen 512K lane is up** (`q-dgx-gateway`, C8/MTP1) — the working lane.
- Nothing is pushed to any remote except the earlier `yarn-qwen-512k` PR #3
  rebase.
