# Morning GPU window — DeepSeek-V4-Flash (world 2) runbook

The night landed everything host-side on branch `dsv4-flash` (see
`docs/deepseek_v4_flash_plan.md` for gate status). This is the sequence for
the quiet GB10 window, in order. Each step lists its evidence (exit code /
expected output) so it can be run unattended or checked at a glance.

## 0. Preconditions
- The operator is up; the GB10 nodes are quiet.
- **One-stack-at-a-time.** The live stack (Lane D vision or Lane Q qwen on
  :8888) must be down before a fabric boot. `cd ~/work/v-dgx-gateway &&
  make down` (or the lane's own `make down`). Do NOT boot v4 while anything
  holds a GPU — `REQUIRE_IDLE_GPU` will refuse, and the 130 GiB unified
  cannot hold two full copies.
- Checkpoint is local: `/data/models/DeepSeek-V4-Flash-0731`. The loader
  walks the HF-cache layout, so either
  `ln -s /data/models/DeepSeek-V4-Flash-0731 /data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731`
  or export `HF_HOME=/data/models` for the boot.

## 1. Finish G4 (the last code gate)
W2b's kernel geometry layer (`src/models/dsv4/csa2_layer.{hpp,cu}` + the 4 CPU
oracle tests). If it has landed:
```
cmake --build --preset ci --target unit_tests -j
./build-ci/unit_tests            # expect the 4 new dsv4_*_oracle tests green, prior 204 still green
```
If W2b is still running or rejected, verify its diff, build, run the 4 CPU
oracle tests, and commit before any GPU step.

## 2. CUDA loader test (G3 GPU half)
```
cmake --build --preset ci --target dsv4_loader_test -j
./build-ci/dsv4_loader_test       # loads the real checkpoint into the w2 plan; expect exit 0
```
This is the first real-GPU touch of the loader. Watch for OOM (the w2 plan
is 78.89 GiB weights + 14 GiB reserve = 92.89 GiB < 130 GiB, so it fits with
~37 GiB to spare).

## 3. Numerics sanity on the new shapes
```
compute-sanitizer --tool memcheck ./build-ci/dsv4_loader_test   # 0 errors
```
Run the family's CUDA oracle tests at the v4 shapes (csa2/indexer/engram/
dspark) if they are GPU-targeted.

## 4. Reference parity (G5)
The checkpoint ships its own reference inference
(`/data/models/DeepSeek-V4-Flash-0731/inference/{model,kernel,generate}.py`).
Run the engine's decode against it on a fixed prompt set and diff the tokens.
This is the parity gate — the engine must match the reference before we trust
the kernels.

## 5. Fabric boot at world 2 (G7)
```
# one-stack-at-a-time: the live stack is down (step 0)
# boot the v4 world-2 cluster from the deploy template
#   deploy/cluster_deepseek-v4-flash_fp4_w2.example.json
# (the dgpp-serve / cluster launcher for the 2-Spark kit)
```
- Validate `kv_capacity` against measured headroom (the template ships
  1048576 = 3.43 GiB/rank; the plan leaves 37.11 GiB, so it can be raised
  toward ~9.7M positions for higher concurrency).
- Smoke: a short generation, then a 1M-context needle (the YaRN ceiling).

## 6. A/B + soak
- A/B against the vLLM 2×-Spark recipe (`/data/work/DeepSeek-v4-Flash-0731-
  DSpark-1M-NVFP4-KV-2x-DGX-Spark`) at matched concurrency: tokens/s, TTFT,
  quality.
- Soak (off-peak): sustained load, watch `MemFree` (head ~3.0 GiB / worker
  ~2.2 GiB steady) + swap activity (`grep pswp /proc/vmstat`, twice, 20 s
  apart — must stay flat). The memory guard (`launch/mem-guard.sh`) is
  idempotent and safe against a live server.

## Abort / restore
- `make down` the v4 stack, then restore the live stack (`make up` in the
  lane's repo). The v4 boot is additive; nothing about Lane D/Q is changed.
