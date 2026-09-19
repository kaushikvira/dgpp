# GPU-window gate runbook — DeepSeek-V4-Flash

The runner is `scripts/dsv4_gates.sh` (committed — read it before arguing with this doc). It
checks the preconditions, starts and stops the dsv4 engine only (direct: a local `dgpp-serve`;
fabric: `dgpp-cluster` up/down on the v4 deployment), and records a verdict per gate. It never
touches the live qwen/vision lane — that is why the "live lane down first" precondition exists.

## 1. Preconditions (satisfy in this order)

1. **The live lane down** (frees the GPU; the 512k lane is the live qwen
   deployment — nginx down + `dgpp-stop.sh dgpp-qwen-w2-yarn512k.json`):
   ```bash
   cd ~/work/q-dgx-gateway && make down-dgpp-512k
   ```
2. **Page caches dropped on both nodes** (head + worker) — the pre-launch
   ritual that `status` prints:
   ```bash
   sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
   ```
3. **The memory guard applied** (idempotent; installs `/etc/sysctl.d/90-laneq.conf`,
   inverts OOM priority so the model is protected):
   ```bash
   cd ~/work/q-dgx-gateway && make mem-guard   # runs launch/mem-guard.sh
   ```
4. **The build present.** The runner expects the build tree at `build-ci` in the repo root it
   is invoked from (`DGPP_BUILD_DIR` overrides, `DGPP_PRESET` defaults to `ci`), with the
   target binaries at the top level of the tree. `build_has` checks `[ -x $BUILD/<target> ]`
   for `dgpp-serve` and the parity/sanitizer targets, and prints the missing ones plus the
   hint `cmake --build --preset ci --target <missing> -j`. (In a worktree the tree lives in
   the main checkout — point `DGPP_BUILD_DIR` at it, or run the runner from there.)
5. **The checkpoint present** at `/data/models/DeepSeek-V4-Flash-0731` (with a `config.json`),
   plus the HF-cache wiring — the loader walks the HF-cache layout, so either the symlink
   `/data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local` → the
   checkpoint, or `HF_HOME=/data/models` for the boot. `checkpoint_ok` checks all three.

Verify with the read-only check (always exits 0; nothing is started or stopped):
`bash scripts/dsv4_gates.sh status`. Real output shape (GPU-quiet case):

```
=== dsv4 GPU-gate preconditions (read-only; nothing is started or stopped) ===
GPU quiet   : YES (no dgpp-serve, no heavy GPU app, the lane's health is down)
page caches : MemFree 58.0 GiB (the pre-launch ritual: sync && echo 3 | sudo tee /proc/sys/vm/drop_caches)
mem-guard   : installed (/etc/sysctl.d/90-laneq.conf, vm.swappiness=10)
build       : /home/kv/work/dgpp/build-ci
checkpoint  : /data/models/DeepSeek-V4-Flash-0731 (HF-cache wiring OK)
torch env: local python3 has no torch
torch env: docker image eugr/spark-vllm:latest is present (probe it with: docker run --rm --entrypoint python3 eugr/spark-vllm:latest -c 'import torch; print(torch.__version__)')
torch env: docker image eugr/spark-vllm-b12x:latest is present (…)
done (status never fails)
```

(In a worktree without its own `build-ci` the per-binary build lines show `MISSING`.) When
the GPU is not quiet, `status` first prints the busy reasons (`dgpp-serve running (pid …)`,
`GPU process … (… MiB)`, `the live lane answers at http://127.0.0.1:8888/health`) plus the
`make down-dgpp-512k` hint. `gpu_quiet` = no `dgpp-serve` + no non-desktop GPU compute app
>512 MiB (via `nvidia-smi --query-compute-apps`) + the lane's health down (`DSV4_LANE_HEALTH_URL`,
default `http://127.0.0.1:8888/health`).

## 2. The gate sequence

```bash
bash scripts/dsv4_gates.sh smoke      # one full /v1/chat/completions response
bash scripts/dsv4_gates.sh parity    # the dsv4 test targets that exist
bash scripts/dsv4_gates.sh sanitizer # compute-sanitizer memcheck pass
bash scripts/dsv4_gates.sh tps       # sustained decode vs the 40-60 tok/s band
bash scripts/dsv4_gates.sh reference # reference parity (degrades offline)
bash scripts/dsv4_gates.sh all       # smoke -> parity -> sanitizer -> tps -> reference
bash scripts/dsv4_gates.sh status    # read-only precondition matrix
```

Add `--build` to build the missing (registered) targets before parity/sanitizer.
**Exit codes (house convention):** `0` = the gate passed (for `all`: every gate passed or was
a designed skip, nothing failed); `1` = a real failure (a test failed, the request failed, the
sanitizer found errors, tps under target); `2` = a designed SKIP — a precondition is not met,
the message says what to do, nothing was run; `64` = usage error (unknown flag or subcommand).

### smoke — the one never captured after the csa2 out-of-bounds fix
Launches the engine, sends the fixed house prompt ("In one sentence, what is a DGX Spark?",
`max_tokens` 64, temperature 0 by default), saves the complete response, stops the engine.
- **SKIP (2):** the GPU is not quiet; the checkpoint/HF wiring is missing; the build tree or
  `dgpp-serve` is missing (direct); the fabric config is missing (fabric); or a memory-plan
  refusal in direct+w1 — designed: w1 does not hold the full model (155.42 GiB of weights against
  130 GiB, `docs/checkpoint_budget_dsv4.md`); the message says: run `DSV4_SMOKE_MODE=fabric`
  for the e2e smoke.
- **FAIL (1):** engine launch failed; not ready within `DSV4_READY_TIMEOUT` (default 900 s —
  process died, or a non-w1 memory-plan refusal); the request returned non-200; an
  unexpected `finish_reason` (not `stop`/`length`/`tool_calls`); empty content.
- **PASS (0):** HTTP 200, a valid finish_reason, non-empty content; the full response is in
  `smoke_response.json`.

### parity — the dsv4 numerics targets that exist in this tree
Runs `DSV4_PARITY_TARGETS` (nine wired): `dsv4_csa2_oracle_test`, `dsv4_dspark_oracle_test`,
`dsv4_mhc_oracle_test`, `dsv4_moe_oracle_test`, `dsv4_compress_tail_test`, `dsv4_fp8_scale_test`,
`dsv4_prompt_test`, `dsv4_tokenizer_test`, `dsv4_loader_test`. Per target: not registered in
`CMakeLists.txt` → designed SKIP; not built → SKIP with the build hint; the GPU subset
(`DSV4_PARITY_GPU_TARGETS`, default `dsv4_loader_test`) additionally needs the quiet GPU (it
allocates device memory). Each target runs under `timeout $DSV4_TARGET_TIMEOUT` (default 600
s), log in `<target>.log`.
- **FAIL (1):** any target failed (its log tail is printed).
- **PASS (0):** every target that ran passed (skips are designed).
- **SKIP (2):** no target ran at all (all skipped, or the build tree is missing).

### sanitizer — no memory errors in the CUDA dsv4 targets
Needs the quiet GPU, the build, the targets built (`DSV4_SANITIZER_TARGETS`, default
`dsv4_loader_test`), and `compute-sanitizer` (PATH or `/usr/local/cuda*/bin`). Runs
`compute-sanitizer --tool memcheck --exit-code $BUILD/<target>`; **PASS (0)** requires exit 0
*and* the sanitizer's own summary line `ERROR SUMMARY: 0 errors` (the witness). **FAIL (1):**
a sanitizer error (the first Invalid hit is printed). **SKIP (2):** no target ran (a
precondition was not met).

### tps — sustained decode against the 40-60 tok/s target
Needs the quiet GPU, the checkpoint, the build (direct) or the fabric config. Streams
`DSV4_TPS_TOKENS` (default 512) tokens from the house decode prompt, saving the raw SSE; the
client stamps every content-delta arrival (the first delta is t0, so prefill and HTTP setup
are excluded from the decode window). The verdict uses the engine's own last stats tick
(`decode X tok/s, Y ms/tok, Z ms/step` in `serve.log`) against `[DSV4_TPS_TARGET_MIN,
DSV4_TPS_TARGET_MAX]` (default 40–60): ≥60 → PASS (above the top of the band — good);
40–60 → PASS (inside the band); <40 → FAIL (under the target floor); when the stats interval
never ticked (a short decode), the client-measured pace is the fallback (neither available → FAIL).
- **SKIP (2):** w1 memory-plan refusal (designed — run in fabric mode).
- **FAIL (1):** launch/ready failure (non-w1), or the pace is under 40.

### reference — the reference-parity harness (degrades offline)
See §4. **PASS (0):** the token streams matched (`parity_diff.txt`).
**FAIL (1):** the dgpp-side capture failed (re-run `dgpp` + `diff` by hand — the reference side
ran), or the token streams diverged (`parity_diff.txt`). **SKIP (2):** the GPU is not quiet
(degraded to the offline prompt + diff path), or no CUDA-capable torch env could run the
reference here.

### all / status
`all` runs the five gates in order: PASS → continue; FAIL → stop the run (exit 1, a later
gate would be meaningless); designed SKIP → continue. Exit 0 only when nothing failed. `status`
is the read-only matrix above and never fails (exit 0).

## 3. Where artifacts land

Each gate invocation writes its own dir under `dsv4-gates/` (`DGPP_DSV4_GATES_OUT`
overrides; gitignored): `dsv4-gates/<YYYYmmdd-HHMMSS>-<gate>/` with:

- `gate-<name>.result` — the verdict line (PASS / FAIL / "SKIP (designed)") plus the reason.
- **smoke:** `smoke_response.json` (the full response), `serve.log`.
- **parity:** `<target>.log` per run target.
- **sanitizer:** `sanitizer-<target>.log`.
- **tps:** `tps_stream.sse` (the raw SSE), `serve.log`.
- **reference:** `reference_prompts.txt`, `reference_tokens.jsonl`, `dgpp_responses.json`,
  `parity_diff.txt`, `serve.log` (fabric teardown: `serve-down.log`).

Evidence to keep: the full smoke response (`smoke_response.json`), the tps numbers vs the
40-60 band (the client's line + the engine's last decode tick), the sanitizer log
(`ERROR SUMMARY: 0 errors`), and the token-stream diff (`parity_diff.txt`).

## 4. Known degradation: no CUDA-capable torch env here

The local `python3` has no torch (`import torch` fails); the docker images
`eugr/spark-vllm:latest` and `eugr/spark-vllm-b12x:latest` are present but **unprobed**
(`status` prints the probe command for each: `docker run --rm --entrypoint python3 <img> -c
'import torch; print(torch.__version__)'`). When no env can run the checkpoint's own
`inference/` reference, the gate degrades to the **offline path** (designed SKIP, exit 2) and
still produces the artifacts + instructions:

1. The prompt file `reference_prompts.txt` (generate.py's `--input-file` format) — regenerated
   on every degraded reference invocation (`prompts --out <gate dir>`; `|| true`, so a
   missing harness does not break the skip).
2. Run the Python reference on another box with that prompt file → the reference JSONL of tokens.
3. Capture dgpp's side: `python3 scripts/dsv4_reference_parity.py dgpp --host H --port P --out DIR`.
4. Diff: `python3 scripts/dsv4_reference_parity.py diff REFERENCE_JSONL DGGP_JSON` → `parity_diff.txt`.

⚠ **The harness `scripts/dsv4_reference_parity.py` is not in the tree yet** (only the runner
references it). Until it lands, `reference` can only SKIP (exit 2): with the script missing,
`python3 … run` exits non-zero and the runner maps that to the designed "no CUDA-capable
torch env" SKIP — the message is torch-flavoured, but the root cause is the missing script.

## 5. What a failure means

- **A parity target fails** (an oracle/host test) → the numerics wiring: the adapted v4 op
  diverges from its naive reference (csa2 64-head select, dspark union, moe router, compress
  tail, fp8 scale, prompt, tokenizer).
- **`dsv4_loader_test` fails** → the resident loader at the v4 geometry (the w2 plan,
  e8m0 decode, MXFP4 slices, hash-table map/gather).
- **A sanitizer hit** (Invalid global read/write) → an out-of-bounds in the CUDA dsv4 target —
  cf. the csa2 six-partials under-allocation (an invalid global read 61,569 B past the scratch).
- **A memory-plan refusal** (`refusing to allocate` in `serve.log`) → headroom: the shape does
  not fit this node. w1: 155.42 GiB of weights against 130 GiB — designed SKIP; e2e needs the world-2 fabric boot.
- **tps under 40** → decode pace (graph replay / kernel performance), not correctness.
- **Reference divergence** → the engine's token stream diverges from the checkpoint's own inference (numerics or sampling).

## 6. Env knobs (all optional; the runner's defaults)

- `DGPP_BUILD_DIR` (`$ROOT/build-ci`) · `DGPP_PRESET` (`ci`) · `DGPP_DSV4_GATES_OUT`
  (`$ROOT/dsv4-gates`) · `DSV4_CHECKPOINT_DIR` (`/data/models/DeepSeek-V4-Flash-0731`)
- `DSV4_SMOKE_MODE` (`direct`) · `DSV4_SMOKE_PORT` (8899) · `DSV4_SMOKE_KV_CAPACITY` (8192) ·
  `DSV4_SMOKE_WORLD` (1) · `DSV4_SMOKE_MAX_TOKENS` (64) · `DSV4_SMOKE_EXTRA_KNOBS` (`--temperature 0`)
- `DSV4_FABRIC_CONFIG` (`deploy/cluster_deepseek-v4-flash_fp4_w2.example.json`) ·
  `DGPP_ENV_FILE` (the site file for fabric mode) · `DSV4_LANE_HEALTH_URL`
  (`http://127.0.0.1:8888/health`)
- `DSV4_TPS_TOKENS` (512) · `DSV4_TPS_TARGET_MIN`/`MAX` (40 / 60) · `DSV4_READY_TIMEOUT`
  (900 s) · `DSV4_TARGET_TIMEOUT` (600 s)
- `DSV4_PARITY_TARGETS` (the nine wired in §2) · `DSV4_PARITY_GPU_TARGETS` and
  `DSV4_SANITIZER_TARGETS` (both default to `dsv4_loader_test`)

Direct mode (the default) runs w1 on one node — which this kit cannot hold (155.42 GiB >
130 GiB), so the e2e gates (smoke / tps / reference) are designed to SKIP until run with
`DSV4_SMOKE_MODE=fabric` (the world-2 fabric boot from the deploy template, `DGPP_ENV_FILE` set).
