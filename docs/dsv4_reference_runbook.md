# DeepSeek-V4-Flash reference runbook — 2× DGX Spark (TP=2)

Exact, verified steps to run the **checkpoint's own Python reference**
(`inference/generate.py`) on this 2× GB10 kit, to produce ground-truth token
streams for the G5 parity gate (diff against the engine's
`/v1/chat/completions`, via `scripts/dsv4_reference_parity.py`).

Every claim below is marked **[measured]** (real output, dated 2026-09-18),
**[computed]** (derived from measured bytes / code), or **[est]** (estimate,
label the uncertainty when you run it).

## Verdict up front

- **Memory: TP=2 FITS.** Per-rank peak ≈ **82–84 GiB** (81 GiB weights
  [computed] + ~0.2 GiB KV + ~1–2 GiB other [est]) against **114 GiB
  available** per node [measured: `free -g` → `available 114`]. ~30 GiB headroom.
- **The reference cannot load the shipped HF shards directly.** `generate.py`
  loads `model{rank}-mp{world_size}.safetensors` (`generate.py:91`), which only
  `convert.py` produces. Conversion is mandatory, **per rank** (see §3 — the
  stock `convert.py --model-parallel 2` OOMs a single Spark; use the
  `convert_rank.py` driver, §5.3 + appendix).
- **Top risk: `fast_hadamard_transform` is missing from both docker images**
  [measured probe, §4] and is required at runtime (21 ratio-4 layers + the
  indexer, `model.py:375,420,256`). Install/build it into the image (or a
  mount) **before** the run, or the first ratio-4 layer crashes.
- **Smallest useful command** (after the one-time §1–§5.3 prep): the §5.5
  rank-0 `docker run … torchrun … generate.py --input-file … --max-new-tokens 128
  --temperature 0`, 1-prompt file. Expected wall-clock: **first run 5–20 min,
  reruns 1–3 min [est]**.

## 0. Kit facts (all [measured] 2026-09-18)

| fact | value |
|---|---|
| nodes | `dgx1` (this box) = 192.168.177.11, `dgx2` = 192.168.177.12 (fabric, `enp1s0f0np0`); second fabric 192.168.178.11/.12 (`enP2p1s0f0np0`); Wi-Fi `wlP9s9` = 192.168.2.40 |
| checkpoint | `/data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local` on **both** nodes: 48 shards, 166,886,535,336 B = **155.4 GiB** payload, plus `inference/`, `encoding/`, `config.json`, tokenizer files |
| `/data` | **local NVMe per node, not shared** (dgx1 2.3 TiB free, dgx2 2.7 TiB free). Anything a rank reads must exist on that node |
| `inference/` on dgx2 | **EMPTY** [measured via ssh] — `generate.py`, `model.py`, `kernel.py`, `convert.py`, `config.json`, `requirements.txt` missing; `encoding/` and all 48 shards present. Sync needed (§1) |
| unified memory | 121.69 GiB/node; `free -g` on dgx1: total 121, **available 114**, swap 15 GiB |
| docker | `eugr/spark-vllm:latest` (id `4bca31bc0105`) present on **both** nodes; `eugr/spark-vllm-b12x:latest` only on dgx1. Same dependency matrix in both (§4) |
| ssh | `kv@192.168.177.12` works over the fabric (verified). The `dgx2` ssh alias resolves over Wi-Fi/LAN (`dgx2.l.vira.rocks`) — **do not** use it for torchrun rendezvous or large transfers |
| engine | one-stack-at-a-time: the dgpp engine (world 2, :8888) must be **down** before this run (it holds the same GB10s) |

## 1. Sync the missing reference tree to dgx2 (once)

From dgx1 (fabric, ~60 KB):

```bash
rsync -av /data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local/inference/ \
  kv@192.168.177.12:/data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local/inference/
```

Expected: 7 files transferred (`model.py` 45 KB, `kernel.py` 22 KB,
`generate.py`, `convert.py`, `config.json`, `requirements.txt`, `README.md`).
Verify:

```bash
ssh kv@192.168.177.12 'ls /data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local/inference/'
```

## 2. How the reference loads the model (evidence)

Reference tree: `/data/hf/hub/models--…-0731/snapshots/local/inference/`
(`generate.py`, `model.py`, `kernel.py`, `convert.py`, `engram.py`,
`requirements.txt`, `config.json`), tokenizer at the snapshot root.

- **Args** (`generate.py:135-144`): `--ckpt-path` (required), `--config`
  (required, the `ModelArgs` JSON = `inference/config.json`), `--input-file`
  (prompts split on a **blank line**, `generate.py:120-121`), `--interactive`
  (off by default → batch mode), `--max-new-tokens` (default 300),
  `--temperature` (default 1.0; **use 0** — `model.py:943` then takes
  `logits.argmax`, deterministic, which is what the parity gate needs).
- **Distributed env** (`generate.py:67-71`): reads `WORLD_SIZE`, `RANK`,
  `LOCAL_RANK` from the environment (exactly what `torchrun` exports);
  `dist.init_process_group("nccl")` when `WORLD_SIZE > 1` — plain NCCL over
  TCP sockets, no RDMA needed.
- **Load path** (`generate.py:89-91`):
  `AutoTokenizer.from_pretrained(ckpt_path)` then
  `load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"), strict=False)`.
  So `--ckpt-path` must be a directory containing
  **`model0-mp2.safetensors` + `model1-mp2.safetensors` + the two tokenizer
  files**. The shipped HF shards (`model-00001-of-00048.safetensors`…) are
  **not** directly loadable — the names/sharding don't match (rank-sharded
  files are what `convert.py` emits, `convert.py:136`).
- **Output** (`generate.py:125-127`): non-interactive mode prints
  `Prompt: …` / `Completion: …` blocks to **rank 0's stdout only**
  (`generate.py:72-73` silences ranks ≠ 0). That is the token stream to diff
  (`scripts/dsv4_reference_parity.py::parse_generate_stdout` already parses
  exactly this).
- **Sequence length**: batch (non-interactive) mode uses the config defaults
  `max_seq_len=4096`, `max_batch_size=4` (`model.py:37-38`;
  `inference/config.json` sets neither) — fine for short prompts. Interactive
  mode would switch to 64 K (`generate.py:84-85`); don't use it.
- **Model shape** (`inference/config.json`): 43 layers, dim 4096, 64 heads,
  head_dim 512 (MLA), 256 routed experts (top-6) + 1 shared, `expert_dtype:
  "fp4"`, `dtype: "fp8"`, `scale_fmt: "ue8m0"`, 3 MTP/DSpark layers.
- **Kernels** (`kernel.py:2`): all GEMM/quant/attention kernels are
  **tilelang** (`tilelang==0.1.8` pinned in `requirements.txt`; image ships
  0.1.12 — API surface checked, §4). **No flash-attn, no einops** — the only
  third-party runtime dep beyond torch/transformers/safetensors/tqdm is
  `fast_hadamard_transform` (lazy import, `model.py:256`).

## 3. Conversion: what `convert.py` produces, and why you need a per-rank driver

**What it does** (`convert.py:83-142`): one CPU-only pass over all 48 HF
shards (`safe_open(…, device="cpu")`), renames tensors
(`self_attn→attn`, `mlp→ffn`, `weight_scale_inv→scale`, …), splits:

- routed experts by index — 256 experts / mp=2 = **128 per rank**
  (`convert.py:108-110`);
- mapped tensors along dim 0/1 (embed, head, wq_b, wo_a, wo_b, attn_sink,
  weights_proj, markov_*) — half per rank (`convert.py:111-115`);
- everything else replicated to every rank;
- `wo_a` upcast FP4→**bf16** (`convert.py:124-127`); expert weights kept as
  int8-viewed-`float4_e2m1fn_x2` (the config's `expert_dtype: "fp4"` — do
  **not** pass `--expert-dtype fp8`);
- writes `model{i}-mp{mp}.safetensors` per rank (`convert.py:136`) and copies
  `tokenizer.json` + `tokenizer_config.json` into `--save-path`
  (`convert.py:138-142`).

**Shard sizes** [computed from the 48 shard headers, exact]:
routed experts 146.6 GiB → 73.31 GiB/rank; split-half non-expert 3.29 GiB;
replicated 2.21 GiB; `wo_a` bf16 upcast +2.16 GiB/rank ⇒

- **per-rank shard ≈ 81 GiB** (≈ 87 GB); **both ranks ≈ 162 GiB** on disk.
- `/data` has 2.3/2.7 TiB free on each node [measured] — disk is a non-issue.

**The OOM problem with stock `convert.py` at mp=2 [computed]:** it holds
*all* `mp` state dicts in host RAM at once (`convert.py:83,116,121`) ⇒
≈ 162 GiB peak > 121.69 GiB unified (114 GiB available + 15 GiB swap =
129 GiB) ⇒ OOM-kill or hours of swap thrash. **Do not run stock
`convert.py --model-parallel 2` on a Spark.** (mp=1 is worse: 155 GiB in one
dict — this is also why `scripts/dsv4_reference_parity.py`'s single-node
`run` path can never work on this kit.)

**Fix: a per-rank driver** (`convert_rank.py`, appendix — same math as
`convert.py`, one rank per process, peak ≈ 81 GiB RAM [computed], CPU-only,
no GPU). Each node converts **only its own rank** (each rank loads only
`model{rank}-mp2.safetensors`, `generate.py:91`):

- dgx1 → rank 0 → `/data/dsv4-ref-mp2/model0-mp2.safetensors` (+ tokenizer files)
- dgx2 → rank 1 → `/data/dsv4-ref-mp2/model1-mp2.safetensors` (+ tokenizer files)

Runtime [est]: one 167 GB read + 81 GB write per node at NVMe speed ≈
**3–8 min**. Verify after: `ls -la /data/dsv4-ref-mp2/` should show
`model{N}-mp2.safetensors` ≈ 87 GB + the two tokenizer files.

## 4. Dependencies (probed in the images, CPU-only, 2026-09-18)

`requirements.txt`: `torch>=2.10.0`, `transformers>=5.0.0`,
`safetensors>=0.7.0`, `fast_hadamard_transform`, `tilelang==0.1.8`.

Probe: `docker run --rm --entrypoint python3 <img> -c "import …"` (no GPU):

| package | `eugr/spark-vllm:latest` | `eugr/spark-vllm-b12x:latest` | needed? |
|---|---|---|---|
| torch | **2.13.0+cu130** ✓ | 2.13.0+cu130 ✓ | yes (≥2.10) |
| transformers | **5.15.0** ✓ | 5.15.0 ✓ | yes (≥5.0, `generate.py:9`) |
| safetensors | **0.8.0** ✓ | 0.8.0 ✓ | yes (`generate.py:10`) |
| tqdm | 4.70.0 ✓ | 4.70.0 ✓ | yes (`convert.py:5`) |
| tilelang | **0.1.12** (pin says 0.1.8) | 0.1.12 | yes (`kernel.py:2`) |
| fast_hadamard_transform | **MISSING** | **MISSING** | **yes, at runtime** (`model.py:256`) |
| flash_attn | missing | missing | not needed |
| einops | 0.8.2 | 0.8.2 | not needed (no import in the reference) |

Notes:

- **tilelang version drift (0.1.12 vs pinned 0.1.8)**: the API surface
  `kernel.py` uses (`set_log_level`, `PassConfigKey.TL_DISABLE_WARP_SPECIALIZED`
  / `TL_DISABLE_TMA_LOWER`, `T.symbolic/Kernel/prim_func/gemm/Pipelined/…`)
  all exist in 0.1.12 [measured probe]. Actual kernel codegen for GB10
  (sm_121) is untested — the first run will JIT-compile 6 kernel families
  (`act_quant`, `fp4_quant`, `fp8_gemm`, `fp4_gemm`, `sparse_attn`,
  `hc_split_sinkhorn`); persist the cache with `-e
  TILELANG_CACHE_DIR=/data/tilelang-cache` (env var exists in tilelang's
  `env.py` [measured]) so reruns skip it.
- **`fast_hadamard_transform` (top risk)**: imported only inside
  `rotate_activation()` (`model.py:256`) but that runs on every forward of
  the 21 `compress_ratio==4` layers (`model.py:375`) and the indexer
  (`model.py:420`) — i.e. it will fire within the first few decode steps.
  Options, in order of preference: (a) `pip install fast-hadamard-transform`
  inside a derived image / `docker run -it` session (CUDA extension build
  for cu130/sm_121 — untested, may need a small patch), (b) prebuild a wheel
  on any CUDA box and `pip install` the wheel, (c) last resort a torch-based
  Hadamard shim (changes numerics slightly — acceptable only with a note in
  the parity report). **Resolve this before the GPU window.**

## 5. The run (TP=2 across both Sparks)

### 5.1 Preconditions

```bash
docker ps                       # no GPU-holding containers (engine down — one stack at a time)
ssh kv@192.168.177.12 'docker ps'   # same on dgx2
```

### 5.2 Prompt file (the cheapest useful set: 1 prompt)

Create the **same** file on both nodes (each container reads its local
`/data`):

```bash
mkdir -p /data/dsv4-ref-mp2 /data/dsv4-artifacts/dsv4-ref
printf 'In one sentence, what is a DGX Spark?' > /data/dsv4-ref-mp2/reference_prompts.txt
ssh kv@192.168.177.12 'mkdir -p /data/dsv4-ref-mp2 && printf "In one sentence, what is a DGX Spark?" > /data/dsv4-ref-mp2/reference_prompts.txt'
```

Notes: **no trailing newline** — `generate.py:121` splits on `"\n\n"` and
keeps the remainder as prompt text. The 5-prompt gate set (same smoke prompt
+ 4 more, `DSV4_REFERENCE_MAX_TOKENS=128` shared cap) is emitted by
`python3 scripts/dsv4_reference_parity.py prompts --out /data/dsv4-artifacts/dsv4-ref`
on dgx1; copy its `reference_prompts.txt` to both nodes the same way if you
want the full set.

### 5.3 Convert (once, per node, CPU-only — can run while the engine is up)

Write `convert_rank.py` from the appendix to `/data/dsv4-artifacts/`
(dgx1), then `rsync -av /data/dsv4-artifacts/convert_rank.py kv@192.168.177.12:/data/dsv4-artifacts/`.

```bash
# dgx1 (rank 0) — ~3-8 min, ~81 GiB peak RAM, ~87 GB written
docker run --rm -v /data:/data eugr/spark-vllm:latest \
  python3 /data/dsv4-artifacts/convert_rank.py \
    --hf-ckpt-path /data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local \
    --save-path /data/dsv4-ref-mp2 --rank 0 --mp 2 --n-experts 256

# dgx2 (rank 1) — same, over the fabric ssh
ssh kv@192.168.177.12 'docker run --rm -v /data:/data eugr/spark-vllm:latest \
  python3 /data/dsv4-artifacts/convert_rank.py \
    --hf-ckpt-path /data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local \
    --save-path /data/dsv4-ref-mp2 --rank 1 --mp 2 --n-experts 256'
```

Both can run in parallel. `tqdm` progress per shard; exit 0. Then:

```bash
ls -la /data/dsv4-ref-mp2/          # model0-mp2.safetensors ≈ 87 GB, tokenizer.json, tokenizer_config.json
ssh kv@192.168.177.12 'ls -la /data/dsv4-ref-mp2/'   # model1-mp2.safetensors ≈ 87 GB
```

### 5.4 Launch rank 1 (dgx2) first, detached

```bash
ssh kv@192.168.177.12 'nohup docker run --rm --gpus all --net=host \
  -v /data:/data \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e TILELANG_CACHE_DIR=/data/tilelang-cache \
  -w /data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local/inference \
  eugr/spark-vllm:latest \
  torchrun --nnodes=2 --node-rank=1 --nproc-per-node=1 \
    --master-addr=192.168.177.11 --master-port=29500 \
    generate.py \
      --ckpt-path /data/dsv4-ref-mp2 \
      --config /data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local/inference/config.json \
      --input-file /data/dsv4-ref-mp2/reference_prompts.txt \
      --max-new-tokens 128 --temperature 0 \
  > /data/dsv4-artifacts/dsv4-ref/rank1.log 2>&1 &'
```

### 5.5 Launch rank 0 (dgx1) in the foreground — this is the run

```bash
docker run --rm --gpus all --net=host \
  -v /data:/data \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e TILELANG_CACHE_DIR=/data/tilelang-cache \
  -w /data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local/inference \
  eugr/spark-vllm:latest \
  torchrun --nnodes=2 --node-rank=0 --nproc-per-node=1 \
    --master-addr=192.168.177.11 --master-port=29500 \
    generate.py \
      --ckpt-path /data/dsv4-ref-mp2 \
      --config /data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local/inference/config.json \
      --input-file /data/dsv4-ref-mp2/reference_prompts.txt \
      --max-new-tokens 128 --temperature 0 \
  2>&1 | tee /data/dsv4-artifacts/dsv4-ref/reference_run.log
```

Why these flags:

- `--net=host`: the container must reach `192.168.177.11:29500` (torchrun
  c10d rendezvous) and the NCCL bootstrap on the fabric; on the default
  bridge network that route would go out the Wi-Fi default route (slow, and
  the `dgx2`-alias path is exactly what we are avoiding).
- `NCCL_SOCKET_IFNAME=enp1s0f0np0`: force NCCL onto the 177 fabric; without
  it NCCL may pick `wlP9s9` (Wi-Fi, 192.168.2.40) or the 178 fabric.
- `--master-addr=192.168.177.11` (dgx1's fabric IP) + `--master-port=29500`
  (clear of the dgpp fabric ports 29970/29971).
- `TILELANG_CACHE_DIR=/data/tilelang-cache` (inside the `/data` mount):
  persists tilelang's JIT cache across `--rm` runs.
- Start both nodes **within ~5 minutes** (torchrun's static rendezvous
  default timeout); if you hit a rendezvous timeout, add
  `--rdzv-conf timeout=1800` to **both** torchrun lines.

Expected stdout on rank 0: the `ModelArgs(...)` dump, `load model`
(`generate.py:88`), `I'm DeepSeek 👋` (`generate.py:93`), then the
`Prompt:`/`Completion:` block(s) (`generate.py:125-127`). The log lands at
`/data/dsv4-artifacts/dsv4-ref/reference_run.log`; rank 1's log at
`/data/dsv4-artifacts/dsv4-ref/rank1.log` (should be quiet — rank ≠ 0
silences prints, `generate.py:72-73`).

### 5.6 Capture + diff against the engine (the parity gate)

The reference log is already the ground-truth stream; convert it to the
harness's JSONL with the harness's own parser (from the repo root on dgx1):

```bash
python3 - <<'EOF'
import sys, json
sys.path.insert(0, 'scripts')
from dsv4_reference_parity import parse_generate_stdout
blocks = parse_generate_stdout(open('/data/dsv4-artifacts/dsv4-ref/reference_run.log').read())
with open('/data/dsv4-artifacts/dsv4-ref/reference_tokens.jsonl', 'w') as f:
    for p, c in blocks:
        f.write(json.dumps({'prompt': p, 'completion': c}) + '\n')
print(len(blocks), 'block(s) parsed -> reference_tokens.jsonl')
EOF
```

Then **bring the engine up** (one-stack-at-a-time: the reference is done by
now) and capture + diff its side:

```bash
python3 scripts/dsv4_reference_parity.py dgpp --host 127.0.0.1 --port 8888 --out /data/dsv4-artifacts/dsv4-ref
python3 scripts/dsv4_reference_parity.py diff \
  /data/dsv4-artifacts/dsv4-ref/reference_tokens.jsonl \
  /data/dsv4-artifacts/dsv4-ref/dgpp_responses.json \
  --out /data/dsv4-artifacts/dsv4-ref
```

Verdict line: `verdict: N/N matched — the token streams agree` (exit 0) or
`DIVERGED at normalized char …` (exit 1). Note the harness compares
**decoded text** (whitespace-normalized), not raw token ids — that is the
gate's design (`scripts/dsv4_reference_parity.py` docstring).

## 6. Memory feasibility (per rank, non-interactive mode)

| component | size | basis |
|---|---|---|
| weights (the converted shard, 1:1 into params) | **≈ 81 GiB** | [computed] from shard headers: 73.31 GiB experts + 3.29 split + 2.21 replicated + 2.16 `wo_a` bf16 upcast |
| KV buffers (bf16, `max_seq_len=4096`, `max_batch_size=4`, `model.py:37-38`): 2×128-entry window layers + 21×1152 + 20×160 (attn, `model.py:479-480`) + 21×1024 indexer (`model.py:405`) + fp32 `kv_state`/`score_state` (`model.py:309-310`) + `freqs_cis` | ≈ 0.2 GiB | [computed] |
| activations (1×~40-token prefill, 1-token decode) | < 0.05 GiB | [est] |
| NCCL buffers, tilelang workspaces, python | ≈ 1–2 GiB | [est] |
| **per-node peak** | **≈ 82–84 GiB** | vs **114 GiB available** [measured] |
| conversion host RAM (per-rank driver) | ≈ 81 GiB peak | [computed] — fits; stock mp=2 would need ≈ 162 GiB and **does not** fit |

**Verdict: TP=2 fits, ~30 GiB headroom per node.** No layer reduction, CPU
offload, or shorter config needed. (If you ever want the interactive 64 K
mode, KV grows to ≈ 0.5 GiB — still fits.)

## 7. Cheapest useful run + wall-clock

Smallest invocation that yields a ground-truth stream: **§5.5 with the
1-prompt file and `--max-new-tokens 128 --temperature 0`** (temperature 0 →
`argmax`, `model.py:943` → reproducible; 128 matches the harness's shared
cap `DSV4_REFERENCE_MAX_TOKENS=128`).

Wall-clock [est — first run vs rerun, tilelang cache warm):

| phase | first | rerun |
|---|---|---|
| conversion (one-time, §5.3) | 3–8 min | — (shards persist) |
| weight load 81 GiB NVMe→GPU | 0.5–2 min | same |
| tilelang JIT (6 kernel families) | 2–15 min | ~0 (cache in `/data/tilelang-cache`) |
| prefill (~40 tokens) | 1–3 s | same |
| decode 128 tokens (Python loop over 128 local experts/layer, `model.py:640`) | 15–90 s | same |
| **total** | **~5–20 min** | **~1–3 min** |

Capture: already in the §5.5 command (`tee /data/dsv4-artifacts/dsv4-ref/reference_run.log`).

## 8. Risks / blockers (ranked)

1. **`fast_hadamard_transform` missing in both images** [measured]. Hard
   runtime requirement (§4). Plan the install/build **before** the GPU
   window; if the CUDA build fails on sm_121/cu130, the torch-Hadamard shim
   is the fallback (flag it in the parity report).
2. **Stock `convert.py --model-parallel 2` OOMs a Spark** (≈ 162 GiB RAM
   [computed] > 121.69 GiB). Use `convert_rank.py` (§5.3/appendix).
3. **tilelang 0.1.12 vs pinned 0.1.8**: API surface verified [measured],
   but GB10 codegen of the 6 kernels is untested; a broken kernel surfaces
   only on first forward (JIT compile or a runtime assert). Mitigation:
   `TILELANG_CACHE_DIR` persistence; if a kernel fails, the log names the
   kernel + shape.
4. **NCCL interface selection**: without `NCCL_SOCKET_IFNAME=enp1s0f0np0`
   NCCL may route over Wi-Fi (`wlP9s9`) — slow and flaky. Both nodes have
   `enp1s0f0np0` UP on 192.168.177.x [measured].
5. **`/data` is not shared**: converted shards must exist per node (§5.3
   converts per node); the `inference/` tree was missing on dgx2 (§1).
6. **One stack at a time**: the dgpp engine must be down for the whole
   reference run (same GB10s); the `dgpp`+`diff` half of §5.6 runs after the
   engine is back up.
7. **torchrun rendezvous window**: start rank 1 and rank 0 within ~5 min
   (else add `--rdzv-conf timeout=1800` to both).
8. **Reference decode is slow by design** (Python expert loop,
   `model.py:640`): fine for a 1-prompt/128-token ground truth; do **not**
   use it for long generations.
9. **The harness's `run` subcommand is single-node MP=1 only**
   (`scripts/dsv4_reference_parity.py::cmd_run` — one `docker run`, no
   torchrun; its own hint even says "run inference/convert.py first"). MP=1
   needs 155 GiB > 121.69 GiB, so on this kit the reference half must be the
   manual 2-node flow of §5; `prompts`/`dgpp`/`diff` subcommands work as-is.

## Appendix: `convert_rank.py` (write to `/data/dsv4-artifacts/`)

Per-rank version of the reference's `convert.py` (§3). Same math — it
imports the reference's own `mapping` / `cast_e2m1fn_to_e4m3fn` from
`inference/convert.py` so the numerics can't drift — but builds and saves
**one rank per process** (peak RAM ≈ one shard ≈ 81 GiB instead of ≈ 162
GiB at mp=2), so it fits one GB10 node. CPU-only.

```bash
cat > /data/dsv4-artifacts/convert_rank.py <<'PYEOF'
#!/usr/bin/env python3
"""Per-rank checkpoint converter for the DeepSeek-V4-Flash reference.

Same conversion as inference/convert.py (imports its mapping + fp4->fp8
cast), but processes ONE rank per process so host RAM stays ~one shard's
size (~81 GiB at mp=2) instead of mp shards' (~162 GiB) — which does not
fit a GB10's 121.69 GiB unified memory. CPU-only, no GPU.

Usage (inside eugr/spark-vllm, /data mounted):
  python3 convert_rank.py --hf-ckpt-path <snapshot dir> --save-path /data/dsv4-ref-mp2 \
      --rank 0 --mp 2 --n-experts 256
"""
import gc
import importlib.util
import os
import shutil
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm


def load_reference_converter(hf_ckpt_path):
    """Import the reference's convert.py for its mapping + cast (single source of truth)."""
    path = os.path.join(hf_ckpt_path, "inference", "convert.py")
    spec = importlib.util.spec_from_file_location("ref_convert", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # top level is defs only; its __main__ guard keeps main() out
    return mod


def convert_rank(rank, mp, n_experts, hf_ckpt_path, save_path, ref):
    n_local = n_experts // mp
    state_dict = {}
    for file_path in tqdm(sorted(glob(os.path.join(hf_ckpt_path, "*.safetensors"))),
                         desc=f"rank {rank}"):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                param = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                if name.startswith("mtp.") and ("emb" in name or name.endswith("head.weight")):
                    continue
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                if any(x in name for x in ["hc", "attn_sink", "tie2eid", "ape"]):
                    key = name.split(".")[-1]
                else:
                    key = name.split(".")[-2]
                if key in ref.mapping:
                    new_key, dim = ref.mapping[key]
                else:
                    new_key, dim = key, None
                name = name.replace(key, new_key)
                if "experts" in name and "shared_experts" not in name:
                    idx = int(name.split(".")[-3])
                    if idx < rank * n_local or idx >= (rank + 1) * n_local:
                        continue
                    new_param = param
                elif dim is not None:
                    assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                    shard_size = param.size(dim) // mp
                    new_param = param.narrow(dim, rank * shard_size, shard_size).contiguous()
                else:
                    new_param = param
                state_dict[name] = new_param

    # Post-processing: identical to the reference's save loop (convert.py:120-136)
    for name in list(state_dict.keys()):
        if name.endswith("wo_a.weight"):
            weight = state_dict[name]
            scale = state_dict.pop(name.replace("weight", "scale"))
            weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() \
                * scale[:, None, :, None].float()
            state_dict[name] = weight.flatten(2, 3).flatten(0, 1).bfloat16()
        elif "experts" in name and state_dict[name].dtype == torch.int8:
            state_dict[name] = state_dict[name].view(torch.float4_e2m1fn_x2)

    os.makedirs(save_path, exist_ok=True)
    out = os.path.join(save_path, f"model{rank}-mp{mp}.safetensors")
    save_file(state_dict, out)
    print(f"rank {rank}: wrote {out} ({os.path.getsize(out) / 1e9:.1f} GB, "
          f"{len(state_dict)} tensors)")
    del state_dict
    gc.collect()


def main():
    p = ArgumentParser()
    p.add_argument("--hf-ckpt-path", required=True)
    p.add_argument("--save-path", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--mp", type=int, required=True)
    p.add_argument("--n-experts", type=int, required=True)
    a = p.parse_args()
    assert a.n_experts % a.mp == 0 and 0 <= a.rank < a.mp
    ref = load_reference_converter(a.hf_ckpt_path)
    convert_rank(a.rank, a.mp, a.n_experts, a.hf_ckpt_path, a.save_path, ref)
    for file in ["tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(a.hf_ckpt_path, file)
        dst = os.path.join(a.save_path, file)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
    print(f"rank {a.rank}: done ({a.save_path})")


if __name__ == "__main__":
    main()
PYEOF
```

(If you'd rather not run the driver: the reference's own
`python3 inference/convert.py --hf-ckpt-path … --save-path … --n-experts 256
--model-parallel 2` is the stock command — but see the OOM note in §3; it
will not complete on this hardware.)
