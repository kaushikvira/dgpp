#!/usr/bin/env bash
# The DeepSeek-V4-Flash GPU-window gate suite (docs/dsv4_gpu_gate_runbook.md).
#
# Runs the gates for the quiet-GB10 window, in order:
#   smoke       one full /v1/chat/completions response (the one never captured
#               after the csa2 out-of-bounds fix) — launches the engine, sends
#               the fixed smoke prompt, saves the complete response, stops it
#   parity      the GPU parity gate: the dsv4 test targets that exist (the
#               csa2 64-head select / dspark union / moe router oracles, the
#               compress-tail + fp8-scale + prompt + tokenizer host tests, the
#               CUDA dsv4_loader_test)
#   sanitizer   a compute-sanitizer memcheck pass over the CUDA dsv4 targets
#   tps         a sustained-decode tps benchmark against the 40-60 tok/s target
#               (client timestamps + the engine's own stats line,
#               "decode X tok/s, Y ms/tok, Z ms/step")
#   reference   the reference-parity harness (the checkpoint's own inference/;
#               scripts/dsv4_reference_parity.py) — degrades to the offline
#               prompt + token-stream diff when no CUDA-capable torch env
#               can run it here
#   all         smoke -> parity -> sanitizer -> tps -> reference
#   status      read-only precondition matrix (always exits 0)
#
# Exit codes (house convention, cf. SKIP_RETURN_CODE 2):
#   0 = the gate passed (or, for `all`, every gate passed or was a designed
#       skip and nothing failed)
#   1 = a gate failed (a real failure: a test failed, the request failed,
#       the sanitizer found errors, tps is under target)
#   2 = a designed SKIP: a precondition is not met (the GPU is not quiet,
#       the live lane is up, the build or the checkpoint is missing, no torch
#       env, the direct shape has no resident image so the loader is
#       streaming, the fabric config's shape does not match the resident
#       images, the fabric binary is missing) — the message says what to do,
#       and nothing was run
#
# Every subcommand is dry-runnable without a GPU: precondition checks exit 2
# with the designed message, they never crash. `status` is pure read-only.
#
# This script starts and stops the dsv4 engine only (direct mode: a local
# dgpp-serve; fabric mode: dgpp-cluster up/down on the dsv4 deployment).
# It never touches the live qwen/vision lane — that is why the GPU-quiet
# precondition (live lane down first) exists.
#
# Env knobs (all optional; the runbook documents each):
#   DGPP_BUILD_DIR            build tree (default: $ROOT/build-ci)
#   DGPP_PRESET               the CMake preset for --build (default: ci)
#   DSV4_CHECKPOINT_DIR       (default: /data/models/DeepSeek-V4-Flash-0731)
#   DSV4_SMOKE_MODE           direct | fabric (default: direct)
#   DSV4_SMOKE_PORT           direct-mode port (default: 8899)
#   DSV4_SMOKE_KV_CAPACITY    direct-mode --kv-capacity (default: 8192)
#   DSV4_SMOKE_WORLD          direct-mode --world (default: 1)
#   DSV4_SMOKE_MAX_TOKENS     smoke prompt's max_tokens (default: 64)
#   DSV4_SMOKE_EXTRA_KNOBS    extra dgpp-serve flags (e.g. "--temperature 0")
#   DSV4_FABRIC_CONFIG        fabric-mode deployment (default: the site's w2
#                             dsv4 deployment /home/kv/work/q-dgx-gateway/config/
#                             dgpp-dsv4-w2.json when present — world 2 / kv 262144 /
#                             max_concurrency 2, mtp 1 (or off — the mtp flag does
#                             not enter the image's key) — the shape the resident
#                             images were captured for — else the repo's
#                             deploy/cluster_deepseek-v4-flash_fp4_w2.example.json;
#                             a config whose shape does not match the resident
#                             images is a designed SKIP, with the reason)
#   DSV4_FABRIC_BIN           the dgpp-serve binary fabric mode launches
#                             (default: $DGPP_BUILD_DIR/dgpp-serve — the repo's own
#                             build, the same dir build_has checks; passed to
#                             dgpp-cluster as --bin, so the site's DGPP_RELEASE pin
#                             — a master build with no dsv4 model code — is never
#                             launched)
#   DGPP_ENV_FILE             site file for fabric mode
#   DSV4_TPS_TOKENS           sustained-decode length N (default: 512)
#   DSV4_TPS_TARGET_MIN/MAX   the target band (default: 40 / 60)
#   DSV4_READY_TIMEOUT        seconds to wait for readiness (default: 900)
#   DSV4_TARGET_TIMEOUT       per-test timeout seconds (default: 600)
#   DSV4_PARITY_TARGETS       space-separated test targets (default: the seven
#                             wired below)
#   DSV4_PARITY_GPU_TARGETS   the subset that needs a live GPU
#                             (default: dsv4_loader_test)
#   DSV4_SANITIZER_TARGETS   sanitizer targets (default: dsv4_loader_test)
#   DSV4_REFERENCE_*          see scripts/dsv4_reference_parity.py --help
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${DGPP_BUILD_DIR:-$ROOT/build-ci}"
PRESET="${DGPP_PRESET:-ci}"
CKPT="${DSV4_CHECKPOINT_DIR:-/data/models/DeepSeek-V4-Flash-0731}"
OUT_ROOT="${DGPP_DSV4_GATES_OUT:-$ROOT/dsv4-gates}"

# The wired parity targets (the ones registered in CMakeLists.txt; a target
# that is not registered is a designed SKIP, not a failure).
DEFAULT_PARITY_TARGETS="dsv4_csa2_oracle_test dsv4_dspark_oracle_test dsv4_mhc_oracle_test dsv4_moe_oracle_test dsv4_compress_tail_test dsv4_fp8_scale_test dsv4_prompt_test dsv4_tokenizer_test dsv4_loader_test"
PARITY_TARGETS="${DSV4_PARITY_TARGETS:-$DEFAULT_PARITY_TARGETS}"
PARITY_GPU_TARGETS="${DSV4_PARITY_GPU_TARGETS:-dsv4_loader_test}"
SANITIZER_TARGETS="${DSV4_SANITIZER_TARGETS:-dsv4_loader_test}"

SMOKE_MODE="${DSV4_SMOKE_MODE:-direct}"
SMOKE_PORT="${DSV4_SMOKE_PORT:-8899}"
SMOKE_KV="${DSV4_SMOKE_KV_CAPACITY:-8192}"
SMOKE_WORLD="${DSV4_SMOKE_WORLD:-1}"
SMOKE_MAX_TOKENS="${DSV4_SMOKE_MAX_TOKENS:-64}"
SMOKE_EXTRA_KNOBS="${DSV4_SMOKE_EXTRA_KNOBS:---temperature 0}"
# The fabric deployment's default: the site's w2 dsv4 shape (world 2, kv
# 262144, max_concurrency 2, mtp 1 or off — the deployment the resident
# images in ~/.cache/dgpp/resident/ were captured for; the 83 GB
# 5c67fe8d85a3cf87.img is that image), falling back to the repo's example
# when the site file is absent. fabric_shape_ok() refuses a config whose
# shape does not match the images (designed SKIP, exit 2).
SITE_FABRIC_CONFIG="/home/kv/work/q-dgx-gateway/config/dgpp-dsv4-w2.json"
if [ -z "${DSV4_FABRIC_CONFIG:-}" ] && [ -f "$SITE_FABRIC_CONFIG" ]; then
  FABRIC_CONFIG="$SITE_FABRIC_CONFIG"
else
  FABRIC_CONFIG="${DSV4_FABRIC_CONFIG:-$ROOT/deploy/cluster_deepseek-v4-flash_fp4_w2.example.json}"
fi
# The fabric mode's binary: explicit, so dgpp-cluster never falls back to the
# site's DGPP_RELEASE pin (a master build that has no dsv4 model code at all).
FABRIC_BIN="${DSV4_FABRIC_BIN:-$BUILD/dgpp-serve}"
TPS_TOKENS="${DSV4_TPS_TOKENS:-512}"
TPS_MIN="${DSV4_TPS_TARGET_MIN:-40}"
TPS_MAX="${DSV4_TPS_TARGET_MAX:-60}"
READY_TIMEOUT="${DSV4_READY_TIMEOUT:-900}"
TARGET_TIMEOUT="${DSV4_TARGET_TIMEOUT:-600}"

LANE_HEALTH_URL="${DSV4_LANE_HEALTH_URL:-http://127.0.0.1:8888/health}"
LANE_DOWN_CMD="cd ~/work/q-dgx-gateway && make down-dgpp-512k"
DO_BUILD=0

# The house smoke prompt (the q-dgx-gateway Makefile's smoke-dgpp prompt).
SMOKE_PROMPT="In one sentence, what is a DGX Spark?"
# The house sustained-decode prompt (serve_bench.py's default).
TPS_PROMPT="Explain, in a few paragraphs, why a CUDA graph replay can be faster than launching the same kernels eagerly, and what it costs."

SERVE_PID=""
GATE_OUT=""

# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
log() { printf '%s\n' "$*"; }

gate_start() {  # gate_start NAME
  GATE_OUT="$OUT_ROOT/$(date +%Y%m%d-%H%M%S)-$1"
  mkdir -p "$GATE_OUT"
  log "=== gate $1 -> $GATE_OUT"
}

gate_result() {  # gate_result NAME PASS|FAIL|SKIP [reason]
  local name=$1 verdict=$2 reason=${3:-}
  printf '%s\n' "$verdict" > "$GATE_OUT/gate-$name.result"
  [ -n "$reason" ] && printf '%s\n' "$reason" >> "$GATE_OUT/gate-$name.result"
  local tag="PASS"
  case "$verdict" in
    PASS) tag="PASS" ;;
    FAIL) tag="FAIL" ;;
    SKIP) tag="SKIP (designed)" ;;
  esac
  log "=== gate $name: $tag${reason:+ — $reason}"
}

# Designed skip: the precondition is not met. The message says what to do.
skip_gate() {  # skip_gate NAME reason...
  local name=$1; shift
  gate_result "$name" SKIP "$*"
  exit 2
}

# ---------------------------------------------------------------------------
# preconditions (read-only)
# ---------------------------------------------------------------------------

# The GPU is quiet when: no dgpp-serve process, no non-desktop GPU compute
# app above the desktop's own footprint, and the live lane's health is down.
# Returns 0 (quiet) or 1 (not quiet; the reasons are printed).
gpu_quiet() {
  local busy=""
  local pids
  pids=$(pgrep -x dgpp-serve 2>/dev/null || true)
  if [ -z "$pids" ]; then
    # `pgrep -f` would also match any unrelated shell whose command line merely
    # mentions the binary (a `ls …/dgpp-serve` is enough) — only trust it for a
    # command line that actually STARTS the engine.
    pids=$(pgrep -f 'dgpp-serve([[:space:]]|$)' 2>/dev/null | while read -r p; do
             [ "$p" = "$$" ] && continue
             tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qE '(^|/)dgpp-serve( |$)' && echo "$p"
           done | tr '\n' ' ' || true)
  fi
  if [ -n "$pids" ]; then
    busy="$busy dgpp-serve running (pid $(echo $pids | tr '\n' ' '))"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local line pid pname mem
    while IFS=',' read -r pid pname mem; do
      pname=$(echo "$pname" | sed 's|.*/||; s/^ *//; s/ *$//')
      mem=$(echo "$mem" | tr -d ' MiB' | tr -d ' ')
      case "$pname" in
        Xorg|gnome-shell|gnome*|plasm*) : ;;  # the desktop's own footprint
        *) if [ -n "$mem" ] && [ "$mem" -gt 512 ] 2>/dev/null; then
             busy="$busy GPU process $pname (pid $pid, ${mem} MiB)"
           fi ;;
      esac
    done < <(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
  fi
  local code
  code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' "$LANE_HEALTH_URL" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    busy="$busy the live lane answers at $LANE_HEALTH_URL"
  fi
  if [ -n "$busy" ]; then
    log "GPU is NOT quiet:$busy"
    log "  -> stop the live stack first: $LANE_DOWN_CMD (or the lane's own make down)"
    return 1
  fi
  return 0
}

# The build tree exists and the named targets' binaries are present.
# Prints the missing ones. Returns 0 when all are present.
build_has() {  # build_has target...
  local missing="" t
  [ -d "$BUILD" ] || { log "build tree $BUILD is missing"; return 1; }
  for t in "$@"; do
    [ -x "$BUILD/$t" ] || missing="$missing $t"
  done
  if [ -n "$missing" ]; then
    log "missing binaries:$missing"
    log "  -> build them: cmake --build --preset $PRESET --target $(echo $missing | tr ' ' ',') -j"
    return 1
  fi
  return 0
}

checkpoint_ok() {
  [ -d "$CKPT" ] || { log "checkpoint $CKPT is missing"; return 1; }
  [ -f "$CKPT/config.json" ] || { log "checkpoint $CKPT has no config.json"; return 1; }
  local hf_link=/data/hf/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/local
  if [ ! -e "$hf_link" ] && [ -z "${HF_HOME:-}" ]; then
    log "checkpoint $CKPT is present, but the HF-cache symlink $hf_link is missing and HF_HOME is unset"
    log "  -> ln -sfn $CKPT $hf_link (or export HF_HOME=/data/models for the boot)"
    return 1
  fi
  return 0
}

# The fabric config's shape must be the deployment the resident images were
# captured for (world 2 / kv 262144 / max_concurrency 2, mtp 1 or off — the
# per-rank tensor set the on-disk image holds; the mtp flag does not enter
# the image's key, so the mtp-1 and nomtp variants share the image, cf. the
# image-key comment above serve_log_streaming). A different shape's first
# run would have to capture a new image — a full checkpoint build, far
# longer than the gate window — so the gate refuses the mismatch (designed
# SKIP) with the reason, instead of launching a world that cannot come up
# in time.
fabric_shape_ok() {
  [ -f "$FABRIC_CONFIG" ] || { log "fabric config $FABRIC_CONFIG is missing"; return 1; }
  local shape w kv mc mtp md
  shape=$(python3 - "$FABRIC_CONFIG" <<'PYSHAPE'
import json, sys
d = json.load(open(sys.argv[1]))
e = d.get("engine", {})
print("%s %s %s %s %s" % (d.get("world_size"), e.get("kv_capacity"), e.get("max_concurrency"), e.get("mtp"), e.get("mtp_depth")))
PYSHAPE
) || { log "cannot read the shape out of $FABRIC_CONFIG (is it a deployment JSON?)"; return 1; }
  read -r w kv mc mtp md <<< "$shape"
  # The image's key (resident_image_key()) does not include mtp: the w2
  # dsv4 shape's mtp-1 and nomtp variants share the same resident image
  # (only which layers a run captures differs — a run captures whatever is
  # missing). Both variants match the image.
  if [ "$w" = "2" ] && [ "$kv" = "262144" ] && [ "$mc" = "2" ] && { [ "$mtp" = "True" ] && [ "$md" = "1" ] || [ "$mtp" = "False" ]; }; then
    return 0
  fi
  log "fabric config $FABRIC_CONFIG is world $w / kv $kv / max_concurrency $mc / mtp $mtp (depth $md) — the resident images in ${DGPP_RESIDENT_CACHE_DIR:-$HOME/.cache/dgpp/resident} were captured for the w2 dsv4 shape (world 2 / kv 262144 / max_concurrency 2, mtp 1 or off)"
  log "  -> point DSV4_FABRIC_CONFIG at the w2 dsv4 deployment (the site's $SITE_FABRIC_CONFIG when present), or capture an image for this shape first"
  return 1
}

find_compute_sanitizer() {
  local c
  command -v compute-sanitizer >/dev/null 2>&1 && { command -v compute-sanitizer; return 0; }
  for c in /usr/local/cuda*/bin/compute-sanitizer /usr/local/cuda/bin/compute-sanitizer; do
    [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

# Is the target registered in this tree's CMakeLists.txt?
target_registered() {
  grep -q "add_executable($1 " "$ROOT/CMakeLists.txt" 2>/dev/null
}

# A torch env probe (report-only, no docker run): the local interpreter first,
# then the docker image candidates. Prints what it found.
probe_torch_env() {
  local found=0
  if python3 -c 'import torch, transformers, safetensors' >/dev/null 2>&1; then
    local v
    v=$(python3 -c 'import torch; print(torch.__version__, "cuda_available=" + str(torch.cuda.is_available()))' 2>/dev/null)
    log "torch env: local python3 — $v"
    found=1
  else
    log "torch env: local python3 has no torch"
  fi
  if command -v docker >/dev/null 2>&1; then
    local img
    for img in ${DSV4_REFERENCE_IMAGE_CANDIDATES:-eugr/spark-vllm:latest eugr/spark-vllm-b12x:latest}; do
      if docker image inspect "$img" >/dev/null 2>&1; then
        log "torch env: docker image $img is present (probe it with: docker run --rm --entrypoint python3 $img -c 'import torch; print(torch.__version__)')"
        found=1
      fi
    done
  fi
  if [ "$found" -eq 0 ]; then
    log "torch env: NONE found — the reference degrades to the offline path (run the Python reference on another box, diff the token streams here)"
  fi
  return 0
}

# ---------------------------------------------------------------------------
# the engine launch (direct | fabric)
# ---------------------------------------------------------------------------

# The streaming-fallback guard. dgpp-serve's world-1 boot is the "local
# reference" path: it never uses the resident image, so it prints
# "model constructed in Xs (streaming, …)" and every forward pass re-reads
# the layer weights off disk (streaming residency reuses one layer
# allocation, rebuilt per pass — src/loaders/resident_stream.hpp). A single
# prefill then outlasts the whole gate window: the request hangs to the
# curl timeout and the gate reports HTTP 000 — a FAIL that says nothing
# about dsv4's correctness. When the marker is in the serve log the case is
# a designed SKIP (exit 2), not a hang.
#
# Which shape the resident images correspond to: the images in the default
# dir ~/.cache/dgpp/resident/ (DGPP_RESIDENT_CACHE_DIR) are keyed by the
# shape — format version, loader format, world, rank, head sharding, the
# checkpoint's config + shard headers (resident_image_key(),
# src/loaders/resident_stream.hpp). The on-disk 83 GB 5c67fe8d85a3cf87.img
# is the world-2 / kv-262144 deployment shape (the site's
# dgpp-dsv4-w2.json: world 2, kv 262144, max_concurrency 2, mtp 1 — or its
# nomtp variant; the mtp flag does not enter the image's key).
#
# How a new shape's image gets captured: the loader captures it on a run.
# In RESIDENT mode (the world-2 fabric boot) the image file is opened
# O_CREAT — created fresh, or "header mismatch … replacing" when the key
# changed (src/loaders/resident_image.cpp) — and every layer the image does
# not hold is built from the checkpoint and captured into the image
# durably (fdatasync'd) before its entry is published (load_layer →
# capture_layer_to_image → write_layer, src/loaders/resident_stream.hpp).
# The first run of a shape is a full checkpoint build + capture (slow);
# later runs restore from the image. The w1 streaming path never captures.
serve_log_streaming() {  # serve_log_streaming LOGFILE -> 0 when the marker is present
  grep 'model constructed in' "$1" 2>/dev/null | grep -q '(streaming,'
}

# launch_serve MODE PORT KNOBS LOGFILE -> prints the serve pid (direct) or
# "fabric" (fabric mode; readiness and teardown go through dgpp-cluster).
launch_serve() {
  local mode=$1 port=$2 knobs=$3 logfile=$4
  case "$mode" in
    direct)
      "$BUILD/dgpp-serve" --checkpoint-dir "$CKPT" --port "$port" \
        --kv-capacity "$SMOKE_KV" --world "$SMOKE_WORLD" --rank 0 \
        $knobs > "$logfile" 2>&1 &
      SERVE_PID=$!
      log "launched dgpp-serve (pid $SERVE_PID, port $port, kv $SMOKE_KV, world $SMOKE_WORLD)"
      ;;
    fabric)
      export DGPP_ENV_FILE="${DGPP_ENV_FILE:-/home/kv/work/q-dgx-gateway/.env.dgpp}"
      # --bin is explicit: without it dgpp-cluster falls back to the site's
      # DGPP_RELEASE pin — a master build with no dsv4 model code. Default:
      # the repo's own build (DSV4_FABRIC_BIN overrides).
      log "launching fabric world: dgpp-cluster up --bin $FABRIC_BIN --config $FABRIC_CONFIG (env: $DGPP_ENV_FILE)"
      python3 "$ROOT/scripts/dgpp-cluster" up --bin "$FABRIC_BIN" --config "$FABRIC_CONFIG" >> "$logfile" 2>&1 \
        || { log "dgpp-cluster up failed — see $logfile"; return 1; }
      SERVE_PID="fabric"
      ;;
  esac
  return 0
}

stop_serve() {
  case "$SERVE_PID" in
    fabric)
      log "stopping fabric world: dgpp-cluster down --config $FABRIC_CONFIG"
      DGPP_ENV_FILE="${DGPP_ENV_FILE:-/home/kv/work/q-dgx-gateway/.env.dgpp}" \
        python3 "$ROOT/scripts/dgpp-cluster" down --config "$FABRIC_CONFIG" >> "$GATE_OUT/serve-down.log" 2>&1 \
        || log "WARNING: dgpp-cluster down reported a problem — see $GATE_OUT/serve-down.log"
      ;;
    "") : ;;
    *)
      kill "$SERVE_PID" 2>/dev/null
      for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "$SERVE_PID" 2>/dev/null || break
        sleep 1
      done
      kill -0 "$SERVE_PID" 2>/dev/null && kill -9 "$SERVE_PID" 2>/dev/null
      log "stopped dgpp-serve (pid $SERVE_PID)"
      ;;
  esac
  SERVE_PID=""
}

serve_cleanup() { [ -n "$SERVE_PID" ] && stop_serve; }
trap serve_cleanup EXIT INT TERM

# wait_ready URL TIMEOUT: polls /health then /v1/models; watches the serve log
# for the memory-plan refusal. Returns 0 on ready, 1 on refusal, 2 on death.
wait_ready() {
  local url=$1 timeout=$2 logfile=${3:-}
  local deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if [ -n "$logfile" ] && grep -q "refusing to allocate" "$logfile" 2>/dev/null; then
      log "the engine refused the memory plan (itemized in $logfile):"
      grep -A4 "refusing to allocate" "$logfile" | head -8
      return 1
    fi
    [ -n "$logfile" ] && [ ! -e "$logfile" ] && : # the file is created by the launcher
    if [ "$SERVE_PID" != "fabric" ] && ! kill -0 "$SERVE_PID" 2>/dev/null; then
      log "the serve process died during boot — tail of $logfile:"
      tail -20 "$logfile" 2>/dev/null
      return 2
    fi
    local code
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$url/health" 2>/dev/null || echo 000)
    if [ "$code" = "200" ]; then
      log "ready at $url"
      return 0
    fi
    # Some shapes answer /v1/models before /health is wired; accept either.
    local models_code
    models_code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$url/v1/models" 2>/dev/null || echo 000)
    if [ "$models_code" = "200" ]; then
      log "ready at $url"
      return 0
    fi
    sleep 5
  done
  log "not ready after ${timeout}s — tail of $logfile:"
  tail -20 "$logfile" 2>/dev/null
  return 2
}

# client_host_port MODE -> "HOST PORT" for the running (or to-be) serve.
client_host_port() {
  if [ "$SMOKE_MODE" = "fabric" ]; then
    local h p
    h=$(python3 "$ROOT/scripts/site_env.py" client-host 2>/dev/null || echo 127.0.0.1)
    p=$(python3 "$ROOT/scripts/site_env.py" http-port 2>/dev/null || echo 8888)
    echo "$h $p"
  else
    echo "127.0.0.1 $SMOKE_PORT"
  fi
}

served_model_id() {  # served_model_id HOST PORT -> the model id (or "deepseek-v4-flash")
  local host=$1 port=$2
  curl -s -m 5 "http://$host:$port/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys
try:
    d = json.load(sys.stdin)
    print(d["data"][0]["id"])
except Exception:
    print("deepseek-v4-flash")' 2>/dev/null || echo "deepseek-v4-flash"
}

# ---------------------------------------------------------------------------
# gate: smoke
# ---------------------------------------------------------------------------
gate_smoke() {
  gate_start smoke
  gpu_quiet || skip_gate smoke "the GPU is not quiet — the live lane (or a serve process) is holding it"
  checkpoint_ok || skip_gate smoke "the checkpoint / HF-cache wiring is not in place"
  if [ "$SMOKE_MODE" = "direct" ]; then
    [ -d "$BUILD" ] || skip_gate smoke "the build tree $BUILD is missing"
    maybe_build_targets dgpp-serve || true
    build_has dgpp-serve || skip_gate smoke "dgpp-serve is not built (cmake --build --preset $PRESET --target dgpp_serve_app -j, or scripts/ci-local.sh)"
  else
    fabric_shape_ok || skip_gate smoke "the fabric config's shape does not match the resident images — point DSV4_FABRIC_CONFIG at the w2 dsv4 deployment (world 2 / kv 262144 / max_concurrency 2, mtp 1 or off), or capture an image for this shape first"
    [ -x "$FABRIC_BIN" ] || skip_gate smoke "the fabric binary $FABRIC_BIN is missing (DSV4_FABRIC_BIN; default the repo's build — cmake --build --preset $PRESET --target dgpp_serve_app -j, or scripts/ci-local.sh)"
  fi

  local host port
  read -r host port < <(client_host_port)
  local logfile="$GATE_OUT/serve.log"

  launch_serve "$SMOKE_MODE" "$port" "$SMOKE_EXTRA_KNOBS" "$logfile" || {
    gate_result smoke FAIL "engine launch failed"; exit 1; }
  local rc=0
  wait_ready "http://$host:$port" "$READY_TIMEOUT" "$logfile" || rc=$?
  if [ "$SMOKE_MODE" = "direct" ] && serve_log_streaming "$logfile"; then
    # The streaming-fallback guard: this direct shape has no resident image,
    # so the loader is streaming and a single prefill will not finish — a
    # designed SKIP, not a request that hangs to the curl timeout (HTTP 000).
    stop_serve
    gate_result smoke SKIP "this shape (w$SMOKE_WORLD, kv $SMOKE_KV) has no resident image; the loader is streaming (marker in $logfile) and a single prefill will not finish — use DSV4_SMOKE_MODE=fabric with the world-2 shape, or capture an image for this shape first (a resident-mode run captures the missing layers)"
    exit 2
  fi
  if [ "$rc" -ne 0 ]; then
    stop_serve
    if [ "$rc" -eq 1 ]; then
      # The designed refusal: the shape does not fit this node.
      if [ "$SMOKE_MODE" = "direct" ] && [ "$SMOKE_WORLD" -eq 1 ]; then
        gate_result smoke SKIP "w1 does not hold the full model (the census: 155.42 GiB of weights against 130 GiB — docs/checkpoint_budget_dsv4.md)"
        log "  -> for the e2e smoke, run the world-2 fabric boot: DSV4_SMOKE_MODE=fabric $0 smoke"
        exit 2
      fi
      gate_result smoke FAIL "the engine refused the memory plan"
      exit 1
    fi
    gate_result smoke FAIL "the engine did not become ready within ${READY_TIMEOUT}s"
    exit 1
  fi

  local model
  model=$(served_model_id "$host" "$port")
  log "sending the smoke prompt (max_tokens $SMOKE_MAX_TOKENS, model $model)"
  local resp="$GATE_OUT/smoke_response.json"
  local http_code
  http_code=$(curl -s -m 600 -o "$resp" -w '%{http_code}' \
    "http://$host:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"'"$model"'","messages":[{"role":"user","content":"'"$SMOKE_PROMPT"'"}],"max_tokens":'"$SMOKE_MAX_TOKENS"',"temperature":0}')
  stop_serve

  if [ "$http_code" != "200" ]; then
    gate_result smoke FAIL "the smoke request returned HTTP $http_code (body in $resp)"
    exit 1
  fi
  local verdict
  verdict=$(python3 - "$resp" "$SMOKE_MAX_TOKENS" <<'EOF'
import json, sys
path, max_tokens = sys.argv[1], int(sys.argv[2])
d = json.load(open(path))
ch = d["choices"][0]
msg = ch["message"]
text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
if ch.get("finish_reason") not in ("stop", "length", "tool_calls"):
    print("FAIL finish_reason=" + str(ch.get("finish_reason")))
elif not text.strip():
    print("FAIL empty content")
else:
    usage = d.get("usage") or {}
    print("PASS finish_reason=%s content_tokens=%s" % (ch["finish_reason"], usage.get("completion_tokens")))
    print("content: " + text[:300].replace("\n", " "))
EOF
)
  local first=${verdict%%$'\n'*}
  if [[ "$first" == PASS* ]]; then
    log "$verdict"
    gate_result smoke PASS "one full /v1/chat/completions response captured in $resp"
    exit 0
  fi
  gate_result smoke FAIL "$verdict"
  exit 1
}

# ---------------------------------------------------------------------------
# gate: parity
# ---------------------------------------------------------------------------
# gate: parity
# ---------------------------------------------------------------------------
gate_parity() {
  gate_start parity
  [ -d "$BUILD" ] || skip_gate parity "the build tree $BUILD is missing (cmake -S $ROOT -B $BUILD --preset $PRESET, or scripts/ci-local.sh)"
  maybe_build_targets $PARITY_TARGETS || log "WARNING: the build step failed — the missing targets will be skipped"

  local gpu_targets="" t
  for t in $PARITY_TARGETS; do
    case " $PARITY_GPU_TARGETS " in *" $t "*) gpu_targets="$gpu_targets $t" ;; esac
  done

  local failed=0 ran=0 skipped=0
  for t in $PARITY_TARGETS; do
    if ! target_registered "$t"; then
      log "SKIP $t: not registered in CMakeLists.txt (the target does not exist in this tree)"
      skipped=$((skipped + 1)); continue
    fi
    if [ ! -x "$BUILD/$t" ]; then
      log "SKIP $t: not built (cmake --build --preset $PRESET --target $t -j)"
      skipped=$((skipped + 1)); continue
    fi
    case " $gpu_targets " in
      *" $t "*)
        gpu_quiet || { log "SKIP $t: the GPU is not quiet (this target allocates device memory) — $LANE_DOWN_CMD"; skipped=$((skipped + 1)); continue; }
        ;;
    esac
    log "run $t"
    local out="$GATE_OUT/$t.log"
    if timeout "$TARGET_TIMEOUT" "$BUILD/$t" > "$out" 2>&1; then
      log "PASS $t"
      ran=$((ran + 1))
    else
      log "FAIL $t (exit $?, log $out)"
      tail -15 "$out"
      failed=$((failed + 1))
    fi
  done

  if [ "$failed" -gt 0 ]; then
    gate_result parity FAIL "$failed target(s) failed, $ran passed, $skipped skipped"
    exit 1
  fi
  if [ "$ran" -eq 0 ]; then
    gate_result parity SKIP "no target ran ($skipped skipped)"
    exit 2
  fi
  gate_result parity PASS "$ran passed, $skipped skipped (designed)"
  exit 0
}

# ---------------------------------------------------------------------------
# gate: sanitizer
# ---------------------------------------------------------------------------
gate_sanitizer() {
  gate_start sanitizer
  gpu_quiet || skip_gate sanitizer "the GPU is not quiet — the sanitizer pass needs the quiet window"
  [ -d "$BUILD" ] || skip_gate sanitizer "the build tree $BUILD is missing"
  maybe_build_targets $SANITIZER_TARGETS || log "WARNING: the build step failed — the missing targets will be skipped"
  build_has $SANITIZER_TARGETS || skip_gate sanitizer "the sanitizer targets are not built"
  local cs
  cs=$(find_compute_sanitizer) || skip_gate sanitizer "compute-sanitizer is not installed (CUDA toolkit's bin on PATH)"

  local failed=0 ran=0 t out rc
  for t in $SANITIZER_TARGETS; do
    out="$GATE_OUT/sanitizer-$t.log"
    log "run compute-sanitizer --tool memcheck $t"
    if "$cs" --tool memcheck --exit-code "$BUILD/$t" > "$out" 2>&1; then
      rc=0
    else
      rc=$?
    fi
    # compute-sanitizer's own summary line is the witness: "ERROR SUMMARY: 0
    # errors from 0 device" (a nonzero exit with --exit-code means errors).
    if [ "$rc" -eq 0 ] && grep -q "ERROR SUMMARY: 0 errors" "$out"; then
      log "PASS $t: $(grep 'ERROR SUMMARY' "$out" | head -1)"
      ran=$((ran + 1))
    else
      log "FAIL $t (sanitizer exit $rc) — $(grep 'ERROR SUMMARY' "$out" | head -1)"
      grep -m4 -B2 "Invalid\|invalid" "$out" || tail -15 "$out"
      failed=$((failed + 1))
    fi
  done

  if [ "$failed" -gt 0 ]; then
    gate_result sanitizer FAIL "$failed target(s) had sanitizer errors"
    exit 1
  fi
  if [ "$ran" -eq 0 ]; then
    gate_result sanitizer SKIP "no target ran"
    exit 2
  fi
  gate_result sanitizer PASS "$ran target(s) clean (0 errors each)"
  exit 0
}

# ---------------------------------------------------------------------------
# gate: tps
# ---------------------------------------------------------------------------
gate_tps() {
  gate_start tps
  gpu_quiet || skip_gate tps "the GPU is not quiet — the tps benchmark needs the quiet window"
  checkpoint_ok || skip_gate tps "the checkpoint / HF-cache wiring is not in place"
  if [ "$SMOKE_MODE" = "direct" ]; then
    [ -d "$BUILD" ] || skip_gate tps "the build tree $BUILD is missing"
    maybe_build_targets dgpp-serve || true
    build_has dgpp-serve || skip_gate tps "dgpp-serve is not built"
  else
    fabric_shape_ok || skip_gate tps "the fabric config's shape does not match the resident images — point DSV4_FABRIC_CONFIG at the w2 dsv4 deployment (world 2 / kv 262144 / max_concurrency 2, mtp 1 or off), or capture an image for this shape first"
    [ -x "$FABRIC_BIN" ] || skip_gate tps "the fabric binary $FABRIC_BIN is missing (DSV4_FABRIC_BIN; default the repo's build — cmake --build --preset $PRESET --target dgpp_serve_app -j, or scripts/ci-local.sh)"
  fi

  local host port
  read -r host port < <(client_host_port)
  local logfile="$GATE_OUT/serve.log"

  launch_serve "$SMOKE_MODE" "$port" "$SMOKE_EXTRA_KNOBS" "$logfile" || {
    gate_result tps FAIL "engine launch failed"; exit 1; }
  local rc=0
  wait_ready "http://$host:$port" "$READY_TIMEOUT" "$logfile" || rc=$?
  if [ "$SMOKE_MODE" = "direct" ] && serve_log_streaming "$logfile"; then
    # The streaming-fallback guard (cf. gate_smoke): designed SKIP, not a
    # request that hangs to the curl timeout.
    stop_serve
    gate_result tps SKIP "this shape (w$SMOKE_WORLD, kv $SMOKE_KV) has no resident image; the loader is streaming (marker in $logfile) and a single prefill will not finish — use DSV4_SMOKE_MODE=fabric with the world-2 shape, or capture an image for this shape first (a resident-mode run captures the missing layers)"
    exit 2
  fi
  if [ "$rc" -ne 0 ]; then
    stop_serve
    if [ "$rc" -eq 1 ] && [ "$SMOKE_MODE" = "direct" ] && [ "$SMOKE_WORLD" -eq 1 ]; then
      gate_result tps SKIP "w1 does not hold the full model (the census: 155.42 GiB against 130 GiB) — run in fabric mode: DSV4_SMOKE_MODE=fabric $0 tps"
      exit 2
    fi
    gate_result tps FAIL "the engine did not come up (launch refused or died within ${READY_TIMEOUT}s)"
    exit 1
  fi

  local model
  model=$(served_model_id "$host" "$port")
  log "sustained decode: $TPS_TOKENS tokens, model $model"
  local resp="$GATE_OUT/tps_stream.sse" client_line
  # The client-side witness: stream live, stamp every content delta's arrival
  # (the serve_bench.py pattern) and save the raw SSE. The decode pace is the
  # span of the content deltas over (n-1) gaps — the first content delta is
  # t0, so prefill and HTTP setup are excluded from the decode window.
  client_line=$(python3 - "$host" "$port" "$model" "$TPS_TOKENS" "$resp" "$TPS_PROMPT" <<'PYCLIENT'
import json, sys, time, urllib.request
host, port, model, n_tokens, out_path, prompt = sys.argv[1:7]
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(n_tokens),
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
}).encode()
req = urllib.request.Request("http://%s:%s/v1/chat/completions" % (host, port),
                             data=body, headers={"Content-Type": "application/json"})
raw = open(out_path, "wb")
stamps = []
usage = None
with urllib.request.urlopen(req, timeout=1800) as r:
    buf = b""
    while True:
        chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
        if not chunk:
            break
        raw.write(chunk)
        buf += chunk
        now = time.perf_counter()
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("content") or d.get("reasoning_content"):
                        stamps.append(now)
raw.close()
n = len(stamps)
if n > 1:
    pace_ms = (stamps[-1] - stamps[0]) / (n - 1) * 1e3
    print("client: %d content deltas, decode pace %.2f ms/tok (%.1f tok/s)"
          % (n, pace_ms, 1000.0 / pace_ms))
else:
    print("client: %d content deltas (not enough for a pace)" % n)
if usage:
    print("client: usage %s" % usage)
PYCLIENT
)
  stop_serve
  log "$client_line"

  # The engine's own witness: its stats line, in its own format.
  local stats
  stats=$(grep -o "decode [0-9.]* tok/s, [0-9.]* ms/tok, [0-9.]* ms/step" "$logfile" | tail -3)
  if [ -z "$stats" ]; then
    stats="(no engine stats line yet in $logfile — the client's pace above is the witness)"
  fi
  log "engine stats (last ticks):"
  while IFS= read -r l; do log "  $l"; done <<< "$stats"

  # Verdict: the engine's last decode tok/s against the [TPS_MIN, TPS_MAX]
  # band; when the stats interval never ticked (a short decode), fall back to
  # the client's measured pace.
  local tps_verdict
  tps_verdict=$(python3 - "$logfile" "$client_line" "$TPS_MIN" "$TPS_MAX" <<'PYVERDICT'
import re, sys
path, client, lo, hi = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
rates = []
for line in open(path, errors="replace"):
    m = re.search(r"decode ([0-9.]+) tok/s, ([0-9.]+) ms/tok, ([0-9.]+) ms/step", line)
    if m:
        rates.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))
if rates:
    r, ms_tok, ms_step = rates[-1]
    line = "decode %.1f tok/s, %.1f ms/tok, %.1f ms/step" % (r, ms_tok, ms_step)
else:
    m = re.search(r"decode pace ([0-9.]+) ms/tok \(([0-9.]+) tok/s\)", client)
    if not m:
        print("FAIL no engine stats line and no client decode pace (check the stream)")
        sys.exit(0)
    line = "decode %s tok/s, %s ms/tok, n/a ms/step (client-measured)" % (m.group(2), m.group(1))
    r = float(m.group(2))
if r >= hi:
    print("PASS " + line + " (above the %.0f tok/s top of the target band — good)" % hi)
elif r >= lo:
    print("PASS " + line + " (inside the %.0f-%.0f tok/s target band)" % (lo, hi))
else:
    print("FAIL " + line + " (under the %.0f tok/s target floor)" % lo)
PYVERDICT
)
  local first=${tps_verdict%%$'\n'*}
  if [[ "$first" == PASS* ]]; then
    gate_result tps PASS "$tps_verdict"
    exit 0
  fi
  gate_result tps FAIL "$tps_verdict"
  exit 1
}

# ---------------------------------------------------------------------------
# gate: reference
# ---------------------------------------------------------------------------
gate_reference() {
  gate_start reference
  # The degraded path (no quiet GPU, or no torch env) still produces the
  # fixed-prompt artifacts and the offline diff instructions — that is the
  # designed behavior, not a crash.
  local py="$ROOT/scripts/dsv4_reference_parity.py"
  if ! gpu_quiet; then
    log "the GPU is not quiet — the reference cannot run here right now (designed degradation)"
    python3 "$py" prompts --out "$GATE_OUT" || true
    probe_torch_env
    log "offline path: run the Python reference on another box with the prompt file"
    log "  $GATE_OUT/reference_prompts.txt (generate.py's --input-file format),"
    log "  then capture dgpp's side (dsv4_reference_parity.py dgpp) and diff:"
    log "  python3 $py diff REFERENCE_JSONL DGGP_JSON"
    gate_result reference SKIP "the GPU is not quiet — degraded to the offline prompt + diff path (artifacts in $GATE_OUT)"
    exit 2
  fi
  if ! python3 "$py" run --out "$GATE_OUT"; then
    # probe_torch_env inside `run` already reported what it found.
    gate_result reference SKIP "no CUDA-capable torch env could run the reference here — degraded to the offline path"
    exit 2
  fi
  # The dgpp side of the diff: launch the engine (the same launch helper the
  # smoke/tps gates use), capture the fixed prompts, stop it, diff.
  checkpoint_ok || skip_gate reference "the checkpoint / HF-cache wiring is not in place"
  if [ "$SMOKE_MODE" = "direct" ]; then
    [ -d "$BUILD" ] || skip_gate reference "the build tree $BUILD is missing"
    maybe_build_targets dgpp-serve || true
    build_has dgpp-serve || skip_gate reference "dgpp-serve is not built"
  else
    fabric_shape_ok || skip_gate reference "the fabric config's shape does not match the resident images — point DSV4_FABRIC_CONFIG at the w2 dsv4 deployment (world 2 / kv 262144 / max_concurrency 2, mtp 1 or off), or capture an image for this shape first"
    [ -x "$FABRIC_BIN" ] || skip_gate reference "the fabric binary $FABRIC_BIN is missing (DSV4_FABRIC_BIN; default the repo's build — cmake --build --preset $PRESET --target dgpp_serve_app -j, or scripts/ci-local.sh)"
  fi
  local host port
  read -r host port < <(client_host_port)
  local logfile="$GATE_OUT/serve.log"
  launch_serve "$SMOKE_MODE" "$port" "$SMOKE_EXTRA_KNOBS" "$logfile" || {
    gate_result reference FAIL "engine launch failed"; exit 1; }
  local rc=0
  wait_ready "http://$host:$port" "$READY_TIMEOUT" "$logfile" || rc=$?
  if [ "$SMOKE_MODE" = "direct" ] && serve_log_streaming "$logfile"; then
    # The streaming-fallback guard (cf. gate_smoke): designed SKIP, not a
    # capture that hangs to the curl timeout.
    stop_serve
    gate_result reference SKIP "this shape (w$SMOKE_WORLD, kv $SMOKE_KV) has no resident image; the loader is streaming (marker in $logfile) and a single prefill will not finish — use DSV4_SMOKE_MODE=fabric with the world-2 shape, or capture an image for this shape first (a resident-mode run captures the missing layers)"
    exit 2
  fi
  if [ "$rc" -ne 0 ]; then
    stop_serve
    if [ "$rc" -eq 1 ] && [ "$SMOKE_MODE" = "direct" ] && [ "$SMOKE_WORLD" -eq 1 ]; then
      gate_result reference SKIP "w1 does not hold the full model (the census: 155.42 GiB against 130 GiB) — run in fabric mode: DSV4_SMOKE_MODE=fabric $0 reference"
      exit 2
    fi
    gate_result reference FAIL "the engine did not come up within ${READY_TIMEOUT}s"
    exit 1
  fi
  local dgpp_rc=0
  python3 "$py" dgpp --host "$host" --port "$port" --out "$GATE_OUT" || dgpp_rc=$?
  stop_serve
  if [ "$dgpp_rc" -ne 0 ]; then
    gate_result reference FAIL "the dgpp-side capture failed (the reference side ran; re-run 'dgpp' + 'diff' by hand)"
    exit 1
  fi
  if python3 "$py" diff "$GATE_OUT/reference_tokens.jsonl" "$GATE_OUT/dgpp_responses.json" --out "$GATE_OUT"; then
    gate_result reference PASS "token streams matched (see $GATE_OUT/parity_diff.txt)"
    exit 0
  fi
  gate_result reference FAIL "token streams diverged (see $GATE_OUT/parity_diff.txt)"
  exit 1
}

# ---------------------------------------------------------------------------
# status (read-only) and all
# ---------------------------------------------------------------------------
cmd_status() {
  log "=== dsv4 GPU-gate preconditions (read-only; nothing is started or stopped) ==="
  if gpu_quiet; then
    log "GPU quiet   : YES (no dgpp-serve, no heavy GPU app, the lane's health is down)"
  else
    log "GPU quiet   : NO (see above)"
  fi
  local free
  free=$(python3 -c '
import re
m = re.search(r"MemFree:\s+(\d+)", open("/proc/meminfo").read())
print("%.1f GiB" % (int(m.group(1)) / 1048576) if m else "n/a")')
  log "page caches : MemFree $free (the pre-launch ritual: sync && echo 3 | sudo tee /proc/sys/vm/drop_caches)"
  if [ -f /etc/sysctl.d/90-laneq.conf ]; then
    log "mem-guard   : installed (/etc/sysctl.d/90-laneq.conf, vm.swappiness=$(sysctl -n vm.swappiness 2>/dev/null || echo n/a))"
  else
    log "mem-guard   : NOT installed — bash ~/work/q-dgx-gateway/launch/mem-guard.sh (idempotent)"
  fi
  log "build       : ${BUILD}"
  for t in dgpp-serve dsv4_loader_test; do
    [ -x "$BUILD/$t" ] && log "  $t: present" || log "  $t: MISSING"
  done
  if checkpoint_ok; then
    log "checkpoint  : $CKPT (HF-cache wiring OK)"
  else
    log "checkpoint  : PROBLEM (see above)"
  fi
  probe_torch_env
  log "done (status never fails)"
  exit 0
}

cmd_all() {
  local failed_gate="" rc=0
  local gates="smoke parity sanitizer tps reference"
  local g
  for g in $gates; do
    case "$g" in
      smoke)      gate_smoke ;;
      parity)     gate_parity ;;
      sanitizer)  gate_sanitizer ;;
      tps)        gate_tps ;;
      reference)  gate_reference ;;
    esac
    rc=$?
    case "$rc" in
      0) log "gate $g: PASS — continuing" ;;
      1) log "gate $g: FAIL — stopping the run (a later gate would be meaningless)"
         failed_gate="$g"; exit 1 ;;
      2) log "gate $g: designed SKIP — continuing to the next gate" ;;
      *)  log "gate $g: unexpected exit $rc — stopping"
         failed_gate="$g"; exit 1 ;;
    esac
  done
  if [ -n "$failed_gate" ]; then
    exit 1
  fi
  exit 0
}

# --build: build the missing (but registered) test targets before running
# them. A one-time configure when the build tree is not configured yet.
maybe_build_targets() {  # maybe_build_targets target...
  [ "$DO_BUILD" -eq 1 ] || return 0
  local missing="" t
  for t in "$@"; do
    target_registered "$t" && [ ! -x "$BUILD/$t" ] && missing="$missing $t"
  done
  [ -n "$missing" ] || return 0
  if [ ! -f "$BUILD/CMakeCache.txt" ]; then
    log "configuring $BUILD (preset $PRESET)"
    cmake -S "$ROOT" -B "$BUILD" --preset "$PRESET" || return 1
  fi
  log "building:$missing"
  cmake --build "$BUILD" --target $(echo $missing | tr ' ' ',') -j "$(nproc)" || return 1
  return 0
}

usage() {
  # The header comment (up to `set -u`) is the documentation: print it whole,
  # whatever length it has grown to.
  local end
  end=$(grep -n '^set -u$' "$0" | head -1 | cut -d: -f1)
  [ -n "$end" ] && sed -n "2,$((end - 1))p" "$0" | sed 's/^# \{0,1\}//' | sed '/^$/d'
  log "usage: $0 {smoke|parity|sanitizer|tps|reference|all|status|help} [--build]"
  log "  --build  build the missing (registered) test targets before parity/sanitizer"
}

# ---------------------------------------------------------------------------
main() {
  local cmd="" arg
  local args=()
  for arg in "$@"; do
    case "$arg" in
      --build) DO_BUILD=1 ;;
      -h|--help|help|usage)
        usage
        exit 0 ;;
      -*)
        log "unknown flag '$arg' (try: help)"
        exit 64 ;;
      *)
        [ -z "$cmd" ] && cmd="$arg" || log "extra argument '$arg' ignored"
        ;;
    esac
  done
  [ -z "$cmd" ] && cmd="help"
  case "$cmd" in
    smoke)      gate_smoke ;;
    parity)     gate_parity ;;
    sanitizer)  gate_sanitizer ;;
    tps)        gate_tps ;;
    reference)  gate_reference ;;
    all)        cmd_all ;;
    status)     cmd_status ;;
    help)       usage ;;
    *)
      log "unknown subcommand '$cmd' (try: help)"
      exit 64 ;;
  esac
}

main "$@"
