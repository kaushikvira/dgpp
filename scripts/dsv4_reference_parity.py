#!/usr/bin/env python3
"""The DeepSeek-V4-Flash reference-parity harness (the reference gate in
scripts/dsv4_gates.sh; the runbook's G5: "run the engine's decode against the
checkpoint's own inference/ on a fixed prompt set and diff the tokens").

The reference half is the checkpoint's own inference/generate.py. It emits
decoded text only (Prompt:/Completion: blocks on stdout), so the "token
stream" compared here is the decoded completion text, whitespace-normalized;
both sides see the same prompt file and the same max_tokens cap, so the
parity is prompt-aligned by construction. The dgpp half is captured over the
engine's OpenAI-compatible /v1/chat/completions. All artifacts land in the
gate's output directory; the degraded offline path (no CUDA-capable torch
env here) is first-class: the prompt file is always emitted, and the diff
runs anywhere once both sides' artifacts exist.

    dsv4_reference_parity.py prompts --out DIR [--prompts-file FILE]
        Emit DIR/reference_prompts.txt in generate.py's --input-file format
        (prompts separated by a blank line). Pure emitter: no GPU, no torch.
    dsv4_reference_parity.py run --out DIR
        Run the checkpoint's own generate.py over the prompt file. A
        CUDA-capable torch env is probed (local python3 first, then the
        docker image candidates, the latter only with
        DSV4_REFERENCE_ALLOW_DOCKER=1); its Prompt/Completion stdout is
        parsed into DIR/reference_tokens.jsonl.
    dsv4_reference_parity.py dgpp --host H --port P --out DIR [--model M]
        Capture dgpp's side: the same prompts through /v1/chat/completions
        (temperature 0, the shared max_tokens cap) -> DIR/dgpp_responses.json.
    dsv4_reference_parity.py diff REFERENCE_JSONL DGGP_JSON [--out DIR]
        Diff the two token streams, prompt-aligned, whitespace-normalized;
        the report goes to DIR/parity_diff.txt.

Exit codes (house convention):
    0  pass — the artifacts were written, the side ran, the streams matched
    1  fail — the reference process failed, a request failed, or the token
              streams diverged / cannot be aligned
    2  designed SKIP — a precondition is not met (the checkpoint's reference
              side is missing, no CUDA-capable torch env, a diff input is
              missing, the endpoint is unreachable); the message says what
              to do, and nothing was run

Stdlib only. The env knobs are documented in --help (the runner's header
points DSV4_REFERENCE_* here).
"""
import argparse
import http.client
import json
import os
import shutil
import subprocess
import sys

DEFAULT_CKPT = "/data/models/DeepSeek-V4-Flash-0731"
DEFAULT_MAX_TOKENS = 128
DEFAULT_IMAGE_CANDIDATES = "eugr/spark-vllm:latest eugr/spark-vllm-b12x:latest"

PROMPTS_NAME = "reference_prompts.txt"
REF_TOKENS_NAME = "reference_tokens.jsonl"
DGGP_NAME = "dgpp_responses.json"
DIFF_NAME = "parity_diff.txt"
RUN_LOG_NAME = "reference_run.log"

# The fixed prompt set: two house anchors (the smoke prompt, the tps
# sustained-decode prompt) plus three fixed shapes (math, code, structured
# output). Single-line, so generate.py's "Prompt:/Completion:" stdout blocks
# parse cleanly.
FIXED_PROMPTS = [
    "In one sentence, what is a DGX Spark?",
    "Solve for x: 3x + 7 = 22. Show your work in two short steps.",
    "Write a Python function, fib(n), that returns the nth Fibonacci number iteratively. Code only, no explanation.",
    "Explain, in a few paragraphs, why a CUDA graph replay can be faster than launching the same kernels eagerly, and what it costs.",
    'List the three largest oceans by area as a JSON array of {"name": ..., "area_km2": ...} objects. No prose.',
]

ENV_DOC = """env knobs:
  DSV4_CHECKPOINT_DIR              the checkpoint directory (default: %s;
                                   the same knob the gate runner uses)
  DSV4_REFERENCE_MAX_TOKENS        the shared max_tokens cap for both sides (default: %d)
  DSV4_REFERENCE_IMAGE_CANDIDATES docker images probed for a torch env,
                                   space-separated (default: %s)
  DSV4_REFERENCE_ALLOW_DOCKER      1: let `run` execute the reference inside
                                   the first candidate image (docker run
                                   --gpus all, the checkpoint mounted ro)

exit codes: 0 pass / 1 fail / 2 designed skip (precondition not met)
""" % (DEFAULT_CKPT, DEFAULT_MAX_TOKENS, DEFAULT_IMAGE_CANDIDATES)


def read_prompt_file(path):
    """Parse a prompt file exactly like generate.py's --input-file does
    (split on a blank line, no other normalization) so both sides see the
    same prompt text."""
    with open(path) as f:
        return f.read().split("\n\n")


def write_prompt_file(path, prompts):
    with open(path, "w") as f:
        f.write("\n\n".join(prompts))


# ---------------------------------------------------------------------------
# prompts: the pure emitter (the degraded offline path's first artifact)
# ---------------------------------------------------------------------------

def cmd_prompts(args):
    if args.prompts_file:
        prompts = read_prompt_file(args.prompts_file)
        source = "from %s" % args.prompts_file
    else:
        prompts = list(FIXED_PROMPTS)
        source = "the built-in fixed set"
    if not any(p.strip() for p in prompts):
        print("no prompts to emit (%s is empty) — nothing written" % (args.prompts_file or "the prompt set"))
        return 2
    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as e:
        print("cannot create the output directory %s: %s" % (args.out, e))
        return 2
    path = os.path.join(args.out, PROMPTS_NAME)
    write_prompt_file(path, prompts)
    print("wrote %d prompts %s to %s (generate.py's --input-file format: blank-line separated)"
          % (len(prompts), source, path))
    print("offline flow: run the reference on a CUDA box with this file, capture dgpp's side (`dgpp`), then `diff`.")
    return 0


# ---------------------------------------------------------------------------
# run: the checkpoint's own Python reference (needs a CUDA-capable torch env)
# ---------------------------------------------------------------------------

def probe_torch_env():
    """Report-only probe (the runner's probe_torch_env, mirrored): the local
    interpreter first, then the docker image candidates. Returns "local",
    "docker:IMAGE" or None."""
    probe = ("import torch, transformers, safetensors\n"
             "assert torch.cuda.is_available(), 'torch has no CUDA device'")
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True)
    if r.returncode == 0:
        print("torch env: local python3 (CUDA available)")
        return "local"
    print("torch env: local python3 has no CUDA-capable torch (import failed or no CUDA device)")
    docker = shutil.which("docker")
    if not docker:
        print("torch env: docker is not on PATH (no image candidates probeable)")
        return None
    candidates = [c for c in os.environ.get("DSV4_REFERENCE_IMAGE_CANDIDATES", DEFAULT_IMAGE_CANDIDATES).split()]
    for img in candidates:
        if subprocess.run([docker, "image", "inspect", img], capture_output=True).returncode == 0:
            print("torch env: docker image %s is present (probe it with: docker run --rm --entrypoint python3 %s -c 'import torch; print(torch.__version__)')"
                  % (img, img))
            return "docker:" + img
    print("torch env: no candidate docker image present (tried: %s)"
          % (", ".join(candidates) if candidates else "none configured"))
    return None


def parse_generate_stdout(text):
    """Parse generate.py's non-interactive stdout into (prompt, completion)
    pairs. generate.py prints each pair as 'Prompt: X', 'Completion: Y...' ,
    then a blank line, so a new block starts only at a 'Prompt: ' line that
    follows a blank line; any other 'Prompt: ' line (or any line) after the
    completion starts belongs to the completion, including blank lines
    inside a multi-paragraph answer. Preamble lines before the first block
    (the ModelArgs dump, 'load model', the greeting) are ignored. Prompts
    must be single-line (the fixed set is)."""
    blocks = []
    cur = None  # [prompt, completion_lines]
    in_completion = False
    prev_blank = True  # the stream start is a boundary
    for line in text.splitlines():
        is_blank = line.strip() == ""
        if cur is None:
            if not is_blank and line.startswith("Prompt: "):
                cur = [line[len("Prompt: "):], []]
                in_completion = False
        elif in_completion:
            if line.startswith("Prompt: ") and prev_blank:
                blocks.append((cur[0], "\n".join(cur[1]).rstrip("\n")))
                cur = [line[len("Prompt: "):], []]
                in_completion = False
                prev_blank = False
                continue
            cur[1].append(line)
        else:
            if line.startswith("Completion: "):
                in_completion = True
                cur[1].append(line[len("Completion: "):])
        prev_blank = is_blank
    if cur is not None:
        blocks.append((cur[0], "\n".join(cur[1]).rstrip("\n")))
    return blocks


def cmd_run(args):
    ckpt = os.environ.get("DSV4_CHECKPOINT_DIR", DEFAULT_CKPT)
    generate_py = os.path.join(ckpt, "inference", "generate.py")
    config = os.path.join(ckpt, "config.json")
    missing = [p for p in (generate_py, config) if not os.path.isfile(p)]
    if missing:
        print("the checkpoint's reference side is missing: " + ", ".join(missing))
        print("  -> put the checkpoint at %s (DSV4_CHECKPOINT_DIR), or degrade to the offline path:" % ckpt)
        print("     python3 %s prompts --out %s, run the reference on a CUDA box, then `diff`."
              % (sys.argv[0], args.out))
        return 2
    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as e:
        print("cannot create the output directory %s: %s" % (args.out, e))
        return 2

    max_tokens = int(os.environ.get("DSV4_REFERENCE_MAX_TOKENS", DEFAULT_MAX_TOKENS))
    prompts_path = os.path.join(args.out, PROMPTS_NAME)
    write_prompt_file(prompts_path, FIXED_PROMPTS)
    prompts = FIXED_PROMPTS
    print("prompt file: %s (%d fixed prompts, the shared max_tokens cap %d)"
          % (prompts_path, len(prompts), max_tokens))

    env = probe_torch_env()
    allow_docker = os.environ.get("DSV4_REFERENCE_ALLOW_DOCKER", "") in ("1", "true", "yes")
    if env is None or (env.startswith("docker:") and not allow_docker):
        print("no CUDA-capable torch env could run the reference here (designed skip, exit 2)")
        if env is not None:
            print("  -> a docker image was found; re-run with DSV4_REFERENCE_ALLOW_DOCKER=1 to run the reference inside it")
        print("degraded offline path (the prompt file is already written above):")
        print("  1. run the reference on a CUDA box:")
        print("     python3 %s --ckpt-path %s --config %s --input-file %s --max-new-tokens %d --temperature 0"
              % (generate_py, ckpt, config, prompts_path, max_tokens))
        print("  2. save its Prompt/Completion blocks as %s"
              % os.path.join(args.out, REF_TOKENS_NAME))
        print("     (one JSON object per prompt: {\"prompt\": ..., \"completion\": ...})")
        print("  3. capture dgpp's side: python3 %s dgpp --host H --port P --out %s" % (sys.argv[0], args.out))
        print("  4. diff: python3 %s diff %s %s --out %s"
              % (sys.argv[0], os.path.join(args.out, REF_TOKENS_NAME),
                 os.path.join(args.out, DGGP_NAME), args.out))
        return 2

    common = ["--max-new-tokens", str(max_tokens), "--temperature", "0"]
    if env == "local":
        cmd = [sys.executable, generate_py, "--ckpt-path", ckpt, "--config", config,
               "--input-file", prompts_path] + common
        how = "local python3"
    else:
        img = env.split(":", 1)[1]
        cmd = ["docker", "run", "--rm", "--gpus", "all",
               "-v", ckpt + ":/ckpt:ro", "-v", args.out + ":/refout:ro", img,
               "python3", "/ckpt/inference/generate.py",
               "--ckpt-path", "/ckpt", "--config", "/ckpt/config.json",
               "--input-file", "/refout/" + PROMPTS_NAME] + common
        how = "docker image %s" % img
    print("running the reference (%s): %s" % (how, " ".join(cmd)))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    run_log = os.path.join(args.out, RUN_LOG_NAME)
    with open(run_log, "w") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        print("the reference run failed (exit %d) — log: %s" % (proc.returncode, run_log))
        for line in (proc.stdout or "").splitlines()[-6:]:
            print("  | " + line)
        if "model0-mp1.safetensors" in (proc.stdout + (proc.stderr or "")):
            print("hint: the reference wants model{rank}-mp{world}.safetensors but the checkpoint ships shards "
                  "(model-0000X-of-00048.safetensors) — run inference/convert.py first")
        return 1
    blocks = parse_generate_stdout(proc.stdout)
    if len(blocks) != len(prompts):
        print("the reference produced %d Prompt/Completion blocks for %d prompts — log: %s"
              % (len(blocks), len(prompts), run_log))
        return 1
    ref_path = os.path.join(args.out, REF_TOKENS_NAME)
    with open(ref_path, "w") as f:
        for file_prompt, (block_prompt, completion) in zip(prompts, blocks):
            if block_prompt != file_prompt:
                print("warning: the reference echoed a different prompt for entry %d (the prompt file was modified between the two sides?)" % (prompts.index(file_prompt) + 1))
            f.write(json.dumps({"prompt": block_prompt, "completion": completion}) + "\n")
    print("reference ran (%s) — %d/%d blocks parsed -> %s" % (how, len(blocks), len(prompts), ref_path))
    return 0


# ---------------------------------------------------------------------------
# dgpp: capture the engine's side over the OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

def http_json(host, port, path, body=None, timeout=600):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    if body is None:
        conn.request("GET", path)
    else:
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read().decode("utf-8", "replace")
    conn.close()
    try:
        return resp.status, json.loads(raw)
    except json.JSONDecodeError:
        return resp.status, raw


def cmd_dgpp(args):
    try:
        os.makedirs(args.out, exist_ok=True)
    except OSError as e:
        print("cannot create the output directory %s: %s" % (args.out, e))
        return 2
    prompts_path = os.path.join(args.out, PROMPTS_NAME)
    if os.path.isfile(prompts_path):
        prompts = read_prompt_file(prompts_path)
        prompt_src = prompts_path
    else:
        prompts = list(FIXED_PROMPTS)
        prompt_src = ("the built-in fixed set (no %s in %s — the gate flow writes it first via `prompts`/`run`)"
                      % (PROMPTS_NAME, args.out))
    if not any(p.strip() for p in prompts):
        print("no prompts to capture (%s is empty) — run `prompts` first" % prompts_path)
        return 2
    max_tokens = int(os.environ.get("DSV4_REFERENCE_MAX_TOKENS", DEFAULT_MAX_TOKENS))

    model = args.model
    if not model:
        try:
            st, data = http_json(args.host, args.port, "/v1/models", timeout=15)
        except (OSError, http.client.HTTPException) as e:
            print("the dgpp side is unreachable at http://%s:%s (%s) — is the engine up? "
                  "(designed skip: re-run `dgpp` once it serves)" % (args.host, args.port, e))
            return 2
        if st != 200 or not isinstance(data, dict):
            print("GET /v1/models returned HTTP %s — falling back to the model id 'deepseek-v4-flash'" % st)
            model = "deepseek-v4-flash"
        else:
            data_list = data.get("data") or [{}]
            model = data_list[0].get("id", "deepseek-v4-flash")

    responses = []
    for i, prompt in enumerate(prompts, 1):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0}
        try:
            st, data = http_json(args.host, args.port, "/v1/chat/completions", body)
        except (OSError, http.client.HTTPException) as e:
            if responses:
                print("the dgpp side died mid-capture (request %d/%d: %s) — %d responses were captured before the failure"
                      % (i, len(prompts), e, len(responses)))
                return 1
            print("the dgpp side is unreachable at http://%s:%s (%s) — is the engine up? "
                  "(designed skip: re-run `dgpp` once it serves)" % (args.host, args.port, e))
            return 2
        if st != 200 or not isinstance(data, dict) or "choices" not in data:
            print("dgpp request %d/%d failed (HTTP %s): %s" % (i, len(prompts), st, str(data)[:300]))
            return 1
        ch = data["choices"][0]
        msg = ch.get("message") or {}
        completion = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        responses.append({"prompt": prompt, "completion": completion,
                         "finish_reason": ch.get("finish_reason"), "usage": data.get("usage")})
        print("captured %d/%d (finish %s, %d completion chars)" % (i, len(prompts), ch.get("finish_reason"), len(completion)))

    out_path = os.path.join(args.out, DGGP_NAME)
    with open(out_path, "w") as f:
        json.dump(responses, f, indent=2)
        f.write("\n")
    print("captured %d responses from http://%s:%s (model %s, prompts from %s) -> %s"
          % (len(responses), args.host, args.port, model, prompt_src, out_path))
    return 0


# ---------------------------------------------------------------------------
# diff: the token-stream comparison (the degraded offline path's verdict)
# ---------------------------------------------------------------------------

def norm(s):
    return " ".join((s or "").split())


def load_ref_jsonl(path):
    entries = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError("%s line %d is not a JSON object: %s" % (path, n, e))
            if not isinstance(obj, dict) or "completion" not in obj:
                raise ValueError("%s line %d has no 'completion' field" % (path, n))
            entries.append(obj)
    return entries


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n  # one stream is a prefix of the other


def snippet(s, at, width=60):
    lo = max(0, at - 20)
    hi = min(len(s), at + width)
    return ("..." if lo > 0 else "") + s[lo:hi] + ("..." if hi < len(s) else "")


def cmd_diff(args):
    ref_path, dgpp_path = args.reference, args.dgpp
    missing = []
    if not os.path.isfile(ref_path):
        missing.append("reference JSONL %s (run the reference side first — or `prompts`, then the reference on a CUDA box)" % ref_path)
    if not os.path.isfile(dgpp_path):
        missing.append("dgpp JSON %s (capture it with `dgpp --host H --port P --out DIR`)" % dgpp_path)
    if missing:
        for m in missing:
            print("missing diff input: " + m)
        return 2
    try:
        ref_entries = load_ref_jsonl(ref_path)
    except (ValueError, OSError) as e:
        print("malformed reference input: %s" % e)
        return 2
    try:
        with open(dgpp_path) as f:
            dgpp_entries = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print("malformed dgpp input %s: %s" % (dgpp_path, e))
        return 2
    if not isinstance(dgpp_entries, list) or any(not isinstance(o, dict) or "completion" not in o for o in dgpp_entries):
        print("malformed dgpp input: %s must be a JSON array of objects with a 'completion' field" % dgpp_path)
        return 2
    if not ref_entries or not dgpp_entries:
        print("empty side(s): %d reference vs %d dgpp entries — the reference and/or dgpp side never produced a stream"
              % (len(ref_entries), len(dgpp_entries)))
        return 2

    total = max(len(ref_entries), len(dgpp_entries))
    lines = ["dsv4 reference parity diff",
             "reference: %s (%d entries)" % (ref_path, len(ref_entries)),
             "dgpp:      %s (%d entries)" % (dgpp_path, len(dgpp_entries)),
             "---"]
    diverged = 0
    for i in range(min(len(ref_entries), len(dgpp_entries))):
        r, d = ref_entries[i], dgpp_entries[i]
        rp, dp = norm(str(r.get("prompt", ""))), norm(str(d.get("prompt", "")))
        rc, dc = norm(str(r.get("completion", ""))), norm(str(d.get("completion", "")))
        if rp and dp and rp != dp:
            diverged += 1
            lines.append("[%d/%d] DIVERGED (prompt mismatch)" % (i + 1, total))
            lines.append("  reference prompt: %s" % snippet(rp, 0))
            lines.append("  dgpp prompt:      %s" % snippet(dp, 0))
        elif rc == dc:
            lines.append("[%d/%d] MATCH (%d normalized chars)" % (i + 1, total, len(rc)))
        else:
            diverged += 1
            at = first_diff(rc, dc)
            lines.append("[%d/%d] DIVERGED at normalized char %d" % (i + 1, total, at))
            lines.append("  reference: %s" % snippet(rc, at))
            lines.append("  dgpp:      %s" % snippet(dc, at))
    aligned = len(ref_entries) == len(dgpp_entries)
    if not aligned:
        diverged += 1
        lines.append("CANNOT COMPARE: %d reference vs %d dgpp entries (the prompt sets are not aligned)"
                     % (len(ref_entries), len(dgpp_entries)))
    lines.append("---")
    ok = aligned and diverged == 0
    if ok:
        lines.append("verdict: %d/%d matched — the token streams agree" % (total, total))
    else:
        lines.append("verdict: %d of %d diverged" % (diverged, total))
    if args.out:
        try:
            os.makedirs(args.out, exist_ok=True)
            diff_path = os.path.join(args.out, DIFF_NAME)
            with open(diff_path, "w") as f:
                f.write("\n".join(lines) + "\n")
            print("\n".join(lines))
            print("wrote %s" % diff_path)
        except OSError as e:
            print("cannot write the diff report to %s: %s" % (args.out, e))
            return 2
    else:
        print("\n".join(lines))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv):
    p = argparse.ArgumentParser(
        prog="dsv4_reference_parity.py",
        description="The DeepSeek-V4-Flash reference-parity harness: the fixed-prompt "
                    "artifacts, the checkpoint's own generate.py side, the dgpp endpoint "
                    "side, and the token-stream diff. The runner's reference gate "
                    "(scripts/dsv4_gates.sh) calls prompts / run / dgpp / diff.",
        epilog=ENV_DOC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("prompts", help="emit reference_prompts.txt (generate.py's --input-file format)")
    sp.add_argument("--out", required=True, help="the gate's output directory (the runner's $GATE_OUT)")
    sp.add_argument("--prompts-file", help="use this file's prompts instead of the built-in fixed set "
                                            "(parsed exactly like generate.py: split on a blank line)")
    sp.set_defaults(func=cmd_prompts)

    sr = sub.add_parser("run", help="run the checkpoint's own generate.py in a CUDA-capable torch env")
    sr.add_argument("--out", required=True, help="the gate's output directory (artifacts land here)")
    sr.set_defaults(func=cmd_run)

    sd = sub.add_parser("dgpp", help="capture dgpp's side over /v1/chat/completions")
    sd.add_argument("--host", required=True, help="the engine's host (the runner's $host)")
    sd.add_argument("--port", required=True, type=int, help="the engine's port (the runner's $port)")
    sd.add_argument("--out", required=True, help="the gate's output directory (dgpp_responses.json lands here)")
    sd.add_argument("--model", help="the served model id (default: discovered from /v1/models)")
    sd.set_defaults(func=cmd_dgpp)

    sdiff = sub.add_parser("diff", help="diff the two token streams (whitespace-normalized)")
    sdiff.add_argument("reference", help="the reference JSONL (one {prompt, completion} object per line)")
    sdiff.add_argument("dgpp", help="the dgpp JSON (an array of {prompt, completion, ...} objects)")
    sdiff.add_argument("--out", help="write the report to <out>/parity_diff.txt (also printed)")
    sdiff.set_defaults(func=cmd_diff)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
