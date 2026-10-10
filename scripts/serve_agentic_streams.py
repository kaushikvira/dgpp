#!/usr/bin/env python3
"""Concurrent multi-turn agentic streams against one dgpp-serve endpoint
(2026-09-07): the call pattern an agent client such as Hermes produces —
several conversations at once, each a tool loop of many turns, the model's
reasoning stripped from the history it sends back, single-message side
requests between turns — with the prefix cache's answer per turn read from
the response's usage (prompt_tokens_details.cached_tokens).
    serve_agentic_streams.py HOST PORT [--streams N] [--turns T]
        [--system-words W] [--tool-words K] [--side-every M]
        [--keep-reasoning] [--mutate-system-at TURN] [--reasoning-effort E]
        [--max-tokens N] [--stagger S] [--out DIR] [--label L]
Each stream: a system prompt of ~W words unique to the stream, a user
request, then T turns of `tool_choice: required` calls answered with a
synthetic ~K-word tool result. Reasoning is dropped from the assistant
messages sent back (Hermes's behavior) unless --keep-reasoning. Every
--side-every turns a single-message request runs beside the stream (the
agent's summary or memory calls). --mutate-system-at T appends a line to
the system prompt before turn T (a saved memory injected by the client):
the turn must miss and the server's INFO line must name the divergence.
Per request: prompt, cached and computed tokens, TTFT, completion tokens,
the decode pace, the finish reason, the tool called, and the response id;
a summary per stream and per turn; the misses listed. Artifacts:
OUT/requests.jsonl, summary.txt.
Exit status: nonzero when a turn that should have hit did not.

Exact token capture (2026-10-10). The response id recorded here joins the
turn to the serving process's token dump, which is the ground truth an
offline drafter study replays (the n-gram copy drafter's Phase 0 gate):
    DGPP_DUMP_TOKENS=$OUT/tokens.jsonl dgpp-serve ...
    serve_agentic_streams.py HOST PORT ... --out $OUT
Each dump line carries that request's exact prompt ids (the rendered
template) and committed ids, keyed by the same `id` this script writes as
`response_id` — the tokenizer's real ids, not a re-encoding of the text.
Only arm the dump for a capture run: it perturbs latency.
"""
import argparse
import http.client
import json
import os
import random
import sys
import threading
import time
from serve_client import served_model

ap = argparse.ArgumentParser()
ap.add_argument("host")
ap.add_argument("port", type=int)
ap.add_argument("--model")
ap.add_argument("--streams", type=int, default=3)
ap.add_argument("--turns", type=int, default=6)
ap.add_argument("--system-words", type=int, default=1200)
ap.add_argument("--tool-words", type=int, default=600)
ap.add_argument("--side-every", type=int, default=0)
ap.add_argument("--keep-reasoning", action="store_true")
ap.add_argument("--mutate-system-at", type=int, default=0)
ap.add_argument("--reasoning-effort", default="low")
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--stagger", type=float, default=0.0, help="seconds between stream starts")
ap.add_argument("--out", default="agentic_out")
ap.add_argument("--label", default="")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--prompt-mode", choices=["tools", "essay"], default="tools",
                help="essay: no tools, each turn asks for a long answer (sustained decode)")
args = ap.parse_args()
args.model = args.model or served_model(args.host, args.port)
os.makedirs(args.out, exist_ok=True)
lock = threading.Lock()
requests_f = open(os.path.join(args.out, "requests.jsonl"), "a", encoding="utf-8")
records = []
failures = []

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron "
         "pi rho sigma tau upsilon phi chi psi omega ledger orbit quartz saffron timber "
         "velvet willow zenith anchor beacon cipher dune ember fjord glacier harbor").split()


def words(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


TOOLS = [
    {"type": "function", "function": {
        "name": "run_step",
        "description": "Runs one step of the plan and returns its output.",
        "parameters": {"type": "object", "properties": {
            "step": {"type": "integer", "minimum": 1, "maximum": 1000,
                     "description": "The step index, starting at 1."},
            "note": {"type": "string", "description": "What this step does."}},
            "required": ["step"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "read_log",
        "description": "Reads lines of the run log.",
        "parameters": {"type": "object", "properties": {
            "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 2000}},
            "required": ["offset"], "additionalProperties": False}}},
]


def post_stream(body):
    """Streams a chat completion; returns (status, content, reasoning,
    tool_calls, usage, finish, rid, t_first, t_last, n_deltas)."""
    conn = http.client.HTTPConnection(args.host, args.port, timeout=1800)
    t0 = time.time()
    conn.request("POST", "/v1/chat/completions", json.dumps(body),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raw = resp.read().decode("utf-8", "replace")
        conn.close()
        return resp.status, raw, "", [], None, None, "", None, None, 0
    content, reasoning, usage, finish, rid = "", "", None, None, ""
    calls = {}
    t_first = t_last = None
    n_deltas = 0
    buf = b""
    while True:
        chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].decode("utf-8", "replace")
                if payload == "[DONE]":
                    continue
                ev = json.loads(payload)
                if ev.get("id") and not rid:
                    rid = ev["id"]
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {})
                    if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                        now = time.time()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        n_deltas += 1
                    content += d.get("content") or ""
                    reasoning += d.get("reasoning_content") or ""
                    for tc in d.get("tool_calls") or []:
                        slot = calls.setdefault(tc.get("index", 0),
                                                {"id": None, "name": "", "arguments": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    conn.close()
    tool_calls = [calls[k] for k in sorted(calls)]
    return 200, content, reasoning, tool_calls, usage, finish, rid, (t_first or t0) - t0, \
        ((t_last - t_first) if (t_first and t_last) else 0.0), n_deltas


def record(rec):
    with lock:
        records.append(rec)
        requests_f.write(json.dumps(rec) + "\n")
        requests_f.flush()
        tag = "%-3s %-4s t%-2d" % (rec["stream"], rec["kind"], rec["turn"])
        print("%s prompt %6d cached %6d computed %5d ttft %6.2fs out %4d %s %5.1f ms/tok %s%s" % (
            tag, rec["prompt_tokens"], rec["cached_tokens"], rec["computed"], rec["ttft_s"],
            rec["completion_tokens"], rec["finish"] or "?", rec["pace_ms"],
            rec["tool"] or "", "  <-- MISS" if rec["expected_hit"] and rec["cached_tokens"] == 0 else ""),
            flush=True)


def side_request(stream, turn, rng):
    body = {"model": args.model, "max_tokens": 64, "temperature": 0,
            "reasoning_effort": "low",
            "messages": [{"role": "user", "content": "Summarize in one line: " + words(rng, 120)}],
            "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.time()
    st, content, reasoning, calls, usage, finish, rid, ttft, dec, nd = post_stream(body)
    u = usage or {}
    record({"stream": stream, "kind": "side", "turn": turn, "status": st,
            "response_id": rid,
            "prompt_tokens": u.get("prompt_tokens", 0),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "computed": u.get("prompt_tokens", 0) - (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0), "ttft_s": ttft,
            "pace_ms": 1000.0 * dec / max(1, u.get("completion_tokens", 1) - 1),
            "finish": finish, "tool": "", "expected_hit": False,
            "t_start": t0, "t_end": time.time(), "label": args.label})


def run_stream(idx):
    rng = random.Random(args.seed * 1000 + idx)
    name = "S%d" % idx
    system = ("You are an autonomous operator working through a plan with tools. "
              "Call exactly one tool per turn: run_step with the next step index, or "
              "read_log to inspect the log. Keep any reasoning brief.\n\nReference notes for this run:\n"
              + words(rng, args.system_words))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Begin the plan. Start with step 1 and keep going; "
                                            "after each result, call the next step."}]
    prev_prompt = 0
    if args.prompt_mode == "essay":
        messages[1] = {"role": "user", "content": "Write a long, detailed essay (many paragraphs) "
                                                  "about the history of navigation at sea."}
    for turn in range(1, args.turns + 1):
        if args.mutate_system_at and turn == args.mutate_system_at:
            messages[0]["content"] += "\n\nSaved memory: the operator prefers concise notes."
        body = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens,
                "temperature": 0, "reasoning_effort": args.reasoning_effort,
                "stream": True, "stream_options": {"include_usage": True}}
        if args.prompt_mode == "tools":
            body["tools"] = TOOLS
            body["tool_choice"] = "required"
        t0 = time.time()
        st, content, reasoning, calls, usage, finish, rid, ttft, dec, nd = post_stream(body)
        if st != 200:
            with lock:
                failures.append("%s turn %d: HTTP %s %s" % (name, turn, st, str(content)[:200]))
            print("%s turn %d: HTTP %s %s" % (name, turn, st, str(content)[:200]), flush=True)
            return
        u = usage or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        expected_hit = turn > 1 and not (args.mutate_system_at and turn == args.mutate_system_at)
        rec = {"stream": name, "kind": "turn", "turn": turn, "status": st,
               "response_id": rid,
               "prompt_tokens": u.get("prompt_tokens", 0), "cached_tokens": cached,
               "computed": u.get("prompt_tokens", 0) - cached,
               "prev_prompt_tokens": prev_prompt,
               "completion_tokens": u.get("completion_tokens", 0),
               "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
               "ttft_s": ttft, "pace_ms": 1000.0 * dec / max(1, u.get("completion_tokens", 1) - 1),
               "finish": finish, "tool": ",".join(c["name"] for c in calls),
               "expected_hit": expected_hit, "t_start": t0, "t_end": time.time(),
               "label": args.label}
        record(rec)
        if expected_hit and cached == 0:
            with lock:
                failures.append("%s turn %d: expected a cache hit, got 0 cached of %d" %
                                (name, turn, rec["prompt_tokens"]))
        prev_prompt = rec["prompt_tokens"]
        # The assistant message as the client sends it back.
        assistant = {"role": "assistant", "content": content or None}
        if calls:
            assistant["tool_calls"] = [
                {"id": c["id"] or ("call_%d_%d" % (idx, turn)), "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                for c in calls]
        if args.keep_reasoning and reasoning:
            assistant["reasoning_content"] = reasoning
        messages.append(assistant)
        for c in calls:
            try:
                a = json.loads(c["arguments"] or "{}")
            except json.JSONDecodeError:
                a = {}
            if c["name"] == "run_step":
                result = "step %s output:\n%s" % (a.get("step"), words(rng, args.tool_words))
            elif c["name"] == "read_log":
                result = "log lines %s..:\n%s" % (a.get("offset"), words(rng, args.tool_words))
            else:
                result = "unknown tool"
            messages.append({"role": "tool", "tool_call_id": c["id"] or ("call_%d_%d" % (idx, turn)),
                             "content": result})
        if not calls:
            messages.append({"role": "user", "content":
                             "Now write another long, detailed essay (many paragraphs) about topic %d: %s."
                             % (turn + 1, ["the printing press", "glaciers", "the telegraph", "coral reefs",
                                          "bread", "the compass", "tides", "lighthouses"][turn % 8])
                             if args.prompt_mode == "essay" else
                             "Continue with the next step by calling a tool."})
        if args.side_every and turn % args.side_every == 0:
            side_request(name, turn, rng)


threads = []
t_all = time.time()
for i in range(args.streams):
    t = threading.Thread(target=run_stream, args=(i,), daemon=True)
    t.start()
    threads.append(t)
    if args.stagger > 0:
        time.sleep(args.stagger)
for t in threads:
    t.join()
wall = time.time() - t_all

# ---- the summary ------------------------------------------------------------
turns = [r for r in records if r["kind"] == "turn"]
sides = [r for r in records if r["kind"] == "side"]
lines = []
lines.append("%s: %d streams x %d turns, %d side requests, %.0f s wall%s" % (
    args.label or "run", args.streams, args.turns, len(sides), wall,
    ", reasoning kept" if args.keep_reasoning else ", reasoning stripped"))
hits = [r for r in turns if r["turn"] > 1 and r["cached_tokens"] > 0]
misses = [r for r in turns if r["turn"] > 1 and r["cached_tokens"] == 0]
lines.append("turns after the first: %d hit, %d missed; prompt tokens %d, cached %d (%.0f %%)" % (
    len(hits), len(misses), sum(r["prompt_tokens"] for r in turns),
    sum(r["cached_tokens"] for r in turns),
    100.0 * sum(r["cached_tokens"] for r in turns) / max(1, sum(r["prompt_tokens"] for r in turns))))
if hits:
    # Where the attach landed: at the previous prompt's end (a header-cut
    # attach re-prefills the previous answer) or past it (a close entry).
    at_close = [r for r in hits if r["cached_tokens"] > r["prev_prompt_tokens"]]
    lines.append("hits attached past the previous prompt (close entry): %d; at or before it (header cut): %d" % (
        len(at_close), len(hits) - len(at_close)))
    suffix = sorted(r["computed"] for r in hits)
    lines.append("re-prefilled suffix per hit: min %d, median %d, max %d tokens; ttft median %.2f s" % (
        suffix[0], suffix[len(suffix) // 2], suffix[-1],
        sorted(r["ttft_s"] for r in hits)[len(hits) // 2]))
for r in misses:
    lines.append("MISS %s turn %d: prompt %d tokens, ttft %.1f s%s" % (
        r["stream"], r["turn"], r["prompt_tokens"], r["ttft_s"],
        " (expected: the system prompt was mutated)" if not r["expected_hit"] else ""))
paces = [r["pace_ms"] for r in turns if r["completion_tokens"] > 8]
if paces:
    paces.sort()
    lines.append("decode pace over turns with > 8 tokens: median %.1f ms/tok, p90 %.1f" % (
        paces[len(paces) // 2], paces[int(len(paces) * 0.9)]))
tools = {}
for r in turns:
    tools[r["tool"] or "(none)"] = tools.get(r["tool"] or "(none)", 0) + 1
lines.append("tools called: " + ", ".join("%s x%d" % kv for kv in sorted(tools.items())))
if failures:
    lines.append("FAILURES:")
    lines.extend("  " + f for f in failures)
summary = "\n".join(lines)
print(summary, flush=True)
with open(os.path.join(args.out, "summary.txt"), "a", encoding="utf-8") as f:
    f.write(summary + "\n\n")
sys.exit(1 if failures else 0)
