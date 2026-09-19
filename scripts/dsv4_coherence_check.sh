#!/usr/bin/env bash
# dsv4_coherence_check.sh — one-command coherence readout for the dsv4 engine.
#
# The vLLM lane serves the SAME checkpoint (`make up-base` in
# ~/work/v-dgx-gateway), so a correct engine answers the smoke prompt with the
# reference's opening: "The DGX Spark is NVIDIA's compact, personal AI
# supercomputer...". This script asks OUR engine the same prompt (greedy) and
# reports how many of the reference's first tokens we reproduce — the metric to
# move when fixing the numerics. 0 means the first token is already wrong.
#
#   bash scripts/dsv4_coherence_check.sh [HOST:PORT] [MAX_TOKENS]
set -uo pipefail
HOSTPORT="${1:-127.0.0.1:8888}"
MAXTOK="${2:-40}"
MODEL="${DSV4_MODEL:-deepseek-v4-flash-dspark}"
PROMPT="In one sentence, what is a DGX Spark?"
TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

curl -s -m 300 "http://$HOSTPORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":$MAXTOK,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}" \
  -o "$TMP" 2>/dev/null

python3 - "$TMP" <<'PYEOF'
import json, sys
path = sys.argv[1]
raw = open(path, 'rb').read()
if not raw:
    print("no response (is the engine up?)")
    raise SystemExit(2)
try:
    d = json.loads(raw.decode('utf-8', 'replace'))
except Exception as e:
    print("unparseable response:", e)
    raise SystemExit(2)
if 'error' in d:
    print("engine error:", str(d['error'])[:200])
    raise SystemExit(1)
m = d.get('choices', [{}])[0].get('message', {})
text = (m.get('content') or '') + (m.get('reasoning_content') or '')
print("our text :", repr(text[:220]))
print("reference: 'The DGX Spark is NVIDIA\\'s compact, personal AI supercomputer ...'")
ref = ["The", " DG", "X", " Spark", " is", " NVIDIA", "'s", " compact"]
i = 0
for tok in ref:
    if text[i:i + len(tok)] != tok:
        break
    i += len(tok)
print("common prefix with the reference: %d chars (0 = the first token is wrong)" % i)
print("VERDICT:", "coherent opening" if i >= 8 else "still degenerate")
PYEOF
