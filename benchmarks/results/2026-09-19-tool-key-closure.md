# Tool-call argument keys: closed by default, never repeated — 2026-09-19

Two defects in the constrained tool-call decoding surfaced together on a
served request, and both are silent: the turn is well-formed, the call
parses, and the client sees arguments the model never declared.

## What the engine returned

A Qwen3.8 request (the Qwen XML tool-call format, constrained decoding on,
`tool_choice: required`) with one tool whose parameters were
`{"properties": {"path": {"type": "string"}, "limit": {"type": "number"}},
"required": ["path"]}` — the ordinary client shape, no
`additionalProperties` — produced

```json
{"limit": 5, "content": true, "offset": 1, "limit": 5, "filePath": "/etc/hostname"}
```

Adding only `"additionalProperties": false` to the same schema produced

```json
{"path": "/etc/hostname", "limit": 5}
```

The wire capture came from the running service (`dgpp-serve
0.1.0+gf7c06630fd7b`, git `f7c06630fd7b`) on the two-rank Qwen3.8
deployment, `config: model=nvidia/Qwen3.8-Flash-Next-NVFP4 world=2
fabric=29970 journal=29971 conc=8 kv=524288 kvdt=bf16 ngt=mmap dw=fp8
bfw=checkpoint pf=bounded emsh=replicated maxtok=65536 queue=8 eos=1
graph=1 mtp=1 mtpd=3 ... batchmin=2 cand=128 pcgib=4 adm=full win=256`,
rank 0 log `serve_r0.log`. The same failure class appears without any schema
trick at all: `limit` is written twice in the first object, which no JSON
object may do.

## Why it is an engine defect

The `/v1/chat/completions` contract promises "every call is well-formed by
constrained decoding ... a declared function name, the closed keys under
`additionalProperties: false`". The qualifier made the guarantee vacuous on
the common path: a client that declares `properties` and omits
`additionalProperties` — which is what clients do, JSON Schema's default
being open — left the `<parameter=NAME>` slot **free text**. The grammar
then let the model write whatever name its prior held for that tool, and
`grammar_tool_from_function` recorded no key set to constrain it. Two
independent consequences of one open slot:

1. **Undeclared and invented names** (`content`, `offset`, `filePath` in
   the capture above) — the model falling back on memorized argument names
   for a familiar tool.
2. **A repeated name.** The Qwen XML parser wrote every `(key, value)` pair
   into the object verbatim, so `"limit": 5` appeared twice; the DSML
   parser has always rejected a repeat (`a duplicate parameter`), so the
   two formats disagreed about what a valid call is.

The open *validation* default is not an argument for an open *decoding*
grammar. Validation accepts what the model produced; decoding decides what
it can produce, and the engine advertises the second. Comparable engines
(vLLM's Qwen tool parsers, OpenAI's structured outputs, llama.cpp's
grammar sampling) pin the key slot to the declared properties; JSON
Schema's open default says nothing about what a decoder should allow.

## The change

- `grammar_tool_from_function` closes the key set whenever the schema
  declares `properties` and does not opt out with an explicit
  `additionalProperties: true`. Any other value — `false` as before, or a
  subschema this grammar cannot type per name — closes it. The explicit
  opt-out keeps the free key and emits one `notes` line naming the tool
  (`additionalProperties: true on '<tool>': the keys are free text, not
  the declared properties`), which
  `log_tool_schema_note` prints at INFO once, then at debug.
- `parse_qwen_block` rejects a repeated `<parameter=NAME>` — the parser's
  ledger, matching DSML. The grammar already offers a declared key at most
  once per call; the parser is what makes a duplicate impossible for a key
  the grammar cannot name (the explicit opt-out, and a request whose
  grammar is inactive).

## Validation

The two regressions were written first and fail on the unmodified tree:

```text
$ DGPP_TEST_FILTER=open_schema_closes ./build-ci/unit_tests
[FAIL] tool_grammar_open_schema_closes_the_parameter_names: an unspecified additionalProperties closes the declared names
1 tests, 1 failed

$ DGPP_TEST_FILTER=RepeatedParameter ./build-ci/unit_tests
[FAIL] tool_parser_rejects_aRepeatedParameterName: a repeated parameter is not a call
1 tests, 1 failed
```

After the change, the same commands report `[ OK ] ... 1 tests, 0 failed`.
`tool_grammar_open_schema_closes_the_parameter_names` walks the Qwen XML
grammar from a schema that declares `city` and `days` and no
`additionalProperties`: no undeclared name starts a key, a declared name is
offered once, `days` (a number) opens the JSON machine, and
`additionalProperties: true` keeps the free slot with its note.
`tool_parser_rejects_aRepeatedParameterName` feeds a block with `city`
twice — it becomes content, not a call — and checks the distinct-name and
DSML paths alongside it.

Host suite, per CONTRIBUTING (no GPU, no fabric, production serving left
running):

```text
$ cmake --build build-ci -j 4 --target unit_tests http_server_test serve_test fabric_serve_test scheduler_test roster_check
$ ctest --test-dir build-ci -L host -LE checkpoint --output-on-failure
100% tests passed, 0 tests failed out of 16
```

`unit_tests` alone: `213 tests, 0 failed`. The service gate's grammar
assertion for a schema without `additionalProperties` now requires the
closed key set `{"city", "days"}` where it previously required
`!constrain_keys`.

The chat-template golden gates could not run here: `chat_template_test`
exits 2 with "checkpoint unavailable" for every model, because this box has
no HF cache for them. They are prompt-rendering goldens, and key closure
changes the decoding grammar, not rendering, so they are not expected to
move. Checked directly instead: the four `tests/data/*chat_template_goldens.jsonl`
corpora contain schemas with `properties` and no `additionalProperties`,
and **zero** golden tool calls use a key outside the declared properties,
so no golden's expected arguments depend on the old open key slot.

## Not covered here

The post-change behaviour on the wire (a real Qwen3.8 decode with a tool
schema on the common path) was not re-measured: production serving owns the
GPUs, and CONTRIBUTING forbids GPU/fabric runs alongside it. The evidence
above is host-side, and the wire evidence is the before-state capture in
the issue.
