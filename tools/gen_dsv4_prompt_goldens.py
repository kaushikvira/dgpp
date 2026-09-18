#!/usr/bin/env python3
"""Generates tests/data/dsv4_prompt_goldens.jsonl — the differential goldens
for the DeepSeek-V4-Flash prompt renderer (docs/deepseek_v4_flash_plan.md G6).

The GOLDEN SOURCE is the checkpoint's own encoder, `encoding/encoding_dsv4.py`
in the local checkpoint (no chat_template.jinja ships with this model),
applied to the SERVICE'S template globals: "messages" (OpenAI messages:
system / user / assistant with reasoning_content and tool_calls / tool with
tool_call_id), optional "tools" (the OpenAI array), "reasoning_effort"
(the reference's levels: "low" is the default and adds nothing, "high" and
"max" prepend their prompt), "clear_thinking" and "enable_thinking". The
mapping from the globals to the reference's call is the renderer's contract,
written here once:
  * tools attach to the first system message ("tools" field), or to an empty
    one inserted at the front;
  * enable_thinking false -> thinking_mode "chat", else "thinking";
  * clear_thinking -> drop_thinking (default true);
  * reasoning_effort: passed through (the reference asserts low/high/max);
    absent -> the reference's default (low).
Each case records the render and the tokenizer ids of the render (HF
tokenizers on the checkpoint's tokenizer.json). The header carries the
encoder's FNV-1a-64 hash and the tokenizer.json's FNV-1a-64; the C++ gate
refuses a different tokenizer.json.

Regenerating (a venv with tokenizers installed):
  python3 tools/gen_dsv4_prompt_goldens.py \
      [--model-dir /data/models/DeepSeek-V4-Flash-0731] \
      [--out tests/data/dsv4_prompt_goldens.jsonl]
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path


def fnv1a64(data: bytes) -> int:
    h = 1469598103934665603
    for b in data:
        h = (h ^ b) * 1099511628211 & 0xFFFFFFFFFFFFFFFF
    return h


def load_encoder(model_dir: Path):
    path = model_dir / "encoding" / "encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("encoding_dsv4", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, fnv1a64(path.read_bytes())


def weather_tool():
    return {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city (城市天气查询).",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer", "minimum": 0}}, "required": ["city"]}}}


def search_tool():
    return {"type": "function", "function": {
        "name": "search",
        "description": "Search the web.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}}


def namespaced_tool():
    return {"type": "function", "function": {
        "namespace": "data",
        "name": "lookup",
        "description": "Look up a fact in the data store.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string"}}, "required": ["key"]}}}


def msg(role, content, **kw):
    d = {"role": role, "content": content}
    d.update(kw)
    return d


def tool_call(cid, name, arguments):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def cases():
    C = []
    C.append(("simple_user", {"messages": [
        msg("user", "The capital of France is")]}))
    C.append(("system_user", {"messages": [
        msg("system", "You are a concise geography assistant."),
        msg("user", "The capital of France is")]}))
    C.append(("chat_mode", {"messages": [
        msg("system", "You are a helpful assistant."),
        msg("user", "Hi there")], "enable_thinking": False}))
    C.append(("effort_low", {"messages": [msg("user", "Hi")],
                             "reasoning_effort": "low"}))
    C.append(("effort_high", {"messages": [msg("user", "Hi")],
                              "reasoning_effort": "high"}))
    C.append(("effort_max_system", {"messages": [
        msg("system", "Be brief."), msg("user", "Hi")],
        "reasoning_effort": "max"}))
    C.append(("multi_turn_drop_thinking", {"messages": [
        msg("system", "You are a helpful assistant."),
        msg("user", "What is 2+2?"),
        msg("assistant", "4.", reasoning_content="2+2 is 4."),
        msg("user", "And 3+3?"),
        msg("assistant", "6.", reasoning_content="3+3 is 6.")]}))
    C.append(("multi_turn_keep_thinking", {"messages": [
        msg("user", "Hello"),
        msg("assistant", "Hi!", reasoning_content="The user said hello."),
        msg("user", "How are you?")], "clear_thinking": False}))
    C.append(("multi_turn_chat_mode", {"messages": [
        msg("user", "Hello"),
        msg("assistant", "Hi!", reasoning_content="ignored in chat mode"),
        msg("user", "How are you?")], "enable_thinking": False}))
    C.append(("mid_system", {"messages": [
        msg("system", "Root."), msg("user", "a"), msg("assistant", "b"),
        msg("system", "Override."), msg("user", "c")]}))
    C.append(("tools_no_system", {"messages": [msg("user", "Weather in Paris?")],
                                  "tools": [weather_tool()]}))
    C.append(("tools_with_system", {"messages": [
        msg("system", "Use tools when helpful."),
        msg("user", "Weather in Paris?")],
        "tools": [weather_tool(), search_tool()]}))
    C.append(("tool_call_and_result", {"messages": [
        msg("system", "Use tools."),
        msg("user", "Weather in Paris?"),
        msg("assistant", "",
            tool_calls=[tool_call("call_1", "get_weather",
                                  '{"city": "Paris", "days": 1}')]),
        msg("tool", "Sunny, 18C.", tool_call_id="call_1"),
        msg("assistant", "It is sunny in Paris.")],
        "tools": [weather_tool()]}))
    C.append(("parallel_calls_results_reordered", {"messages": [
        msg("user", "Weather in Paris and Rome?"),
        msg("assistant", "", tool_calls=[
            tool_call("call_1", "get_weather", '{"city": "Paris"}'),
            tool_call("call_2", "get_weather", '{"city": "Rome"}')]),
        msg("tool", "Rome: rainy.", tool_call_id="call_2"),
        msg("tool", "Paris: sunny.", tool_call_id="call_1"),
        msg("assistant", "Paris is sunny, Rome is rainy.")],
        "tools": [weather_tool()]}))
    C.append(("namespaced_call", {"messages": [
        msg("user", "Look it up."),
        msg("assistant", "",
            tool_calls=[tool_call("call_1", "data::lookup",
                                  '{"key": "x"}')]),
        msg("tool", "found.", tool_call_id="call_1"),
        msg("assistant", "It is x.")],
        "tools": [namespaced_tool()]}))
    C.append(("call_args_object_and_double_encoded", {"messages": [
        msg("user", "Go"),
        msg("assistant", "",
            tool_calls=[tool_call("call_1", "search", '{"query": "pi"}')]),
        msg("tool", "3.14", tool_call_id="call_1"),
        msg("assistant", "pi is 3.14")],
        "tools": [search_tool()]}))
    C.append(("content_parts", {"messages": [
        msg("user", [{"type": "text", "text": "第 一 条"},
                     {"type": "text", "text": "and a second part"}])]}))
    C.append(("assistant_last_no_header", {"messages": [
        msg("user", "Hi"),
        msg("assistant", "Hello!", reasoning_content="greeting")]}))
    C.append(("empty_content", {"messages": [msg("user", "")]}))
    C.append(("unicode_and_code", {"messages": [
        msg("system", "Répondez en français, s'il vous plaît."),
        msg("user", "def f(x):\n    return x * 2  # double"),
        msg("assistant", "Voici la fonction.",
            reasoning_content="simple")]}))
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/data/models/DeepSeek-V4-Flash-0731")
    ap.add_argument("--out", default="tests/data/dsv4_prompt_goldens.jsonl")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    enc, encoder_hash = load_encoder(model_dir)
    tok_bytes = (model_dir / "tokenizer.json").read_bytes()

    import tokenizers
    tok = tokenizers.Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    cs = cases()
    lines = [json.dumps({
        "model": "DeepSeek-V4-Flash-0731",
        "encoder_hash": format(encoder_hash, "012x"),
        "tokenizer_hash": format(fnv1a64(tok_bytes), "012x"),
        "tokenizers": tokenizers.__version__,
        "cases": len(cs),
    })]
    for name, kwargs in cs:
        messages = copy.deepcopy(kwargs["messages"])
        thinking = "chat" if kwargs.get("enable_thinking") is False else "thinking"
        drop = kwargs.get("clear_thinking", True)
        effort = kwargs.get("reasoning_effort")
        if kwargs.get("tools"):
            sys_msgs = [x for x in messages if x.get("role") == "system"]
            target = sys_msgs[0] if sys_msgs else None
            if target is None:
                messages.insert(0, {"role": "system", "content": ""})
                target = messages[0]
            target["tools"] = kwargs["tools"]
        # The reference's user content is a string (merge_tool_messages
        # rebuilds content_blocks from it, dropping any content_blocks we
        # pass); the service's OpenAI text parts therefore map to a single
        # string. The renderer's content_text joins text parts with "\n\n"
        # (src/text/dsv4_prompt.cpp), so the golden must use the same join
        # to stay byte-exact.
        for x in messages:
            c = x.get("content")
            if isinstance(c, list):
                x["content"] = "\n\n".join(
                    p.get("text", "") if p.get("type") == "text"
                    else json.dumps(p, ensure_ascii=False) for p in c)
        render = enc.encode_messages(
            messages, thinking, drop_thinking=drop, reasoning_effort=effort)
        lines.append(json.dumps({
            "name": name, "kwargs": kwargs, "render": render,
            "ids": tok.encode(render).ids,
        }, ensure_ascii=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines, encoder {encoder_hash:012x}, "
          f"tokenizer {fnv1a64(tok_bytes):012x})")


if __name__ == "__main__":
    main()
