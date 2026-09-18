#pragma once
// The DeepSeek-V4-Flash prompt renderer (docs/deepseek_v4_flash_plan.md G6):
// the checkpoint ships no chat_template.jinja — `encoding/encoding_dsv4.py`
// (its `encode_messages`, text-only) is the reference, and this is that
// function over the service's template globals. The prompt format
// (encoding/README.md):
//   <｜begin▁of▁sentence｜>[effort prefix]{system}<｜User｜>{user}
//   <｜Assistant｜><think>{reasoning}</think>{content}[\n\n<｜DSML｜ calls>...]<｜end▁of▁sentence｜>
//   ... the generation header after the last user (or mid-conversation
//   system) message: <｜Assistant｜> then <think> (thinking mode) or
//   </think> (chat mode).
// NOTE the v4 differences from the V4.1 renderer (this is a TWIN of
// dsv41_prompt with those deltas):
//   * the system message carries NO <｜System｜> token — its content (and the
//     effort prefix before it) rides the BOS directly (the reference's
//     system_msg_template is bare {content});
//   * `reasoning_effort` takes the reference's string levels — "low" (the
//     default, adds nothing), "high" and "max" (each prepends its prompt
//     at the conversation's start in thinking mode); a numeric budget or
//     any other level is refused;
//   * OpenAI content parts (a content array) map to the reference's
//     string content as the "\n\n"-joined text parts (content_text);
//     anything but a text part is rendered as its JSON.
// Tool results are <tool_result>...</tool_result> blocks inside the user
// turn (a `tool` message merges into the preceding user turn or opens
// one; consecutive user messages merge likewise), ordered by the previous
// assistant turn's call ids; tools are the "## Tools" block appended to
// the system message (the service's `tools` global attaches to the first
// system message, or to an empty one inserted at the front); earlier
// turns' reasoning is dropped in thinking mode unless the conversation
// carries tools (`clear_thinking`, default true); `enable_thinking` false
// is the reference's chat mode.
//
// Validation: tests/host/dsv4_prompt_test.cpp renders
// tests/data/dsv4_prompt_goldens.jsonl (tools/gen_dsv4_prompt_goldens.py:
// the checkpoint's own encoder over the same globals) byte-exact, and the
// tokenizer ids of every render; tests/host/dsv4_tokenizer_test.cpp is the
// standalone encoder gate on tests/data/dsv4_tokenizer_goldens.jsonl.
#include <cstdint>
#include <string>
#include <string_view>

#include "loaders/minijson.hpp"

namespace dgpp::text {

class Dsv4Prompt {
 public:
  // The service's globals: "messages" (the normalized OpenAI array; roles
  // system / user / assistant / tool), optional "tools" (the OpenAI
  // array), "reasoning_effort" (a string: low, high or max),
  // "clear_thinking" and "enable_thinking" (bools). Throws
  // std::runtime_error naming the offending field on anything the
  // reference refuses (an unknown role, image content, a namespace
  // conflict, a reasoning_effort outside low/high/max).
  static std::string render(const minijson::Value& globals);
  // One OpenAI tool entry's name as the schema lists it and the model
  // writes it in a DSML invoke: "namespace::name" when a namespace is
  // present (the entry's or the function's; a "ns::name" in the name
  // itself must agree), else the bare name. Throws std::invalid_argument
  // on a malformed entry (the renderer's own refusals).
  static std::string qualified_tool_name(const minijson::Value& tool);
  // The knobs this renderer reads (the service's chat_template_kwargs gate).
  static bool reads(std::string_view name);
  // The renderer's revision (the prefix cache's key stands on it, as on a
  // Jinja template's source hash): bumped whenever the render changes.
  static uint64_t source_hash();

  static constexpr const char* kBos = "<｜begin▁of▁sentence｜>";
  static constexpr const char* kEos = "<｜end▁of▁sentence｜>";
  static constexpr const char* kUser = "<｜User｜>";
  static constexpr const char* kAssistant = "<｜Assistant｜>";
  static constexpr const char* kThinkOpen = "<think>";
  static constexpr const char* kThinkClose = "</think>";
  static constexpr const char* kDsml = "｜DSML｜";
};

}  // namespace dgpp::text
