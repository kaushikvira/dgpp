// M6 6g: the tool-call grammar over a fake vocabulary (host-only, always
// runs). Pins: the masks of every state (which ids a position allows and
// the allowed count the sampler treats as the vocabulary), the name/key
// automaton over token texts (any tokenization of a name, nothing else,
// prefix-sharing names), the modes (required / named / auto-single /
// forbid, parallel or not), the EOS discipline while a call is owed, a
// disallowed id killing the grammar, and a complete valid turn being
// accepted position by position.
#include <algorithm>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/tool_grammar.hpp"

namespace {

using dgpp::text::ChatMarker;
using dgpp::text::ChatMarkers;
using dgpp::text::GrammarArg;
using dgpp::text::GrammarSpec;
using dgpp::text::GrammarState;
using dgpp::text::GrammarTool;
using dgpp::text::GrammarVocab;
using dgpp::text::TokenMask;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// The fake vocabulary: ids 0..255 are single bytes, then a few multi-byte
// word tokens, the eight markers, and three EOS ids; vocab_size pads it.
constexpr int64_t kGet = 256, kWeather = 257, kGetWeather = 258,
                  kUnderscore = 259, kWea = 260, kTher = 261, kCity = 262,
                  kGetT = 263, kIme = 264;
constexpr int64_t kThinkOpen = 300, kThinkClose = 301, kToolOpen = 302,
                  kToolClose = 303, kKeyOpen = 304, kKeyClose = 305,
                  kValueOpen = 306, kValueClose = 307;
constexpr int64_t kEosText = 310, kEosUser = 311, kEosObs = 312;
constexpr int kVocab = 320;

GrammarVocab fake_vocab() {
  std::vector<std::string> texts(static_cast<size_t>(kVocab));
  for (int b = 0; b < 256; ++b) texts[static_cast<size_t>(b)] = std::string(1, static_cast<char>(b));
  texts[kGet] = "get";
  texts[kWeather] = "_weather";
  texts[kGetWeather] = "get_weather";
  texts[kUnderscore] = "_";
  texts[kWea] = "wea";
  texts[kTher] = "ther";
  texts[kCity] = "city";
  texts[kGetT] = "get_t";
  texts[kIme] = "ime";
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  m.arg_key_open = ChatMarker{kKeyOpen, "<arg_key>"};
  m.arg_key_close = ChatMarker{kKeyClose, "</arg_key>"};
  m.arg_value_open = ChatMarker{kValueOpen, "<arg_value>"};
  m.arg_value_close = ChatMarker{kValueClose, "</arg_value>"};
  return GrammarVocab(std::move(texts), m, {kEosText, kEosUser, kEosObs}, kVocab,
                      kEosObs);
}

GrammarSpec spec_of(GrammarSpec::Mode mode, bool parallel = true,
                    const std::string& named = "") {
  GrammarSpec s;
  s.mode = mode;
  s.parallel = parallel;
  s.named = named;
  GrammarTool weather;
  weather.name = "get_weather";
  weather.constrain_keys = true;
  weather.keys = {"city", "days"};
  GrammarTool time;
  time.name = "get_time";  // shares the "get_" prefix
  time.constrain_keys = false;
  GrammarTool bare;
  bare.name = "ping";
  bare.constrain_keys = true;  // closed and empty: no arguments at all
  s.tools = {weather, time, bare};
  return s;
}

std::vector<int64_t> allowed_ids(const GrammarState& g) {
  TokenMask m;
  g.mask(&m);
  std::vector<int64_t> out;
  if (!m.constrained()) return out;
  for (int64_t id = 0; id < kVocab; ++id)
    if (m.allows(id)) out.push_back(id);
  require(static_cast<int>(out.size()) == m.allowed,
          "allowed count equals the set bits: " + std::to_string(out.size()) +
              " vs " + std::to_string(m.allowed));
  return out;
}

bool same(std::vector<int64_t> a, std::vector<int64_t> b) {
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  return a == b;
}

std::string show(const std::vector<int64_t>& v) {
  std::string s = "[";
  for (size_t i = 0; i < v.size() && i < 12; ++i)
    s += (i ? "," : "") + std::to_string(v[i]);
  if (v.size() > 12) s += ",…";
  return s + "]";
}

void feed(GrammarState& g, const std::vector<int64_t>& ids) {
  for (const int64_t id : ids) {
    require(g.allows(id), "id " + std::to_string(id) + " allowed in state " +
                              g.state_name());
    g.advance(id);
  }
}

// ---- the Qwen3.8 XML format --------------------------------------
GrammarVocab qwen_vocab() {
  std::vector<std::string> texts(static_cast<size_t>(kVocab));
  for (int b = 0; b < 256; ++b) texts[static_cast<size_t>(b)] = std::string(1, static_cast<char>(b));
  texts[kGet] = "get";
  texts[kWeather] = "_weather";
  texts[kGetWeather] = "get_weather";
  texts[kUnderscore] = "_";
  texts[kWea] = "wea";
  texts[kTher] = "ther";
  texts[kCity] = "city";
  texts[kGetT] = "get_t";
  texts[kIme] = "ime";
  texts[kThinkOpen] = "<think>";
  texts[kThinkClose] = "</think>";
  texts[kToolOpen] = "<tool_call>";
  texts[kToolClose] = "</tool_call>";
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.tool_call_open = ChatMarker{kToolOpen, "<tool_call>"};
  m.tool_call_close = ChatMarker{kToolClose, "</tool_call>"};
  return GrammarVocab(std::move(texts), m, {kEosText, kEosUser}, kVocab, kEosUser);
}
std::vector<int64_t> bytes_of(const std::string& s) {
  std::vector<int64_t> out;
  for (const char c : s) out.push_back(static_cast<unsigned char>(c));
  return out;
}

DGPP_TEST(tool_grammar_qwen_required_call_walks_the_xml_shape) {
  const GrammarVocab vocab = qwen_vocab();
  require(vocab.usable() && vocab.markers().tool_format() == dgpp::text::ToolFormat::kQwenXml,
          "the two-marker vocabulary is the Qwen format");
  GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kRequired), /*prompt_opens_thinking=*/false);
  // A call is owed: only <tool_call> opens the turn.
  // Text may precede the call (the Qwen format's optional reasoning); the
  // turn may not end while the call is owed.
  require(g.allows('H') && g.allows(kToolOpen) && !g.allows(kEosUser) && !g.allows(kEosText) &&
              !g.allows(kToolClose),
          "top: free text or <tool_call>, never EOS while owed");
  g.advance('H');
  g.advance(kToolOpen);
  // "\n<function=": any tokenization of the literal, nothing else.
  require(g.allows('\n') && !g.allows('x') && !g.allows(kToolClose) && !g.allows(kEosUser),
          "q-name starts with the literal's newline");
  feed(g, bytes_of("\n<function="));
  // The name: get_weather / get_time / ping over token texts.
  require(g.allows(kGet) && g.allows(kGetWeather) && g.allows('p') && g.allows(kGetT) && !g.allows('x'),
          "names over token texts");
  feed(g, {kGet, kWeather});
  require(g.allows('>') && !g.allows('\n'), "the name closes with '>\\n'");
  feed(g, bytes_of(">\n"));
  require(std::string(g.state_name()) == "q-key-or-close", std::string("after the name: ") + g.state_name());
  // Closed keys city/days, or the close: everything starts with '<'.
  require(same(allowed_ids(g), {'<'}), "key-or-close: '<' only: " + show(allowed_ids(g)));
  feed(g, bytes_of("<parameter="));
  require(g.allows('c') && g.allows('d') && !g.allows('x'), "closed keys city/days");
  feed(g, bytes_of("city>\n"));
  require(std::string(g.state_name()) == "q-value", "a free value follows the key");
  // The free value: anything but the markers, the think markers and EOS.
  const std::vector<int64_t> v = allowed_ids(g);
  require(!v.empty() && std::find(v.begin(), v.end(), kToolClose) == v.end() &&
              std::find(v.begin(), v.end(), kEosUser) == v.end() &&
              std::find(v.begin(), v.end(), kThinkOpen) == v.end() &&
              std::find(v.begin(), v.end(), 'P') != v.end(),
          "free value: text only");
  feed(g, bytes_of("Paris\n</parameter>\n"));
  require(std::string(g.state_name()) == "q-key-or-close", "the terminator closes the value");
  // city is used: after "<parameter=" only days remains.
  feed(g, bytes_of("<parameter="));
  require(g.allows('d') && !g.allows('c'), "a closed key is offered once");
  feed(g, bytes_of("days>\n3\n</parameter>\n"));
  // Every key used: only the close remains.
  require(g.allows('<') && !g.allows('\n'), "only </function> remains");
  feed(g, bytes_of("</function>"));
  require(!g.allows('p'), "after </function> the newline");
  feed(g, bytes_of("\n"));
  require(same(allowed_ids(g), {kToolClose}), "q-close: </tool_call> only: " + show(allowed_ids(g)));
  g.advance(kToolClose);
  // Required (parallel): free text, another call, or the turn's end.
  require(g.allows(kToolOpen) && g.allows(kEosUser) && g.allows('x') && !g.allows(kToolClose),
          "after the call: text, another call or EOS");
  g.advance(kEosUser);
  require(std::string(g.state_name()) == "done", "done after EOS");
}

DGPP_TEST(tool_grammar_qwen_free_keys_typed_values_and_named_single) {
  const GrammarVocab vocab = qwen_vocab();
  // get_time: an open key set — free text through ">\n".
  GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kNamed, false, "get_time"), false);
  g.advance(kToolOpen);
  feed(g, bytes_of("\n<function="));
  require(!g.allows(kGetWeather) && g.allows(kGetT), "named: only get_time");
  feed(g, {kGetT, kIme});
  feed(g, bytes_of(">\n<parameter="));
  require(std::string(g.state_name()) == "q-free-key", "an open key set is free text");
  require(g.allows('t') && !g.allows(kToolClose) && !g.allows(kEosUser), "free key text");
  feed(g, bytes_of("tz>\nUTC\n</parameter>\n</function>\n"));
  g.advance(kToolClose);
  // Named: exactly one call, then the turn ends.
  require(same(allowed_ids(g), {kEosUser}), "named single: EOS only: " + show(allowed_ids(g)));
  // Typed values: a JSON integer and an enum string.
  GrammarSpec spec = spec_of(GrammarSpec::Mode::kRequired);
  GrammarArg days;
  days.key = "days";
  days.kind = GrammarArg::Kind::kJson;
  days.schema = R"({"type": "integer"})";
  GrammarArg unit;
  unit.key = "unit";
  unit.kind = GrammarArg::Kind::kText;
  unit.texts = {"celsius", "fahrenheit"};
  spec.tools[0].keys = {"days", "unit"};
  spec.tools[0].args = {days, unit};
  GrammarState t(&vocab, spec, false);
  t.advance(kToolOpen);
  feed(t, bytes_of("\n<function=get_weather>\n<parameter=days>\n"));
  // A JSON text may open with whitespace; a letter never.
  require(t.allows('3') && !t.allows('x') && !t.allows('<'), "a JSON integer: digits first");
  feed(t, bytes_of("3"));
  require(t.allows('4') && t.allows('\n'), "more digits, or the terminator once complete");
  feed(t, bytes_of("\n"));
  require(t.allows('<') && !t.allows('4') && !t.allows('\n'), "inside the terminator only its bytes");
  feed(t, bytes_of("</parameter>\n<parameter=unit>\n"));
  require(t.allows('c') && t.allows('f') && !t.allows('k'), "an enum value: its texts");
  feed(t, bytes_of("celsius"));
  require(t.allows('\n') && !t.allows('c'), "the enum text then its terminator");
  feed(t, bytes_of("\n</parameter>\n</function>\n"));
  t.advance(kToolClose);
  require(t.active() && std::string(t.state_name()) == "top", "the call closed cleanly");
}

// A schema that declares properties and does not opt out with an explicit
// `additionalProperties: true` closes the parameter names: an open name slot
// is free text, which is what lets the model write an undeclared or a
// repeated name (the wire evidence in
// benchmarks/results/2026-09-19-tool-key-closure.md).
DGPP_TEST(tool_grammar_open_schema_closes_the_parameter_names) {
  const GrammarVocab vocab = qwen_vocab();
  const dgpp::minijson::ParseResult def = dgpp::minijson::parse(
      R"({"name":"get_weather","parameters":{"type":"object","properties":)"
      R"({"city":{"type":"string"},"days":{"type":"number"}},"required":["city"]}})");
  std::vector<std::string> notes;
  GrammarSpec spec;
  spec.mode = GrammarSpec::Mode::kRequired;
  spec.tools.push_back(dgpp::text::grammar_tool_from_function(def.root, nullptr, &notes));
  require(spec.tools[0].constrain_keys &&
              spec.tools[0].keys == std::vector<std::string>{"city", "days"} && notes.empty(),
          "an unspecified additionalProperties closes the declared names");
  GrammarState g(&vocab, spec, /*prompt_opens_thinking=*/false);
  g.advance(kToolOpen);
  feed(g, bytes_of("\n<function=get_weather>\n<parameter="));
  require(std::string(g.state_name()) == "q-key-or-close",
          std::string("after <parameter=: ") + g.state_name());
  require(g.allows('c') && g.allows('d') && !g.allows('l') && !g.allows('x'),
          "only the declared names start a key: " + show(allowed_ids(g)));
  feed(g, bytes_of("city>\nRome\n</parameter>\n<parameter="));
  require(g.allows('d') && !g.allows('c'),
          "a declared name is offered once: " + show(allowed_ids(g)));
  // days is a number: its value is the JSON machine, not free text.
  feed(g, bytes_of("days>\n"));
  require(g.allows('5') && !g.allows('x'),
          "a number property opens the JSON machine: " + show(allowed_ids(g)));
  feed(g, bytes_of("5\n</parameter>\n</function>\n"));
  require(same(allowed_ids(g), {kToolClose}),
          "every declared name used: only </tool_call>: " + show(allowed_ids(g)));
  g.advance(kToolClose);
  require(g.active() && std::string(g.state_name()) == "top", "the call closed cleanly");
  // An explicit opt-out keeps the free name slot, and says so once.
  const dgpp::minijson::ParseResult open =
      dgpp::minijson::parse(R"({"name":"get_weather","parameters":{"type":"object","properties":)"
                            R"({"city":{"type":"string"}},"additionalProperties":true}})");
  std::vector<std::string> free_notes;
  const GrammarTool f = dgpp::text::grammar_tool_from_function(open.root, nullptr, &free_notes);
  require(!f.constrain_keys && f.keys.empty() && free_notes.size() == 1 &&
              free_notes[0].find("get_weather") != std::string::npos &&
              free_notes[0].find("additionalProperties") != std::string::npos,
          "an explicit additionalProperties:true keeps the keys free, noted");
}

// ---- the DeepSeek-V4.1 DSML format --------------------------------
// One marker id (the tag token, empty text like every special token) inside
// text tags: "<" TAG " calls>\n", "<" TAG " invoke name=\"NAME\">\n", the
// parameters "<" TAG " parameter name=\"K\" string=\"true|false\">" V "</" TAG
// " parameter>\n", "</" TAG " invoke>\n", "</" TAG " calls>" then EOS.
constexpr int64_t kDsmlTag = 308;
constexpr int64_t kQuotedLt = 265;
GrammarVocab dsml_vocab() {
  std::vector<std::string> texts(static_cast<size_t>(kVocab));
  for (int b = 0; b < 256; ++b) texts[static_cast<size_t>(b)] = std::string(1, static_cast<char>(b));
  texts[kGet] = "get";
  texts[kWeather] = "_weather";
  texts[kGetWeather] = "get_weather";
  texts[kUnderscore] = "_";
  texts[kWea] = "wea";
  texts[kTher] = "ther";
  texts[kCity] = "city";
  texts[kGetT] = "get_t";
  texts[kIme] = "ime";
  texts[kThinkOpen] = "<think>";
  texts[kThinkClose] = "</think>";
  texts[kQuotedLt] = "\"x<";
  ChatMarkers m;
  m.think_open = ChatMarker{kThinkOpen, "<think>"};
  m.think_close = ChatMarker{kThinkClose, "</think>"};
  m.dsml = ChatMarker{kDsmlTag, "｜DSML｜"};
  return GrammarVocab(std::move(texts), m, {kEosText}, kVocab, kEosText);
}

DGPP_TEST(tool_grammar_dsml_required_call_walks_the_tagged_shape) {
  const GrammarVocab vocab = dsml_vocab();
  require(vocab.usable() && vocab.markers().tool_format() == dgpp::text::ToolFormat::kDsml,
          "the tag-token vocabulary is the DSML format");
  GrammarState g(&vocab, spec_of(GrammarSpec::Mode::kRequired), /*prompt_opens_thinking=*/false);
  // The top is free text (the content before the block); the tag needs
  // its "<"; the turn may not end while the call is owed.
  require(g.allows('H') && !g.allows(kDsmlTag) && !g.allows(kEosText), "top: text, no bare tag, no EOS while owed");
  feed(g, bytes_of("Sure.\n\n"));
  require(!g.allows(kDsmlTag), "the tag is not offered until a '<'");
  g.advance('<');
  require(g.allows(kDsmlTag) && g.allows('a'), "after '<': the tag (or more text)");
  g.advance('a');
  require(!g.allows(kDsmlTag), "text after the '<' withdraws the tag");
  feed(g, bytes_of(" <"));
  g.advance(kDsmlTag);
  require(std::string(g.state_name()) == "d-calls", std::string("after the tag: ") + g.state_name());
  require(same(allowed_ids(g), {' '}), "the block opens with ' calls>': " + show(allowed_ids(g)));
  feed(g, bytes_of(" calls>\n"));
  require(std::string(g.state_name()) == "d-invoke", "then an invoke");
  // "<" TAG " invoke name=\"" NAME "\">\n": the '<', the tag alone, the literal, the names.
  require(same(allowed_ids(g), {'<'}), "an invoke opens with '<'");
  g.advance('<');
  require(same(allowed_ids(g), {kDsmlTag}), "the tag alone after the '<' in the block: " + show(allowed_ids(g)));
  g.advance(kDsmlTag);
  require(!g.allows(kDsmlTag) && g.allows(' '), "no token runs across the tag");
  feed(g, bytes_of(" invoke name=\""));
  require(g.allows(kGet) && g.allows(kGetWeather) && g.allows('p') && g.allows(kGetT) && !g.allows('x'),
          "names over token texts");
  feed(g, {kGet, kWeather});
  feed(g, bytes_of("\">\n"));
  require(std::string(g.state_name()) == "d-param-or-close", std::string("after the name: ") + g.state_name());
  // Closed keys city/days, or the invoke's close: everything starts with '<'.
  require(same(allowed_ids(g), {'<'}), "param-or-close: '<' only");
  g.advance('<');
  require(g.allows(kDsmlTag) && g.allows('/') && !g.allows('x'), "the tag (a parameter) or '/' (the close)");
  g.advance(kDsmlTag);
  feed(g, bytes_of(" parameter name=\""));
  require(g.allows('c') && g.allows('d') && !g.allows('x'), "closed keys city/days");
  feed(g, bytes_of("city\" string=\""));
  require(std::string(g.state_name()) == "d-flag", "the string flag follows a key");
  require(g.allows('t') && g.allows('f'), "a free value: string true or false");
  feed(g, bytes_of("true\">"));
  require(std::string(g.state_name()) == "d-value", "a free value follows the flag");
  // The free value: anything but the markers, the think markers, EOS and
  // the tag — until a "</" makes the tag the closer's start.
  const std::vector<int64_t> v = allowed_ids(g);
  require(!v.empty() && std::find(v.begin(), v.end(), kDsmlTag) == v.end() &&
              std::find(v.begin(), v.end(), kEosText) == v.end() &&
              std::find(v.begin(), v.end(), kThinkOpen) == v.end() &&
              std::find(v.begin(), v.end(), 'P') != v.end(),
          "free value: text only");
  feed(g, bytes_of("Paris </b> <"));
  require(!g.allows(kDsmlTag), "a lone '<' in the value is text");
  g.advance('/');
  require(g.allows(kDsmlTag) && g.allows('x'), "after \"</\" the tag opens the closer (or the text goes on)");
  g.advance(kDsmlTag);
  require(same(allowed_ids(g), {' '}), "after the closer's tag: its literal only: " + show(allowed_ids(g)));
  feed(g, bytes_of(" parameter>\n"));
  require(std::string(g.state_name()) == "d-param-or-close", "the closer ends the value");
  // days is JSON-typed: the flag is forced false; the value a JSON integer.
  GrammarSpec typed = spec_of(GrammarSpec::Mode::kRequired);
  (void)typed;
  feed(g, bytes_of("<"));
  g.advance(kDsmlTag);
  feed(g, bytes_of(" parameter name=\""));
  require(g.allows('d') && !g.allows('c'), "a closed key is offered once");
  feed(g, bytes_of("days\" string=\"false\">3</"));
  g.advance(kDsmlTag);
  feed(g, bytes_of(" parameter>\n"));
  // Every key used: only the invoke's close remains.
  require(same(allowed_ids(g), {'<'}), "only the close remains");
  g.advance('<');
  require(same(allowed_ids(g), {'/'}), "the close's '/'");
  g.advance('/');
  require(same(allowed_ids(g), {kDsmlTag}), "the close's tag");
  g.advance(kDsmlTag);
  feed(g, bytes_of(" invoke>\n"));
  require(std::string(g.state_name()) == "d-invoke-or-close", std::string("after the invoke: ") + g.state_name());
  // Required (parallel): another invoke or the block's close, then EOS.
  g.advance('<');
  require(g.allows(kDsmlTag) && g.allows('/'), "another invoke or the close");
  g.advance('/');
  g.advance(kDsmlTag);
  feed(g, bytes_of(" calls>"));
  require(same(allowed_ids(g), {kEosText}), "after the block: EOS only: " + show(allowed_ids(g)));
  g.advance(kEosText);
  require(std::string(g.state_name()) == "done", "done after EOS");
}

DGPP_TEST(tool_grammar_dsml_typed_values_named_single_and_open_keys) {
  const GrammarVocab vocab = dsml_vocab();
  // Typed values: a JSON integer (the flag forced false) and an enum
  // string (forced true); named: exactly one invoke.
  GrammarSpec spec = spec_of(GrammarSpec::Mode::kNamed, false, "get_weather");
  GrammarArg days;
  days.key = "days";
  days.kind = GrammarArg::Kind::kJson;
  days.schema = R"({"type": "integer"})";
  GrammarArg unit;
  unit.key = "unit";
  unit.kind = GrammarArg::Kind::kText;
  unit.texts = {"celsius", "fahrenheit"};
  spec.tools[0].keys = {"days", "unit"};
  spec.tools[0].args = {days, unit};
  GrammarState t(&vocab, spec, false);
  t.advance('<');
  t.advance(kDsmlTag);
  feed(t, bytes_of(" calls>\n<"));
  t.advance(kDsmlTag);
  feed(t, bytes_of(" invoke name=\""));
  require(t.allows(kGetWeather) && !t.allows(kGetT) && !t.allows('p'), "named: only get_weather");
  feed(t, {kGetWeather});
  feed(t, bytes_of("\">\n<"));
  t.advance(kDsmlTag);
  feed(t, bytes_of(" parameter name=\"days\" string=\""));
  require(t.allows('f') && !t.allows('t'), "a JSON-typed value is not a string");
  feed(t, bytes_of("false\">"));
  require(t.allows('3') && !t.allows('x') && !t.allows('<'), "a JSON integer: digits first");
  feed(t, bytes_of("3"));
  require(t.allows('4') && t.allows('<') && !t.allows(kDsmlTag), "more digits, or the closer's '<' once complete");
  feed(t, bytes_of("<"));
  require(same(allowed_ids(t), {'/'}), "inside the closer only its bytes");
  t.advance('/');
  require(same(allowed_ids(t), {kDsmlTag}), "then the tag");
  t.advance(kDsmlTag);
  feed(t, bytes_of(" parameter>\n<"));
  t.advance(kDsmlTag);
  feed(t, bytes_of(" parameter name=\"unit\" string=\""));
  require(t.allows('t') && !t.allows('f'), "an enum value is a string");
  feed(t, bytes_of("true\">"));
  require(t.allows('c') && t.allows('f') && !t.allows('k'), "an enum value: its texts");
  feed(t, bytes_of("celsius"));
  require(t.allows('<') && !t.allows('c'), "the enum text then its closer");
  feed(t, bytes_of("</"));
  t.advance(kDsmlTag);
  feed(t, bytes_of(" parameter>\n</"));
  t.advance(kDsmlTag);
  feed(t, bytes_of(" invoke>\n"));
  // Named: no second invoke — the block closes.
  require(same(allowed_ids(t), {'<'}), "the block's close only");
  t.advance('<');
  require(same(allowed_ids(t), {'/'}), "no second invoke under named");
  feed(t, bytes_of("/"));
  t.advance(kDsmlTag);
  feed(t, bytes_of(" calls>"));
  require(same(allowed_ids(t), {kEosText}), "named single: EOS only");
  // An open key set (get_time): free text through "\" string=\"".
  GrammarState o(&vocab, spec_of(GrammarSpec::Mode::kNamed, false, "get_time"), false);
  o.advance('<');
  o.advance(kDsmlTag);
  feed(o, bytes_of(" calls>\n<"));
  o.advance(kDsmlTag);
  feed(o, bytes_of(" invoke name=\""));
  feed(o, {kGetT, kIme});
  feed(o, bytes_of("\">\n<"));
  o.advance(kDsmlTag);
  feed(o, bytes_of(" parameter name=\""));
  require(std::string(o.state_name()) == "d-free-key", "an open key set is free text");
  require(o.allows('t') && !o.allows(kDsmlTag) && !o.allows(kEosText), "free key text");
  feed(o, bytes_of("tz\" string=\"true\">UTC</"));
  o.advance(kDsmlTag);
  feed(o, bytes_of(" parameter>\n</"));
  o.advance(kDsmlTag);
  feed(o, bytes_of(" invoke>\n</"));
  o.advance(kDsmlTag);
  feed(o, bytes_of(" calls>"));
  require(same(allowed_ids(o), {kEosText}), "the open-key call closed cleanly");
  // Auto: text may end the turn without a call; a block, once opened,
  // completes; a disallowed id kills the grammar.
  GrammarState a(&vocab, spec_of(GrammarSpec::Mode::kAuto), false);
  require(a.allows(kEosText) && a.allows('x') && !a.allows(kDsmlTag), "auto: free, EOS allowed, no bare tag");
  a.advance('<');
  a.advance(kDsmlTag);
  require(std::string(a.state_name()) == "d-calls" && !a.allows(kEosText), "a block owes its close");
  a.advance(kEosText);
  require(!a.active() && std::string(a.state_name()) == "dead", "a disallowed id kills the grammar");
}

DGPP_TEST(tool_grammar_requiredOwesACallAndForcesItsShape) {
  const GrammarVocab v = fake_vocab();
  GrammarState g(&v, spec_of(GrammarSpec::Mode::kRequired), /*thinking=*/true);
  // Thinking: free except the three EOS ids (the turn may not end).
  {
    const std::vector<int64_t> a = allowed_ids(g);
    require(a.size() == static_cast<size_t>(kVocab - 3), "think allows all but EOS: " +
                                                             std::to_string(a.size()));
    require(g.allows('a') && g.allows(kThinkClose) && g.allows(kToolOpen) &&
                !g.allows(kEosObs) && !g.allows(kEosUser),
            "think mask membership");
  }
  feed(g, {'h', 'm', kThinkClose});
  // After </think> under required: only <tool_call>.
  require(same(allowed_ids(g), {kToolOpen}), "top owes a call: " + show(allowed_ids(g)));
  feed(g, {kToolOpen});
  // The name: any token that continues a tool name from its start.
  require(same(allowed_ids(g), {'g', 'p', kGet, kGetWeather, kGetT}),
          "name starts: " + show(allowed_ids(g)));
  feed(g, {kGet});
  // "get" emitted: "_" (either name), "_weather", "_t" is not a token here.
  require(same(allowed_ids(g), {'_', kUnderscore, kWeather}),
          "after 'get': " + show(allowed_ids(g)));
  feed(g, {kUnderscore, kWea});  // "get_wea"
  require(same(allowed_ids(g), {'t', kTher}), "after 'get_wea': " + show(allowed_ids(g)));
  feed(g, {kTher});  // "get_weather" complete, no longer name extends it
  require(same(allowed_ids(g), {kKeyOpen, kToolClose}),
          "complete name: <arg_key> or </tool_call>: " + show(allowed_ids(g)));
  feed(g, {kKeyOpen});
  // Closed keys: "city" or "days" — by their tokens.
  require(same(allowed_ids(g), {'c', 'd', kCity}), "keys: " + show(allowed_ids(g)));
  feed(g, {kCity});
  require(same(allowed_ids(g), {kKeyClose}), "complete key");
  feed(g, {kKeyClose});
  require(same(allowed_ids(g), {kValueOpen}), "after key");
  feed(g, {kValueOpen});
  // Values are free text: everything but the markers (and EOS: a call is
  // still owed until it closes), plus the closer.
  {
    const std::vector<int64_t> a = allowed_ids(g);
    require(g.allows('P') && g.allows(kValueClose) && g.allows(kThinkOpen) &&
                !g.allows(kKeyOpen) && !g.allows(kToolOpen) && !g.allows(kEosObs),
            "value mask membership");
    require(a.size() == static_cast<size_t>(kVocab - 5 - 3), "value allows all but 5 "
            "markers and 3 EOS: " + std::to_string(a.size()));
  }
  feed(g, {'P', 'a', kValueClose});
  require(same(allowed_ids(g), {kKeyOpen, kToolClose}), "after value");
  feed(g, {kToolClose});
  // One call closed, parallel: another call or the turn end (only the
  // call-turn EOS).
  require(same(allowed_ids(g), {kToolOpen, kEosObs}),
          "after a call (parallel): " + show(allowed_ids(g)));
  feed(g, {kEosObs});
  require(same(allowed_ids(g), {kEosObs}), "done: EOS again");
  require(g.active(), "still active (never died)");
}

DGPP_TEST(tool_grammar_namedSingleCallThenEos) {
  const GrammarVocab v = fake_vocab();
  GrammarState g(&v, spec_of(GrammarSpec::Mode::kNamed, true, "get_time"),
                 /*thinking=*/false);
  require(same(allowed_ids(g), {kToolOpen}), "no think: a call right away");
  feed(g, {kToolOpen});
  require(same(allowed_ids(g), {'g', kGet, kGetT}),
          "only get_time's starts: " + show(allowed_ids(g)));
  feed(g, {kGetT, kIme});
  // get_time's keys are open: free text or the closers.
  require(same(allowed_ids(g), {kKeyOpen, kToolClose}), "name complete");
  feed(g, {kKeyOpen});
  require(g.allows('z') && g.allows(kKeyClose) && !g.allows(kValueOpen),
          "open keys are free text");
  feed(g, {'z', kKeyClose, kValueOpen, '1', kValueClose, kToolClose});
  // Named: exactly one call, then the turn ends.
  require(same(allowed_ids(g), {kEosObs}), "named: EOS after the one call");
  // A tool without arguments closes right after its name.
  GrammarState p(&v, spec_of(GrammarSpec::Mode::kNamed, true, "ping"), false);
  feed(p, {kToolOpen, 'p', 'i', 'n', 'g'});
  require(same(allowed_ids(p), {kToolClose}), "closed empty keys: no <arg_key>");
}

DGPP_TEST(tool_grammar_autoSingleAndForbidModes) {
  const GrammarVocab v = fake_vocab();
  // auto + parallel false: free until a call opens, then that call, then EOS.
  GrammarState a(&v, spec_of(GrammarSpec::Mode::kAuto, false), true);
  require(!a.active() || allowed_ids(a).empty(), "think is unconstrained (no call owed)");
  feed(a, {'x', kThinkClose});
  require(a.allows('H') && a.allows(kEosUser) && a.allows(kToolOpen) &&
              !a.allows(kKeyOpen),
          "top: free, calls may open, stray markers may not");
  feed(a, {'H', 'i', kToolOpen, kGetWeather, kToolClose});
  require(same(allowed_ids(a), {kEosObs}), "auto single: EOS after the one call");
  // forbid: everything but <tool_call> (and the other markers).
  GrammarState f(&v, spec_of(GrammarSpec::Mode::kForbidCalls), true);
  feed(f, {kThinkClose});
  require(f.allows('H') && f.allows(kEosUser) && !f.allows(kToolOpen) &&
              !f.allows(kToolClose),
          "forbid: no call may open");
  {
    TokenMask m;
    f.mask(&m);
    require(m.allowed == kVocab - 6, "forbid allows all but the six markers");
  }
  // required + parallel false: exactly one call.
  GrammarState r(&v, spec_of(GrammarSpec::Mode::kRequired, false), false);
  feed(r, {kToolOpen, kGetWeather, kToolClose});
  require(same(allowed_ids(r), {kEosObs}), "required single: EOS");
}

DGPP_TEST(tool_grammar_disallowedIdKillsTheGrammar) {
  const GrammarVocab v = fake_vocab();
  GrammarState g(&v, spec_of(GrammarSpec::Mode::kRequired), false);
  require(!g.allows('H'), "text before the call is not allowed");
  g.advance('H');  // the MTP draft may propose it; the verify rejects it
  require(!g.active(), "dead after a disallowed id");
  TokenMask m;
  g.mask(&m);
  require(!m.constrained(), "a dead grammar constrains nothing");
  // An inactive spec is never active.
  GrammarState none(&v, GrammarSpec{}, true);
  require(!none.active(), "kNone is inactive");
  none.mask(&m);
  require(!m.constrained(), "inactive: no mask");
  // A named function outside the tools refuses.
  bool threw = false;
  try {
    GrammarState bad(&v, spec_of(GrammarSpec::Mode::kNamed, true, "nope"), false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "named outside tools refuses");
}

DGPP_TEST(tool_grammar_jsonModeSpellsOneTextThenEos) {
  // response_format (M6 6h) as a grammar: thinking stays free but the
  // turn cannot end; </think> opens the JSON body, where the machine's
  // mask rules, the markers never appear, and EOS comes only once the
  // text is complete.
  const GrammarVocab v = fake_vocab();
  GrammarSpec spec;
  spec.mode = GrammarSpec::Mode::kJson;  // json_schema "" = json_object
  GrammarState g(&v, spec, /*prompt_opens_thinking=*/true);
  require(g.active() && std::string(g.state_name()) == "think", "starts thinking");
  require(g.allows('x') && g.allows(kThinkClose) && g.allows(kToolOpen) &&
              !g.allows(kEosText) && !g.allows(kEosObs),
          "thinking is free, EOS withheld");
  g.advance(kThinkClose);
  require(std::string(g.state_name()) == "json-value", "the body opens");
  require(g.allows('{') && g.allows(' ') && g.allows('\n') && !g.allows('"') &&
              !g.allows('[') && !g.allows('1') && !g.allows(kEosText) &&
              !g.allows(kThinkOpen) && !g.allows(kThinkClose) && !g.allows(kToolOpen),
          "json_object: an object opens, nothing else");
  TokenMask m;
  g.mask(&m);
  require(m.constrained() && m.allows('{') && !m.allows(kEosText) && m.allowed >= 5,
          "the body mask is a real constraint");
  for (const char c : std::string("{\"a\":1")) g.advance(static_cast<unsigned char>(c));
  require(g.active() && !g.allows(kEosText) && g.allows(',') && g.allows('}'),
          "inside the object: no EOS yet");
  g.advance('}');
  require(g.allows(kEosText) && g.allows(kEosObs) && g.allows(' ') && !g.allows(',') &&
              !g.allows('{'),
          "complete: EOS or whitespace only");
  g.mask(&m);
  require(m.allows(kEosText) && m.allows(kEosUser) && !m.allows('}'), "done mask");
  // Trailing whitespace is capped: past sixteen bytes only EOS remains.
  for (int i = 0; i < 16; ++i) g.advance(' ');
  g.mask(&m);
  require(m.allows(kEosText) && !m.allows(' ') && !m.allows('\n') && m.allowed == 3,
          "the whitespace cap leaves the EOS ids alone");
  g.advance(kEosText);
  require(std::string(g.state_name()) == "done", "EOS ends the turn");

  // A schema: closed keys spelled from the declared names, an integer
  // value, the closer only once the required key is in.
  GrammarSpec typed;
  typed.mode = GrammarSpec::Mode::kJson;
  typed.json_schema =
      "{\"type\":\"object\",\"properties\":{\"k\":{\"type\":\"integer\"},"
      "\"s\":{\"enum\":[\"on\",\"off\"]}},\"required\":[\"k\"],"
      "\"additionalProperties\":false}";
  GrammarState t(&v, typed, /*prompt_opens_thinking=*/false);
  require(std::string(t.state_name()) == "json-value", "no think block: body at once");
  t.advance('{');
  require(t.allows('"') && !t.allows('}'), "the required key is owed");
  t.advance('"');
  require(t.allows('k') && t.allows('s') && !t.allows('z') && !t.allows('"'),
          "keys from the declared names");
  for (const char c : std::string("k\":")) t.advance(static_cast<unsigned char>(c));
  require(t.allows('-') && t.allows('7') && !t.allows('"') && !t.allows('t'),
          "an integer value");
  t.advance('4');
  require(!t.allows('.') && !t.allows('e') && t.allows('}') && t.allows(','),
          "integer: no fraction; the object may close");
  for (const char c : std::string(",\"s\":\"o")) t.advance(static_cast<unsigned char>(c));
  require(t.allows('n') && t.allows('f') && !t.allows('x') && !t.allows('"'),
          "an enum string is spelled from its targets");
  for (const char c : std::string("ff\"}")) t.advance(static_cast<unsigned char>(c));
  require(t.allows(kEosText), "done under the schema");
  // A disallowed id kills the grammar (the sampler never produces one).
  GrammarState k(&v, typed, false);
  k.advance('[');
  require(!k.active(), "a disallowed id kills the JSON grammar");
  // A schema text that does not parse, or is outside the subset, refuses
  // at construction.
  GrammarSpec bad = typed;
  bad.json_schema = "{not json";
  bool threw = false;
  try {
    GrammarState b(&v, bad, false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "unparsable schema text refused");
  bad.json_schema = "{\"type\":\"string\",\"pattern\":\"[\"}";
  threw = false;
  try {
    GrammarState b(&v, bad, false);
  } catch (const std::invalid_argument& e) {
    threw = std::string(e.what()).rfind("schema.pattern", 0) == 0;
  }
  require(threw, "an unsupported keyword refused by name");
  // The spec's equality covers the schema text.
  require(!(spec == typed), "specs differ by schema");
  GrammarSpec same = typed;
  require(same == typed, "equal specs");
}

DGPP_TEST(tool_grammar_jsonOrToolsCommitsToOneBranch) {
  for (const GrammarVocab& v : {fake_vocab(), qwen_vocab(), dsml_vocab()}) {
    const bool dsml = v.markers().tool_format() == dgpp::text::ToolFormat::kDsml;
    const bool qwen = v.markers().tool_format() == dgpp::text::ToolFormat::kQwenXml;
    for (const bool parallel : {false, true}) {
      GrammarSpec spec = spec_of(GrammarSpec::Mode::kJsonOrTools, parallel);
      spec.json_schema = R"({"type":"object","properties":{"n":{"type":"number","minimum":0,"maximum":10}},"required":["n"],"additionalProperties":false})";
      GrammarState json(&v, spec, true);
      require(json.allows('x') && !json.allows(kEosText), "thinking precedes JSON or tools");
      feed(json, {kThinkClose, ' ', '\n'});
      require(json.allows('{') && json.allows(dsml ? '<' : kToolOpen) &&
                  !json.allows('x') && !json.allows(kEosText), "choose JSON or a call");
      feed(json, bytes_of("{\"n\":0.5}"));
      require(json.allows(kEosText) && !json.allows(kToolOpen) && !json.allows('<') &&
                  !json.allows('{'), "JSON answer cannot become a call");
      GrammarState call(&v, spec, false);
      if (dsml) {
        feed(call, {'<', kDsmlTag});
        feed(call, bytes_of(" calls>\n<"));
        feed(call, {kDsmlTag});
        feed(call, bytes_of(" invoke name=\"ping\">\n</"));
        feed(call, {kDsmlTag});
        feed(call, bytes_of(" invoke>\n</"));
        feed(call, {kDsmlTag});
        feed(call, bytes_of(" calls>"));
      } else {
        feed(call, {kToolOpen});
        feed(call, bytes_of(qwen ? "\n<function=ping>\n</function>\n" : "ping"));
        feed(call, {kToolClose});
      }
      require(call.allows(v.call_turn_eos()) && !call.allows('{') && !call.allows('x'),
              "tool turn ends without an unconstrained answer");
      if (!dsml) require(call.allows(kToolOpen) == parallel, "parallel flag still enforced");
    }
  }
  const auto v = dsml_vocab();
  GrammarSpec scalar = spec_of(GrammarSpec::Mode::kJsonOrTools);
  scalar.json_schema = R"({"type":"string"})";
  GrammarState string(&v, scalar, false);
  feed(string, {kQuotedLt, '"', kEosText});
  require(string.active(), "a '<' inside a JSON string is not a DSML opener");
}

DGPP_TEST(tool_grammar_typedValuesFollowTheSchema) {
  // Typed arguments (M6 6i): a JSON-typed value runs the JSON machine
  // under the property's schema with </arg_value> only when complete; an
  // enum string is spelled from its texts; a plain string and an unknown
  // key stay free; auto mode arms the shape with calls at will.
  const GrammarVocab v = fake_vocab();
  GrammarSpec spec;
  spec.mode = GrammarSpec::Mode::kAuto;
  spec.parallel = true;
  GrammarTool w;
  w.name = "get_weather";
  w.constrain_keys = false;  // open keys: an unknown key is allowed and free
  w.args.push_back(GrammarArg{"city", GrammarArg::Kind::kFree, "", {}});
  w.args.push_back(GrammarArg{"days", GrammarArg::Kind::kJson, "{\"type\":\"integer\"}", {}});
  w.args.push_back(GrammarArg{"unit", GrammarArg::Kind::kText, "", {"celsius", "fahrenheit"}});
  w.args.push_back(GrammarArg{"opts", GrammarArg::Kind::kJson,
                              "{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"boolean\"}},"
                              "\"required\":[\"a\"],\"additionalProperties\":false}", {}});
  spec.tools.push_back(w);
  GrammarState g(&v, spec, /*prompt_opens_thinking=*/false);
  const auto feed = [&](const std::string& text) {
    for (const char c : text) {
      require(g.allows(static_cast<unsigned char>(c)),
              std::string("byte '") + c + "' refused in state " + g.state_name());
      g.advance(static_cast<unsigned char>(c));
    }
  };
  const auto marker = [&](int64_t id) {
    require(g.allows(id), "marker refused in state " + std::string(g.state_name()));
    g.advance(id);
  };
  require(g.allows(kToolOpen) && g.allows(kEosText) && g.allows('x'), "auto: free top");
  marker(kToolOpen);
  feed("get_weather");
  marker(kKeyOpen);
  feed("days");
  marker(kKeyClose);
  marker(kValueOpen);
  // An integer: digits, no quote, no fraction; the closer once complete.
  require(g.allows('-') && g.allows('7') && !g.allows('"') && !g.allows('t') &&
              !g.allows(kValueClose),
          "integer value: the closer waits");
  feed("12");
  require(g.allows('3') && !g.allows('.') && !g.allows('e') && g.allows(kValueClose) &&
              !g.allows(kToolClose) && !g.allows(kEosText),
          "12 is complete: only the closer or more digits");
  TokenMask m;
  g.mask(&m);
  require(m.constrained() && m.allows(kValueClose) && m.allows('3') && !m.allows('.'),
          "the value mask agrees with allows()");
  marker(kValueClose);
  // An enum string: spelled from its texts.
  marker(kKeyOpen);
  feed("unit");
  marker(kKeyClose);
  marker(kValueOpen);
  require(g.allows('c') && g.allows('f') && !g.allows('k') && !g.allows(kValueClose),
          "enum: first bytes only");
  feed("celsiu");
  require(!g.allows(kValueClose) && g.allows('s'), "enum: incomplete");
  feed("s");
  require(g.allows(kValueClose) && !g.allows('x'), "enum: complete");
  marker(kValueClose);
  // A nested object under its schema: closed keys, a boolean, required.
  marker(kKeyOpen);
  feed("opts");
  marker(kKeyClose);
  marker(kValueOpen);
  require(g.allows('{') && !g.allows('[') && !g.allows('1'), "an object value");
  feed("{\"");
  require(g.allows('a') && !g.allows('b'), "closed key");
  feed("a\":");
  require(g.allows('t') && g.allows('f') && !g.allows('1'), "boolean");
  feed("true");
  require(!g.allows(kValueClose) && g.allows('}'), "not closed yet");
  feed("}");
  require(g.allows(kValueClose) && !g.allows(','), "complete object");
  marker(kValueClose);
  // A free string, then an unknown key (open schema): free text.
  marker(kKeyOpen);
  feed("city");
  marker(kKeyClose);
  marker(kValueOpen);
  require(g.allows('P') && g.allows('"') && g.allows(' ') && g.allows(kValueClose) &&
              !g.allows(kToolClose),
          "a string is free (closer allowed at once)");
  feed("Paris 75");
  marker(kValueClose);
  marker(kKeyOpen);
  feed("zzz");
  marker(kKeyClose);
  marker(kValueOpen);
  require(g.allows('{') && g.allows('x') && g.allows(kValueClose), "unknown key: free value");
  marker(kValueClose);
  marker(kToolClose);
  // auto with parallel: another call may open, or the turn may end.
  require(g.allows(kToolOpen) && g.allows(kEosText) && g.allows('x'), "auto: calls at will");
  g.advance(kEosText);
  require(std::string(g.state_name()) == "done", "done");
  // A disallowed byte inside a typed value kills the grammar.
  GrammarState k(&v, spec, false);
  k.advance(kToolOpen);
  for (const char c : std::string("get_weather")) k.advance(static_cast<unsigned char>(c));
  k.advance(kKeyOpen);
  for (const char c : std::string("days")) k.advance(static_cast<unsigned char>(c));
  k.advance(kKeyClose);
  k.advance(kValueOpen);
  k.advance('x');
  require(!k.active(), "a non-integer byte kills the grammar");
  // A JSON-typed argument's schema tolerates a keyword that only narrows
  // the value (2026-09-06: the value stays typed — a letter still kills the
  // grammar) and refuses one outside the subset in shape at construction.
  GrammarSpec lax = spec;
  lax.tools[0].args[1].schema = "{\"type\":\"integer\",\"minimum\":0}";
  {
    GrammarState l(&v, lax, false);
    l.advance(kToolOpen);
    for (const char c : std::string("get_weather")) l.advance(static_cast<unsigned char>(c));
    l.advance(kKeyOpen);
    for (const char c : std::string("days")) l.advance(static_cast<unsigned char>(c));
    l.advance(kKeyClose);
    l.advance(kValueOpen);
    require(l.allows('7') && !l.allows('x'),
            "a tolerated minimum: the value is still an integer");
    l.advance('x');
    require(!l.active(), "a non-integer byte kills the grammar under a tolerated keyword");
  }
  GrammarSpec bad = spec;
  bad.tools[0].args[1].schema = "{\"type\":\"integer\",\"$ref\":\"#/x\"}";
  bool threw = false;
  try {
    GrammarState b(&v, bad, false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "an unsupported argument schema refused");
  // Equality covers the arguments.
  GrammarSpec same = spec;
  require(same == spec && !(bad == spec), "spec equality over arguments");
}

// The 6i follow-on: a closed key is offered ONCE per call (a duplicate key
// is never a valid object), and a strict tool's call cannot close while a
// required key is missing — OpenAI's strict guarantee, enforced by the
// mask rather than hoped for. A non-strict tool keeps the closer open (its
// required keys are the client's business) but still spends its keys.
DGPP_TEST(tool_grammar_closedKeysOnceAndStrictRequiredKeysGateTheClose) {
  const GrammarVocab v = fake_vocab();
  GrammarSpec s = spec_of(GrammarSpec::Mode::kNamed, false, "get_weather");
  s.tools[0].strict = true;
  s.tools[0].required_keys = {"city"};
  GrammarState g(&v, s, /*thinking=*/false);
  feed(g, {kToolOpen, kGetWeather});
  // The name is complete: a strict tool owing a key cannot close yet.
  require(same(allowed_ids(g), {kKeyOpen}),
          "strict name: only <arg_key>: " + show(allowed_ids(g)));
  feed(g, {kKeyOpen});
  require(same(allowed_ids(g), {'c', 'd', kCity}), "keys: " + show(allowed_ids(g)));
  feed(g, {'d', 'a', 'y', 's', kKeyClose, kValueOpen, '3', kValueClose});
  // "days" used, "city" (required) still missing: another key, no close.
  require(same(allowed_ids(g), {kKeyOpen}),
          "after days: city still owed: " + show(allowed_ids(g)));
  feed(g, {kKeyOpen});
  // "days" is spent: only "city" continues, and 'd' never starts again.
  require(same(allowed_ids(g), {'c', kCity}),
          "keys less the used one: " + show(allowed_ids(g)));
  require(!g.allows('d'), "a duplicate key never starts");
  feed(g, {kCity, kKeyClose, kValueOpen, 'X', kValueClose});
  // Every closed key used: no <arg_key>; every required key present: close.
  require(same(allowed_ids(g), {kToolClose}),
          "all keys spent: only </tool_call>: " + show(allowed_ids(g)));
  feed(g, {kToolClose});
  require(same(allowed_ids(g), {kEosObs}), "named: the turn ends");
  require(g.active(), "still active");

  // The same tool, non-strict: the closer stays open from the name on ...
  GrammarSpec ns = s;
  ns.tools[0].strict = false;
  GrammarState h(&v, ns, /*thinking=*/false);
  feed(h, {kToolOpen, kGetWeather});
  require(same(allowed_ids(h), {kKeyOpen, kToolClose}),
          "non-strict: closable at once: " + show(allowed_ids(h)));
  // ... but a closed key is still offered once.
  feed(h, {kKeyOpen, kCity, kKeyClose, kValueOpen, kValueClose, kKeyOpen});
  require(same(allowed_ids(h), {'d'}),
          "non-strict: the used key is gone: " + show(allowed_ids(h)));
  feed(h, {'d', 'a', 'y', 's', kKeyClose, kValueOpen, kValueClose});
  require(same(allowed_ids(h), {kToolClose}),
          "non-strict, keys spent: only the close: " + show(allowed_ids(h)));

  // The derivation from a function definition: strict rides, required keys
  // are the declared ones, an undeclared name warns and is dropped.
  {
    const dgpp::minijson::ParseResult def = dgpp::minijson::parse(
        R"({"name":"get_weather","strict":true,"parameters":{"type":"object",)"
        R"("properties":{"city":{"type":"string"},"days":{"type":"integer"}},)"
        R"("required":["city","city"],"additionalProperties":false}})");
    std::vector<std::string> warnings;
    const GrammarTool t = dgpp::text::grammar_tool_from_function(def.root, &warnings);
    require(t.strict && t.constrain_keys &&
                t.required_keys == std::vector<std::string>{"city"},
            "derived: strict, closed, the declared required key once");
    require(warnings.empty(), "valid strict schema has no warnings");
    bool invalid_required = false;
    try {
      const auto bad = dgpp::minijson::parse(R"({"name":"f","strict":true,"parameters":{"type":"object","properties":{},"required":["missing"]}})");
      dgpp::text::grammar_tool_from_function(bad.root, nullptr);
    } catch (const std::invalid_argument&) { invalid_required = true; }
    require(invalid_required, "strict tools reject unsatisfiable required keys");
    const dgpp::minijson::ParseResult lax = dgpp::minijson::parse(
        R"({"name":"f","parameters":{"type":"object","properties":{"a":{"type":"string"}},"required":["a"]}})");
    const GrammarTool u = dgpp::text::grammar_tool_from_function(lax.root, nullptr);
    require(!u.strict && u.constrain_keys && u.keys == std::vector<std::string>{"a"} &&
                u.required_keys == std::vector<std::string>{"a"},
            "derived: a non-strict tool closes its declared names and records its "
            "required keys unenforced");
  }
}

}  // namespace
