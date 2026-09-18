// The DeepSeek-V4-Flash tokenizer's standalone gate: the engine's BPE
// encoder vs the goldens HF tokenizers produced on the checkpoint's own
// tokenizer.json (tests/data/dsv4_tokenizer_goldens.jsonl). The tokenizer
// loads from the local model dir (DGPP_DSV4_MODEL_DIR, default
// /data/models/DeepSeek-V4-Flash-0731) — this checkpoint is not in the HF
// hub cache, so the shared tokenizer_test (which resolves via the HF cache)
// does not cover it. The render->ids integration is separately gated by
// dsv4_prompt_test; this is the focused "encode any string" half.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common/log.hpp"

int g_argc = 0;
char** g_argv = nullptr;
#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "text/tokenizer.hpp"

namespace {

std::string golden_path(int argc, char** argv) {
  return argc > 1 ? argv[1] : "tests/data/dsv4_tokenizer_goldens.jsonl";
}

std::string model_dir() {
  if (const char* env = std::getenv("DGPP_DSV4_MODEL_DIR"); env && *env) return env;
  return "/data/models/DeepSeek-V4-Flash-0731";
}

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

std::string read_text_file(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

DGPP_TEST(dsv4_tokenizer_differential_goldens) {
  const std::string kGoldenPath = golden_path(g_argc, g_argv);
  const std::vector<std::string> lines = [&] {
    std::vector<std::string> out;
    std::istringstream f(read_text_file(kGoldenPath));
    std::string line;
    while (std::getline(f, line))
      if (!line.empty()) out.push_back(line);
    return out;
  }();
  require(!lines.empty(), "golden corpus is empty");

  const std::filesystem::path tok_path = std::filesystem::path(model_dir()) / "tokenizer.json";
  if (!std::filesystem::exists(tok_path)) {
    DGPP_LOG_WARN("dsv4_tokenizer_test: {} not found; skipping (set DGPP_DSV4_MODEL_DIR)",
                  tok_path.string());
    std::exit(2);
  }
  const dgpp::text::Tokenizer tok = dgpp::text::Tokenizer::load(tok_path.string());

  size_t checked = 0;
  for (const std::string& line : lines) {
    const dgpp::minijson::ParseResult parsed = dgpp::minijson::parse(line);
    const dgpp::minijson::Value& rec = parsed.root;
    if (rec.find("text") == nullptr) continue;  // the header line, if any
    const std::string text(rec.at("text").as_string());
    std::vector<int64_t> want;
    for (const auto& v : rec.at("ids").items()) want.push_back(v.as_int());
    const std::vector<int64_t> got = tok.encode(text);
    require(got == want, "the engine's tokenizer encodes '" + text + "' differently from HF's");
    require(tok.decode(got, false) == text, "round-trip decode of '" + text + "' is not the source");
    ++checked;
  }
  DGPP_LOG_INFO("dsv4 tokenizer differential: {} cases encode + round-trip clean", checked);
}

int main(int argc, char** argv) {
  g_argc = argc;
  g_argv = argv;
  return dgpp::test::run_all();
}
