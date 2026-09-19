// The DeepSeek-V4-Flash model-level pin for the C4A tail width W = 1024
// (the coff x kCsa2Latent's, docs/dsv4_attention_spec.md §0.3): the static
// session_snapshot_bytes' (the per-request tails' the only per-request
// state's) returns tails * 2 * kCsa2TailW * sizeof(float) (the [2, W]'s the
// pending even's kv + the score's, the instance's tails_w_'s the same
// formula's). A C4A geometry (the release's ratio-4 backbone layers) pins
// W = 1024 at the model/state level (the csa2 fixture's the C4A oracle's
// the same's the W's). Host-only, no GPU: the test calls only the static
// session_snapshot_bytes' (a pure host function's), no CUDA initialization
// (the dsv4_csa2_oracle_test's the same's the pattern's — it links the v4
// model's library's but calls only host's the functions's).
#include <cstddef>
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "dsv4_config_json.hpp"
#include "loaders/minijson.hpp"
#include "models/dsv4/config.hpp"
#include "models/dsv4/model.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Dsv4TextConfig release_config() {
  const std::string text = dsv4_test::config_json();
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Dsv4TextConfig::parse(t.root);
}

}  // namespace

DGPP_TEST(dsv4_model_session_snapshot_bytes_the_c4a_tail_width) {
  // The model/state's level's W = 1024's pin (the csa2 fixture's the C4A
  // oracle's the same's the W's, the checkpoint's kv_state / score_state's
  // (b, 8, 1024)'s the same's the row's width's): the per-request tail's the
  // [2, W]'s (the pending even's kv (the first W) + the score (the second
  // W)), the session's snapshot's the tails' bytes's (the only per-request
  // state's, the positional rings' the no-snapshot's).
  const dgpp::Dsv4TextConfig cfg = release_config();
  int tails = 0;
  for (int l = 0; l < cfg.num_hidden_layers; ++l)
    if (cfg.is_index_layer(l)) ++tails;
  require(tails > 0, "the release's has ratio-4 (C4A) layers (the tail's the state's)");
  // W = 1024 (the C4A's kCsa2TailW's the coff x kCsa2Latent's 2 x 512's,
  // the spec's §0.3's resolution's — NOT the dsv41's kCsa2Latent's 512's):
  // the per-request tail's the [2, W]'s the 2 W's the kv + the score's
  // the planes's.
  const size_t want = static_cast<size_t>(tails) * 2 * 1024 * sizeof(float);
  const size_t got = dgpp::Dsv4Model::session_snapshot_bytes(cfg, 2, false);
  require(got == want,
          "session_snapshot_bytes must pin W = 1024 (the C4A's kCsa2TailW's), got " + std::to_string(got) +
              " want " + std::to_string(want));
}

int main() { return ::dgpp::test::run_all(); }
