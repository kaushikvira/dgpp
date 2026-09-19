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
  // The model/state's level's per-ordinal tail's bytes' pin (the 2026-09-21's
  // the 0731's reference's form's the G-c128a-compressor's + the G-tail-
  // cadence / G-tail-pool / G-tail-ape's the closed's, docs/dsv4_attention_
  // spec.md §5): the per-request tail's the checkpoint's kv_state /
  // score_state's (coff * ratio, coff * head_dim)'s the fp32's — the C4A's
  // (ratio 4's coff 2's) the 8 x 1024's the 2 overlapping's windows' the
  // C128A's (ratio 128's coff 1's) the 128 x 512's the 128-token's ring's —
  // the session's snapshot's the tails' bytes's (the only per-request
  // state's, the positional rings' the no-snapshot's). The C4A geometry's
  // (the release's ratio-4 backbone layers) pins the 8 x 1024's + the C128A's
  // (the release's ratio-128 backbone layers) the 128 x 512's at the
  // model/state's level's (the csa2 fixture's the C4A oracle's the same's
  // the W's). Host-only, no GPU: the test calls only the static
  // session_snapshot_bytes' (a pure host function's), no CUDA initialization
  // (the dsv4_csa2_oracle_test's the same's the pattern's — it links the v4
  // model's library's but calls only host's the functions's).
  const dgpp::Dsv4TextConfig cfg = release_config();
  int c4a = 0, c128a = 0;
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const int ratio = cfg.compress_ratio(l);
    if (ratio == 4) ++c4a;
    if (ratio == 128) ++c128a;
  }
  require(c4a > 0, "the release's has ratio-4 (C4A) layers (the tail's the state's)");
  require(c128a > 0, "the release's has ratio-128 (C128A) layers (the tail's the state's the G-c128a's the closed's)");
  // The per-ordinal's state's (the checkpoint's kv_state / score_state's
  // (coff * ratio, coff * head_dim)'s the fp32's): the C4A's 2 x 8 x 1024's
  // the 2 overlapping's windows' the C128A's 2 x 128 x 512's the 128-token's
  // ring's.
  const size_t c4a_bytes = static_cast<size_t>(2) * 8 * 1024 * sizeof(float);  // the C4A's 8 x 1024's
  const size_t c128a_bytes = static_cast<size_t>(2) * 128 * 512 * sizeof(float);  // the C128A's 128 x 512's
  const size_t want = static_cast<size_t>(c4a) * c4a_bytes + static_cast<size_t>(c128a) * c128a_bytes;
  const size_t got = dgpp::Dsv4Model::session_snapshot_bytes(cfg, 2, false);
  require(got == want,
          "session_snapshot_bytes must pin the per-ordinal's tail's (the C4A's 8 x 1024's + the C128A's 128 x 512's), got " +
              std::to_string(got) + " want " + std::to_string(want));
}

DGPP_TEST(dsv4_model_union_attn_out_bytes_the_dspark_64_head_shape) {
  // The DSpark union attention's out latent's shape's pin (the 2026-09-20's
  // dsv4 S3's union-attn's q-latent's + the block-kv's wiring's): the
  // [max_rows, 64, 512]'s the bf16's (the Dsv4DsparkConfig's kHeads's x
  // kHeadDim's the DSpark's 64-head's the 512-dim's output's) — the model's
  // staging's (the q fallback's + the out's the 2x's the constructor's the
  // scratch's the same's formula's the union_attn_out_bytes's static's).
  // A pure host's function's (the no CUDA's, the dsv4_model_state_test's the
  // same's pattern's — it links the v4 model's library's but calls only
  // host's the functions's).
  const int max_rows = 16;  // a decode batch's row count
  const size_t want = static_cast<size_t>(max_rows) * static_cast<size_t>(dgpp::Dsv4DsparkConfig::kHeads) *
                      static_cast<size_t>(dgpp::Dsv4DsparkConfig::kHeadDim) * sizeof(uint16_t);
  const size_t got = dgpp::Dsv4Model::union_attn_out_bytes(max_rows);
  require(got == want,
          "union_attn_out_bytes must pin the DSpark 64-head out latent's shape, got " + std::to_string(got) +
              " want " + std::to_string(want));
  // The degenerate's max_rows == 0's the no-rows's the 1's row's bound's
  // (the std::max(1, max_rows)'s the constructor's the no-zero's alloc's):
  // the formula's floors at 1's row's (the never's a zero-byte's alloc's).
  const size_t want0 =
      static_cast<size_t>(1) * static_cast<size_t>(dgpp::Dsv4DsparkConfig::kHeads) *
      static_cast<size_t>(dgpp::Dsv4DsparkConfig::kHeadDim) * sizeof(uint16_t);
  const size_t got0 = dgpp::Dsv4Model::union_attn_out_bytes(0);
  require(got0 == want0, "union_attn_out_bytes(0) must floor at 1 row (the no-zero's alloc's)");
}

int main() { return ::dgpp::test::run_all(); }
