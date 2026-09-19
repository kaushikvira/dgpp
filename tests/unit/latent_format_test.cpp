// The latent cache's storage formats: the e2m1 codec, the
// fp8/fp4 row quantizers and their dequantizers — the host reference the
// device append kernel and the attention tile loaders are pinned to.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/latent_format.hpp"

namespace {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::LatentFormat;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Deterministic pseudo-random bf16 rows in [-4, 4) with a few zeros and
// one large outlier per row (the block-scale case that matters).
std::vector<uint16_t> row(uint64_t seed, int n) {
  std::vector<uint16_t> r(static_cast<size_t>(n));
  uint64_t x = seed * 0x9E3779B97F4A7C15ull + 1;
  for (int i = 0; i < n; ++i) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    const uint32_t h = static_cast<uint32_t>((x * 0x2545F4914F6CDD1Dull) >> 32);
    float v = (static_cast<float>(h & 0xFFFF) / 65536.0f - 0.5f) * 8.0f;
    if ((h >> 16) % 17 == 0) v = 0.0f;
    if (i == static_cast<int>(seed % static_cast<uint64_t>(n))) v *= 20.0f;
    r[static_cast<size_t>(i)] = float_to_bf16_bits(v);
  }
  return r;
}

double rel_l2(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  double d2 = 0, n2 = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = bf16_bits_to_float(a[i]), y = bf16_bits_to_float(b[i]);
    d2 += (x - y) * (x - y);
    n2 += x * x;
  }
  return std::sqrt(d2 / std::max(n2, 1e-30));
}

}  // namespace

DGPP_TEST(latent_format_names_and_row_bytes) {
  require(dgpp::latent_format_from_string("bf16") == LatentFormat::kBf16 &&
              dgpp::latent_format_from_string("fp8") == LatentFormat::kFp8 &&
              dgpp::latent_format_from_string("fp4") == LatentFormat::kFp4 &&
              dgpp::latent_format_from_string("fp8_block_rope") == LatentFormat::kFp8BlockRope &&
              !dgpp::latent_format_from_string("int8") &&
              !dgpp::latent_format_from_string(""),
          "bf16 / fp8 / fp4 / fp8_block_rope parse and nothing else does");
  require(std::string(dgpp::latent_format_name(LatentFormat::kFp4)) == "fp4",
          "the name round-trips");
  require(std::string(dgpp::latent_format_name(LatentFormat::kFp8BlockRope)) == "fp8_block_rope",
          "the mixed record's name round-trips");
  // The real geometry: 512-wide rows.
  require(dgpp::latent_row_bytes(LatentFormat::kBf16, 512) == 1024, "bf16 row");
  require(dgpp::latent_row_bytes(LatentFormat::kFp8, 512) == 512, "fp8 row");
  require(dgpp::latent_row_bytes(LatentFormat::kFp4, 512) == 288, "fp4 row: 256 codes + 32 scales");
  // The mixed record (the 584 B envelope — 448 e4m3 + 128 bf16 RoPE +
  // 7 e8m0 + 1 pad; 8-byte aligned, NOT 16 — the tile loader's uint2's).
  require(dgpp::latent_row_bytes(LatentFormat::kFp8BlockRope, 512) == 584, "the 584 B envelope");
  require(dgpp::latent_row_bytes(LatentFormat::kFp8BlockRope, 512) % 8 == 0 &&
                dgpp::latent_row_bytes(LatentFormat::kFp8BlockRope, 512) % 16 != 0,
          "the 584's stride's 8-aligned's (the uint2's loads's)");
  // Rows keep 16-byte alignment (the tile loaders' vector loads).
  require(dgpp::latent_row_bytes(LatentFormat::kFp4, 32) == 32, "fp4 row padded to 16 bytes");
  require(dgpp::latent_row_bytes(LatentFormat::kFp4, 256) == 144, "fp4 256-wide row");
  require(!dgpp::latent_format_has_row_scale(LatentFormat::kBf16) &&
              dgpp::latent_format_has_row_scale(LatentFormat::kFp8) &&
              dgpp::latent_format_has_row_scale(LatentFormat::kFp4) &&
              !dgpp::latent_format_has_row_scale(LatentFormat::kFp8Block) &&
              !dgpp::latent_format_has_row_scale(LatentFormat::kFp4Block) &&
              !dgpp::latent_format_has_row_scale(LatentFormat::kFp8BlockRope),
          "only the row-scaled formats carry a row scale (the block formats' self-describing's)");
}

DGPP_TEST(latent_format_e2m1_codec_is_exact_and_rounds_to_even) {
  const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  for (int code = 0; code < 16; ++code) {
    const float v = dgpp::fp4_e2m1_bits_to_float(static_cast<uint8_t>(code));
    const float want = (code & 8) ? -grid[code & 7] : grid[code & 7];
    require(v == want, "decode of code " + std::to_string(code));
    // Every grid value encodes back to itself (the zero's sign aside).
    const uint8_t re = dgpp::float_to_fp4_e2m1_bits(v);
    require((re & 7) == (code & 7) && ((code & 7) == 0 || (re & 8) == (code & 8)),
            "encode(decode) of code " + std::to_string(code));
  }
  // Midpoints round to the even code: .25->0, .75->1.0, 1.25->1.0,
  // 1.75->2.0, 2.5->2.0, 3.5->4.0, 5->4.0; beyond 6 saturates.
  const std::pair<float, int> ties[] = {{0.25f, 0}, {0.75f, 2}, {1.25f, 2}, {1.75f, 4},
                                        {2.5f, 4},  {3.5f, 6},  {5.0f, 6},  {7.0f, 7},
                                        {1e9f, 7},  {0.26f, 1}, {0.74f, 1}, {4.99f, 6}};
  for (const auto& [x, code] : ties) {
    require(dgpp::float_to_fp4_e2m1_bits(x) == code,
            "tie/saturation at " + std::to_string(x));
    require(dgpp::float_to_fp4_e2m1_bits(-x) == (code | 8), "negative of it");
  }
  require((dgpp::float_to_fp4_e2m1_bits(NAN) & 7) == 7, "NaN saturates (no NaN code)");
}

DGPP_TEST(latent_format_row_quantizers_bound_the_error_and_handle_zero_rows) {
  const int kv_lora = 512;
  for (uint64_t seed = 1; seed <= 24; ++seed) {
    const std::vector<uint16_t> src = row(seed, kv_lora);
    for (LatentFormat f : {LatentFormat::kBf16, LatentFormat::kFp8, LatentFormat::kFp4}) {
      std::vector<uint8_t> bytes(dgpp::latent_row_bytes(f, kv_lora), 0xAA);
      float scale = -1.0f;
      dgpp::latent_quantize_row_host(f, src.data(), kv_lora, bytes.data(), &scale);
      std::vector<uint16_t> back(static_cast<size_t>(kv_lora));
      dgpp::latent_dequantize_row_host(f, bytes.data(), scale, kv_lora, back.data());
      const double err = rel_l2(src, back);
      const double bound = f == LatentFormat::kBf16 ? 0.0 : f == LatentFormat::kFp8 ? 0.06 : 0.30;
      require(err <= bound, std::string("row ") + std::to_string(seed) + " in " +
                                dgpp::latent_format_name(f) + ": relative l2 " +
                                std::to_string(err) + " above " + std::to_string(bound));
      // The largest element survives within its own quantum: the row scale
      // puts the absmax at the top of the range.
      float amax = 0, back_max = 0;
      for (int i = 0; i < kv_lora; ++i) {
        amax = std::max(amax, std::fabs(bf16_bits_to_float(src[static_cast<size_t>(i)])));
        back_max = std::max(back_max, std::fabs(bf16_bits_to_float(back[static_cast<size_t>(i)])));
      }
      require(std::fabs(back_max - amax) <= amax * (f == LatentFormat::kFp4 ? 0.13 : 0.07) + 1e-6f,
              std::string("absmax preserved in ") + dgpp::latent_format_name(f));
      if (f == LatentFormat::kFp4) {
        // The fp4 row's padding bytes are written (never left as the
        // buffer's old contents).
        for (size_t k = dgpp::latent_fp4_scale_offset(kv_lora) + kv_lora / dgpp::kLatentFp4Block;
             k < bytes.size(); ++k)
          require(bytes[k] == 0, "fp4 padding zeroed");
      }
    }
  }
  // An all-zero row: scale 0, every code 0, exact zeros back.
  const std::vector<uint16_t> zeros(static_cast<size_t>(kv_lora), 0);
  for (LatentFormat f : {LatentFormat::kFp8, LatentFormat::kFp4}) {
    std::vector<uint8_t> bytes(dgpp::latent_row_bytes(f, kv_lora), 0xFF);
    float scale = -1.0f;
    dgpp::latent_quantize_row_host(f, zeros.data(), kv_lora, bytes.data(), &scale);
    require(scale == 0.0f, "zero row's scale");
    for (uint8_t b : bytes) require(b == 0, "zero row's codes");
    std::vector<uint16_t> back(static_cast<size_t>(kv_lora), 1);
    dgpp::latent_dequantize_row_host(f, bytes.data(), scale, kv_lora, back.data());
    for (uint16_t v : back) require(v == 0, "zero row decodes to zeros");
  }
}

DGPP_TEST(latent_format_fp8_row_matches_the_elementwise_recipe) {
  // The row codec is the per-element recipe applied with the row's scale:
  // scale = absmax / 448, code = e4m3(x * 448 / absmax).
  const int kv_lora = 64;
  const std::vector<uint16_t> src = row(7, kv_lora);
  float absmax = 0;
  for (uint16_t v : src) absmax = std::max(absmax, std::fabs(bf16_bits_to_float(v)));
  std::vector<uint8_t> bytes(dgpp::latent_row_bytes(LatentFormat::kFp8, kv_lora));
  float scale = 0;
  dgpp::latent_quantize_row_host(LatentFormat::kFp8, src.data(), kv_lora, bytes.data(), &scale);
  require(scale == absmax * (1.0f / 448.0f), "the row scale");
  for (int i = 0; i < kv_lora; ++i) {
    const uint8_t want =
        dgpp::float_to_fp8_e4m3_bits(bf16_bits_to_float(src[static_cast<size_t>(i)]) * (448.0f / absmax));
    require(bytes[static_cast<size_t>(i)] == want, "code " + std::to_string(i));
    require(dgpp::latent_fp8_decode_bf16(want, scale) ==
                float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(want) * scale),
            "decode " + std::to_string(i));
  }
}

// The window ring's mixed-precision record (the G8's first bullet's close,
// 2026-09-21): the reference's window KV quantizes ONLY the non-RoPE 448
// dims to fp8 (per-64, e8m0) leaving the RoPE 64 in bf16 (ck:model.py:508-
// 512 `act_quant(kv[..., :-rd], 64, ...)` — "FP8-simulate non-rope dims
// to match QAT; rope dims stay bf16 for positional precision"). This
// oracle pins the kFp8BlockRope record's byte layout (the 584 B envelope's,
// the dsv4-native kv_cache.h's + the DSpark union kernel's
// dsv4_dspark_decode_record's) and its round-trip: the RoPE dims bit-exact
// bf16, the NoPE dims the release's per-64 e4m3/e8m0 (an independent
// bit-field ceil-log2 cross-check, not the shared helper's).
DGPP_TEST(latent_format_fp8blockrope_mixed_record_layout_and_roundtrip) {
  const int kv_lora = 512;
  const int n = dgpp::latent_fp8blockrope_nope(kv_lora);  // 448 the NoPE prefix
  const size_t rope_off = dgpp::latent_fp8blockrope_rope_offset(kv_lora);
  const size_t scale_off = dgpp::latent_fp8blockrope_scale_offset(kv_lora);
  // ---- the byte layout (the 584 B envelope's the regions' pin) --------
  require(dgpp::latent_row_bytes(LatentFormat::kFp8BlockRope, kv_lora) == 584, "the 584 B envelope");
  require(n == 448, "the NoPE prefix's 448");
  require(rope_off == 448, "the RoPE tail's at 448");
  require(scale_off == 576, "the 7 scales' at 576");
  require(scale_off + 7 == 583, "the pad's byte's at 583");
  // A hand-crafted record: the known codes / scales / tail decode to the
  // exact bf16 (the layout's pin, independent of the quantizer's).
  {
    std::vector<uint8_t> rec(584, 0);
    for (int i = 0; i < 448; ++i) rec[static_cast<size_t>(i)] = 0x3C;  // e4m3's 1.5
    rec[576] = 127;  // 2^(127 - 127) = 1.0
    rec[577] = 128;  // 2^1 = 2.0
    const uint16_t t0 = float_to_bf16_bits(3.5f);
    for (int i = 0; i < 64; ++i) {
      rec[448 + 2 * i] = static_cast<uint8_t>(t0 & 0xFFu);
      rec[448 + 2 * i + 1] = static_cast<uint8_t>(t0 >> 8);
    }
    std::vector<uint16_t> back(static_cast<size_t>(kv_lora));
    dgpp::latent_dequantize_row_host(LatentFormat::kFp8BlockRope, rec.data(), 1.0f, kv_lora, back.data());
    for (int i = 0; i < 64; ++i) require(back[static_cast<size_t>(i)] == float_to_bf16_bits(1.5f),
                                          "the block-0's decode's 1.5 x 1.0");
    for (int i = 64; i < 128; ++i) require(back[static_cast<size_t>(i)] == float_to_bf16_bits(3.0f),
                                            "the block-1's decode's 1.5 x 2.0");
    for (int i = 448; i < kv_lora; ++i)
      require(back[static_cast<size_t>(i)] == t0, "the RoPE tail's the raw bf16's the bit-exact's");
  }
  // ---- the round-trip (random rows: the RoPE's bit-exact's, the NoPE's
  // the per-64's the e4m3/e8m0's the reference's the act_quant's) -------
  for (uint64_t seed = 1; seed <= 8; ++seed) {
    const std::vector<uint16_t> src = row(seed, kv_lora);
    std::vector<uint8_t> bytes(dgpp::latent_row_bytes(LatentFormat::kFp8BlockRope, kv_lora), 0xAA);
    float scale = -1.0f;
    dgpp::latent_quantize_row_host(LatentFormat::kFp8BlockRope, src.data(), kv_lora, bytes.data(), &scale);
    require(scale == 1.0f, "the self-describing's the row scale's the unused's");
    std::vector<uint16_t> back(static_cast<size_t>(kv_lora));
    dgpp::latent_dequantize_row_host(LatentFormat::kFp8BlockRope, bytes.data(), scale, kv_lora, back.data());
    // The RoPE tail: bit-exact (the format's whole reason for existing —
    // the positional signal the reference keeps exact).
    for (int i = 448; i < kv_lora; ++i)
      require(back[static_cast<size_t>(i)] == src[static_cast<size_t>(i)],
              std::string("row ") + std::to_string(seed) + " RoPE dim " + std::to_string(i) + " bit-exact");
    // The NoPE prefix: per-64 e4m3 with the e8m0 scale, against an
    // independent per-block reference (the bit-field's ceil-log2's, the
    // 1e-4's floor's, the e4m3's the RNE's encode's).
    for (int b = 0; b < 7; ++b) {
      float bmax = 0.0f;
      for (int j = 0; j < 64; ++j) {
        const float a = std::fabs(bf16_bits_to_float(src[static_cast<size_t>(b * 64 + j)]));
        bmax = a > bmax ? a : bmax;
      }
      const float a = bmax > 1e-4f ? bmax : 1e-4f;
      const uint32_t u = std::bit_cast<uint32_t>(a * (1.0f / 448.0f));
      const int k = static_cast<int>((u >> 23) & 0xFFu) - 127 + ((u & 0x7FFFFFu) != 0u ? 1 : 0);
      const int b_exp = k + 127;
      const uint8_t want_sb = static_cast<uint8_t>(b_exp < 0 ? 0 : (b_exp > 254 ? 254 : b_exp));
      require(bytes[scale_off + static_cast<size_t>(b)] == want_sb,
              std::string("row ") + std::to_string(seed) + " block " + std::to_string(b) + "'s scale byte");
      const float s = dgpp::e8m0_byte_to_float(want_sb);
      for (int j = 0; j < 64; ++j) {
        const int i = b * 64 + j;
        const uint8_t want_code =
            dgpp::float_to_fp8_e4m3_bits(bf16_bits_to_float(src[static_cast<size_t>(i)]) / s);
        require(bytes[static_cast<size_t>(i)] == want_code,
                std::string("row ") + std::to_string(seed) + " code " + std::to_string(i));
        const uint16_t want_back =
            float_to_bf16_bits(dgpp::fp8_e4m3_bits_to_float(want_code) * s);
        require(back[static_cast<size_t>(i)] == want_back,
                std::string("row ") + std::to_string(seed) + " dequant " + std::to_string(i));
      }
    }
    // The e4m3 class's error bound over the NoPE (the 3-bit mantissa's).
    require(rel_l2(src, back) <= 0.07, std::string("row ") + std::to_string(seed) + "'s NoPE's the e4m3's class's");
    // The pad's byte's the written's (never the buffer's old contents's,
    // the 0xAA's fill's the initcheck's discipline's).
    require(bytes[583] == 0, "the pad's byte's the zero's");
  }
  // An all-zero row: every code 0, the scale's byte the 1e-4 floor's
  // (2^-22's the 105's the 127 - 22's), the tail's zeros' the back's
  // zeros's.
  {
    const std::vector<uint16_t> zeros(static_cast<size_t>(kv_lora), 0);
    std::vector<uint8_t> bytes(dgpp::latent_row_bytes(LatentFormat::kFp8BlockRope, kv_lora), 0xFF);
    float scale = -1.0f;
    dgpp::latent_quantize_row_host(LatentFormat::kFp8BlockRope, zeros.data(), kv_lora, bytes.data(), &scale);
    for (int i = 0; i < 448; ++i) require(bytes[static_cast<size_t>(i)] == 0, "the zero row's codes");
    // The 1e-4 floor's byte (the a * (1/448)'s the bit-field's ceil's
    // the 105's the 2^-22's) — computed independently here.
    const uint32_t uf = std::bit_cast<uint32_t>(1e-4f * (1.0f / 448.0f));
    const int kf = static_cast<int>((uf >> 23) & 0xFFu) - 127 + ((uf & 0x7FFFFFu) != 0u ? 1 : 0);
    const uint8_t want_floor_byte =
        static_cast<uint8_t>((kf + 127) < 0 ? 0 : ((kf + 127) > 254 ? 254 : (kf + 127)));
    for (int b = 0; b < 7; ++b)
      require(bytes[scale_off + static_cast<size_t>(b)] == want_floor_byte, "the zero row's scale");
    std::vector<uint16_t> back(static_cast<size_t>(kv_lora), 1);
    dgpp::latent_dequantize_row_host(LatentFormat::kFp8BlockRope, bytes.data(), scale, kv_lora, back.data());
    for (uint16_t v : back) require(v == 0, "the zero row decodes to zeros");
  }
}
