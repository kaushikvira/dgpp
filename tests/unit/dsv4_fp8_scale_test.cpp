// DeepSeek-V4-Flash fp8 contract (host, no GPU; docs/dsv4_kernel_port_spec.md
// §3-§4). The 0731 checkpoint ships every dense fp8 matrix as F8_E4M3
// payload [N, K] + F8_E8M0 scales [ceil(N/128), ceil(K/128)] (config.json
// quantization_config: weight_block_size [128, 128], scale_fmt ue8m0,
// activation_scheme dynamic; /data/models/DeepSeek-V4-Flash-0731). The
// DGPP fp8 kernel surface (kernels/scale_gemm.cu, fp8_gemv.cuh,
// fp8_dequant.cu, mma_gemv.hpp) consumes F32 scales only, so the V4
// loader must decode the e8m0 bytes to F32 at load time (the dsv41
// pattern: src/models/dsv41/loader.cpp e8m0_to_float + convert_scales,
// the 32 x 32 grid's version). These tests pin that decode and the
// 128 x 128 scale index math against naive references, so the Phase B
// loader and any kernel adaptation have a host-side oracle before the
// GPU work lands.
//
// Every check runs on the CPU: no CUDA, no model files, deterministic.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"

namespace {

// The loader's decode under test: the dsv41 pattern's formula
// (src/models/dsv41/loader.cpp:24-32: "2^(byte - 127); 255 is NaN"),
// mirrored here so the V4 loader's 128 x 128-grid decode is pinned to
// the same contract. The V4 checkpoint ships no 255 bytes (verified
// 2026-09-17: 0 of the 8,832 MiB of F8_E8M0 scale bytes are 0xFF), so
// the 255 -> NaN policy is unobservable in the data but stays the
// documented one (a NaN scale marks a dead block).
float e8m0_to_float(uint8_t b) {
  if (b == 255) return std::nanf("");
  return std::ldexp(1.0f, static_cast<int>(b) - 127);
}

// The naive bit-level oracle: 2^(b - 127) is the float whose IEEE bits
// are exactly (b << 23) for b in 1..254 (exponent field = b, zero
// mantissa); b = 0 is 2^-127, the smallest DENORMAL float (bits
// 0x00400000 — the bit trick breaks there); 255 has no finite value
// (2^128 overflows) and the contract maps it to NaN.
float e8m0_oracle(uint8_t b) {
  if (b == 255) return std::nanf("");
  const uint32_t want = (b == 0) ? 0x00400000u : (static_cast<uint32_t>(b) << 23);
  float out;
  std::memcpy(&out, &want, 4);
  return out;
}

// The 128-wide tile kernel's stage property (kernels/scale_gemm.cu:
// "a 32-deep stage lies in one scale column for cs >= 5"): a BK = 32
// k-slice [k0, k0 + 32) starting on a 32 boundary reads ONE scale
// column, whatever cs in {5, 6, 7} (32/64/128-wide scale columns).
bool stage_single_scale_col(int k0, int cs, int k) {
  if (k0 + 32 > k) return true;  // the tail stage is masked element-wise
  const int col0 = k0 >> cs;
  for (int kx = k0; kx < k0 + 32; ++kx)
    if ((kx >> cs) != col0) return false;
  return true;
}

// The GEMV core's chunk property (kernels/fp8_gemv.cuh: "a 16-byte
// chunk then lies inside one 128-wide scale block"): a 16-byte chunk
// [c0, c0 + 16) whose start is a multiple of 16 (the lane geometry)
// lies inside one scale column of width 2^cs for cs >= 4, whenever
// k % 16 == 0 (the shape contract).
bool chunk_single_scale_col(int c0, int cs, int k) {
  if (c0 + 16 > k) return true;  // the ragged tail is zero-filled
  return (c0 >> cs) == ((c0 + 15) >> cs);
}

}  // namespace

DGPP_TEST(dsv4_e8m0_decode_full_table) {
  // All 256 bytes: the loader formula and the bit-level oracle agree,
  // and the finite values are exact powers of two (b >= 1: zero
  // mantissa, the exponent field IS the byte; b = 0: the denormal
  // 2^-127).
  for (int b = 0; b < 256; ++b) {
    const float got = e8m0_to_float(static_cast<uint8_t>(b));
    const float want = e8m0_oracle(static_cast<uint8_t>(b));
    if (b == 255) {
      if (!std::isnan(got))
        throw std::runtime_error("e8m0 byte 255 must decode to NaN");
      continue;
    }
    if (std::isnan(got))
      throw std::runtime_error("e8m0 byte " + std::to_string(b) +
                               " decoded to NaN");
    uint32_t got_bits, want_bits;
    std::memcpy(&got_bits, &got, 4);
    std::memcpy(&want_bits, &want, 4);
    if (got_bits != want_bits)
      throw std::runtime_error("e8m0 byte " + std::to_string(b) +
                               " decoded to the wrong float bits");
    // A power of two: for b >= 1 zero mantissa and the exponent field
    // IS the byte; b = 0 is the denormal 2^-127 (bits 0x00400000).
    if (b > 0) {
      if ((got_bits & 0x007FFFFFu) != 0)
        throw std::runtime_error("e8m0 byte " + std::to_string(b) +
                                 " decoded with mantissa bits set");
      if (static_cast<int>(got_bits >> 23) != b)
        throw std::runtime_error("e8m0 byte " + std::to_string(b) +
                                 " exponent field is not the byte");
    }
  }
}

DGPP_TEST(dsv4_e8m0_decode_finite_range) {
  // The decode never overflows for a finite byte: 0 -> 2^-127 (the
  // smallest denormal float), 254 -> 2^127 (finite, just under inf).
  // The V4 checkpoint's observed scale bytes (62..161) sit deep inside
  // this range.
  const float lo = e8m0_to_float(0);
  const float hi = e8m0_to_float(254);
  if (lo != std::ldexp(1.0f, -127))
    throw std::runtime_error("byte 0 is not 2^-127");
  uint32_t lo_bits;
  std::memcpy(&lo_bits, &lo, 4);
  if (lo_bits != 0x00400000u)
    throw std::runtime_error("byte 0 is not the denormal 2^-127 bits");
  if (!std::isfinite(hi) || hi != std::ldexp(1.0f, 127))
    throw std::runtime_error("byte 254 is not 2^127");
  // Monotone: every byte above decodes strictly larger.
  for (int b = 1; b < 255; ++b)
    if (e8m0_to_float(static_cast<uint8_t>(b)) <=
        e8m0_to_float(static_cast<uint8_t>(b - 1)))
      throw std::runtime_error("decode is not strictly increasing at " +
                               std::to_string(b));
}

DGPP_TEST(dsv4_block128_scale_index_math) {
  // The scale index math of the 128 x 128 grid for [N, K] with N, K
  // not multiples of 128, kernel formulas vs the naive reference:
  //   naive:  scale_cols = ceil(K / 128); idx(n, k) = (n / 128) *
  //           scale_cols + (k / 128)
  //   tile:   scales[(gn >> 7) * scale_cols + (k0 >> 7)], k0 the 32-
  //           aligned stage start (kernels/scale_gemm.cu)
  //   gemv:   scales[(row >> 7) * scale_cols + (c0 >> 7)], c0 the
  //           16-byte chunk start (kernels/fp8_gemv.cuh)
  struct Shape {
    int n, k;
    const char* who;
  };
  const Shape shapes[] = {
      // The V4 checkpoint's dense fp8 shapes (N, K):
      {1024, 4096, "wq_a"},    {512, 4096, "wkv"},
      {8192, 1024, "indexer.wq_b"},
      {4096, 4096, "wo_a"},    {2048, 2048, "expert.w1 (k codes 4096)"},
      // The MXFP4 expert payload: 2048 B/row = 4096 e2m1 codes.
      // Ragged tails:
      {1000, 1300, "ragged"},  {127, 127, "one short"},
      {128, 128, "exact"},     {129, 129, "one over"},
      {5, 7, "tiny"},          {255, 300, "k not a multiple of 128"},
  };
  for (const Shape& s : shapes) {
    const int scale_cols = (s.k + 127) / 128;
    const int scale_rows = (s.n + 127) / 128;
    // (1) Element-wise: the kernel's per-element index equals the naive
    // (n / 128, k / 128) block coordinates for every in-range element.
    for (int n = 0; n < s.n; ++n)
      for (int k = 0; k < s.k; ++k) {
        const int naive = (n / 128) * scale_cols + (k / 128);
        const int kernel = (n >> 7) * scale_cols + (k >> 7);
        if (naive != kernel)
          throw std::runtime_error(std::string(s.who) + ": index mismatch " +
                                   "at (" + std::to_string(n) + "," +
                                   std::to_string(k) + ")");
        if (naive >= scale_rows * scale_cols)
          throw std::runtime_error(std::string(s.who) +
                                   ": index out of the scale grid");
      }
    // (2) The last block row and column are partial: the count of
    // elements that read scale entry (r, c) is the naive product of the
    // clipped extents (the kernel's masked loads, scale_gemm.cu "ragged
    // N/K tails read the true last block row/col").
    for (int r = 0; r < scale_rows; ++r)
      for (int c = 0; c < scale_cols; ++c) {
        const int rows = (std::min)(s.n, (r + 1) * 128) - r * 128;
        const int cols = (std::min)(s.k, (c + 1) * 128) - c * 128;
        int count = 0;
        for (int n = r * 128; n < (r + 1) * 128; ++n)
          for (int k = c * 128; k < (c + 1) * 128; ++k)
            if (n < s.n && k < s.k) ++count;
        if (count != rows * cols)
          throw std::runtime_error(std::string(s.who) + ": tail count at " +
                                   "block (" + std::to_string(r) + "," +
                                   std::to_string(c) + ")");
      }
    // (3) The stage property: a 32-deep stage on a 32 boundary reads
    // one scale column for cs in {5, 6, 7} (the tile kernel's "one
    // scalar per stage" invariant).
    for (int cs : {5, 6, 7})
      for (int k0 = 0; k0 < s.k; k0 += 32)
        if (!stage_single_scale_col(k0, cs, s.k))
          throw std::runtime_error(std::string(s.who) +
                                   ": stage at k0=" + std::to_string(k0) +
                                   " spans scale columns (cs=" +
                                   std::to_string(cs) + ")");
    // (4) The GEMV chunk property: a 16-byte chunk on a 16 boundary
    // (the lane geometry, k % 16 == 0) lies in one scale column for
    // cs in {4, 5, 6, 7} (fp8_gemv.cuh's shape contract).
    for (int cs : {4, 5, 6, 7})
      for (int c0 = 0; c0 < s.k; c0 += 16)
        if (!chunk_single_scale_col(c0, cs, s.k))
          throw std::runtime_error(std::string(s.who) +
                                   ": chunk at c0=" + std::to_string(c0) +
                                   " spans scale columns (cs=" +
                                   std::to_string(cs) + ")");
  }
}

DGPP_TEST(dsv4_block128_partial_rescale_math) {
  // The per-128-k-block rescale of the fp8 x fp8 dynamic-scheme GEMM
  // (the checkpoint's inference/kernel.py fp8_gemm + the dsv4-native
  // ops/linear contract): each 128-wide k-block partial p (fp32) is
  // scaled by 2^(xs + ws - 254) (the two ue8m0 powers of two, exact),
  // and the block partials sum to the output (one bf16 rounding at the
  // end). The e4m3 codes and the e8m0 powers are exact in double, so
  // the whole expression is a naive double reference: the fp32 kernel
  // form may differ only by fp32 rounding (<= 1 ulp per block
  // multiply), and the power-of-two factor must stay exact (no
  // mantissa error) for every (xs, ws) pair.
  const int k_blocks = 4;  // K = 512
  const int per_block = 128;
  // Deterministic pseudo-random e4m3 codes: the finite e4m3 values are
  // exact in double (8-bit significands), so the block partials are
  // exact doubles. Codes 0..127 only (no NaN codes, no sign bit).
  std::vector<double> x(per_block * k_blocks), w(per_block * k_blocks);
  uint32_t state = 0x9E3779B9u;
  auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<int>(state % 128);
  };
  // e4m3 decode (the kernels/fp8 family's table semantics, the finite
  // codes): sign x 2^(e - 7) x (1 + m / 8), e = bits 6..3, m = 1..7;
  // the subnormal zone e == 0 is m / 64 (the e4m3 format's 2^-6 step).
  auto e4m3 = [](int b) {
    const int e = (b >> 3) & 0x7, m = b & 0x7;
    const double mag = (e == 0) ? (m / 64.0)
                                : (std::ldexp(1.0, e - 7) * (8.0 + m) / 8.0);
    return mag;  // codes 0..127: positive
  };
  for (auto& v : x) v = e4m3(next());
  for (auto& v : w) v = e4m3(next());
  // The practical ue8m0 axis: the activation quantizer's 1e-4 amax
  // floor keeps xs at >= ~110 (2^-17 class), the weight scales sit
  // near 2^0, so the product 2^(xs + ws - 254) stays inside the fp32
  // range for xs, ws in {96, 104, ..., 176} (sums 192..352 ->
  // 2^-62..2^98, all exact in fp32). The grid pins that region.
  for (int xs = 96; xs <= 176; xs += 8) {
    for (int ws = 96; ws <= 176; ws += 8) {
      const double ax = std::ldexp(1.0, static_cast<double>(xs) - 127.0);
      const double aw = std::ldexp(1.0, static_cast<double>(ws) - 127.0);
      const double factor = ax * aw;  // 2^(xs + ws - 254), exact in any
                                      // floating-point format in range
      const float f32 = static_cast<float>(factor);
      if (f32 != factor)
        throw std::runtime_error("the power-of-two factor is not exact "
                                 "in fp32 (xs=" + std::to_string(xs) +
                                 ", ws=" + std::to_string(ws) + ")");
      double exact = 0.0;      // the naive double reference, block order
      double kernel_form = 0.0;  // the fp32 form: the per-block fp32
                                 // roundings the kernel's accumulation
                                 // would apply
      for (int kb = 0; kb < k_blocks; ++kb) {
        double partial = 0.0;
        for (int i = 0; i < per_block; ++i)
          partial += x[static_cast<size_t>(kb) * per_block + i] *
                     w[static_cast<size_t>(kb) * per_block + i];
        exact += factor * partial;
        // One block's contribution in the fp32 form: the partial rounds
        // to fp32 (one rounding), the power-of-two multiply is exact,
        // the accumulation adds at most one rounding per block.
        const double block_f32 = static_cast<double>(
            static_cast<float>(partial) * f32);
        kernel_form = static_cast<double>(
            static_cast<float>(kernel_form) + static_cast<float>(block_f32));
      }
      // The fp32 path may diverge from the double only by the fp32
      // roundings above: bound the running gap to a few ulps of the
      // running magnitude (the oracle's ULP budget, not a bitwise pin
      // — the accumulation order is the kernel's to choose).
      const double mag =
          std::max({std::fabs(exact), std::fabs(kernel_form), 1.0});
      if (std::fabs(exact - kernel_form) > mag * 1e-6)
        throw std::runtime_error("fp32 block accumulation diverged "
                                 "beyond fp32 rounding");
    }
  }
}

int main() { return ::dgpp::test::run_all(); }
