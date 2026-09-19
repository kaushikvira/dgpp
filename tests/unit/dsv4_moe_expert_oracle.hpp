#pragma once
// The DeepSeek-V4-Flash (deepseek_v4) MXFP4 routed-expert CPU oracle
// (docs/dsv4_kernel_port_spec.md §2.4, dsv4-native src/ops/moe/experts'
// V4 re-expression): the fused W4A16 MXFP4 decode (e2m1 x 2^(byte - 127)
// EXACT in fp32, the per-32 block partial scaled once, the swiglu-clamp,
// the router weight folded into the down epilogue). The naive per-expert
// DOUBLE reference (dsv4-native ops/common/op_test.h:892 moe_mxfp4_ref +
// :740 moe_mxfp4_decode_expert). Host-only, CUDA-free (the GPU gate's
// tests/cuda/dsv4_moe_slot_test.cu's the same reference's, the CPU's
// tests/unit/dsv4_moe_oracle_test.cpp's the pinned's form's).
//
// Borrowed, not rewritten (house rule 2): the expert mirrors the
// dsv4-native ops/moe/experts (src/ops/moe/experts/experts.h:1-60); the
// unit test cross-checks it against an INDEPENDENT formulation (the
// dsv4_moe_mxfp4_expert_decode_and_clamp's per-nibble recompute's), so a
// silent corruption of the borrowed form is caught there, not on the
// GPU.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "common/dtypes.hpp"

namespace dsv4_moe_oracle {

using dgpp::bf16_bits_to_float;

// bf16 word -> exact double.
inline double bf16_to_d64(uint16_t w) { return static_cast<double>(bf16_bits_to_float(w)); }

// e2m1 (MXFP4) nibble -> value: the 8-code table (0, .5, 1, 1.5, 2, 3,
// 4, 6) with the sign bit 0x8 (dsv4-native ops/common/op_test.h
// e2m1_ref_value's kMagnitude table).
inline double e2m1_value(uint8_t nib) {
  static const double kMag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
  const double v = kMag[nib & 0x7];
  return (nib & 0x8) ? -v : v;
}
// ue8m0 byte -> power of two: 2^(b - 127) (the tree's e8m0_to_float
// contract, tests/unit/dsv4_fp8_scale_test.cpp).
inline double e8m0_pow2(uint8_t b) { return std::ldexp(1.0, static_cast<int>(b) - 127); }
// The reference's silu in double (the test's bounded inputs keep exp in
// range; a different code path from a kernel's fp32 expf).
inline double silu_ref(double g) { return g / (1.0 + std::exp(-g)); }

// The MXFP4 routed expert (dsv4-native src/ops/moe/experts): the fused
// W4A16 MXFP4 decode (e2m1 x 2^(byte - 127) EXACT in fp32, the per-32
// block partial scaled once, the swiglu-clamp, the router weight folded
// into the down epilogue). `x` [m, k] bf16, `w1`/`w3` [e, n, k/2] I8
// (two e2m1 per byte, the low nibble the even element), `w2` [e, k, n/2]
// I8, the scales [e, n, k/32] / [e, k, n/32] F8_E8M0. Returns the
// per-token output [m, k] (the sum over the selected experts of
// weight * the expert's SwiGLU down-projection, the reference's
// ascending-slot accumulation).
inline std::vector<double> moe_mxfp4_ref(const std::vector<uint16_t>& x,
                                         const std::vector<int32_t>& topk_indices,
                                         const std::vector<float>& topk_weights,
                                         const std::vector<uint8_t>& w1,
                                         const std::vector<uint8_t>& w1_scale,
                                         const std::vector<uint8_t>& w3,
                                         const std::vector<uint8_t>& w3_scale,
                                         const std::vector<uint8_t>& w2,
                                         const std::vector<uint8_t>& w2_scale,
                                         int m, int e, int n, int k, int topk,
                                         double activation_clamp) {
  const std::size_t nk = static_cast<size_t>(n) * k;
  const std::size_t kn = static_cast<size_t>(k) * n;
  std::vector<double> out(static_cast<size_t>(m) * k, 0.0);
  for (int mm = 0; mm < m; ++mm) {
    const std::size_t xr = static_cast<size_t>(mm) * k;
    // Decode the DISTINCT used experts' planes once (the small-shape
    // path; the double GEMM over the decoded planes). The per-expert
    // down-projection `y` depends only on the expert + x (NOT the
    // weight), so it is cached per distinct expert and the per-slot
    // accumulation folds the router weight in (the reference's
    // `weight * y` over the slots).
    std::vector<bool> used_e(static_cast<size_t>(e), false);
    std::vector<int> used;
    for (int j = 0; j < topk; ++j) {
      const int ei = topk_indices[static_cast<size_t>(mm) * topk + j];
      if (!used_e[static_cast<size_t>(ei)]) {
        used_e[static_cast<size_t>(ei)] = true;
        used.push_back(ei);
      }
    }
    // The per-distinct-expert down-projection y [k] (the expert's SwiGLU
    // output, the router weight NOT folded in yet).
    std::vector<std::vector<double>> y_cache(used.size());
    for (std::size_t u = 0; u < used.size(); ++u) {
      const int ei = used[u];
      const std::size_t w1e = static_cast<size_t>(ei) * n * (k / 2);
      const std::size_t w3e = static_cast<size_t>(ei) * n * (k / 2);
      const std::size_t w2e = static_cast<size_t>(ei) * k * (n / 2);
      const std::size_t w1se = static_cast<size_t>(ei) * n * (k / 32);
      const std::size_t w3se = static_cast<size_t>(ei) * n * (k / 32);
      const std::size_t w2se = static_cast<size_t>(ei) * k * (n / 32);
      std::vector<double> w1f(nk), w3f(nk), w2f(kn);
      for (int nr = 0; nr < n; ++nr)
        for (int kk = 0; kk < k; ++kk) {
          // The per-32-block scale (the MXFP4 blockscale-32 geometry):
          // the block index kk/32 (a per-row scale would silently
          // corrupt any expert whose blocks carry different scales).
          const double s1 = e8m0_pow2(w1_scale[w1se + static_cast<size_t>(nr) * (k / 32) + kk / 32]);
          const double s3 = e8m0_pow2(w3_scale[w3se + static_cast<size_t>(nr) * (k / 32) + kk / 32]);
          const uint8_t b1 = w1[w1e + static_cast<size_t>(nr) * (k / 2) + kk / 2];
          const uint8_t b3 = w3[w3e + static_cast<size_t>(nr) * (k / 2) + kk / 2];
          w1f[static_cast<size_t>(nr) * k + kk] = e2m1_value(kk % 2 == 0 ? b1 & 0xFu : b1 >> 4) * s1;
          w3f[static_cast<size_t>(nr) * k + kk] = e2m1_value(kk % 2 == 0 ? b3 & 0xFu : b3 >> 4) * s3;
        }
      for (int kk = 0; kk < k; ++kk)
        for (int nn = 0; nn < n; ++nn) {
          const double s2 = e8m0_pow2(w2_scale[w2se + static_cast<size_t>(kk) * (n / 32) + nn / 32]);
          const uint8_t b2 = w2[w2e + static_cast<size_t>(kk) * (n / 2) + nn / 2];
          w2f[static_cast<size_t>(kk) * n + nn] = e2m1_value(nn % 2 == 0 ? b2 & 0xFu : b2 >> 4) * s2;
        }
      // The expert's SwiGLU: gate = x @ w1^T, up = x @ w3^T (double
      // over the decoded planes: the e2m1 x bf16 products are exact in
      // double, the k-term sum < 2^53), silu-clamp x up, down = inter
      // @ w2^T.
      std::vector<double> gate(n), up(n), inter(n);
      for (int nr = 0; nr < n; ++nr) {
        double g = 0.0, u = 0.0;
        for (int kk = 0; kk < k; ++kk) {
          const double xv = bf16_to_d64(x[xr + kk]);
          g += xv * w1f[static_cast<size_t>(nr) * k + kk];
          u += xv * w3f[static_cast<size_t>(nr) * k + kk];
        }
        gate[nr] = g;
        up[nr] = u;
      }
      for (int nr = 0; nr < n; ++nr) {
        double sg = silu_ref(gate[nr]);
        sg = std::min(std::max(sg, -activation_clamp), activation_clamp);  // the SwiGLU clamp
        inter[nr] = sg * up[nr];
      }
      y_cache[u].resize(k);
      for (int kk = 0; kk < k; ++kk) {
        double y = 0.0;
        for (int nn = 0; nn < n; ++nn)
          y += inter[nn] * w2f[static_cast<size_t>(kk) * n + nn];
        y_cache[u][kk] = y;
      }
    }
    // The per-slot accumulation: out[mm] += weight_j * y_{expert_j} (the
    // router weight folded into the down epilogue, the reference's
    // ascending-slot order).
    for (int j = 0; j < topk; ++j) {
      const int ei = topk_indices[static_cast<size_t>(mm) * topk + j];
      const std::size_t u = static_cast<std::size_t>(std::find(used.begin(), used.end(), ei) - used.begin());
      const double weight = topk_weights[static_cast<size_t>(mm) * topk + j];
      for (int kk = 0; kk < k; ++kk)
        out[xr + kk] += weight * y_cache[u][kk];
    }
  }
  return out;
}

// A deterministic MXFP4 expert-plane generator (the e2m1 nibbles + the
// e8m0 scale bytes, bounded so the GEMM stays finite).
inline void fill_mxfp4(std::vector<uint8_t>* payload, std::vector<uint8_t>* scale, std::size_t rows,
                        std::size_t cols, std::uint32_t seed) {
  payload->resize(rows * cols / 2);
  scale->resize(rows * ((cols + 31) / 32));
  std::uint32_t s = seed;
  auto next = [&]() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  };
  for (std::size_t i = 0; i < payload->size(); ++i)
    (*payload)[i] = static_cast<uint8_t>(next());  // two e2m1 nibbles per byte
  for (std::size_t i = 0; i < scale->size(); ++i)
    (*scale)[i] = static_cast<uint8_t>(127 + static_cast<int>(next() % 9) - 4);  // 2^-4..2^4
}

}  // namespace dsv4_moe_oracle
