// C.1.0: the MXFP8 scale-factor (SF) fragment spec for
//   mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
// on sm_121a (docs/qwen_fp8_phase_c_plan.md §3, the C.1.0 spike that gates C.1).
//
// The instruction is verified to run on compute_121a (all-ones e4m3 0x38 with
// ue8m0 0x7f scales, C=1 -> 132.0 = 4*(32*1)+4; both scales 0x80 -> 516.0,
// i.e. sfa*sfb apply multiplicatively per k-block) and is rejected on plain
// sm_121, so the 121a arch flag (CMakeLists.txt) is load-bearing.
//
// What this file pins (the governed outcome of C.1.0): which lane's which byte
// of which register supplies the scale for which (row, k-block) of A and which
// (col, k-block) of B, so the C.1a kernel can read/write the SF registers
// without guessing. The layout asserted here (PTX ISA 9.7.14.6.1,
// .kind::mxf8f6f4 .scale_vec::1X):
//
//   lane = 4*g + t (g = lane/4, t = lane%4)
//   SFA: 1 .b32 register per thread:  byte0 = SFA[g]   (row g,   block of 32)
//                                          byte1 = SFA[g+8] (row g+8)
//         (the 4 threads of a group carry the same two bytes)
//   SFB: 1 .b32 register per thread:  byte0 = SFB[2t]  (col 2t)
//                                          byte1 = SFB[2t+1] (col 2t+1)
//   selectors: a PAIR of .b16 registers follows each scale register
//   ({sfa} {sfa_sel0,sfa_sel1} {sfb} {sfb_sel0,sfb_sel1}, the mxf4nvf4 4X
//   operand order); all four are 0 for scale_vec::1X.
//
// A row of the m16n8k32 A tile is exactly one 32-wide k-block, so one mma
// carries ONE scale per A row and ONE per B column; the 8-scales-per-256-dim-row
// design is 8 sequential k-steps, each with its own SFA/SFB registers (no
// k-steps share a byte). The oracle below proves the A side can hold 16
// distinct row scales (rows g and g+8 differ inside one register) -- if the
// hardware forced a uniform pair, the spec mapping would fail and the
// uniform mapping would pass, which the sensitivity test reports explicitly.
//
// Exactness: the e4m3 codes are drawn from {0, +/-0.5, +/-1} (0x00, 0x30, 0x38,
// 0x50, 0x58), so every 32-term dot product is an exact multiple of 0.25 with
// |dot| <= 32; the ue8m0 scales are powers of two (2^-4..2^4 for A, 2^-3..2^3
// for B); the fp32 mma result is therefore exact and the host oracle compares
// bitwise (max diff must be 0).
//
// Run:  ctest -R qwen_fp8_mxfp8_sf   (or the built binary directly)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Minimal device buffer (the kda_test_helpers.hpp one drags in the KDA model
// headers; this test is self-contained).
struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;
  explicit DevBuf(size_t n) {
    if (n == 0) n = 16;
    if (cudaMalloc(&p, n) != cudaSuccess) throw std::runtime_error("test cudaMalloc failed");
    bytes = n;
  }
  DevBuf(DevBuf&&) = delete;
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  ~DevBuf() {
    if (p) cudaFree(p);
  }
  void upload(const void* h, size_t n) {
    if (n > bytes) throw std::runtime_error("upload overruns buffer");
    if (cudaMemcpy(p, h, n, cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("upload failed");
  }
  void download(void* h, size_t n) const {
    if (n > bytes) throw std::runtime_error("download overruns buffer");
    if (cudaMemcpy(h, p, n, cudaMemcpyDeviceToHost) != cudaSuccess)
      throw std::runtime_error("download failed");
  }
  template <typename T>
  T* as() {
    return static_cast<T*>(p);
  }
  template <typename T>
  const T* as() const {
    return static_cast<const T*>(p);
  }
};

// The block-scaled mma. `map` selects the SF byte mapping under test:
//   0 spec      : sfa byte0=SFA[g] byte1=SFA[g+8]; sfb byte0=SFB[2t] byte1=SFB[2t+1]
//   1 sfa_swap  : the two SFA bytes exchanged (a deliberately wrong mapping)
//   2 sfb_swap  : the two SFB bytes exchanged (a deliberately wrong mapping)
//   3 sfa_unif  : both SFA bytes = SFA[g] (the fallback the hardware would use
//                 if A-side scales were uniform across the 8-row pair)
// A/B/C fragment packing is the plain e4m3 m16n8k32 layout (pinned by
// qwen_fp8_mma_a_fragment_oracle): a0=A[g][4t..4t+3], a1=A[g+8][4t..4t+3],
// a2=A[g][4t+16..4t+19], a3=A[g+8][4t+16..4t+19]; b0=B[4t..4t+3][g],
// b1=B[4t+16..4t+19][g]; c0..c3 = D[g][2t], D[g][2t+1], D[g+8][2t], D[g+8][2t+1].
__global__ void mxfp8_sf_kernel(const uint8_t* a, const uint8_t* b, const uint8_t* sfa,
                                const uint8_t* sfb, int map, float* c) {
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  // A is 16x32 row-major (m x k): A[m][k] = a[m*32 + k]; the 4 k-values of a
  // row are contiguous.
  auto packA = [&](int row, int k0) {
    return static_cast<uint32_t>(a[row * 32 + k0]) |
           (static_cast<uint32_t>(a[row * 32 + k0 + 1]) << 8) |
           (static_cast<uint32_t>(a[row * 32 + k0 + 2]) << 16) |
           (static_cast<uint32_t>(a[row * 32 + k0 + 3]) << 24);
  };
  // B is 32x8 row-major (k x n): B[k][n] = b[k*8 + n]; the 4 k-values of a
  // column are spaced 8 apart (NOT contiguous).
  auto packB = [&](int k0, int col) {
    return static_cast<uint32_t>(b[k0 * 8 + col]) |
           (static_cast<uint32_t>(b[(k0 + 1) * 8 + col]) << 8) |
           (static_cast<uint32_t>(b[(k0 + 2) * 8 + col]) << 16) |
           (static_cast<uint32_t>(b[(k0 + 3) * 8 + col]) << 24);
  };
  const uint32_t a0 = packA(g, 4 * t), a1 = packA(g + 8, 4 * t);
  const uint32_t a2 = packA(g, 4 * t + 16), a3 = packA(g + 8, 4 * t + 16);
  const uint32_t b0 = packB(4 * t, g), b1 = packB(4 * t + 16, g);
  uint32_t sfa_r, sfb_r;
  const uint32_t sfa_spec = static_cast<uint32_t>(sfa[g]) | (static_cast<uint32_t>(sfa[g + 8]) << 8);
  const uint32_t sfb_spec = static_cast<uint32_t>(sfb[2 * t]) | (static_cast<uint32_t>(sfb[2 * t + 1]) << 8);
  switch (map) {
    case 1:
      sfa_r = static_cast<uint32_t>(sfa[g + 8]) | (static_cast<uint32_t>(sfa[g]) << 8);
      sfb_r = sfb_spec;
      break;
    case 2:
      sfa_r = sfa_spec;
      sfb_r = static_cast<uint32_t>(sfb[2 * t + 1]) | (static_cast<uint32_t>(sfb[2 * t]) << 8);
      break;
    case 3:
      sfa_r = static_cast<uint32_t>(sfa[g]) | (static_cast<uint32_t>(sfa[g]) << 8);
      sfb_r = sfb_spec;
      break;
    default:
      sfa_r = sfa_spec;
      sfb_r = sfb_spec;
      break;
  }
  float cc[4] = {0.f, 0.f, 0.f, 0.f};
  // The selector operands are PAIRS of .b16 registers interleaved after each
  // scale register (the mxf4nvf4 4X form in moe_w4a4.cu is the in-repo
  // precedent): {sfa} {sfa_sel0,sfa_sel1} {sfb} {sfb_sel0,sfb_sel1}. Both
  // selectors are 0 for scale_vec::1X.
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
      : "+f"(cc[0]), "+f"(cc[1]), "+f"(cc[2]), "+f"(cc[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa_r),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb_r),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
  c[g * 8 + 2 * t + 0] = cc[0];
  c[g * 8 + 2 * t + 1] = cc[1];
  c[(g + 8) * 8 + 2 * t + 0] = cc[2];
  c[(g + 8) * 8 + 2 * t + 1] = cc[3];
}

// The exact host oracle: C[m][n] = 2^(sfa[m]-127) * 2^(sfb[n]-127) * sum_k a[m][k]*b[k][n].
// Exact in fp32 for the value sets used (see the file header).
float oracle_c(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, const std::vector<uint8_t>& sfa,
               const std::vector<uint8_t>& sfb, int m, int n) {
  double dot = 0.0;
  for (int k = 0; k < 32; ++k)
    dot += static_cast<double>(dgpp::fp8_e4m3_bits_to_float(a[static_cast<size_t>(m) * 32 + k])) *
           dgpp::fp8_e4m3_bits_to_float(b[static_cast<size_t>(k) * 8 + n]);
  const double sc =
      std::ldexp(1.0, static_cast<int>(sfa[static_cast<size_t>(m)]) - 127) *
      std::ldexp(1.0, static_cast<int>(sfb[static_cast<size_t>(n)]) - 127);
  return static_cast<float>(dot * sc);
}

// Run one mapping over (a, b, sfa, sfb); return the max |mma - oracle| over the
// 16x8 C tile (0.0 = bitwise match).
float run_mapping(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b,
                  const std::vector<uint8_t>& sfa, const std::vector<uint8_t>& sfb, int map) {
  DevBuf da(a.size()), db(b.size()), dsf(24), dc(16 * 8 * 4);
  da.upload(a.data(), a.size());
  db.upload(b.data(), b.size());
  std::vector<uint8_t> sf(24);
  std::copy(sfa.begin(), sfa.end(), sf.begin());
  std::copy(sfb.begin(), sfb.end(), sf.begin() + 16);
  dsf.upload(sf.data(), sf.size());
  mxfp8_sf_kernel<<<1, 32>>>(da.as<uint8_t>(), db.as<uint8_t>(), dsf.as<uint8_t>(),
                             dsf.as<uint8_t>() + 16, map, dc.as<float>());
  DGPP_CUDA_OK(cudaGetLastError());
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<float> c(16 * 8);
  dc.download(c.data(), c.size() * 4);
  float max_diff = 0.f;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 8; ++n)
      max_diff = std::max(max_diff, std::fabs(c[static_cast<size_t>(m) * 8 + n] - oracle_c(a, b, sfa, sfb, m, n)));
  return max_diff;
}

// The distinct-scale inputs: e4m3 codes from {0, +/-0.5, +/-1} (exact small
// dots) and 16 / 8 DISTINCT ue8m0 bytes (a wrong byte mapping cannot hide).
struct SfInputs {
  std::vector<uint8_t> a, b, sfa, sfb;
};
SfInputs distinct_inputs() {
  SfInputs x;
  x.a.resize(16 * 32);
  x.b.resize(32 * 8);
  const uint8_t vals[5] = {0x00u, 0x30u, 0x38u, 0x50u, 0x58u};  // 0, 0.5, 1, -0.5, -1
  uint64_t seed = 7;
  auto rnd = [&]() {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<size_t>((seed >> 33) % 5);
  };
  for (auto& v : x.a) v = vals[rnd()];
  for (auto& v : x.b) v = vals[rnd()];
  x.sfa.resize(16);
  x.sfb.resize(8);
  for (int m = 0; m < 16; ++m) x.sfa[static_cast<size_t>(m)] = static_cast<uint8_t>(127 - 4 + (m % 9));
  for (int n = 0; n < 8; ++n) x.sfb[static_cast<size_t>(n)] = static_cast<uint8_t>(127 - 3 + (n % 7));
  return x;
}

}  // namespace

// The verified probe from the plan (appendix A): all e4m3 codes 0x38 (1.0),
// ue8m0 scales 0x7f (2^0) -> each C element 32.0 (the plan's 132.0 was with
// C=1: 4*(32*1)+4); scales 0x80 (2^1) -> 128.0 each (the plan's 516.0).
DGPP_TEST(qwen_fp8_mxfp8_sf_sanity) {
  std::vector<uint8_t> a(16 * 32, 0x38u), b(32 * 8, 0x38u);
  for (uint8_t sc : {0x7fu, 0x80u}) {
    std::vector<uint8_t> sfa(16, sc), sfb(8, sc);
    const float d = run_mapping(a, b, sfa, sfb, 0);
    const float expect = sc == 0x7fu ? 32.f : 128.f;
    std::printf("[mxfp8-sf] sanity all-ones scale 0x%02x: max |mma - %.0f| = %.3g\n", sc, expect, d);
    require(d == 0.f, "MXFP8 all-ones sanity disagrees with the verified probe");
  }
}

// The spec mapping over distinct per-row / per-col scales must match the exact
// host oracle bitwise. This is the C.1.0 gate: max diff 0.
DGPP_TEST(qwen_fp8_mxfp8_sf_oracle) {
  const SfInputs x = distinct_inputs();
  const float d = run_mapping(x.a, x.b, x.sfa, x.sfb, 0);
  std::printf("[mxfp8-sf] SF-fragment oracle (spec mapping, 16x8 distinct scales): "
              "max |mma - host oracle| = %.3g\n",
              d);
  require(d == 0.f,
          "MXFP8 SF-fragment spec mapping (sfa byte0=row g / byte1=row g+8; sfb byte0=col 2t / "
          "byte1=col 2t+1) disagrees with the exact host oracle");
}

// Sensitivity: a deliberately wrong byte mapping MUST fail the oracle. If any
// of these comes back 0.0, the oracle is not discriminating (or the hardware
// ignores the bytes it is supposed to use) and the spec is not pinned. The
// uniform-A case is the blocker probe: if spec fails but uniform passes, the
// A side cannot hold distinct scales for rows g and g+8 and C.1's 8-scales-
// per-row design needs the fallback (block scales on the K/B side only).
DGPP_TEST(qwen_fp8_mxfp8_sf_mapping_sensitivity) {
  const SfInputs x = distinct_inputs();
  const float d_swap_a = run_mapping(x.a, x.b, x.sfa, x.sfb, 1);
  const float d_swap_b = run_mapping(x.a, x.b, x.sfa, x.sfb, 2);
  const float d_unif_a = run_mapping(x.a, x.b, x.sfa, x.sfb, 3);
  std::printf("[mxfp8-sf] sensitivity: sfa byte-swap max diff %.3g (must be > 0), "
              "sfb byte-swap %.3g (must be > 0), sfa uniform %.3g\n",
              d_swap_a, d_swap_b, d_unif_a);
  require(d_swap_a > 0.f, "swapped SFA bytes pass the oracle: the test cannot detect a wrong A mapping");
  require(d_swap_b > 0.f, "swapped SFB bytes pass the oracle: the test cannot detect a wrong B mapping");
  require(d_unif_a > 0.f,
          "uniform-A passes the oracle: the hardware may be forcing A-side scales uniform "
          "across the 8-row pair (C.1 blocker: distinct per-row A scales unavailable)");
}

int main() { return dgpp::test::run_all(); }
