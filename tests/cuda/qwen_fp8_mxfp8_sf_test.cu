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
// without guessing.
//
// THE MAP BELOW IS PROBE-VERIFIED, NOT CREDITED TO THE PTX ISA TABLE. The
// earlier doc (and this file's first revision) asserted the PTX ISA 9.7.14.6.1
// table layout (sfa byte0=row g / byte1=row g+8 in the SAME thread; sfb
// byte0=col 2t / byte1=col 2t+1). That mapping was FALSIFIED by running the
// oracle on the GPU (max |mma - oracle| = 6.06e+04, non-zero). The true map was
// then established empirically with a map-independent single-byte-boost probe
// (A/B codes all 0x38=1.0, all SF bytes 0x7f=2^0 except one byte set to
// 0x84=2^5; read all 128 outputs to see which rows/cols get the 32x boost),
// and cross-checked with a distinct-per-row/per-col-scale oracle (max diff 0).
//
// PROBE-VERIFIED TRUE MAP (scale_vec::1X, selectors all 0, byte0 is the active
// scale byte; bytes 1-3 of each register are ignored at sel=0):
//
//   lane = 4*g + t  (g = lane/4, t = lane%4)
//   SFA: A row g   -> thread (g, t=0) register, byte0
//        A row g+8 -> thread (g, t=1) register, byte0
//        (threads t=2, t=3 of a group carry no A scale)
//   SFB: B col g   -> thread (g, t=0) register, byte0
//        (threads t=1, t=2, t=3 of a group carry no B scale)
//
// So the A side is PER-ROW (rows g and g+8 live in DIFFERENT threads of the
// same group, so they can hold distinct scales -- the 8-scales-per-256-dim-row
// design is supported), and the B side is PER-COLUMN (one scale per column, in
// that column's group t=0 thread). This is NOT the uniform-pair / 2-col-per-
// thread layout the PTX ISA table implied.
//
// A row of the m16n8k32 A tile is exactly one 32-wide k-block, so one mma
// carries ONE scale per A row and ONE per B column; the 8-scales-per-256-dim-row
// design is 8 sequential k-steps, each with its own SFA/SFB registers (no
// k-steps share a byte). The oracle below proves the A side can hold 16
// distinct row scales (rows g and g+8 differ, in different threads) -- the
// sensitivity test reports explicitly that the uniform-A and the falsified
// spec mappings FAIL.
//
// Exactness: the e4m3 codes are drawn from {0, 0.5, 1, 8, 16} (0x00, 0x30,
// 0x38, 0x50, 0x58 -- all non-negative; 0x50=+8 and 0x58=+16, sign bit clear),
// so every product a*b is an exact multiple of 2^-2 and the 32-term dot
// product is an exact multiple of 2^-2 with |dot| <= 32*16*16 = 8192 = 2^13;
// write dot = D*2^-2 with integer |D| <= 2^15. The ue8m0 scales are powers of
// two (2^-4..2^4 for A, 2^-3..2^3 for B), so the scale product is a power of
// two in 2^-7..2^7 and the final value is D*2^e with |D| <= 2^15 and e in
// -9..5; since |D| <= 2^15 < 2^24 (the fp32 significand) it is exactly
// representable in fp32 (magnitude 2^-9..2^20, normal, no denormal flush).
// The fp32 mma result is therefore exact and the host oracle (computed in
// double, exact for these values) compares bitwise (max diff must be 0).
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
//   0 true      : PROBE-VERIFIED. sfa: thread(g,0) byte0=row g, thread(g,1)
//                 byte0=row g+8; sfb: thread(g,0) byte0=col g. (bytes 1-3 = 0x7f)
//   1 sfa_swap  : the two A rows exchanged between threads (g,0)/(g,1) -- wrong
//   2 sfb_swap  : the B col scale placed in the ignored thread (g,1) -- wrong
//   3 sfa_unif  : both A rows use row g's scale (uniform across the 8-row pair)
//   4 old_spec  : the FALSIFIED PTX ISA table map (sfa byte0=row g / byte1=row
//                 g+8 in the same thread; sfb byte0=col 2t / byte1=col 2t+1)
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
  const uint32_t sfa_lo = sfa[g];          // row g
  const uint32_t sfa_hi = sfa[g + 8];      // row g+8
  const uint32_t sfb_c = sfb[g];           // col g
  // Pad the ignored bytes 1-3 with 0x7f (2^0, neutral) so the only difference
  // between maps is the placement of the active (byte0) scale.
  auto pad = [](uint32_t v) { return v | 0x7f7f7f00u; };
  switch (map) {
    case 1:  // sfa_swap: exchange which thread holds row g vs row g+8
      sfa_r = pad(t == 0 ? sfa_hi : (t == 1 ? sfa_lo : 0x7f));
      sfb_r = pad(t == 0 ? sfb_c : 0x7f);
      break;
    case 2:  // sfb_swap: put the col scale in the ignored thread (g,1)
      sfa_r = pad(t == 0 ? sfa_lo : (t == 1 ? sfa_hi : 0x7f));
      sfb_r = pad(t == 1 ? sfb_c : 0x7f);
      break;
    case 3:  // sfa_unif: both A rows use row g's scale
      sfa_r = pad((t == 0 || t == 1) ? sfa_lo : 0x7f);
      sfb_r = pad(t == 0 ? sfb_c : 0x7f);
      break;
    case 4:  // old_spec: the falsified PTX ISA table map (same-thread bytes)
      sfa_r = sfa_lo | (sfa_hi << 8);
      sfb_r = static_cast<uint32_t>(sfb[2 * t]) | (static_cast<uint32_t>(sfb[2 * t + 1]) << 8);
      break;
    default:  // 0 = true (probe-verified)
      sfa_r = pad(t == 0 ? sfa_lo : (t == 1 ? sfa_hi : 0x7f));
      sfb_r = pad(t == 0 ? sfb_c : 0x7f);
      break;
  }
  float cc[4] = {0.f, 0.f, 0.f, 0.f};
  // The selector operands are PAIRS of .b16 registers interleaved after each
  // scale register (the mxf4nvf4 4X form in moe_w4a4.cu is the in-repo
  // precedent): {sfa} {sfa_sel0,sfa_sel1} {sfb} {sfb_sel0,sfb_sel1}. Both
  // selectors are 0 for scale_vec::1X (byte0 is the active scale byte).
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

// The distinct-scale inputs: e4m3 codes from {0, 0.5, 1, 8, 16} (exact dyadic
// dots) and 9 / 7 DISTINCT ue8m0 bytes (a wrong byte mapping cannot hide:
// sfa[m] = 127-4+(m%9) yields 9 distinct A scales over 16 rows and
// sfb[n] = 127-3+(n%7) yields 7 distinct B scales over 8 cols; for every pair
// (g, g+8) the two A scales differ (sfa[g] vs sfa[g+8] are never equal under
// this formula) and for every t the B cols 2t and 2t+1 differ, so a swapped,
// uniform, or same-thread (old spec) mapping cannot masquerade as the true map).
struct SfInputs {
  std::vector<uint8_t> a, b, sfa, sfb;
};
SfInputs distinct_inputs() {
  SfInputs x;
  x.a.resize(16 * 32);
  x.b.resize(32 * 8);
  const uint8_t vals[5] = {0x00u, 0x30u, 0x38u, 0x50u, 0x58u};  // 0, 0.5, 1, 8, 16 (all non-negative)
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
// C=1: 4*(32*1)+4); scales 0x80 (2^1) -> 128.0 each (the plan's 516.0). This
// establishes the exponent convention (bias 127, sfa*sfb multiplicative) and
// is map-independent (uniform scales), so it holds under the true map too.
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

// The PROBE-VERIFIED TRUE mapping over distinct per-row / per-col scales must
// match the exact host oracle bitwise. This is the C.1.0 gate: max diff 0.
DGPP_TEST(qwen_fp8_mxfp8_sf_oracle) {
  const SfInputs x = distinct_inputs();
  const float d = run_mapping(x.a, x.b, x.sfa, x.sfb, 0);
  std::printf("[mxfp8-sf] SF-fragment oracle (TRUE map: sfa thread(g,0) byte0=row g / thread(g,1) "
              "byte0=row g+8; sfb thread(g,0) byte0=col g; 16x8 tile, 9/7 distinct scales): "
              "max |mma - host oracle| = %.3g\n",
              d);
  require(d == 0.f,
          "MXFP8 SF-fragment TRUE map (sfa: thread(g,0) byte0=row g, thread(g,1) byte0=row g+8; "
          "sfb: thread(g,0) byte0=col g) disagrees with the exact host oracle");
}

// Sensitivity: a deliberately wrong byte mapping MUST fail the oracle,
// RELATIVE TO THE TRUE MAP. If any of these comes back 0.0, the oracle is not
// discriminating (or the hardware ignores the bytes it is supposed to use) and
// the true map is not pinned. The old_spec case is the regression guard for the
// falsified PTX ISA table map. The uniform-A case is the design probe: if it
// passed, the A side would be uniform across the 8-row pair and C.1's
// 8-scales-per-row design would be a NO-GO (it does NOT pass: rows g and g+8
// live in different threads and hold distinct scales).
DGPP_TEST(qwen_fp8_mxfp8_sf_mapping_sensitivity) {
  const SfInputs x = distinct_inputs();
  const float d_swap_a = run_mapping(x.a, x.b, x.sfa, x.sfb, 1);
  const float d_swap_b = run_mapping(x.a, x.b, x.sfa, x.sfb, 2);
  const float d_unif_a = run_mapping(x.a, x.b, x.sfa, x.sfb, 3);
  const float d_old_spec = run_mapping(x.a, x.b, x.sfa, x.sfb, 4);
  std::printf("[mxfp8-sf] sensitivity (vs TRUE map): sfa byte-swap %.3g, sfb byte-swap %.3g, "
              "sfa uniform %.3g, old-spec-map %.3g (all must be > 0)\n",
              d_swap_a, d_swap_b, d_unif_a, d_old_spec);
  require(d_swap_a > 0.f, "swapped SFA rows pass the oracle: the test cannot detect a wrong A mapping");
  require(d_swap_b > 0.f, "misplaced SFB col scale passes the oracle: the test cannot detect a wrong B mapping");
  require(d_unif_a > 0.f,
          "uniform-A passes the oracle: the hardware may be forcing A-side scales uniform "
          "across the 8-row pair (C.1 blocker: distinct per-row A scales unavailable)");
  require(d_old_spec > 0.f,
          "the falsified old-spec (PTX ISA table) map passes the oracle: regression guard tripped");
}

int main() { return dgpp::test::run_all(); }
