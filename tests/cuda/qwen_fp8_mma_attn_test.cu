// Numerical evidence for the Qwen FP8-MMA attention (Phase B,
// docs/qwen_fp8_mma_plan.md §7). Three parts:
//
//   1. qwen_fp8_mma_projected_study (host fp32, NO kernel): quantize all four
//      operands (Q per-row, K/V the existing per-(slot,head) scales, P
//      V-weighted per-row) to E4M3 exactly as §2.1, compute the *projected*
//      MMA math in fp32 (the exact E4M3 result s·α·βᵏ·Σ q̃·k̃ for the score and
//      γ·Σ p̃·ṽ/l for the PV, isolating the quantization from the tensor-core
//      rounding), and compare to the bf16 reference (qsa_attn_partial +
//      dsa_attn_combine over bf16 K/V). Pre-registers the acceptance band:
//      assert l2_rel <= 0.10 and print the measured value.
//
//   2. qwen_fp8_mma_warp_matches_band (GPU): the E4M3 qsa_attn_prefill_warp
//      output vs the bf16 reference, within the band the study set. The
//      acceptance test for the kernel.
//
//   3. qwen_fp8_mma_a_fragment_oracle (GPU): the E4M3 m16n8k32 A-fragment
//      lane/k -> register packing has no in-repo precedent (§11); this packs
//      a known E4M3 A through the exact recipe the warp kernel uses, runs the
//      mma, and checks the C fragment against a host fp32 dot product.
//
// Run:  ctest -R qwen_fp8_mma_attn   (or the built binary directly)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kda_test_helpers.hpp"
#include "kernels/dsa.hpp"
#include "kernels/latent_format.hpp"
#include "kernels/qsa.hpp"

using namespace dgpp::kda_test;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}
template <class T>
DevBuf up(const std::vector<T>& v) {
  DevBuf b(v.size() * sizeof(T));
  b.upload(v.data(), v.size() * sizeof(T));
  return b;
}
template <class T>
std::vector<T> down(const DevBuf& b, size_t n) {
  std::vector<T> v(n);
  b.download(v.data(), n * sizeof(T));
  return v;
}
template <class T>
const T* ptr(const DevBuf& b) { return static_cast<const T*>(b.p); }
template <class T>
T* mptr(DevBuf& b) { return static_cast<T*>(b.p); }

// e4m3 round (host): the exact code the tensor cores consume.
float e4m3(float x) { return dgpp::fp8_e4m3_bits_to_float(dgpp::float_to_fp8_e4m3_bits(x)); }
// The power-of-two P scale (§5.1): 2^ceil(log2(absmax/448)), the fp8_block
// recipe — RNE relative error is scale-invariant (no mantissa/scale
// interaction). 1.0 for an all-zero row (avoids the div-by-zero; the row's
// output is 0 either way).
float pow2_gamma(float absmax_w) {
  if (absmax_w <= 0.f) return 1.f;
  return dgpp::e8m0_byte_to_float(dgpp::e8m0_ceil_log2_byte(absmax_w / 448.f));
}

// Geometry: a small QSA slice (the real model is 24 q / 2 kv heads, dim 256;
// 6/2 is the test's standard slice). One request, seq tokens, rows queries.
struct Fp8Geo {
  int local_heads = 6, kv_heads = 2, dim = 256;
  int block_tokens = 16, seq = 128, rows = 16;
  int group() const { return local_heads / kv_heads; }
  int width() const { return kv_heads * dim; }
  int blocks_per_request() const { return (seq + block_tokens - 1) / block_tokens; }
  int slots() const { return block_tokens * blocks_per_request(); }
};

// Fill a paged bf16 K/V cache (as qsa_kv_append writes it) from bf16 rows.
void fill_cache(const std::vector<uint16_t>& k, const std::vector<uint16_t>& v, const Fp8Geo& g,
                const int32_t* dtable, uint16_t* kc, uint16_t* vc, cudaStream_t st) {
  std::vector<int32_t> req_ids(static_cast<size_t>(g.seq), 0);
  std::vector<int64_t> pos(static_cast<size_t>(g.seq));
  for (int i = 0; i < g.seq; ++i) pos[static_cast<size_t>(i)] = i;
  const int width = g.width();
  DevBuf dk = up(k), dv = up(v), dreq = up(req_ids), dpos = up(pos);
  dgpp::qsa_kv_append(ptr<uint16_t>(dk), width, ptr<uint16_t>(dv), width, ptr<int32_t>(dreq),
                      ptr<int64_t>(dpos), g.seq, dtable, g.blocks_per_request(), g.block_tokens,
                      g.kv_heads, g.dim, kc, vc, nullptr, nullptr, st);
}

// The bf16 reference: qsa_attn_partial + dsa_attn_combine over a paged bf16
// cache; returns the fp32 pre-gate output [rows, local_heads, dim].
std::vector<float> run_attn_fp32(const uint16_t* qonly, const uint16_t* kc, const uint16_t* vc,
                                 const int32_t* dreq, const int32_t* dtopk, const int32_t* dcounts,
                                 const int32_t* dtable, const Fp8Geo& g, cudaStream_t st) {
  const int width = g.width();
  const size_t part = static_cast<size_t>(g.rows) * 1 * g.local_heads;
  DevBuf m_ws(part * 4), l_ws(part * 4), c_ws(part * g.dim * 4),
      c_out(static_cast<size_t>(g.rows) * g.local_heads * g.dim * 4);
  const int64_t q_row_stride = g.local_heads * g.dim;
  dgpp::qsa_attn_partial(qonly, q_row_stride, kc, vc, dreq, dtopk, g.seq, dcounts, g.rows, 1,
                         g.local_heads, g.kv_heads, g.dim, g.block_tokens, dtable,
                         g.blocks_per_request(), 1.0f / 16.0f, mptr<float>(m_ws), mptr<float>(l_ws),
                         mptr<float>(c_ws), st, nullptr, nullptr);
  dgpp::dsa_attn_combine(ptr<float>(m_ws), ptr<float>(l_ws), ptr<float>(c_ws), g.rows, 1,
                         g.local_heads, g.dim, mptr<float>(c_out), st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  return down<float>(c_out, static_cast<size_t>(g.rows) * g.local_heads * g.dim);
}

// One (geometry, cnt) run of the E4M3 warp kernel against the bf16 reference;
// returns the output l2_rel. Parameterized so the review's untested paths can
// be exercised: group > 8 (the g+8 row half) and partial tiles (cnt not a
// multiple of 16, cnt < 16).
double warp_band_l2(const Fp8Geo& g, int cnt, int seed) {
  cudaStream_t st = test_stream();
  const int width = g.width();
  const int slots = g.slots();
  std::vector<int32_t> table(static_cast<size_t>(g.blocks_per_request()));
  for (int b = 0; b < g.blocks_per_request(); ++b) table[static_cast<size_t>(b)] = b;
  DevBuf dtable = up(table);
  const auto k = random_bf16_normal(seed, static_cast<int64_t>(g.seq) * width, 1.0f);
  const auto v = random_bf16_normal(seed + 1, static_cast<int64_t>(g.seq) * width, 1.0f);
  DevBuf kc(static_cast<size_t>(slots) * width * 2), vc(static_cast<size_t>(slots) * width * 2);
  fill_cache(k, v, g, ptr<int32_t>(dtable), mptr<uint16_t>(kc), mptr<uint16_t>(vc), st);
  // fp8 pool: 1 B/elem codes + [slots, kv_heads] fp32 scales.
  DevBuf kc8(static_cast<size_t>(slots) * width), vc8(static_cast<size_t>(slots) * width);
  DevBuf ks(static_cast<size_t>(slots) * g.kv_heads * 4), vs(static_cast<size_t>(slots) * g.kv_heads * 4);
  {
    std::vector<int32_t> req_ids(static_cast<size_t>(g.seq), 0);
    std::vector<int64_t> pos(static_cast<size_t>(g.seq));
    for (int i = 0; i < g.seq; ++i) pos[static_cast<size_t>(i)] = i;
    DevBuf dk = up(k), dv = up(v), dreq = up(req_ids), dpos = up(pos);
    dgpp::qsa_kv_append(ptr<uint16_t>(dk), width, ptr<uint16_t>(dv), width, ptr<int32_t>(dreq),
                        ptr<int64_t>(dpos), g.seq, ptr<int32_t>(dtable), g.blocks_per_request(),
                        g.block_tokens, g.kv_heads, g.dim,
                        reinterpret_cast<uint16_t*>(mptr<uint8_t>(kc8)),
                        reinterpret_cast<uint16_t*>(mptr<uint8_t>(vc8)), mptr<float>(ks),
                        mptr<float>(vs), st);
  }
  const auto q = random_bf16_normal(seed + 2, static_cast<int64_t>(g.rows) * g.local_heads * g.dim, 1.0f);
  std::vector<int32_t> req_ids(static_cast<size_t>(g.rows), 0);
  std::vector<int32_t> topk(static_cast<size_t>(g.rows) * g.seq);
  for (int r = 0; r < g.rows; ++r)
    for (int i = 0; i < g.seq; ++i) topk[static_cast<size_t>(r) * g.seq + i] = i;
  std::vector<int32_t> counts(static_cast<size_t>(g.rows), cnt);
  DevBuf dq = up(q), dreq = up(req_ids), dtopk = up(topk), dcounts = up(counts);
  const long n = static_cast<long>(g.rows) * g.local_heads * g.dim;
  const auto ref = run_attn_fp32(ptr<uint16_t>(dq), mptr<uint16_t>(kc), mptr<uint16_t>(vc),
                                 ptr<int32_t>(dreq), ptr<int32_t>(dtopk), ptr<int32_t>(dcounts),
                                 ptr<int32_t>(dtable), g, st);
  DevBuf got(static_cast<size_t>(n) * 4);
  const int64_t qrs = static_cast<int64_t>(g.local_heads) * g.dim;
  dgpp::qsa_attn_prefill_warp(ptr<uint16_t>(dq), qrs,
                              reinterpret_cast<uint16_t*>(mptr<uint8_t>(kc8)),
                              reinterpret_cast<uint16_t*>(mptr<uint8_t>(vc8)), ptr<int32_t>(dreq),
                              ptr<int32_t>(dtopk), g.seq, ptr<int32_t>(dcounts), g.rows, g.local_heads,
                              g.kv_heads, g.block_tokens, ptr<int32_t>(dtable),
                              g.blocks_per_request(), 1.0f / 16.0f, mptr<float>(got), st,
                              mptr<float>(ks), mptr<float>(vs));
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  const auto gw = down<float>(got, static_cast<size_t>(n));
  double num = 0, den = 0;
  for (long i = 0; i < n; ++i) {
    const double d = static_cast<double>(gw[static_cast<size_t>(i)]) - ref[static_cast<size_t>(i)];
    num += d * d;
    den += static_cast<double>(ref[static_cast<size_t>(i)]) * ref[static_cast<size_t>(i)];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

// The projected E4M3-MMA math in fp32 (§2.1, §2.2, §2.3): the exact result the
// tensor cores would produce from the quantized codes, isolating the
// quantization from the tensor-core rounding.
std::vector<float> projected_e4m3_attn(const std::vector<float>& q, const std::vector<float>& k,
                                       const std::vector<float>& v, const Fp8Geo& g) {
  const int D = g.dim, H = g.local_heads, KV = g.kv_heads, grp = g.group();
  const float s = 1.0f / 16.0f;
  std::vector<float> out(static_cast<size_t>(g.rows) * H * D, 0.f);
  for (int r = 0; r < g.rows; ++r)
    for (int h = 0; h < H; ++h) {
      const int hkv = h / grp;
      // Q row -> e4m3 codes + alpha (per-row absmax/448, §2.1).
      const float* qr = q.data() + (static_cast<size_t>(r) * H + h) * D;
      float amax = 0.f;
      for (int j = 0; j < D; ++j) amax = std::max(amax, std::fabs(qr[j]));
      const float alpha = amax > 0.f ? amax / 448.f : 1.f;
      std::vector<float> qt(D);
      for (int j = 0; j < D; ++j) qt[static_cast<size_t>(j)] = e4m3(qr[j] / alpha);
      // Per-token K/V codes + scales, the scores, the online-softmax P.
      std::vector<float> sc(g.seq), p(g.seq);
      float M = -INFINITY, l = 0.f;
      for (int n = 0; n < g.seq; ++n) {
        const float* kr = k.data() + (static_cast<size_t>(n) * KV + hkv) * D;
        float kamax = 0.f;
        for (int j = 0; j < D; ++j) kamax = std::max(kamax, std::fabs(kr[j]));
        const float bk = kamax > 0.f ? kamax / 448.f : 1.f;
        float dot = 0.f;
        for (int j = 0; j < D; ++j) dot += qt[static_cast<size_t>(j)] * e4m3(kr[j] / bk);
        sc[static_cast<size_t>(n)] = s * alpha * bk * dot;  // §2.2: s·α·βᵏ·Σ q̃·k̃
        M = std::max(M, sc[static_cast<size_t>(n)]);
      }
      for (int n = 0; n < g.seq; ++n) {
        p[static_cast<size_t>(n)] = expf(sc[static_cast<size_t>(n)] - M);
        l += p[static_cast<size_t>(n)];  // l sums the UNROUNDED P (the DSA pin)
      }
      // V-weighted P (w = P·βᵛ, §2.3) -> e4m3 codes + the power-of-two gamma.
      std::vector<float> w(g.seq), vt(static_cast<size_t>(g.seq) * D);
      float wmax = 0.f;
      for (int n = 0; n < g.seq; ++n) {
        const float* vr = v.data() + (static_cast<size_t>(n) * KV + hkv) * D;
        float vmax = 0.f;
        for (int j = 0; j < D; ++j) vmax = std::max(vmax, std::fabs(vr[j]));
        const float bv = vmax > 0.f ? vmax / 448.f : 1.f;
        for (int j = 0; j < D; ++j)
          vt[(static_cast<size_t>(n) * D + j)] = e4m3(vr[j] / bv);
        w[static_cast<size_t>(n)] = p[static_cast<size_t>(n)] * bv;
        wmax = std::max(wmax, w[static_cast<size_t>(n)]);
      }
      const float gamma = pow2_gamma(wmax);
      // PV: out = gamma · Σ_n p̃·ṽ / l  (§2.3: βᵛ cancels, one copy in P, one in V).
      float* o = out.data() + (static_cast<size_t>(r) * H + h) * D;
      for (int j = 0; j < D; ++j) o[j] = 0.f;
      for (int n = 0; n < g.seq; ++n) {
        const float pt = e4m3(w[static_cast<size_t>(n)] / gamma);
        if (pt == 0.f) continue;
        for (int j = 0; j < D; ++j) o[j] += pt * vt[static_cast<size_t>(n) * D + j];
      }
      const float inv = l > 0.f ? gamma / l : 0.f;
      for (int j = 0; j < D; ++j) o[j] *= inv;
    }
  return out;
}

}  // namespace

DGPP_TEST(qwen_fp8_mma_projected_study) {
  Fp8Geo g;
  cudaStream_t st = test_stream();
  const int width = g.width();

  std::vector<int32_t> table(static_cast<size_t>(g.blocks_per_request()));
  for (int b = 0; b < g.blocks_per_request(); ++b) table[static_cast<size_t>(b)] = b;
  DevBuf dtable = up(table);

  // Realistic bf16 Q/K/V (unit-normal, the test's standard distribution).
  const auto q = random_bf16_normal(101, static_cast<int64_t>(g.rows) * g.local_heads * g.dim, 1.0f);
  const auto k = random_bf16_normal(102, static_cast<int64_t>(g.seq) * width, 1.0f);
  const auto v = random_bf16_normal(103, static_cast<int64_t>(g.seq) * width, 1.0f);

  // bf16 paged cache + the reference (partial + combine, pre-gate fp32).
  DevBuf kc(static_cast<size_t>(g.slots()) * width * 2), vc(static_cast<size_t>(g.slots()) * width * 2);
  fill_cache(k, v, g, ptr<int32_t>(dtable), mptr<uint16_t>(kc), mptr<uint16_t>(vc), st);

  // Queries: every row attends to the full token list (the dense case).
  std::vector<int32_t> req_ids(static_cast<size_t>(g.rows), 0);
  std::vector<int32_t> topk(static_cast<size_t>(g.rows) * g.seq);
  for (int r = 0; r < g.rows; ++r)
    for (int i = 0; i < g.seq; ++i) topk[static_cast<size_t>(r) * g.seq + i] = i;
  std::vector<int32_t> counts(static_cast<size_t>(g.rows), g.seq);
  DevBuf dq = up(q), dreq = up(req_ids), dtopk = up(topk), dcounts = up(counts);

  const auto ref = run_attn_fp32(ptr<uint16_t>(dq), mptr<uint16_t>(kc), mptr<uint16_t>(vc),
                                 ptr<int32_t>(dreq), ptr<int32_t>(dtopk), ptr<int32_t>(dcounts),
                                 ptr<int32_t>(dtable), g, st);

  // The projected E4M3 math (host fp32) over the same (dequantized-to-fp32) Q/K/V.
  std::vector<float> qf(q.size()), kf(k.size()), vf(v.size());
  for (size_t i = 0; i < q.size(); ++i) qf[i] = dgpp::bf16_bits_to_float(q[i]);
  for (size_t i = 0; i < k.size(); ++i) kf[i] = dgpp::bf16_bits_to_float(k[i]);
  for (size_t i = 0; i < v.size(); ++i) vf[i] = dgpp::bf16_bits_to_float(v[i]);
  const auto proj = projected_e4m3_attn(qf, kf, vf, g);

  const long n = static_cast<long>(ref.size());
  double num = 0, den = 0;
  float max_abs = 0.f;
  long match = 0;
  for (long i = 0; i < n; ++i) {
    const float d = proj[static_cast<size_t>(i)] - ref[static_cast<size_t>(i)];
    num += static_cast<double>(d) * d;
    den += static_cast<double>(ref[static_cast<size_t>(i)]) * ref[static_cast<size_t>(i)];
    max_abs = std::max(max_abs, std::fabs(d));
    if (std::fabs(d) / std::max(std::fabs(ref[static_cast<size_t>(i)]), 1e-30f) <= 0.0625) ++match;
  }
  const double l2_rel = std::sqrt(num / std::max(den, 1e-30));
  std::printf(
      "[fp8-mma] projected study (n=%ld): l2_rel %.4f  max_abs %.4g  match(<=2^-4) %.2f%%\n", n,
      l2_rel, max_abs, 100.0 * static_cast<double>(match) / static_cast<double>(n));
  // The pre-registered band (§3): Q+K each contribute 2^-4 to the scores, P
  // contributes 2^-4 to the PV; Phase A measured 0.036 with K/V only, so
  // adding Q+P lands ~0.04-0.08. Assert the band and print the real value.
  require(l2_rel <= 0.10, "projected E4M3-MMA attention l2_rel exceeds the pre-registered 0.10 band");
}

// The E4M3 m16n8k32 A-fragment packing oracle (§11): pack a known E4M3 A
// through the exact recipe the warp kernel uses (4 e4m3 per register:
// a0=A[g][4t..4t+3], a1=A[g+8][4t..4t+3], a2=A[g][4t+16..4t+19],
// a3=A[g+8][4t+16..4t+19]), run the mma against a known B, and check the C
// fragment against a host fp32 dot product.
namespace {
__device__ __forceinline__ uint32_t pack4_e4m3(float a, float b, float c, float d) {
  const uint32_t lo =
      __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi =
      __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}
__global__ void a_fragment_oracle_kernel(const uint8_t* a, const uint8_t* b, float* c) {
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  // A is 16 x 32 e4m3 (row-major), B is 32 x 8 e4m3 (row-major, k x n).
  auto A = [&](int r, int k) { return dgpp::fp8_e4m3_bits_to_float(a[r * 32 + k]); };
  auto B = [&](int k, int n) { return dgpp::fp8_e4m3_bits_to_float(b[k * 8 + n]); };
  uint32_t a0 = pack4_e4m3(A(g, 4 * t), A(g, 4 * t + 1), A(g, 4 * t + 2), A(g, 4 * t + 3));
  uint32_t a1 = pack4_e4m3(A(g + 8, 4 * t), A(g + 8, 4 * t + 1), A(g + 8, 4 * t + 2), A(g + 8, 4 * t + 3));
  uint32_t a2 = pack4_e4m3(A(g, 4 * t + 16), A(g, 4 * t + 17), A(g, 4 * t + 18), A(g, 4 * t + 19));
  uint32_t a3 = pack4_e4m3(A(g + 8, 4 * t + 16), A(g + 8, 4 * t + 17), A(g + 8, 4 * t + 18), A(g + 8, 4 * t + 19));
  uint32_t b0 = pack4_e4m3(B(4 * t, g), B(4 * t + 1, g), B(4 * t + 2, g), B(4 * t + 3, g));
  uint32_t b1 = pack4_e4m3(B(4 * t + 16, g), B(4 * t + 17, g), B(4 * t + 18, g), B(4 * t + 19, g));
  float cc[4] = {0, 0, 0, 0};
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
               "{%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(cc[0]), "+f"(cc[1]), "+f"(cc[2]), "+f"(cc[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  // C fragment (m16n8): c0=D[g][2t] c1=D[g][2t+1] c2=D[g+8][2t] c3=D[g+8][2t+1].
  c[g * 8 + 2 * t + 0] = cc[0];
  c[g * 8 + 2 * t + 1] = cc[1];
  c[(g + 8) * 8 + 2 * t + 0] = cc[2];
  c[(g + 8) * 8 + 2 * t + 1] = cc[3];
}
}  // namespace

DGPP_TEST(qwen_fp8_mma_a_fragment_oracle) {
  // Random E4M3 A (16x32) and B (32x8); the host dot product is the reference.
  // The codes are drawn from {-1, 0, +1} (0x58, 0x00, 0x38) so every dot
  // product is a small exact integer: the fp32 mma accumulation is then
  // exact and the comparison is bitwise (no accumulation-order noise).
  std::vector<uint8_t> a(16 * 32), b(32 * 8);
  uint64_t seed = 201;
  auto rnd = [&]() { seed = seed * 6364136223846793005ULL + 1442695040888963407ULL; return (seed >> 33) & 0x7Fu; };
  const uint8_t vals[3] = {0x58u, 0x00u, 0x38u};
  for (auto& x : a) x = vals[rnd() % 3];
  for (auto& x : b) x = vals[rnd() % 3];
  DevBuf da = up(a), db = up(b), dc(16 * 8 * 4);
  a_fragment_oracle_kernel<<<1, 32>>>(ptr<uint8_t>(da), ptr<uint8_t>(db), mptr<float>(dc));
  DGPP_CUDA_OK(cudaGetLastError());
  const auto c = down<float>(dc, 16 * 8);
  float max_diff = 0.f;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 8; ++n) {
      float ref = 0.f;
      for (int k = 0; k < 32; ++k)
        ref += dgpp::fp8_e4m3_bits_to_float(a[m * 32 + k]) * dgpp::fp8_e4m3_bits_to_float(b[k * 8 + n]);
      max_diff = std::max(max_diff, std::fabs(c[m * 8 + n] - ref));
    }
  std::printf("[fp8-mma] A-fragment oracle: max |mma - host dot| = %.3g\n", max_diff);
  require(max_diff == 0.f, "E4M3 m16n8k32 A-fragment packing disagrees with the host dot product");
}

// The E4M3 warp kernel (the serving default for long prefill) vs the bf16
// reference, within the Step-0 band. The E4M3 path runs when the fp8 pool is
// active (k_scale != nullptr) and DGPP_QSA_FP8_MMA is on (default).
DGPP_TEST(qwen_fp8_mma_warp_matches_band) {
  const double l2_rel = warp_band_l2(Fp8Geo{}, 128, 111);  // group 3, full tiles
  std::printf("[fp8-mma] E4M3 warp kernel vs bf16 (group 3, cnt 128): l2_rel %.4f\n", l2_rel);
  require(l2_rel <= 0.10, "E4M3 warp kernel l2_rel exceeds the Step-0 0.10 band");
}

// The review (docs/qwen_fp8_mma_plan.md §11 follow-up) flagged two paths the
// band test above does not cover: group > 8 (the g+8 half of the m16 A-fragment,
// untested at group 3) and partial tiles (cnt not a multiple of 16, cnt < 16).
DGPP_TEST(qwen_fp8_mma_warp_group16_and_partial_tiles) {
  struct Case { int lh, kv, cnt; const char* label; };
  const Case cases[] = {
      {16, 1, 128, "group16 cnt128 (full tiles)"},
      {16, 1, 17, "group16 cnt17 (partial 16+1)"},
      {16, 1, 5, "group16 cnt5 (cnt < 16)"},
      {6, 2, 17, "group3 cnt17 (partial)"},
  };
  for (int i = 0; i < 4; ++i) {
    Fp8Geo g;
    g.local_heads = cases[i].lh;
    g.kv_heads = cases[i].kv;
    g.rows = 4;
    const double l2_rel = warp_band_l2(g, cases[i].cnt, 200 + 10 * i);
    std::printf("[fp8-mma] E4M3 warp kernel vs bf16 %s: l2_rel %.4f\n", cases[i].label, l2_rel);
    require(l2_rel <= 0.10, "E4M3 warp kernel l2_rel exceeds the 0.10 band");
  }
}

int main() { return dgpp::test::run_all(); }
