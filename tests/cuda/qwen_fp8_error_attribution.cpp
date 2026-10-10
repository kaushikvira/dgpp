// C.1a error attribution study (HOST-ONLY, no GPU) — where does the E4M3
// attention path's ~0.051 l2_rel come from, and is the 0.035 bar reachable?
//
// Exact host-side simulation of the QSA attention (the same math the
// projected study in qwen_fp8_mma_attn_test.cu runs, extended): one factor
// quantized at a time, so each contribution is isolated.
//
//   Q {bf16, e4m3 row, e4m3 block} x K {bf16, e4m3 row, e4m3 block}
//   x P {e4m3 pow2-gamma, bf16}    x V {bf16, e4m3 row, e4m3 block}
//   x {no rotation, Hadamard H2 x H128 on Q+K}
//
// Distributions: (a) unit-normal (the test's standard, the WORST case for a
// block-scale win), (b) heavy-tailed K rows (per-row lognormal scale + two
// 10-100x spikes, a stress regime), (b') RMSNorm-pinned K rows with three
// 5-20x outlier dims (the realistic post-norm regime), (c) sink-like K rows
// (one dim at 32x the row rms).
//
// The quantizers are the production recipes (latent_format.hpp): row scale
// absmax/448, block scale e8m0 ceil per 32 dims (the C.1a append recipe),
// P's power-of-two gamma (qsa_warp.cu pow2_ceil). The reference is the bf16
// pin: fp32 math, P rounded to bf16 (the DSA pin), l unrounded.
//
// Build: plain C++ (no CUDA). Run: ./qwen_fp8_error_attribution
//   ./qwen_fp8_error_attribution                 (the four synthetic regimes)
//   ./qwen_fp8_error_attribution --real <dump>   (the DGPP_DUMP_QK real-data
//                                                mode: the same grid on a
//                                                live forward pass's
//                                                post-norm+RoPE Q/K, plus
//                                                per-row statistics)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"

using dgpp::bf16_bits_to_float;
using dgpp::e8m0_byte_to_float;
using dgpp::e8m0_ceil_log2_byte;
using dgpp::float_to_bf16_bits;
using dgpp::float_to_fp8_e4m3_bits;
using dgpp::fp8_e4m3_bits_to_float;
using dgpp::latent_fp8_block_scale_byte;

namespace {

float e4m3(float x) { return fp8_e4m3_bits_to_float(float_to_fp8_e4m3_bits(x)); }
float pow2_ceil(float x) { return e8m0_byte_to_float(e8m0_ceil_log2_byte(x)); }
float bf16r(float x) { return bf16_bits_to_float(float_to_bf16_bits(x)); }

// The production quantizers (dequantized fp32 values, the tensor cores' view).
// Row: scale = absmax/448, code = e4m3(x * 448/absmax) (latent_fp8_row_scale).
std::vector<float> quant_row(const float* x, int n) {
  float amax = 0.f;
  for (int j = 0; j < n; ++j) amax = std::max(amax, std::fabs(x[j]));
  std::vector<float> out(n);
  if (amax <= 0.f) return out;
  const float scale = amax * (1.0f / 448.0f);
  const float inv = 448.0f / amax;
  for (int j = 0; j < n; ++j) out[j] = e4m3(x[j] * inv) * scale;
  return out;
}
// Block: 8 e8m0 scales per 256-dim row (the C.1a append recipe,
// latent_fp8_block_scale_byte: 1e-4 floor, ceil to a power of two).
std::vector<float> quant_block(const float* x, int n) {
  std::vector<float> out(n);
  for (int b = 0; b < n / 32; ++b) {
    float bmax = 0.f;
    for (int j = 32 * b; j < 32 * b + 32; ++j) bmax = std::max(bmax, std::fabs(x[j]));
    const float s = e8m0_byte_to_float(latent_fp8_block_scale_byte(bmax));
    for (int j = 32 * b; j < 32 * b + 32; ++j) out[j] = e4m3(x[j] / s) * s;
  }
  return out;
}

// H2 x H128 (Sylvester): 7 butterfly stages per 128-half, then a 2-point
// butterfly across halves, scaled by 1/16. The unscaled Kronecker product
// has row norm 16, so the 1/16 makes it orthogonal: q'.k' = q.k exactly
// (the rotation is free mathematically, plan §4).
void hadamard128(float* x) {
  for (int st = 0; st < 7; ++st) {
    const int stride = 1 << st;
    for (int i = 0; i < 128; i += 2 * stride)
      for (int j = 0; j < stride; ++j) {
        const float a = x[i + j], b = x[i + j + stride];
        x[i + j] = a + b;
        x[i + j + stride] = a - b;
      }
  }
}
void hadamard256(float* x) {
  hadamard128(x);
  hadamard128(x + 128);
  for (int j = 0; j < 128; ++j) {
    const float a = x[j], b = x[j + 128];
    x[j] = (a + b) * 0.0625f;
    x[j + 128] = (a - b) * 0.0625f;
  }
}

// Geometry: the tests' standard QSA slice (6 q heads / 2 kv heads, dim 256).
struct Data {
  int rows = 16, H = 6, KV = 2, D = 256, seq = 128;
  std::vector<float> q, k, v;  // fp32
};

enum class QF { BF16 = 0, ROW = 1, BLK = 2 };
enum class PF { E4M3 = 0, BF16 = 1 };
struct Cfg {
  QF q = QF::BLK, k = QF::BLK, v = QF::ROW;
  PF p = PF::E4M3;
  bool had = false;
  const char* name = "";
};

// The bf16 reference (the DSA pin): fp32 math, P bf16-rounded, l unrounded.
std::vector<float> attn_ref(const Data& d) {
  const float s = 1.0f / 16.0f;
  const int grp = d.H / d.KV;
  std::vector<float> out(static_cast<size_t>(d.rows) * d.H * d.D, 0.f);
  for (int r = 0; r < d.rows; ++r)
    for (int h = 0; h < d.H; ++h) {
      const int hkv = h / grp;
      const float* qr = d.q.data() + (static_cast<size_t>(r) * d.H + h) * d.D;
      std::vector<float> sc(static_cast<size_t>(d.seq));
      float M = -INFINITY;
      for (int n = 0; n < d.seq; ++n) {
        const float* kr = d.k.data() + (static_cast<size_t>(n) * d.KV + hkv) * d.D;
        float dot = 0.f;
        for (int j = 0; j < d.D; ++j) dot += qr[j] * kr[j];
        sc[static_cast<size_t>(n)] = s * dot;
        M = std::max(M, sc[static_cast<size_t>(n)]);
      }
      float l = 0.f;
      std::vector<float> p(static_cast<size_t>(d.seq));
      for (int n = 0; n < d.seq; ++n) {
        const float e = expf(sc[static_cast<size_t>(n)] - M);
        p[static_cast<size_t>(n)] = bf16r(e);  // the DSA pin
        l += e;  // l sums the UNROUNDED P
      }
      float* o = out.data() + (static_cast<size_t>(r) * d.H + h) * d.D;
      for (int n = 0; n < d.seq; ++n) {
        if (p[static_cast<size_t>(n)] == 0.f) continue;
        const float* vr = d.v.data() + (static_cast<size_t>(n) * d.KV + hkv) * d.D;
        for (int j = 0; j < d.D; ++j) o[j] += p[static_cast<size_t>(n)] * vr[j];
      }
      const float inv = l > 0.f ? 1.f / l : 0.f;
      for (int j = 0; j < d.D; ++j) o[j] *= inv;
    }
  return out;
}

// The factorized fp8 path: each operand quantized per cfg, the exact codes
// the tensor cores consume (fp32 accumulation isolates the quantization
// from the tensor-core rounding).
std::vector<float> attn_sim(const Data& d, const Cfg& c) {
  const float s0 = 1.0f / 16.0f;
  auto q = d.q, k = d.k;
  float s = s0;
  if (c.had) {
    // H2 x H128 scaled by 1/16 is orthogonal (row norm 16), so q'.k' = q.k
    // exactly and the attention scale is unchanged.
    for (size_t i = 0; i < q.size(); i += d.D) hadamard256(q.data() + i);
    for (size_t i = 0; i < k.size(); i += d.D) hadamard256(k.data() + i);
  }
  const int grp = d.H / d.KV;
  // Per-(kv-head, token) K codes and V codes (shared by the query group).
  // The PV mma's B operand (pv_op) and the P-side weighting scale (w_scale)
  // depend on the V mode: with ROW V the mma sees the CODES and beta^v is
  // absorbed into w = P*beta^v (it cancels: one copy in w, one in the code);
  // with BLOCK V the mma applies the block scales itself, so the operand is
  // the block-dequantized values and w = P (no beta^v); with BF16 V the
  // operand is the raw values and w = P.
  std::vector<float> kd(static_cast<size_t>(d.KV) * d.seq * d.D);
  std::vector<float> pv_op(static_cast<size_t>(d.KV) * d.seq * d.D);
  std::vector<float> vdeq(static_cast<size_t>(d.KV) * d.seq * d.D);
  std::vector<float> w_scale(static_cast<size_t>(d.KV) * d.seq, 1.f);
  for (int hkv = 0; hkv < d.KV; ++hkv)
    for (int n = 0; n < d.seq; ++n) {
      const float* kr = k.data() + (static_cast<size_t>(n) * d.KV + hkv) * d.D;
      if (c.k == QF::BF16) {
        std::copy(kr, kr + d.D, kd.begin() + (static_cast<size_t>(hkv) * d.seq + n) * d.D);
      } else {
        auto kt = c.k == QF::ROW ? quant_row(kr, d.D) : quant_block(kr, d.D);
        std::copy(kt.begin(), kt.end(),
                  kd.begin() + (static_cast<size_t>(hkv) * d.seq + n) * d.D);
      }
      const float* vr = d.v.data() + (static_cast<size_t>(n) * d.KV + hkv) * d.D;
      const size_t i0 = (static_cast<size_t>(hkv) * d.seq + n) * d.D;
      if (c.v == QF::BF16) {
        std::copy(vr, vr + d.D, pv_op.begin() + i0);
        std::copy(vr, vr + d.D, vdeq.begin() + i0);
        w_scale[static_cast<size_t>(hkv) * d.seq + n] = 1.f;
      } else {
        auto vt = c.v == QF::ROW ? quant_row(vr, d.D) : quant_block(vr, d.D);
        std::copy(vt.begin(), vt.end(), vdeq.begin() + i0);
        if (c.v == QF::ROW) {
          float amax = 0.f;
          for (int j = 0; j < d.D; ++j) amax = std::max(amax, std::fabs(vr[j]));
          const float vscale = amax > 0.f ? amax * (1.0f / 448.0f) : 1.f;
          for (int j = 0; j < d.D; ++j)
            pv_op[i0 + static_cast<size_t>(j)] = vt[static_cast<size_t>(j)] / vscale;  // the code
          w_scale[static_cast<size_t>(hkv) * d.seq + n] = vscale;
        } else {
          std::copy(vt.begin(), vt.end(), pv_op.begin() + i0);  // block-dequantized
          w_scale[static_cast<size_t>(hkv) * d.seq + n] = 1.f;
        }
      }
    }
  std::vector<float> out(static_cast<size_t>(d.rows) * d.H * d.D, 0.f);
  for (int r = 0; r < d.rows; ++r)
    for (int h = 0; h < d.H; ++h) {
      const int hkv = h / grp;
      const float* qr = q.data() + (static_cast<size_t>(r) * d.H + h) * d.D;
      auto qd = c.q == QF::BF16 ? std::vector<float>(qr, qr + d.D)
                                : c.q == QF::ROW ? quant_row(qr, d.D)
                                                : quant_block(qr, d.D);
      std::vector<float> sc(static_cast<size_t>(d.seq));
      float M = -INFINITY;
      for (int n = 0; n < d.seq; ++n) {
        const float* kr = kd.data() + (static_cast<size_t>(hkv) * d.seq + n) * d.D;
        float dot = 0.f;
        for (int j = 0; j < d.D; ++j) dot += qd[static_cast<size_t>(j)] * kr[j];
        sc[static_cast<size_t>(n)] = s * dot;
        M = std::max(M, sc[static_cast<size_t>(n)]);
      }
      float l = 0.f;
      std::vector<float> p(static_cast<size_t>(d.seq));
      for (int n = 0; n < d.seq; ++n) {
        const float e = expf(sc[static_cast<size_t>(n)] - M);
        p[static_cast<size_t>(n)] = bf16r(e);
        l += e;
      }
      // PV. V codes (row or block); P either e4m3 with the power-of-two gamma
      // (V-weighted, the kernel's recipe) or bf16 (the DSA pin).
      float* o = out.data() + (static_cast<size_t>(r) * d.H + h) * d.D;
      for (int j = 0; j < d.D; ++j) o[j] = 0.f;
      if (c.p == PF::E4M3) {
        // w = P * w_scale (V-weighted for row V, plan §2.3); gamma = pow2
        // ceil(wmax/448); out = gamma * sum p~ * pv_op / l.
        float wmax = 0.f;
        std::vector<float> w(static_cast<size_t>(d.seq));
        for (int n = 0; n < d.seq; ++n) {
          w[static_cast<size_t>(n)] =
              p[static_cast<size_t>(n)] * w_scale[static_cast<size_t>(hkv) * d.seq + n];
          wmax = std::max(wmax, w[static_cast<size_t>(n)]);
        }
        const float gamma = wmax > 0.f ? pow2_ceil(wmax / 448.0f) : 1.f;
        for (int n = 0; n < d.seq; ++n) {
          const float pt = e4m3(w[static_cast<size_t>(n)] / gamma);
          if (pt == 0.f) continue;
          const float* vo = pv_op.data() + (static_cast<size_t>(hkv) * d.seq + n) * d.D;
          for (int j = 0; j < d.D; ++j) o[j] += pt * vo[j];
        }
        const float inv = l > 0.f ? gamma / l : 0.f;
        for (int j = 0; j < d.D; ++j) o[j] *= inv;
      } else {
        for (int n = 0; n < d.seq; ++n) {
          if (p[static_cast<size_t>(n)] == 0.f) continue;
          const float* vr = vdeq.data() + (static_cast<size_t>(hkv) * d.seq + n) * d.D;
          for (int j = 0; j < d.D; ++j) o[j] += p[static_cast<size_t>(n)] * vr[j];
        }
        const float inv = l > 0.f ? 1.f / l : 0.f;
        for (int j = 0; j < d.D; ++j) o[j] *= inv;
      }
    }
  return out;
}

double l2_rel(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    num += static_cast<double>(a[i] - b[i]) * (a[i] - b[i]);
    den += static_cast<double>(b[i]) * b[i];
  }
  return std::sqrt(num / std::max(den, 1e-30));
}

// ---- distributions -------------------------------------------------------
// (a) unit-normal: the test's standard distribution (the WORST case for a
// block-scale win, the best case for row scales).
// (b) heavy-tailed K: per-row lognormal scale (sigma 1) + two 10-100x spikes
// in random dims (a stress regime, not what post-RMSNorm K looks like).
// (b') heavy-tailed K, RMSNorm-pinned: the row norm is pinned to 16 (what
// qsa_norm_rope leaves), with three outlier dims at 5-20x the typical
// element -- the realistic "outlier dims" regime FA3's incoherent
// processing targets.
// (c) sink K: one dim at 32x the row rms (a sink-token-like row).
enum class Dist { kUnit, kHeavy, kHeavyNorm, kSink };
const char* dist_name(Dist x) {
  switch (x) {
    case Dist::kUnit: return "unit-normal";
    case Dist::kHeavy: return "heavy-tailed K (lognormal rows + 10-100x spikes)";
    case Dist::kHeavyNorm: return "RMSNorm-pinned K + 3 outlier dims (5-20x)";
    case Dist::kSink: return "sink K (one dim at 32x row rms)";
  }
  return "?";
}

void gen(Data& d, Dist dist, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nrm(0.f, 1.f);
  std::lognormal_distribution<float> logn(0.f, 1.f);  // per-row scale, ~10-100x spread
  std::uniform_int_distribution<int> dimd(0, d.D - 1);
  std::uniform_real_distribution<float> spk(10.f, 100.f);
  std::uniform_real_distribution<float> spkn(5.f, 20.f);
  std::uniform_int_distribution<int> signd(0, 1);
  d.q.assign(static_cast<size_t>(d.rows) * d.H * d.D, 0.f);
  d.k.assign(static_cast<size_t>(d.seq) * d.KV * d.D, 0.f);
  d.v.assign(static_cast<size_t>(d.seq) * d.KV * d.D, 0.f);
  for (auto& x : d.q) x = nrm(rng);
  for (auto& x : d.v) x = nrm(rng);
  for (size_t i = 0; i < static_cast<size_t>(d.seq) * d.KV; ++i) {
    float* kr = d.k.data() + i * d.D;
    if (dist == Dist::kHeavy) {
      const float r = logn(rng);  // per-row lognormal scale
      for (int j = 0; j < d.D; ++j) kr[j] = r * nrm(rng);
      for (int s = 0; s < 2; ++s) kr[dimd(rng)] *= spk(rng);  // two 10-100x spikes
    } else if (dist == Dist::kHeavyNorm) {
      for (int j = 0; j < d.D; ++j) kr[j] = nrm(rng);
      double ms = 0;
      for (int j = 0; j < d.D; ++j) ms += static_cast<double>(kr[j]) * kr[j];
      const float sc = 16.0f / static_cast<float>(std::sqrt(ms));  // pin the norm at 16
      for (int j = 0; j < d.D; ++j) kr[j] *= sc;
      for (int s = 0; s < 3; ++s) kr[dimd(rng)] *= spkn(rng);  // three 5-20x outlier dims
    } else if (dist == Dist::kSink) {
      float ms = 0.f;
      for (int j = 1; j < d.D; ++j) {
        kr[j] = nrm(rng);
        ms += static_cast<double>(kr[j]) * kr[j];
      }
      const float rms = std::sqrt(static_cast<float>(ms / (d.D - 1)));
      kr[0] = 32.f * rms * (signd(rng) ? 1.f : -1.f);  // one dim dominates the row
    } else {
      for (int j = 0; j < d.D; ++j) kr[j] = nrm(rng);
    }
  }
}

const Cfg kConfigs[] = {
    // The C.1a baseline (Q blk, K blk, P e4m3 pow2-gamma, V row) + the
    // single-factor deltas + the Hadamard (C.2) columns + the best combos.
    {QF::BLK, QF::BLK, QF::ROW, PF::E4M3, false, "base   Q blk  K blk  P e4m3 V row "},
    // The DEPLOYED DEFAULT (DGPP_QSA_FP8_MX off: row scales on Q and K, the
    // P e4m3 pow2-gamma, V row) -- the config that actually ships, so its
    // bar status is the one that matters.
    {QF::ROW, QF::ROW, QF::ROW, PF::E4M3, false, "depdef Q row  K row  P e4m3 V row "},
    {QF::ROW, QF::BLK, QF::ROW, PF::E4M3, false, "q_row  Q row  K blk  P e4m3 V row "},
    {QF::BF16, QF::BLK, QF::ROW, PF::E4M3, false, "q_bf16 Q bf16 K blk  P e4m3 V row "},
    {QF::BLK, QF::ROW, QF::ROW, PF::E4M3, false, "k_row  Q blk  K row  P e4m3 V row "},
    {QF::BLK, QF::BLK, QF::ROW, PF::BF16, false, "p_bf16 Q blk  K blk  P bf16 V row "},
    {QF::BLK, QF::BLK, QF::BLK, PF::E4M3, false, "v_blk  Q blk  K blk  P e4m3 V blk "},
    {QF::BLK, QF::BLK, QF::BF16, PF::E4M3, false, "v_bf16 Q blk  K blk  P e4m3 V bf16"},
    // The Phase A dequant path (the 0.0360 bar's origin: K/V e4m3 row, Q and P
    // bf16, bf16 math) + its block-K variant.
    {QF::BF16, QF::ROW, QF::ROW, PF::BF16, false, "deq    Q bf16 K row  P bf16 V row "},
    {QF::BF16, QF::BLK, QF::ROW, PF::BF16, false, "deqkb  Q bf16 K blk  P bf16 V row "},
    {QF::BLK, QF::ROW, QF::ROW, PF::BF16, false, "qbkrow Q blk  K row  P bf16 V row "},
    // Hadamard (C.2) columns.
    {QF::BLK, QF::BLK, QF::ROW, PF::E4M3, true, "H+base H  Q blk  K blk  P e4m3 V row "},
    {QF::ROW, QF::ROW, QF::ROW, PF::E4M3, true, "H+row  H  Q row  K row  P e4m3 V row "},
    {QF::BLK, QF::ROW, QF::ROW, PF::E4M3, true, "H+krow H  Q blk  K row  P e4m3 V row "},
    {QF::BLK, QF::BLK, QF::ROW, PF::BF16, true, "H+pbf  H  Q blk  K blk  P bf16 V row "},
    {QF::BLK, QF::ROW, QF::ROW, PF::BF16, true, "H+qbk  H  Q blk  K row  P bf16 V row "},
    {QF::BF16, QF::ROW, QF::ROW, PF::BF16, true, "H+deq  H  Q bf16 K row  P bf16 V row "},
    // Per-factor isolation (one factor quantized, everything else bf16): the
    // pure contribution of each quantizer to the end-to-end l2_rel.
    {QF::BLK, QF::BF16, QF::BF16, PF::BF16, false, "iso_q  Q blk  K bf16 P bf16 V bf16"},
    {QF::BF16, QF::BLK, QF::BF16, PF::BF16, false, "iso_k  Q bf16 K blk  P bf16 V bf16"},
    {QF::BF16, QF::BF16, QF::BF16, PF::E4M3, false, "iso_p  Q bf16 K bf16 P e4m3 V bf16"},
    {QF::BF16, QF::BF16, QF::ROW, PF::BF16, false, "iso_v  Q bf16 K bf16 P bf16 V row "},
};

// ---- real-data mode (the DGPP_DUMP_QK dump) ---------------------------------
// The engine's instrument (QwenQsaLayer::dump_qk, src/models/qwen/layers.cpp)
// writes one layer's post-norm+RoPE Q/K (bf16) and raw V (bf16) per prefill
// chunk. Format: an 80-byte header (magic "DGPPQK01", layer, lh, lkv, D,
// n_segments, truncated, cap_bytes, total_raw_bytes, reserved[4]), then per
// segment a 24-byte record {pos0, tokens, lh, lkv, D, pad} + raw bf16 q
// (tokens*lh*D), k (tokens*lkv*D) and v (tokens*lkv*D).
struct RealDump {
  int layer = -1, lh = 0, lkv = 0, D = 0;
  uint32_t n_segments = 0, truncated = 0;
  uint64_t cap_bytes = 0, total_raw_bytes = 0;
  size_t tokens = 0;
  std::vector<float> q, k, v;  // fp32 [tokens, lh, D] / [tokens, lkv, D]
};

bool load_dump(const char* path, RealDump& d) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "[fp8-attn] cannot open dump %s\n", path);
    return false;
  }
  struct Header {
    char magic[8];
    uint32_t layer, lh, lkv, D;
    uint32_t n_segments, truncated;
    uint64_t cap_bytes, total_raw_bytes;
    uint64_t reserved[4];
  } h;
  f.read(reinterpret_cast<char*>(&h), sizeof(h));
  if (!f || std::memcmp(h.magic, "DGPPQK01", 8) != 0) {
    std::fprintf(stderr, "[fp8-attn] %s: bad magic (not a DGPP_DUMP_QK dump?)\n", path);
    return false;
  }
  d.layer = h.layer;
  d.lh = h.lh;
  d.lkv = h.lkv;
  d.D = h.D;
  d.n_segments = h.n_segments;
  d.truncated = h.truncated;
  d.cap_bytes = h.cap_bytes;
  d.total_raw_bytes = h.total_raw_bytes;
  struct Rec {
    uint32_t pos0, tokens, lh, lkv, D, pad;
  };
  // Walk segments to EOF (the header's n_segments is patched per segment by
  // the writer, but a killed run may leave it stale): a segment whose data
  // runs past EOF is dropped, not fatal.
  auto push = [&](std::vector<float>& dst, const uint16_t* src, size_t n) {
    dst.reserve(dst.size() + n);
    for (size_t i = 0; i < n; ++i) dst.push_back(bf16_bits_to_float(src[i]));
  };
  uint32_t s = 0;
  while (true) {
    Rec r;
    f.read(reinterpret_cast<char*>(&r), sizeof(r));
    if (!f) break;  // clean EOF (or a cut record: the partial record is dropped)
    const size_t qe = static_cast<size_t>(r.tokens) * r.lh * r.D;
    const size_t ke = static_cast<size_t>(r.tokens) * r.lkv * r.D;
    std::vector<uint16_t> buf(qe + 2 * ke);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>((qe + 2 * ke) * 2));
    if (!f) {
      std::fprintf(stderr, "[fp8-attn] %s: segment %u's data is cut off; using %u complete segments\n",
                   path, s, s);
      break;
    }
    push(d.q, buf.data(), qe);
    push(d.k, buf.data() + qe, ke);
    push(d.v, buf.data() + qe + ke, ke);
    d.tokens += r.tokens;
    ++s;
  }
  if (s != h.n_segments)
    std::fprintf(stderr, "[fp8-attn] %s: header says %u segments, file carries %u\n", path,
                 h.n_segments, s);
  d.n_segments = s;
  return d.tokens > 0;
}

// Per-row distribution statistics: the features the synthetic regimes model
// (row norm, within-row dynamic range, kurtosis, the outlier-dim fraction).
struct RowStats {
  size_t rows = 0;
  double norm_p50 = 0, norm_p99 = 0;  // row L2 norm
  double dr_p50 = 0, dr_p99 = 0;      // within-row dynamic range: absmax / rms
  double kurt = 0;                   // mean per-row excess kurtosis
  double outlier_frac = 0;          // fraction of elements with |x| > 10x row rms
};

RowStats row_stats(const std::vector<float>& rows, int D) {
  RowStats s;
  const size_t n = rows.size() / static_cast<size_t>(D);
  s.rows = n;
  if (n == 0) return s;
  std::vector<double> norms(n), drs(n), kurt(n, 0);
  double out_num = 0;
  for (size_t i = 0; i < n; ++i) {
    const float* r = rows.data() + i * static_cast<size_t>(D);
    double ms = 0, m4 = 0;
    float amax = 0;
    for (int j = 0; j < D; ++j) {
      const double x = r[j];
      ms += x * x;
      m4 += x * x * x * x;
      amax = std::max(amax, std::fabs(r[j]));
    }
    const double rms = std::sqrt(ms / D);
    norms[i] = rms * std::sqrt(static_cast<double>(D));
    drs[i] = rms > 0 ? amax / rms : 0;
    const double m2 = ms / D;
    kurt[i] = m2 > 0 ? m4 / D / (m2 * m2) - 3.0 : 0;
    for (int j = 0; j < D; ++j)
      if (std::fabs(r[j]) > 10.0 * rms) ++out_num;
  }
  std::sort(norms.begin(), norms.end());
  std::sort(drs.begin(), drs.end());
  s.norm_p50 = norms[n / 2];
  s.norm_p99 = norms[static_cast<size_t>(n * 0.99 + 0.5)];
  s.dr_p50 = drs[n / 2];
  s.dr_p99 = drs[static_cast<size_t>(n * 0.99 + 0.5)];
  double ksum = 0;
  for (double x : kurt) ksum += x;
  s.kurt = ksum / static_cast<double>(n);
  s.outlier_frac = static_cast<double>(out_num) / static_cast<double>(n * static_cast<size_t>(D));
  return s;
}

void print_row_stats(const char* label, const RowStats& s) {
  std::printf(
      "  %-30s rows %-7zu norm p50/p99 %8.2f/%8.2f  dr p50/p99 %7.2f/%7.2f  kurt %8.2f  "
      "outlier(>10x rms) %7.4f\n",
      label, s.rows, s.norm_p50, s.norm_p99, s.dr_p50, s.dr_p99, s.kurt, s.outlier_frac);
}

// Score-spread statistics: the per-(row, head) pre-softmax score vector (the
// bf16 reference's math, the unquantized q.k/16), the softmax sharpness the
// PV error dilution depends on. spread = max - mean of the row's scores;
// ent = the softmax entropy in nats (log(seq) is the flat maximum). This is
// the feature that distinguishes "the seq/sharpness dependence explains the
// real-vs-synthetic gap" from "the synthetic sampling model (i.i.d. rows /
// random V, uncorrelated Q/K) is the mismatch": a sharp real softmax
// (low entropy, high spread) that the i.i.d. synthetic does not reproduce is
// the structural mismatch.
struct ScoreStats {
  size_t n = 0;
  double spread_p50 = 0, spread_p99 = 0;
  double ent_p50 = 0, ent_p99 = 0;
};

ScoreStats score_stats(const Data& d) {
  const float s = 1.0f / 16.0f;
  const int grp = d.H / d.KV;
  std::vector<double> spreads, ents;
  for (int r = 0; r < d.rows; ++r)
    for (int h = 0; h < d.H; ++h) {
      const int hkv = h / grp;
      const float* qr = d.q.data() + (static_cast<size_t>(r) * d.H + h) * d.D;
      std::vector<float> sc(static_cast<size_t>(d.seq));
      float M = -INFINITY;
      for (int n = 0; n < d.seq; ++n) {
        const float* kr = d.k.data() + (static_cast<size_t>(n) * d.KV + hkv) * d.D;
        float dot = 0.f;
        for (int j = 0; j < d.D; ++j) dot += qr[j] * kr[j];
        sc[static_cast<size_t>(n)] = s * dot;
        M = std::max(M, sc[static_cast<size_t>(n)]);
      }
      double mn = 0;
      for (float x : sc) mn += x;
      mn /= d.seq;
      spreads.push_back(*std::max_element(sc.begin(), sc.end()) - mn);
      double l = 0;
      for (float x : sc) l += exp(x - M);
      double H = 0;
      for (float x : sc) {
        const double p = exp(x - M) / l;
        if (p > 0) H -= p * std::log(p);
      }
      ents.push_back(H);
    }
  ScoreStats st;
  st.n = spreads.size();
  std::sort(spreads.begin(), spreads.end());
  std::sort(ents.begin(), ents.end());
  st.spread_p50 = spreads[spreads.size() / 2];
  st.spread_p99 = spreads[static_cast<size_t>(spreads.size() * 0.99 + 0.5)];
  st.ent_p50 = ents[ents.size() / 2];
  st.ent_p99 = ents[static_cast<size_t>(ents.size() * 0.99 + 0.5)];
  return st;
}

void print_score_stats(const char* label, const ScoreStats& st, int seq) {
  std::printf("  %-30s %-6zu rows x heads  score max-mean p50/p99 %8.2f/%8.2f  "
              "softmax entropy p50/p99 %7.3f/%7.3f nats (flat %.3f)\n",
              label, st.n, st.spread_p50, st.spread_p99, st.ent_p50, st.ent_p99,
              std::log(static_cast<double>(seq)));
}

// ---- the grid runner (shared by the synthetic and real-data modes) ---------
struct Row {
  const Cfg* cfg;
  double l2;
};
bool is_base(const Cfg& c) { return c.name[0] == 'b' && c.name[1] == 'a'; }
bool is_tc(const Cfg& c) {  // the tensor-core path: both mmas stay on e4m3
  return c.q != QF::BF16 && c.k != QF::BF16 && c.p == PF::E4M3 && c.v != QF::BF16;
}
bool is_iso(const Cfg& c) { return c.name[0] == 'i'; }

struct GridOut {
  std::vector<Row> rows;
  double base = -1;
  const Row* best_tc = nullptr;
  const Row* best_any = nullptr;
};

GridOut run_grid(const char* title, int seeds, const std::function<void(Data&)>& make) {
  GridOut g;
  for (const Cfg& c : kConfigs) {
    double sum = 0;
    for (int sd = 0; sd < seeds; ++sd) {
      Data d;
      make(d);
      const auto ref = attn_ref(d);
      sum += l2_rel(attn_sim(d, c), ref);
    }
    const double l2 = sum / seeds;
    if (is_base(c)) g.base = l2;
    g.rows.push_back({&c, l2});
  }
  std::printf("[fp8-attn] === %s (mean of %d) ===\n", title, seeds);
  std::printf("  %-42s %8s %9s\n", "config", "l2_rel", "vs base");
  for (const Row& r : g.rows) {
    if (is_base(*r.cfg))
      std::printf("  %-42s %8.4f  (base)\n", r.cfg->name, r.l2);
    else
      std::printf("  %-42s %8.4f %+9.4f\n", r.cfg->name, r.l2, r.l2 - g.base);
  }
  // Verdict: the best TENSOR-CORE path (both mmas e4m3) and the best
  // DEPLOYABLE path (the iso_* rows are diagnostics -- one factor alone --
  // not a real pool configuration, so they are excluded from best_any).
  for (const Row& r : g.rows) {
    if (is_tc(*r.cfg) && (g.best_tc == nullptr || r.l2 < g.best_tc->l2)) g.best_tc = &r;
    if (!is_iso(*r.cfg) && (g.best_any == nullptr || r.l2 < g.best_any->l2)) g.best_any = &r;
  }
  std::printf("  best tensor-core: %-38s %.4f | best overall: %-38s %.4f | bar 0.035: %s\n",
              g.best_tc->cfg->name, g.best_tc->l2, g.best_any->cfg->name, g.best_any->l2,
              g.best_any->l2 <= 0.035 ? "REACHABLE" : "NOT reached");
  return g;
}

}  // namespace

int main(int argc, char** argv) {
  // Self-check: hadamard256 must be orthogonal (preserves norms and dots),
  // or every H+ row below is meaningless.
  {
    std::mt19937_64 rng(7);
    std::normal_distribution<float> nrm(0.f, 1.f);
    std::vector<float> a(256), b(256), ha(256), hb(256);
    for (int i = 0; i < 256; ++i) {
      a[i] = nrm(rng);
      b[i] = nrm(rng);
      ha[i] = a[i];
      hb[i] = b[i];
    }
    hadamard256(ha.data());
    hadamard256(hb.data());
    double da = 0, db = 0, dab = 0, dh = 0, dhb = 0, dhbh = 0;
    for (int i = 0; i < 256; ++i) {
      da += a[i] * a[i];
      db += b[i] * b[i];
      dab += a[i] * b[i];
      dh += ha[i] * ha[i];
      dhb += hb[i] * hb[i];
      dhbh += ha[i] * hb[i];
    }
    if (std::fabs(da - dh) > 1e-3 * da || std::fabs(db - dhb) > 1e-3 * db ||
        std::fabs(dab - dhbh) > 1e-3 * std::max(1.0, std::fabs(dab))) {
      std::fprintf(stderr, "hadamard256 self-check FAILED (not orthogonal)\n");
      return 1;
    }
  }
  const int seeds = 3;
  // Argument scan: --real <dump> arms the real-data mode, --seq <n> overrides
  // the real mode's K-token count, --synth-seq <n> runs the synthetic grids at
  // a K-token count other than the test's standard 128 slice (the seq/sharpness
  // sweep: does the synthetic l2_rel collapse at long seq the way the real
  // one does?), a bare number is the seed offset (re-draw every distribution
  // for a seed-to-seed stability check of the key numbers).
  const char* real_path = nullptr;
  int real_seq = 4096;
  int synth_seq = 128;
  uint64_t offset = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--real") == 0 && i + 1 < argc) {
      real_path = argv[++i];
    } else if (std::strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
      real_seq = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--synth-seq") == 0 && i + 1 < argc) {
      synth_seq = std::atoi(argv[++i]);
    } else {
      offset = static_cast<uint64_t>(std::strtoull(argv[i], nullptr, 10));
    }
  }
  const Dist dists[] = {Dist::kUnit, Dist::kHeavy, Dist::kHeavyNorm, Dist::kSink};
  const uint64_t seed_bases[] = {1000, 2000, 4000, 3000};
  for (int di = 0; di < 4; ++di) {
    const Dist dist = dists[di];
    const uint64_t seed0 = seed_bases[di] + offset;
    const std::string title = std::string(dist_name(dist)) + " (seq=" + std::to_string(synth_seq) + ")";
    run_grid(title.c_str(), seeds,
             [&dist, seed0, synth_seq](Data& d) { d.seq = synth_seq; gen(d, dist, seed0); });
    // The features that distinguish the real-vs-synthetic gap: the score
    // spread (softmax sharpness) and the V row norms the PV product sees, on
    // the same data the grid ran.
    Data probe;
    probe.seq = synth_seq;
    gen(probe, dist, seed0);
    print_score_stats(dist_name(dist), score_stats(probe), synth_seq);
    print_row_stats("V rows", row_stats(probe.v, probe.D));
  }
  if (real_path != nullptr) {
    RealDump rd;
    if (!load_dump(real_path, rd)) return 1;
    std::printf("[fp8-attn] === real dump %s: layer %d, %zu tokens in %u segments%s, %llu B raw ===\n",
                real_path, rd.layer, rd.tokens, rd.n_segments,
                rd.truncated ? " (TRUNCATED at the cap)" : "",
                static_cast<unsigned long long>(rd.total_raw_bytes));
    // Per-row statistics: the real rows vs the four synthetic regimes (the
    // features the NO-GO's distribution assumptions rest on).
    std::printf("[fp8-attn] per-row statistics (real vs synthetic K rows):\n");
    print_row_stats("real Q rows", row_stats(rd.q, rd.D));
    print_row_stats("real K rows", row_stats(rd.k, rd.D));
    for (int di = 0; di < 4; ++di) {
      Data d;
      gen(d, dists[di], seed_bases[di] + offset);
      print_row_stats(dist_name(dists[di]), row_stats(d.k, d.D));
    }
    // The grid on the real data: 16 strided query rows, all local q heads,
    // one kv head, 4096 strided K tokens (the attention the prefill walks).
    Data real;
    real.rows = 16;
    real.H = rd.lh;
    real.KV = rd.lkv;
    real.D = rd.D;
    real.seq = static_cast<int>(std::min<size_t>(real_seq, rd.tokens));
    real.q.assign(static_cast<size_t>(real.rows) * real.H * real.D, 0.f);
    real.k.assign(static_cast<size_t>(real.seq) * real.KV * real.D, 0.f);
    real.v.assign(static_cast<size_t>(real.seq) * real.KV * real.D, 0.f);
    const size_t tq_stride = rd.tokens > static_cast<size_t>(real.rows)
                                 ? rd.tokens / real.rows : 1;
    const size_t tk_stride = rd.tokens > static_cast<size_t>(real.seq)
                                 ? rd.tokens / real.seq : 1;
    for (int r = 0; r < real.rows; ++r) {
      const size_t t = static_cast<size_t>(r) * tq_stride;
      for (int h = 0; h < real.H; ++h)
        std::copy_n(rd.q.data() + (t * rd.lh + h) * rd.D, rd.D,
                    real.q.data() + (static_cast<size_t>(r) * real.H + h) * real.D);
    }
    for (int n = 0; n < real.seq; ++n) {
      const size_t t = static_cast<size_t>(n) * tk_stride;
      for (int hkv = 0; hkv < real.KV; ++hkv) {
        std::copy_n(rd.k.data() + (t * rd.lkv + hkv) * rd.D, rd.D,
                    real.k.data() + (static_cast<size_t>(n) * real.KV + hkv) * real.D);
        std::copy_n(rd.v.data() + (t * rd.lkv + hkv) * rd.D, rd.D,
                    real.v.data() + (static_cast<size_t>(n) * real.KV + hkv) * real.D);
      }
    }
    std::printf("[fp8-attn] grid geometry: rows=%d H=%d KV=%d seq=%d (strided from %zu tokens)\n",
                real.rows, real.H, real.KV, real.seq, rd.tokens);
    // The same distinguishing features on the real slice the grid runs: the
    // score spread (softmax sharpness) and the V row norms (the full dump, the
    // counterpart of the synthetic V rows above).
    print_score_stats("real scores", score_stats(real), real.seq);
    print_row_stats("real V rows", row_stats(rd.v, rd.D));
    run_grid("real post-RoPE Q/K (DGPP_DUMP_QK)", 1, [&real](Data& d) { d = real; });
  }
  return 0;
}
