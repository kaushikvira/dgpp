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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
  struct Row { const Cfg* cfg; double l2; };
  auto is_base = [](const Cfg& c) { return c.name[0] == 'b' && c.name[1] == 'a'; };
  // The tensor-core path: both mmas stay on e4m3 (Q/K quantized, P e4m3,
  // V quantized) -- the only regime where the Phase B/C work pays off.
  auto is_tc = [](const Cfg& c) {
    return c.q != QF::BF16 && c.k != QF::BF16 && c.p == PF::E4M3 && c.v != QF::BF16;
  };
  // Optional seed offset (argv[1]): re-draw every distribution for a
  // seed-to-seed stability check of the key numbers.
  const uint64_t offset =
      argc > 1 ? static_cast<uint64_t>(std::strtoull(argv[1], nullptr, 10)) : 0;
  const Dist dists[] = {Dist::kUnit, Dist::kHeavy, Dist::kHeavyNorm, Dist::kSink};
  const uint64_t seed_bases[] = {1000, 2000, 4000, 3000};
  for (int di = 0; di < 4; ++di) {
    const Dist dist = dists[di];
    std::vector<Row> rows;
    double base = -1;
    for (const Cfg& c : kConfigs) {
      double sum = 0;
      for (int sd = 0; sd < seeds; ++sd) {
        Data d;
        gen(d, dist, seed_bases[di] + offset + static_cast<uint64_t>(sd) * 97);
        const auto ref = attn_ref(d);
        sum += l2_rel(attn_sim(d, c), ref);
      }
      const double l2 = sum / seeds;
      if (is_base(c)) base = l2;
      rows.push_back({&c, l2});
    }
    std::printf("[fp8-attn] === %s (mean of %d seeds) ===\n", dist_name(dist), seeds);
    std::printf("  %-42s %8s %9s\n", "config", "l2_rel", "vs base");
    for (const Row& r : rows) {
      if (is_base(*r.cfg))
        std::printf("  %-42s %8.4f  (base)\n", r.cfg->name, r.l2);
      else
        std::printf("  %-42s %8.4f %+9.4f\n", r.cfg->name, r.l2, r.l2 - base);
    }
    // Verdict: the best TENSOR-CORE path (both mmas e4m3) and the best
    // DEPLOYABLE path (the iso_* rows are diagnostics -- one factor alone --
    // not a real pool configuration, so they are excluded from best_any).
    auto is_iso = [](const Cfg& c) { return c.name[0] == 'i'; };
    const Row* best_tc = nullptr, *best_any = nullptr;
    for (const Row& r : rows) {
      if (is_tc(*r.cfg) && (best_tc == nullptr || r.l2 < best_tc->l2)) best_tc = &r;
      if (!is_iso(*r.cfg) && (best_any == nullptr || r.l2 < best_any->l2)) best_any = &r;
    }
    std::printf("  best tensor-core: %-38s %.4f | best overall: %-38s %.4f | bar 0.035: %s\n",
                best_tc->cfg->name, best_tc->l2, best_any->cfg->name, best_any->l2,
                best_any->l2 <= 0.035 ? "REACHABLE" : "NOT reached");
  }
  return 0;
}
