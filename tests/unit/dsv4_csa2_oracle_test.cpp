// DeepSeek-V4-Flash (deepseek_v4) CSA2 kernel-geometry CPU oracles
// (docs/dsv4_kernel_port_spec.md §2.1). Host-only, no GPU: the naive
// FP64/FP32 references that pin the numerics of the V4 CSA2 surface —
// the C4A/C128A compressor (state store + boundary compress into the
// 584 B record), the 64-head indexer (the q-side fold + the per-
// (compressed-)token topk-512) and the sparse/union MLA attention — so
// the Phase B v4 layer files (src/models/dsv4/csa2_layer.cu) have a
// host-side oracle before any forward path consumes them.
//
// Every reference below is borrowed, not rewritten (house rule 2): the
// math mirrors the dsv4-native (~/work/dsv4-native, read-only) oracle
// that the spec names as the numeric source, cited at the borrow
// point. "Looks right" is not done: each test cross-checks the naive
// reference against an INDEPENDENT formulation (a different summation /
// closed-form path) and pins the documented edge cases (the sink's
// exactly-once, the +inf degenerate limit, the topk tie-break, the
// causal bound), so a silent corruption of the borrowed form is caught
// here, not on the GPU.
//
// The one genuinely new numerical surface (the fp8 x fp8 dynamic-scheme
// block-scaled GEMM, the spec §3.2) is pinned by the naive double form
// here; its fp32 block-rescale contract already lives in
// tests/unit/dsv4_fp8_scale_test.cpp (dsv4_block128_partial_rescale_math).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/csa2.hpp"
#include "models/dsv4/csa2_layer.hpp"

namespace {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
using dgpp::fp8_e4m3_bits_to_float;
using dgpp::float_to_fp8_e4m3_bits;

// ---- the exact codecs (the tree's host helpers + the fp4/e8m0 tables) ----
// e2m1 (MXFP4) nibble -> value: the 8-code table (0, .5, 1, 1.5, 2, 3, 4, 6)
// with the sign bit 0x8. Mirrors dsv4-native ops/common/op_test.h
// e2m1_ref_value (the kMagnitude table).
// ue8m0 byte -> power of two: 2^(b - 127); 255 is the NaN sentinel (the
// tree's e8m0_to_float contract, tests/unit/dsv4_fp8_scale_test.cpp).
double e8m0_value(uint8_t b) {
  if (b == 255) return std::nan("");
  return std::ldexp(1.0, static_cast<int>(b) - 127);
}
// The exact e4m3 decode as a double (the kernel's fp32 decode is the
// documented rounding point; the double is the oracle's value).
double e4m3_value(uint8_t b) { return static_cast<double>(fp8_e4m3_bits_to_float(b)); }

// The fp8 x fp8 dynamic-scheme block-scaled GEMM (the spec §3.2, the
// checkpoint's inference/kernel.py fp8_gemm + the dsv4-native ops/linear
// contract, src/ops/linear/linear.h:1-40):
//   out[r, m] = bf16( Σ_kb 2^(xs[r,kb] + ws[m,kb] - 254) *
//                       Σ_{k in kb} x[r,k] * w[m,k] )
// where x / w are the EXACT e4m3 values (double) and xs / ws are the
// per-128-k-block ue8m0 scale bytes. The e4m3 values and the power-of-
// two factors are exact in double, so the whole expression is exact and
// the op's single bf16 rounding is applied once at the end. This is the
// "no fp32 intermediate in global memory" bandwidth rule restated as a
// naive double reference (dsv4-native tests/test_op_linear.cpp's
// linear_fp8_ref, ops/common/op_test.h:515).
std::vector<uint16_t> fp8_block_scaled_gemm(const double* x, const uint8_t* x_scale, int n_rows,
                                             const double* w, const uint8_t* w_scale, int n_out, int k) {
  const int nkb = k / 128;
  std::vector<uint16_t> out(static_cast<size_t>(n_rows) * n_out);
  for (int r = 0; r < n_rows; ++r)
    for (int m = 0; m < n_out; ++m) {
      double acc = 0.0;
      for (int kb = 0; kb < nkb; ++kb) {
        double partial = 0.0;
        for (int kx = kb * 128; kx < (kb + 1) * 128; ++kx)
          partial += x[static_cast<size_t>(r) * k + kx] * w[static_cast<size_t>(m) * k + kx];
        // 2^(xs + ws - 254): exact power of two in double (the two ue8m0
        // powers multiplied), the fp32 kernel's per-block rescale factor.
        const double factor = std::ldexp(1.0, static_cast<int>(x_scale[static_cast<size_t>(r) * nkb + kb]) +
                                                 static_cast<int>(w_scale[static_cast<size_t>(m / 128) * nkb + kb]) -
                                             254);
        acc += factor * partial;
      }
      out[static_cast<size_t>(r) * n_out + m] = float_to_bf16_bits(static_cast<float>(acc));
    }
  return out;
}
// bf16 word -> exact double (a bf16 word is an fp32 word with the low 16
// bits zero).
double bf16_to_d64(uint16_t w) { return static_cast<double>(bf16_bits_to_float(w)); }

// A deterministic e4m3 activation / weight plane generator (no NaN codes,
// bounded magnitudes so the GEMM stays in the fp32 range): a small
// LCG over the finite e4m3 codes, symmetric so the sums do not drift.
void fill_plane(std::vector<double>* vals, std::vector<uint8_t>* scale,
                std::size_t rows, std::size_t cols, std::uint32_t seed) {
  vals->resize(rows * cols);
  scale->resize(rows * ((cols + 127) / 128));
  std::uint32_t s = seed;
  auto next = [&]() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  };
  for (std::size_t i = 0; i < rows * cols; ++i) {
    // A finite e4m3 code in 1..0x7E (skip the 0x7F/0xFF NaN patterns),
    // sign applied by the LCG's low bit so the plane is symmetric. The
    // GEMM reference consumes the DECODED double value (exact e4m3).
    const int mag = 1 + static_cast<int>(next() % 0x7E);
    const int code = (next() & 1) ? mag : (mag | 0x80);
    (*vals)[i] = e4m3_value(static_cast<uint8_t>(code & 0xFF));
  }
  for (std::size_t i = 0; i < rows * ((cols + 127) / 128); ++i)
    (*scale)[i] = static_cast<uint8_t>(127 + static_cast<int>(next() % 7) - 3);  // 2^-3..2^3
}

// ---- the V4 584 B record (dsv4-native src/ops/kv_cache/kv_cache.h:83-108) ----
//   [0,448)   448 x fp8 e4m3 (7 groups of 64)
//   [448,576) 64 x bf16 RoPE (unquantized)
//   [576,584) 7 x ue8m0 scale + 1 pad byte
constexpr int kRecordBytes = 584;
constexpr int kNoPE = 448;
constexpr int kNoPEGroups = 7;
constexpr int kGroupEls = 64;
constexpr int kRoPE = 64;
constexpr int kHeadDim = 512;
// Decode one 584 B record into the 512-dim latent (double): the 448 NoPE
// dims are e4m3(code) * 2^(scale_byte - 127) (the per-group ue8m0 power,
// exact), the 64 RoPE dims are the bf16 words. Mirrors dsv4-native's
// kv_slot_ref_latent (the table decode, a different path from the kernel's
// in-register e4m3 x ue8m0 dequant).
void record_latent(const uint8_t* rec, double* lat) {
  for (int g = 0; g < kNoPEGroups; ++g) {
    const double sc = e8m0_value(rec[576 + g]);
    for (int i = 0; i < kGroupEls; ++i)
      lat[g * kGroupEls + i] = e4m3_value(rec[g * kGroupEls + i]) * sc;
  }
  for (int i = 0; i < kRoPE; ++i)
    lat[kNoPE + i] = bf16_to_d64(reinterpret_cast<const uint16_t*>(rec + 448)[i]);
}
// The reference's per-64-group fp8 quant of a 448-dim bf16-rounded NoPE
// vector (dsv4-native kv_cache.h:46-52 + compressor.h:1-60 step 5): the
// 1e-4 absmax floor, exp = ceil(log2(absmax / 448)), the e4m3 RNE codes
// (clamp +/-448), the ue8m0 scale bytes (exp + 127 clamped 0..255; 7 real
// + 1 pad). Returns the 8 scale bytes (the 8th the pad).
// Assemble the 584 B record from the 512-dim (NoPE double, RoPE bf16
// words) latent, applying the per-group fp8 quant to the NoPE 448 and
// the single bf16 RNE to the RoPE 64. The NoPE codes are the e4m3 RNE
// of the bf16-ROUND-TRIPPED value (the production's parity step — the
// quant input is the bf16-rounded normed value, NOT the fp32 normed; do
// NOT "optimize" it away, compressor.h:1-60 step 5).
void assemble_record(const double* nope, const uint16_t* rope_words, uint8_t* rec) {
  std::memset(rec, 0, kRecordBytes);
  for (int g = 0; g < kNoPEGroups; ++g) {
    double amax = 1e-4;
    for (int i = 0; i < kGroupEls; ++i) amax = std::max(amax, std::fabs(nope[g * kGroupEls + i]));
    int e = 0;
    double q = amax / 448.0;
    if (q > 0) while (std::ldexp(1.0, e) < q) ++e;
    rec[576 + g] = static_cast<uint8_t>(std::min<std::int32_t>(255, std::max<std::int32_t>(0, e + 127)));
    const double sc = std::ldexp(1.0, e);
    for (int i = 0; i < kGroupEls; ++i) {
      // The bf16 round-trip parity step: the quant input is the bf16-
      // rounded value (a bf16 word is an exact fp32 word), then the e4m3
      // RNE of (value / scale) clamped to +/-448.
      const double v = bf16_to_d64(float_to_bf16_bits(static_cast<float>(nope[g * kGroupEls + i])));
      const double cl = std::min(std::max(v / sc, -448.0), 448.0);
      rec[g * kGroupEls + i] = float_to_fp8_e4m3_bits(static_cast<float>(cl));
    }
  }
  std::memcpy(rec + 448, rope_words, kRoPE * 2);  // the 64 bf16 RoPE words
}

// ---- the compressor (dsv4-native src/ops/compressor/compressor.h) -------
// The two-step contract: state_store (EVERY token: the kv / score GEMM +
// the APE on the score half) and compress (the group boundary: the
// per-dim softmax pooling + RMSNorm + the GPT-J RoPE on the last 64 + the
// fp8 quant -> the 584 B record). The reference is the independent
// DOUBLE path (dsv4-native tests/test_op_compressor.cpp's
// compressor_ref, ops/common/op_test.h:2608).
struct CmpCfg {
  int ratio;  // 4 (C4A, coff 2) or 128 (C128A, coff 1)
  int coff() const { return ratio == 4 ? 2 : 1; }
  int block_size() const { return ratio == 4 ? 4 : 8; }
  int state_row() const { return 2 * coff() * 512; }
  int n_gather() const { return coff() * ratio; }
};
// The state store: state[blk(p), off(p), 0:W] = kv (the GEMM's bf16
// value, untouched), state[blk(p), off(p), W:2W] = score + ape[p % ratio]
// (the APE on the SCORE half only, indexed position % ratio). The
// physical block is state_block_table[blk] (the 1-row decode's identity
// table). Returns the state as a flat double [n_state_blocks, block_size,
// 2*coff*512] (the unwritten rows 0).
std::vector<double> compressor_state_store(const std::vector<double>& x, const std::vector<uint8_t>& x_scale,
                                           const std::vector<double>& wkv, const std::vector<uint8_t>& wkv_scale,
                                           const std::vector<double>& wgate, const std::vector<uint8_t>& wgate_scale,
                                           const std::vector<double>& ape, const std::vector<int>& positions,
                                           const std::vector<int>& block_table, int n_state_blocks, int W, int bs,
                                           int ratio, int k) {
  const int n = static_cast<int>(positions.size());
  std::vector<uint16_t> kv = fp8_block_scaled_gemm(x.data(), x_scale.data(), n, wkv.data(), wkv_scale.data(), W, k);
  std::vector<uint16_t> sc = fp8_block_scaled_gemm(x.data(), x_scale.data(), n, wgate.data(), wgate_scale.data(), W, k);
  std::vector<double> state(static_cast<size_t>(n_state_blocks) * bs * 2 * W, 0.0);
  for (int t = 0; t < n; ++t) {
    const int p = positions[t];
    const int blk = p / bs, off = p % bs;
    const int phys = block_table[blk];
    double* row = state.data() + (static_cast<size_t>(phys) * bs * 2 * W + static_cast<size_t>(off) * 2 * W);
    const int arow = p % ratio;  // the ape row = position % ratio (the reference's
                                 // save_partial_states' APE index, compressor.h)
    for (int d = 0; d < W; ++d) {
      row[d] = bf16_to_d64(kv[static_cast<size_t>(t) * W + d]);
      row[W + d] = bf16_to_d64(sc[static_cast<size_t>(t) * W + d]) + ape[static_cast<size_t>(arow) * W + d];
    }
  }
  return state;
}
// The boundary compress: the group's per-dim softmax pooling (the -inf-
// masked rows for pos < 0 get weight 0; the C4A head-offset DIMENSION
// split, the (row >= ratio) gate), the fp32 RMSNorm, the GPT-J RoPE on
// the last 64 at the group-start position, the fp8 quant -> the 584 B
// record. `p` is the boundary position ((p + 1) % ratio == 0). Returns
// the assembled record.
std::vector<uint8_t> compressor_compress(const std::vector<double>& state, const std::vector<int>& block_table, int p,
                                         const std::vector<double>& cos_sin, const std::vector<uint16_t>& rms_w,
                                         double rms_eps, const CmpCfg& cfg) {
  const int W = cfg.coff() * 512;
  const int bs = cfg.block_size();
  const int n_gather = cfg.n_gather();
  const int start = p - n_gather + 1;  // the group's first position
  const int gpos = (p / cfg.ratio) * cfg.ratio;  // the group-start position (the RoPE position)
  std::vector<double> pooled(512, 0.0);
  for (int d = 0; d < 512; ++d) {
    // The two-pass max-shift softmax over the group's score states (the
    // kernel's chunked online-softmax is algebraically identical, within
    // the fp32 budget — the C5.5 oracle's documented class).
    double M = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n_gather; ++i) {
      const int pos_i = start + i;
      if (pos_i < 0) continue;  // the masked row (the weight 0)
      const int head_off = (cfg.coff() == 2 && i >= cfg.ratio) ? 512 : 0;
      const double* row = state.data() +
                          static_cast<size_t>(block_table[pos_i / bs]) * bs * 2 * W +
                          static_cast<size_t>(pos_i % bs) * 2 * W + head_off;
      const double s = row[W + d];  // the score half
      if (s > M) M = s;
    }
    double L = 0.0, A = 0.0;
    for (int i = 0; i < n_gather; ++i) {
      const int pos_i = start + i;
      if (pos_i < 0) continue;
      const int head_off = (cfg.coff() == 2 && i >= cfg.ratio) ? 512 : 0;
      const double* row = state.data() +
                          static_cast<size_t>(block_table[pos_i / bs]) * bs * 2 * W +
                          static_cast<size_t>(pos_i % bs) * 2 * W + head_off;
      const double w = std::exp(row[W + d] - M);
      L += w;
      A += w * row[d];  // the weighted kv
    }
    pooled[d] = A / L;
  }
  // The one-rounding RMSNorm (fp32 interior, the weight in fp32, ONE bf16
  // rounding — but the oracle keeps the double, the kernel's rounding
  // point is the documented budget).
  double mean = 0.0;
  for (int d = 0; d < 512; ++d) mean += pooled[d] * pooled[d];
  const double rsqrt = 1.0 / std::sqrt(mean / 512.0 + rms_eps);
  for (int d = 0; d < 512; ++d)
    pooled[d] = pooled[d] * rsqrt * static_cast<double>(bf16_bits_to_float(rms_w[d]));
  // The GPT-J FORWARD, NON-NEOX interleaved-pair RoPE on the LAST 64 dims
  // (the rope_head_dim), at the group-start position, over the caller's
  // cos_sin table (per row: the first 32 = cos + the last 32 = sin, the
  // per-pair layout). The 64 rotated values -> bf16 (the single bf16 RNE).
  std::vector<uint16_t> rope_words(kRoPE);
  const double* crow = cos_sin.data() + static_cast<size_t>(gpos) * 64;
  for (int j = 0; j < 32; ++j) {
    const double c = crow[j], s = crow[32 + j];
    const int a = 448 + 2 * j, b = 448 + 2 * j + 1;
    const double x0 = pooled[a], x1 = pooled[b];
    rope_words[j * 2] = float_to_bf16_bits(static_cast<float>(x0 * c - x1 * s));
    rope_words[j * 2 + 1] = float_to_bf16_bits(static_cast<float>(x1 * c + x0 * s));
  }
  std::vector<uint8_t> rec(kRecordBytes);
  assemble_record(pooled.data(), rope_words.data(), rec.data());
  return rec;
}

// ---- the indexer (dsv4-native src/ops/indexer_token/indexer_token.h) ----
// The q-side fused op: the GPT-J RoPE on each head's last 64 dims, the
// 64-term learned-weight fold (the weights_proj output), the q-side
// scaling (128^-0.5 * 64^-0.5), the per-row amax + the e4m3 fp8 quant.
// The independent DOUBLE path (dsv4-native tests/test_op_indexer_token.cpp
// + ops/common/op_test.h:1601 indexer_token_ref).
struct IndexerQOut {
  std::vector<uint8_t> q_fp8;  // [n_rows, 128]
  std::vector<float> q_scale;  // [n_rows]
};
IndexerQOut indexer_q_fused(const std::vector<uint16_t>& q, const std::vector<double>& cos,
                            const std::vector<double>& sin, const std::vector<uint16_t>& weights,
                            float sm_scale, float head_scale) {
  const int n_rows = static_cast<int>(q.size() / (64 * 128));
  IndexerQOut out;
  out.q_fp8.resize(static_cast<size_t>(n_rows) * 128);
  out.q_scale.resize(n_rows);
  const float sf = sm_scale * head_scale;
  for (int r = 0; r < n_rows; ++r) {
    // The 64-term weight fold (double — exact: the bf16 x rope'd
    // products are <= 40 significant bits, the 64-term sum < 2^53).
    std::vector<double> t(128, 0.0);
    for (int h = 0; h < 64; ++h) {
      const double w = bf16_to_d64(weights[static_cast<size_t>(r) * 64 + h]);
      for (int d = 0; d < 128; ++d) {
        double x = bf16_to_d64(q[(static_cast<size_t>(r) * 64 + h) * 128 + d]);
        if (d >= 64) {  // the rope block (the last 64 dims): the GPT-J
                        // adjacent pairs, the partner the adjacent dim.
          const int j = (d - 64) / 2;
          const double c = cos[static_cast<size_t>(r) * 32 + j];
          const double s = sin[static_cast<size_t>(r) * 32 + j];
          const int partner = d + ((d & 1) == 0 ? 1 : -1);
          const double p = bf16_to_d64(q[(static_cast<size_t>(r) * 64 + h) * 128 + partner]);
          x = ((d & 1) == 0) ? (x * c - p * s) : (x * c + p * s);
        }
        t[d] += w * x;
      }
    }
    // The fp32 rounding points (the kernel's): the fold sum (double
    // exact, rounded ONCE), the scaling multiply, the amax, the fp32/448
    // division.
    std::vector<float> qsum(128);
    double amax = 0.0;
    for (int d = 0; d < 128; ++d) {
      qsum[d] = static_cast<float>(sf * t[d]);
      amax = std::max(amax, std::fabs(static_cast<double>(qsum[d])));
    }
    const float scale = static_cast<float>(std::max(amax, 1e-4) / 448.0);
    out.q_scale[r] = scale;
    for (int d = 0; d < 128; ++d) {
      const float v = qsum[d] / scale;
      const double cl = std::min(std::max(static_cast<double>(v), -448.0), 448.0);
      out.q_fp8[static_cast<size_t>(r) * 128 + d] = float_to_fp8_e4m3_bits(static_cast<float>(cl));
    }
  }
  return out;
}
// The per-(compressed-)token topk: the score is the dequantized 128-dim
// dot (q_scale * k_scale * Σ e4m3(q) * e4m3(k)), the top-`topk` in
// (score desc, index asc) order over the first valid_lens entries (the
// causal bound), the -1 padding (dsv4-native indexer_token.h:1-150 +
// ops/common/op_test.h:1601).
std::vector<int> indexer_topk(const IndexerQOut& qo, const std::vector<uint8_t>& cache, int r, int n_comp, int topk,
                              int valid_len) {
  // The cache's per-entry fp32 scales (the 132 B V3.2 layout: 128 e4m3 +
  // ONE fp32 scale, the exact dequant scale amax/448, NOT a power of two).
  std::vector<float> k_scale(n_comp);
  for (int j = 0; j < n_comp; ++j)
    std::memcpy(&k_scale[j], cache.data() + static_cast<size_t>(j) * 132 + 128, 4);
  std::vector<double> sc(n_comp, 0.0);
  for (int j = 0; j < n_comp; ++j) {
    double s = 0.0;
    for (int d = 0; d < 128; ++d)
      s += e4m3_value(qo.q_fp8[static_cast<size_t>(r) * 128 + d]) * e4m3_value(cache[static_cast<size_t>(j) * 132 + d]);
    sc[j] = static_cast<double>(qo.q_scale[r]) * static_cast<double>(k_scale[j]) * s;
  }
  const int n_valid = valid_len;
  std::vector<std::pair<double, int>> sel;
  sel.reserve(n_valid);
  for (int j = 0; j < n_valid; ++j) sel.emplace_back(sc[j], j);
  std::sort(sel.begin(), sel.end(),
            [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
  std::vector<int> topk_out(topk, -1);
  for (int s = 0; s < topk && s < n_valid; ++s) topk_out[s] = sel[s].second;
  return topk_out;
}

// ---- the sparse / union MLA attention (dsv4-native src/ops/sparse_attn/
// sparse_attn.h + src/ops/dspark_attn/dspark_attn.h) ---------------------
// The single max-shift softmax over the (possibly multi-phase) union of
// KV latents, the D25 sink's exactly-once (the sink's mass enters the
// denominator ONCE). The independent DOUBLE path (dsv4-native
// ops/common/op_test.h:1900 sparse_attn_ref / :1990 dspark_attn_ref).
// `latents` [n_tokens, 512] double (the caller decodes the records);
// `q` [64, 512] double (one row). Returns the [64, 512] double output.
std::vector<double> sparse_attn_ref(const double* q, const double* latents, int n_tokens, const float* attn_sink) {
  constexpr int kHeads = 64, kHeadDim = 512;
  const double softmax_scale = 0.04419417382415922;  // 1/sqrt(512)
  std::vector<double> out(static_cast<size_t>(kHeads) * kHeadDim, 0.0);
  std::vector<double> S(n_tokens);
  for (int h = 0; h < kHeads; ++h) {
    double mx = -std::numeric_limits<double>::infinity();
    for (int t = 0; t < n_tokens; ++t) {
      double s = 0.0;
      for (int d = 0; d < kHeadDim; ++d) s += q[static_cast<size_t>(h) * kHeadDim + d] * latents[static_cast<size_t>(t) * kHeadDim + d];
      s *= softmax_scale;
      S[t] = s;
      if (s > mx) mx = s;
    }
    const double sink = (attn_sink != nullptr) ? static_cast<double>(attn_sink[h]) : -std::numeric_limits<double>::infinity();
    if (sink == std::numeric_limits<double>::infinity()) {
      // The +inf degenerate limit: the sink's mass dominates the
      // denominator, the EXACTLY-zero output (the kernel's NaN-safe
      // rescale's documented edge).
      continue;  // out stays 0
    }
    const double m = (sink > mx) ? sink : mx;  // m' = max(mx, sink)
    double sum = 0.0;
    for (int t = 0; t < n_tokens; ++t) sum += std::exp(S[t] - m);
    sum += std::exp(sink - m);  // the sink's mass (0 at -inf, 1 at m' == sink)
    for (int d = 0; d < kHeadDim; ++d) {
      double acc = 0.0;
      for (int t = 0; t < n_tokens; ++t) acc += std::exp(S[t] - m) * latents[static_cast<size_t>(t) * kHeadDim + d];
      out[static_cast<size_t>(h) * kHeadDim + d] = (sum > 0.0) ? acc / sum : 0.0;
    }
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
DGPP_TEST(dsv4_csa2_compressor_state_store) {
  // The state store's layout + the APE-on-the-score-half-only contract:
  // the kv half is the GEMM's bf16 value (untouched), the score half is
  // the GEMM's bf16 value + ape[p % ratio]. A tiny C4A geometry
  // (ratio 4, coff 2, block 4, W = 1024) with a K = 512 slice so the
  // GEMM is cheap (the contract is block/row addressing, not K).
  const int ratio = 4, W = 1024, bs = 4, K = 512, n = 6;
  std::vector<double> x, wkv, wgate;
  std::vector<uint8_t> xs, wks, wgs;
  fill_plane(&x, &xs, n, K, 0xA11CE);  // x: the e4m3-decoded values + ue8m0 scales
  fill_plane(&wkv, &wks, W, K, 0xB0B);
  fill_plane(&wgate, &wgs, W, K, 0xC0D);
  std::vector<double> ape(static_cast<size_t>(ratio) * W);
  for (auto& v : ape) v = 0.5;
  std::vector<int> positions(n), block_table(2, 0);
  for (int t = 0; t < n; ++t) {
    positions[t] = t;
    block_table[t / bs] = t / bs;  // the identity table
  }
  const auto state = compressor_state_store(x, xs, wkv, wks, wgate, wgs, ape, positions, block_table, 2, W, bs, ratio, K);
  // Independent check: recompute the kv / score GEMM directly and verify
  // the state's halves (the kv half the bf16 value, the score half + ape).
  const auto kv = fp8_block_scaled_gemm(x.data(), xs.data(), n, wkv.data(), wks.data(), W, K);
  const auto sc = fp8_block_scaled_gemm(x.data(), xs.data(), n, wgate.data(), wgs.data(), W, K);
  for (int t = 0; t < n; ++t) {
    const int off = positions[t] % bs;
    const double* row = state.data() + static_cast<size_t>(block_table[positions[t] / bs]) * bs * 2 * W +
                        static_cast<size_t>(off) * 2 * W;
    for (int d = 0; d < W; ++d) {
      const double want_kv = bf16_to_d64(kv[static_cast<size_t>(t) * W + d]);
      const double want_sc = bf16_to_d64(sc[static_cast<size_t>(t) * W + d]) + ape[static_cast<size_t>((positions[t] % ratio) * W + d)];
      if (row[d] != want_kv)
        throw std::runtime_error("state_store: the kv half is not the GEMM's bf16 value at t=" + std::to_string(t) +
                                 " d=" + std::to_string(d));
      if (row[W + d] != want_sc)
        throw std::runtime_error("state_store: the score half is not GEMM + ape at t=" + std::to_string(t) +
                                 " d=" + std::to_string(d));
    }
  }
}

DGPP_TEST(dsv4_csa2_compressor_compress_record) {
  // The boundary compress: run a full C4A state store (n = 16 tokens, the
  // identity table) and compress the boundary at p = 15 ((15+1) % 4 == 0,
  // the group [8..15] + the overlap [8..15] = the 8-row gather). Verify
  // the 584 B record's layout + the pooled value against an INDEPENDENT
  // max-shift softmax formulation (a different loop order) and the
  // per-group fp8 quant's invariants (the 7 real scale bytes + the pad,
  // the e4m3 codes' dequant * scale reproducing the bf16-rounded NoPE
  // within one e4m3 quantum).
  const int ratio = 4, W = 1024, bs = 4, K = 512, n = 16;
  std::vector<double> x, wkv, wgate;
  std::vector<uint8_t> xs, wks, wgs;
  fill_plane(&x, &xs, n, K, 0x1111);
  fill_plane(&wkv, &wks, W, K, 0x2222);
  fill_plane(&wgate, &wgs, W, K, 0x3333);
  std::vector<double> ape(static_cast<size_t>(ratio) * W, 0.25);
  std::vector<int> positions(n), block_table((n + bs - 1) / bs);
  for (int t = 0; t < n; ++t) {
    positions[t] = t;
    block_table[t / bs] = t / bs;
  }
  const auto state = compressor_state_store(x, xs, wkv, wks, wgate, wgs, ape, positions, block_table, (n + bs - 1) / bs, W, bs, ratio, K);
  // The cos_sin table (the group-start position 12: (15 / 4) * 4 = 12).
  const int gpos = (15 / ratio) * ratio;
  std::vector<double> cos_sin(static_cast<size_t>(gpos + 1) * 64);
  for (int p = 0; p <= gpos; ++p)
    for (int j = 0; j < 32; ++j) {
      const double ang = static_cast<double>(p) * (0.01 * (j + 1));  // a deterministic table
      cos_sin[static_cast<size_t>(p) * 64 + j] = std::cos(ang);
      cos_sin[static_cast<size_t>(p) * 64 + 32 + j] = std::sin(ang);
    }
  std::vector<uint16_t> rms_w(512, 0x3F80);  // bf16 1.0
  const CmpCfg cfg{ratio};
  const auto rec = compressor_compress(state, block_table, 15, cos_sin, rms_w, 1e-6, cfg);
  // Layout invariants: the pad byte is 0, the 7 real scale bytes are
  // finite powers-of-two exponents, the record is 584 B.
  if (rec.size() != kRecordBytes) throw std::runtime_error("the record is not 584 B");
  if (rec[583] != 0) throw std::runtime_error("the record's pad byte is not zero");
  for (int g = 0; g < kNoPEGroups; ++g)
    if (rec[576 + g] == 255) throw std::runtime_error("a record scale byte is the NaN sentinel");
  // Decode the record's latent and cross-check the NoPE 448 against an
  // INDEPENDENT recompute of the full pipeline (the per-dim max-shift
  // pooled -> the global RMSNorm -> the bf16 round-trip) from the state:
  // the record's NoPE dequant (e4m3(code) * the group's power-of-two
  // scale) must equal bf16(normed) within one e4m3 quantum (the
  // quant's documented budget). The recompute is a different code path
  // (the full 512-dim pooled vector, not the dim-major sample) so a
  // record-assembly corruption is caught.
  std::vector<double> lat(kHeadDim);
  record_latent(rec.data(), lat.data());
  // The full 512-dim pooled vector (the per-dim max-shift softmax, the
  // reference's math).
  const int start = 15 - cfg.n_gather() + 1;  // 8
  std::vector<double> pooled(512, 0.0);
  for (int d = 0; d < 512; ++d) {
    double M = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < cfg.n_gather(); ++i) {
      const int pos_i = start + i;
      if (pos_i < 0) continue;
      const int head_off = (i >= ratio) ? 512 : 0;
      const double* row = state.data() + static_cast<size_t>(block_table[pos_i / bs]) * bs * 2 * W +
                          static_cast<size_t>(pos_i % bs) * 2 * W + head_off;
      M = std::max(M, row[W + d]);
    }
    double L = 0.0, A = 0.0;
    for (int i = 0; i < cfg.n_gather(); ++i) {
      const int pos_i = start + i;
      if (pos_i < 0) continue;
      const int head_off = (i >= ratio) ? 512 : 0;
      const double* row = state.data() + static_cast<size_t>(block_table[pos_i / bs]) * bs * 2 * W +
                          static_cast<size_t>(pos_i % bs) * 2 * W + head_off;
      const double w = std::exp(row[W + d] - M);
      L += w;
      A += w * row[d];
    }
    pooled[d] = A / L;
  }
  // The global RMSNorm (the rms_w is 1.0, so the norm is the pooled
  // value scaled by the global rsqrt over all 512 dims).
  double mean = 0.0;
  for (int d = 0; d < 512; ++d) mean += pooled[d] * pooled[d];
  const double rsqrt = 1.0 / std::sqrt(mean / 512.0 + 1e-6);
  for (int d : {0, 128, 320, 447}) {  // the NoPE dims (the RoPE 64 are rotated)
    const double normed = pooled[d] * rsqrt;  // rms_w is bf16 1.0
    const double want = bf16_to_d64(float_to_bf16_bits(static_cast<float>(normed)));
    const double got = lat[d];
    const double mag = std::max({std::fabs(want), std::fabs(got), 1e-3});
    if (std::fabs(want - got) > mag / 8.0)
      throw std::runtime_error("compress: the NoPE dequant diverged from the independent normed value at d=" +
                               std::to_string(d) + " (want " + std::to_string(want) + ", got " + std::to_string(got) + ")");
  }
}

DGPP_TEST(dsv4_indexer_q_fused_fold) {
  // The 64-term weight fold + the per-row fp8 quant: the q-scale is the
  // row's amax (the 1e-4 floor) / 448, and the dequant(code) * scale
  // reproduces the bf16-rounded fold within one e4m3 quantum (the
  // quant's documented budget). Cross-check the fold with an INDEPENDENT
  // (head-major, not dim-major) summation.
  const int n_rows = 2;
  std::vector<uint16_t> q(static_cast<size_t>(n_rows) * 64 * 128), weights(static_cast<size_t>(n_rows) * 64);
  std::vector<double> cos(static_cast<size_t>(n_rows) * 32), sin(static_cast<size_t>(n_rows) * 32);
  for (auto& w : q) w = static_cast<uint16_t>(0x3F00 + (w >> 8));  // a deterministic bf16
  for (auto& w : weights) w = 0x3F80;  // bf16 1.0
  for (int r = 0; r < n_rows; ++r)
    for (int j = 0; j < 32; ++j) {
      const double ang = 0.05 * (j + 1) + 0.1 * r;
      cos[static_cast<size_t>(r) * 32 + j] = std::cos(ang);
      sin[static_cast<size_t>(r) * 32 + j] = std::sin(ang);
    }
  const auto qo = indexer_q_fused(q, cos, sin, weights, std::pow(128.0, -0.5), std::pow(64.0, -0.5));
  for (int r = 0; r < n_rows; ++r) {
    if (qo.q_scale[r] < 1e-4 / 448.0 - 1e-30)
      throw std::runtime_error("indexer q: the row scale is below the 1e-4 floor / 448");
    // The dequant of a sample dim must reproduce the INDEPENDENT fold:
    // the fold at dim d is the 64-HEAD sum (one term per head), NOT a
    // sum over dims. The fp32 rounding point is the documented budget,
    // so a loose e4m3-quantum bound (the 3 mantissa bits -> /8).
    for (int d : {0, 64, 65, 127}) {  // a NoPE dim + the rope block's edges
      double fold = 0.0;
      for (int h = 0; h < 64; ++h) {
        double x = bf16_to_d64(q[(static_cast<size_t>(r) * 64 + h) * 128 + d]);
        double xr = x;  // the NoPE dims (d < 64) pass through unrotated
        if (d >= 64) {
          const int j = (d - 64) / 2;
          const int partner = d + ((d & 1) == 0 ? 1 : -1);
          const double p = bf16_to_d64(q[(static_cast<size_t>(r) * 64 + h) * 128 + partner]);
          const double c = cos[static_cast<size_t>(r) * 32 + j], s = sin[static_cast<size_t>(r) * 32 + j];
          xr = ((d & 1) == 0) ? (x * c - p * s) : (x * c + p * s);
        }
        fold += bf16_to_d64(weights[static_cast<size_t>(r) * 64 + h]) * xr;
      }
      const double sf = std::pow(128.0, -0.5) * std::pow(64.0, -0.5);
      const double want = bf16_to_d64(float_to_bf16_bits(static_cast<float>(sf * fold)));
      const double got = e4m3_value(qo.q_fp8[static_cast<size_t>(r) * 128 + d]) * static_cast<double>(qo.q_scale[r]);
      const double mag = std::max({std::fabs(want), std::fabs(got), 1e-3});
      if (std::fabs(want - got) > mag / 8.0)
        throw std::runtime_error("indexer q: the fold dequant diverged from the independent recompute at row " +
                                 std::to_string(r) + " dim " + std::to_string(d) + " (want " + std::to_string(want) +
                                 ", got " + std::to_string(got) + ")");
    }
  }
}

DGPP_TEST(dsv4_indexer_topk_tiebreak_and_causal) {
  // The topk selection: (score desc, index asc) tie-break + the causal
  // bound (only the first valid_lens are eligible) + the -1 padding.
  // A contrived cache whose entries' e4m3 codes make THREE entries tie
  // on the score (the lower indices must win in order) and a valid_len
  // that excludes the single highest-scoring stale entry (the causal
  // exclusion).
  const int topk = 4, n_comp = 6;
  // q_fp8 / q_scale: a single row. Make q a unit vector in dim 0 (e4m3
  // code for 1.0 = 0x38, scale 1.0) so the score is proportional to the
  // cache's dim-0 code.
  IndexerQOut qo;
  qo.q_fp8.assign(128, 0x00);
  qo.q_fp8[0] = 0x38;  // e4m3 1.0
  qo.q_scale = {1.0f};
  // The cache: 132 B entries. dim-0 codes chosen so entries 1 and 4 TIE
  // (same code) and entry 5 (past valid_len) is the highest (excluded by
  // the causal bound). The fp32 scale is 1.0 for every entry.
  std::vector<uint8_t> cache(static_cast<size_t>(n_comp) * 132, 0);
  const uint8_t dim0[] = {2, 5, 1, 5, 5, 9};  // entry 5 (index 5) highest
  for (int j = 0; j < n_comp; ++j) {
    cache[static_cast<size_t>(j) * 132 + 0] = dim0[j];
    float one = 1.0f;
    std::memcpy(cache.data() + static_cast<size_t>(j) * 132 + 128, &one, 4);
  }
  // valid_len = 5: entries 0..4 are eligible (entry 5 is stale). The
  // scores: entry 1 and 4 tie (code 5), entry 0 (code 2), entry 2 (code
  // 1). The top-4 in (score desc, index asc): [1, 4, 0, 2] (the tie 1/4
  // broken to index 1 first; entry 5 excluded by the causal bound).
  const auto got = indexer_topk(qo, cache, 0, n_comp, topk, /*valid_len=*/5);
  // The scores: entries 1, 3, 4 TIE (code 5), entry 0 (code 2), entry 2
  // (code 1); entry 5 (code 9, the highest) is EXCLUDED by the causal
  // bound (valid_len 5). The top-4 in (score desc, index asc): the tie
  // 1/3/4 broken to ascending index, then entry 0 -> [1, 3, 4, 0].
  const int want[4] = {1, 3, 4, 0};
  for (int i = 0; i < topk; ++i)
    if (got[i] != want[i])
      throw std::runtime_error("indexer topk: selection [" + std::to_string(got[0]) + "," + std::to_string(got[1]) + "," +
                               std::to_string(got[2]) + "," + std::to_string(got[3]) + "] != the (score desc, index asc) causal top-4 "
                                     "[1,3,4,0]");
  // The -1 padding: a topk larger than the valid_len pads with -1.
  const auto padded = indexer_topk(qo, cache, 0, n_comp, /*topk=*/6, /*valid_len=*/2);
  if (padded[2] != -1 || padded[5] != -1)
    throw std::runtime_error("indexer topk: the -1 padding is missing past the valid_len");
}

DGPP_TEST(dsv4_sparse_attn_sink_exactly_once) {
  // The D25 sink's exactly-once + the degenerate limits: a single KV
  // latent (n_tokens = 1) so the softmax is a known closed form. The
  // null sink and the -inf sink are bit-identical (the no-op), the +inf
  // sink is the EXACTLY-zero output, and a finite sink adds exactly ONE
  // exp(sink - m') to the denominator (cross-checked against the closed
  // form m' = max(mx, sink)).
  constexpr int kHeads = 64, kHeadDim = 512;
  std::vector<double> q(static_cast<size_t>(kHeads) * kHeadDim, 1.0);
  std::vector<double> latents(kHeadDim, 2.0);  // one token, latent 2.0
  const double softmax_scale = 0.04419417382415922;
  const double s = static_cast<double>(q[0]) * static_cast<double>(latents[0]) * kHeadDim * softmax_scale;  // head 0's logit
  // The null / -inf sink: the single-token softmax is 1.0 (the one token
  // is the whole mass), so the output is the latent (2.0) for every
  // dim of every head.
  const auto out_null = sparse_attn_ref(q.data(), latents.data(), 1, nullptr);
  for (int i = 0; i < kHeads * kHeadDim; ++i)
    if (out_null[i] != 2.0)
      throw std::runtime_error("sparse attn: the null-sink single-token output is not the latent (got " +
                               std::to_string(out_null[i]) + ")");
  float neg_inf[kHeads];
  for (int h = 0; h < kHeads; ++h) neg_inf[h] = -std::numeric_limits<float>::infinity();
  const auto out_neginf = sparse_attn_ref(q.data(), latents.data(), 1, neg_inf);
  for (int i = 0; i < kHeads * kHeadDim; ++i)
    if (out_neginf[i] != out_null[i])
      throw std::runtime_error("sparse attn: the -inf sink is not bit-identical to the null sink (the no-op)");
  // The +inf sink: the EXACTLY-zero output (the degenerate limit).
  float pos_inf[kHeads];
  for (int h = 0; h < kHeads; ++h) pos_inf[h] = std::numeric_limits<float>::infinity();
  const auto out_posinf = sparse_attn_ref(q.data(), latents.data(), 1, pos_inf);
  for (int i = 0; i < kHeads * kHeadDim; ++i)
    if (out_posinf[i] != 0.0)
      throw std::runtime_error("sparse attn: the +inf sink is not the EXACTLY-zero output");
  // A finite sink: the closed form m' = max(mx, sink). With one token,
  // mx = s. For sink < s: m' = s, denom = 1 + exp(sink - s), out =
  // latent / denom. Verify the exactly-once (the sink's mass enters the
  // denominator ONCE, not per-head-duplicated).
  const float sink = static_cast<float>(s - 1.0);  // sink < s
  float one_sink[kHeads];
  for (int h = 0; h < kHeads; ++h) one_sink[h] = sink;
  const auto out_sink = sparse_attn_ref(q.data(), latents.data(), 1, one_sink);
  const double denom = 1.0 + std::exp(static_cast<double>(sink) - s);
  for (int d = 0; d < kHeadDim; ++d)
    for (int h = 0; h < 4; ++h) {  // a sample of heads
      const double want = 2.0 / denom;
      if (std::fabs(out_sink[static_cast<size_t>(h) * kHeadDim + d] - want) > 1e-12)
        throw std::runtime_error("sparse attn: the finite sink's exactly-once denominator is wrong at head " +
                                 std::to_string(h) + " (want " + std::to_string(want) + ", got " +
                                 std::to_string(out_sink[static_cast<size_t>(h) * kHeadDim + d]) + ")");
    }
  (void)s;
}

// ---- the six decode attention partials' layout (the scratch's the
// (row, split)'s indexing's, the dsa_attn_partial / dsa_attn_listed's
// m_ws / l_ws / c_ws's, the csa2_attn_finish's merge's) -----------------
// The six partials (m_main_ / l_main_ / c_main_ + the m_win_ / l_win_ /
// c_win_) are sized by ws_slots (the max_decode_rows's x the decode_n_split's
// (row, split)'s pairs' PRODUCT's, NOT the max's — the 32x's under-
// allocation's the 2026-09-18 GPU window's OOB's regression) and indexed
// [r * n_split + s]. This pins the contract on a small synthetic shape (the
// CPU-qualifiable's: the scratch_bytes's the host's, the indexing's the
// arithmetic's) so the under-allocation's regression's caught here, not on
// the GPU.
DGPP_TEST(dsv4_csa2_partial_layout) {
  using dgpp::Dsv4Csa2Config;
  using dgpp::Dsv4Csa2Layer;
  Dsv4Csa2Config cfg;
  cfg.hidden = 256;
  cfg.q_lora = 128;
  cfg.o_lora = 128;
  cfg.num_heads = 8;
  cfg.o_groups = 2;
  cfg.index_heads = 64;  // the V4's 64 (the NEW fold width's the validate's pin's)
  cfg.index_topk = 8;
  cfg.window = 8;
  cfg.ring_slots = 32;  // >= window + 16
  cfg.block_tokens = 8;  // positive even
  cfg.eps = 1e-6f;
  cfg.tp = 2;  // local_heads 4 (power of two >= 4), local_groups 1
  Dsv4Csa2Config::validate(cfg);
  const int max_decode_rows = 4, decode_n_split = 4;
  const int lh = cfg.local_heads();  // 4
  // The six partials' ws_slots (the (row, split)'s pairs' the PRODUCT's —
  // the 32x's under-allocation's the regression's guard's).
  const size_t ws_slots = std::max<size_t>(max_decode_rows, 8) * std::max<size_t>(decode_n_split, 8);
  const size_t m_bytes = ws_slots * static_cast<size_t>(lh) * 4;
  const size_t c_bytes = ws_slots * static_cast<size_t>(lh) * static_cast<size_t>(dgpp::kCsa2Latent) * 4;
  const size_t six = 4 * m_bytes + 2 * c_bytes;  // m / l x the 2 sources + c x the 2 sources
  const size_t total = Dsv4Csa2Layer::scratch_bytes(cfg, /*max_tokens=*/16, /*max_cache_tokens=*/256, max_decode_rows,
                                                    decode_n_split, 64ull << 20);
  if (total < six)
    throw std::runtime_error("the six partials' layout under-allocated (the total's " + std::to_string(total) +
                              " < the six's " + std::to_string(six) + " — the 32x's under-allocation's regression's)");
  // The (row, split)'s indexing's [r * n_split + s]'s the max offset's fits
  // the region's (the rows's <= max_decode_rows's, the n_split's <=
  // decode_n_split's, so r * n_split < ws_slots's).
  const size_t max_split_idx = static_cast<size_t>(max_decode_rows) * static_cast<size_t>(decode_n_split);
  if (max_split_idx > ws_slots)
    throw std::runtime_error("the (row, split)'s indexing's the max offset's out of the ws_slots's bound's");
}

// ---- the 64-head select's output layout (the topk_out's [rows, select_k]'s
// the -1's padding's, the counts's, the (score desc, index asc)'s total
// order's) the host's dsv4_csa2_select_prefill's (the real code's, the CPU-
// qualifiable's — no CUDA initialization) ------------------------------
DGPP_TEST(dsv4_csa2_select_prefill_layout) {
  const int rows = 2, n_entries = 8, select_k = 4, heads = 64;
  const int64_t dot_stride = n_entries;
  // The dot's [rows * heads, dot_stride] (row r head h at r * heads + h).
  // The logit's entry j's = sum_h w_folded[r,h] * relu(dot) * k_scale[j]'s,
  // so with w_folded = 1's + k_scale = 1's, logit_j = heads * dot[r*heads+h, j]'s.
  // The dyadic values (x / 64's) are exact in fp32, so the ties' exact's
  // (the 2/7's tie's the index's asc's preserved's).
  std::vector<float> dot(static_cast<size_t>(rows) * heads * dot_stride, 0.0f);
  const float row0_logit[8] = {3, 1, 2, 1, 5, 0, 4, 2};  // entry 4's 5's the top's, the 2/7's tie's
  for (int j = 0; j < n_entries; ++j)
    for (int h = 0; h < heads; ++h)
      dot[static_cast<size_t>(h) * dot_stride + j] = row0_logit[j] / heads;
  const float row1_logit[2] = {2, 1};  // the visible's 2's (pos_sel 1's), the rest's -1's the padding's
  for (int j = 0; j < 2; ++j)
    for (int h = 0; h < heads; ++h)
      dot[static_cast<size_t>(heads + h) * dot_stride + j] = row1_logit[j] / heads;
  std::vector<float> w_folded(static_cast<size_t>(rows) * heads, 1.0f);
  std::vector<float> k_scale(n_entries, 1.0f);
  std::vector<int64_t> pos_sel = {n_entries - 1, 1};  // row 0's visible 8's, row 1's visible 2's
  std::vector<int32_t> topk_out(static_cast<size_t>(rows) * select_k, -77);
  std::vector<int32_t> counts(rows, -77);
  dgpp::dsv4_csa2_select_prefill(dot.data(), dot_stride, w_folded.data(), k_scale.data(), pos_sel.data(), rows,
                                 n_entries, select_k, topk_out.data(), counts.data(), nullptr);
  // Row 0: the (score desc, index asc)'s top-4's [4, 6, 0, 2]'s (the tie's
  // 2/7's the index's asc's the 2's first's), the counts's 4's.
  const int32_t want0[4] = {4, 6, 0, 2};
  for (int j = 0; j < select_k; ++j)
    if (topk_out[static_cast<size_t>(j)] != want0[j])
      throw std::runtime_error("the select's row 0's topk's [" + std::to_string(topk_out[0]) + "," +
                               std::to_string(topk_out[1]) + "," + std::to_string(topk_out[2]) + "," +
                               std::to_string(topk_out[3]) + "] != the (score desc, index asc)'s [4,6,0,2]");
  if (counts[0] != 4) throw std::runtime_error("the select's row 0's counts's != 4's");
  // Row 1: the visible's 2's < select_k's 4's, so the top-2's [0, 1]'s +
  // the -1's padding's [0, 1, -1, -1]'s, the counts's 2's.
  const int32_t want1[4] = {0, 1, -1, -1};
  for (int j = 0; j < select_k; ++j)
    if (topk_out[static_cast<size_t>(1) * select_k + j] != want1[j])
      throw std::runtime_error("the select's row 1's topk's != the (top-2's + the -1's padding's) [0,1,-1,-1]");
  if (counts[1] != 2) throw std::runtime_error("the select's row 1's counts's != 2's");
}

// CPU-qualifiable: the planar main + index cache geometry (the no-pool
// positional state, the dsv41 Csa2StatePool's planes' raw-pointer re-
// expression — the dsv4 seam S1b's): the allocation sizes (the
// Dsv4Csa2Layer's static's, the cache_tokens' a multiple of block_tokens'),
// the record stride (the kFp4Block main's 288 B/row's the self-describing's
// + the e4m3 index's 128 B/row's + the fp32 row-scale's 4 B/row's), the
// identity block table mapping (entry e at slot e's, one block per
// request's the block b is physical b's). The dsv4_csa2_oracle_test's the
// real layer code's (the CPU-qualifiable's, no CUDA initialization).
DGPP_TEST(dsv4_csa2_planar_cache_geometry) {
  using dgpp::Dsv4Csa2Layer;
  // The record stride (the G-cache-format's csa2's physical layout's):
  // the kFp4Block main's 288 B/row's (the 512 e2m1 nibbles' 2/byte's 256
  // + the 32 e4m3 block-scales' 32's, the self-describing's the scales'
  // inside the row's — NOT the DSpark's 584 B kFp8 pool's), the e4m3
  // index's 128 B/row's (the kCsa2IndexDim's), the fp32 row-scale's 4
  // B/row's (the per-row's, one scale per entry row's).
  const size_t main_row = dgpp::latent_row_bytes(dgpp::LatentFormat::kFp4Block, dgpp::kCsa2Latent);
  if (main_row != 288)
    throw std::runtime_error("the kFp4Block main row's 288 B (the 512 e2m1's 2/byte's 256 + the 32 e4m3's 32's), got " +
                             std::to_string(main_row));
  if (dgpp::kCsa2IndexDim != 128)
    throw std::runtime_error("the e4m3 index row's 128 B (the kCsa2IndexDim's), got " + std::to_string(dgpp::kCsa2IndexDim));
  const size_t scale_row = sizeof(float);
  if (scale_row != 4) throw std::runtime_error("the fp32 row-scale's 4 B, got " + std::to_string(scale_row));
  // The allocation sizes (the Dsv4Csa2Layer's static's, the cache_tokens'
  // a multiple of block_tokens' the 128-token block's): the C4A's (ratio
  // 4) the epb's 128/4 = 32 entries/128-token block's (the spec's §1.3's
  // epb's), the C128A's (ratio 128) the epb's 128/128 = 1's.
  const int block_tokens = 128;
  const int64_t cache_tokens = int64_t{128} * 256;  // 256 blocks, 32768 tokens
  for (const int ratio : {4, 128}) {
    const int64_t entries = Dsv4Csa2Layer::cache_entries(ratio, cache_tokens, block_tokens);
    // The (cache_tokens / block_tokens) blocks' each (block_tokens / ratio)
    // entries's = cache_tokens / ratio's.
    const int64_t want_entries = cache_tokens / ratio;
    if (entries != want_entries)
      throw std::runtime_error("the cache entries' (ratio " + std::to_string(ratio) + ")'s " + std::to_string(entries) +
                               " != cache_tokens / ratio's " + std::to_string(want_entries));
    const size_t main_bytes = Dsv4Csa2Layer::main_cache_bytes(ratio, cache_tokens, block_tokens);
    if (main_bytes != static_cast<size_t>(entries) * main_row)
      throw std::runtime_error("the main cache bytes' (ratio " + std::to_string(ratio) + ")'s " + std::to_string(main_bytes) +
                               " != entries * 288's " + std::to_string(static_cast<size_t>(entries) * main_row));
    const size_t index_bytes = Dsv4Csa2Layer::index_cache_bytes(ratio, cache_tokens, block_tokens);
    if (index_bytes != static_cast<size_t>(entries) * dgpp::kCsa2IndexDim)
      throw std::runtime_error("the index cache bytes' (ratio " + std::to_string(ratio) + ")'s " + std::to_string(index_bytes) +
                               " != entries * 128's " + std::to_string(static_cast<size_t>(entries) * dgpp::kCsa2IndexDim));
    const size_t scale_bytes = Dsv4Csa2Layer::index_scale_bytes(ratio, cache_tokens, block_tokens);
    if (scale_bytes != static_cast<size_t>(entries) * scale_row)
      throw std::runtime_error("the index scale bytes' (ratio " + std::to_string(ratio) + ")'s " + std::to_string(scale_bytes) +
                               " != entries * 4's " + std::to_string(static_cast<size_t>(entries) * scale_row));
    // The epb's (the entries per 128-token block's the spec's §1.3's):
    // the C4A's 32's, the C128A's 1's.
    if (ratio == 4 && entries != cache_tokens / 4)
      throw std::runtime_error("the C4A's entries' != cache_tokens / 4's");
    if (ratio == 128 && entries != cache_tokens / 128)
      throw std::runtime_error("the C128A's entries' != cache_tokens / 128's");
  }
  // The identity block table mapping (the no-pool positional state's): the
  // main block table's [max_decode_rows, max_blocks]'s the identity's (one
  // block per request's, block b is physical b's), so the dsa_attn_listed's
  // / the dsa_latent_append's block table's resolution's: block = e / epb,
  // off = e % epb, phys_block = block (the identity's), slot = phys_block
  // * epb + off = e's (entry e at slot e's).
  const int epb = 128 / 4;  // the C4A's 32 entries/128-token block's
  const int max_blocks = static_cast<int>(cache_tokens / block_tokens);  // the 256 blocks's
  const std::vector<int64_t> test_entries = {0, 1, 31, 32, 63, 64, 1000};
  for (const int64_t e : test_entries) {
    const int block = static_cast<int>(e / epb);
    const int off = static_cast<int>(e % epb);
    const int32_t phys_block = block;  // the identity block table's (block b is physical b's)
    const int64_t slot = int64_t(phys_block) * epb + off;  // the physical slot's
    if (slot != e)
      throw std::runtime_error("the identity block table's mapping's entry " + std::to_string(e) + " at slot " +
                               std::to_string(slot) + " != e's");
    if (block >= max_blocks)
      throw std::runtime_error("the entry's block's out of the max_blocks' bound's (e " + std::to_string(e) + ")");
  }
}

// ---- the per-head q re-normalization (the checkpoint's ad-hoc rescale, the
// G-q-renorm's) — the contract's the CPU's pin's (the device's kernel's
// csa2_q_renorm_bf16's the GPU's; the host's dsv4_q_renorm_scale's the
// shared's host's + device's the formula's the driven's, no CUDA
// initialization) -------------------------------------------------------
DGPP_TEST(dsv4_q_renorm_contract) {
  const int dim = dgpp::kCsa2Latent;  // 512 (the full head's: the 448's NoPE's + the 64's RoPE's)
  const float eps = 1e-6f;  // the layer's norm_eps's (the reference's args.norm_eps's)
  // A deterministic bf16 head (an O(1)'s magnitude's the LCG's).
  std::vector<uint16_t> q(static_cast<size_t>(dim));
  std::uint32_t s = 0x51AC;
  auto next = [&]() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  };
  double ss_d = 0.0;
  float ss_f32 = 0.0f;  // the kernel's documented fp32 interior's (the sequential's
                        // order's; the kernel's strided's + the block's reduction's
                        // differ by the fp32 epsilon's the documented budget's).
  for (int d = 0; d < dim; ++d) {
    const float v = static_cast<float>(next() % 256) / 128.0f - 1.0f;  // -1.0..~0.996
    q[static_cast<size_t>(d)] = float_to_bf16_bits(v);
    const double x = bf16_to_d64(q[static_cast<size_t>(d)]);
    ss_d += x * x;
    ss_f32 += bf16_bits_to_float(q[static_cast<size_t>(d)]) * bf16_bits_to_float(q[static_cast<size_t>(d)]);
  }
  // The independent DOUBLE reference (the reference's formula's the
  // `q *= rsqrt(mean(q^2) + eps)'s the per-head's the FULL's dim's mean's):
  // the scale's + the out's.
  const double scale_d = 1.0 / std::sqrt(ss_d / dim + eps);
  // The kernel's documented contract (the fp32 interior's the ONE's bf16
  // rounding's the end's): the scale's the shared's dsv4_q_renorm_scale's
  // (the host's + device's the same's the formula's), the out's the bf16's
  // the fp32's product's.
  const float scale_f32 = dgpp::dsv4_q_renorm_scale(ss_f32, dim, eps);
  for (int d = 0; d < dim; ++d) {
    const float v = bf16_bits_to_float(q[static_cast<size_t>(d)]);
    const uint16_t want_word = float_to_bf16_bits(v * scale_f32);
    // The double reference's out's bf16-rounded (the reference's ONE's
    // bf16 rounding's the documented's output's the class's).
    const double out_d = bf16_to_d64(q[static_cast<size_t>(d)]) * scale_d;
    const uint16_t want_d_word = float_to_bf16_bits(static_cast<float>(out_d));
    const double got = bf16_to_d64(want_word);
    const double want_d = bf16_to_d64(want_d_word);
    const double mag = std::max({std::fabs(want_d), std::fabs(got), 1e-30});
    // The fp32 interior's the double's the + the bf16 rounding's the
    // budget's: a few's the bf16's ulp's (the 2^-9's the mantissa's
    // the quantum's — the fp32's accumulation's order's the kernel's
    // strided's the + the block's reduction's the difference's the
    // documented's class's).
    if (std::fabs(got - want_d) > mag / 64.0)
      throw std::runtime_error("q renorm: the bf16 output diverged from the double reference at d=" +
                               std::to_string(d) + " (want " + std::to_string(want_d) + ", got " + std::to_string(got) +
                               ")");
  }
  // The fp32 scale's the double's within the fp32 budget's (the 512-term's
  // sum's + the rsqrt's the documented class's).
  {
    const double mag = std::max({scale_d, static_cast<double>(scale_f32), 1e-30});
    if (std::fabs(scale_f32 - scale_d) > mag * 1e-5)
      throw std::runtime_error("q renorm: the fp32 scale diverged from the double beyond the fp32 budget (want " +
                               std::to_string(scale_d) + ", got " + std::to_string(scale_f32) + ")");
  }
  // The FULL-head-dim's semantics' pin's (the mean's over all 512's, NOT
  // the RoPE's 64's tail's only — the reference's mean(-1)'s the
  // unflattened's head's): a head's the NoPE's 448's the 1.0's the RoPE's
  // 64's the 0.0's -> the mean's 448/512's the scale's rsqrt(448/512 + eps)'s
  // ≈ 1.069's (a RoPE-tail-only's the mean's 0.0's the scale's rsqrt(eps)'s
  // 1000's the would's the wrong's).
  {
    const double mean_nope = 448.0 / 512.0;
    const double want_scale = 1.0 / std::sqrt(mean_nope + eps);
    const float got_scale = dgpp::dsv4_q_renorm_scale(448.0f, dim, eps);
    if (std::fabs(got_scale - want_scale) > std::fabs(want_scale) * 1e-5)
      throw std::runtime_error("q renorm: the full-dim mean is not the 512-dim mean (want " +
                               std::to_string(want_scale) + ", got " + std::to_string(got_scale) + ")");
  }
  // The unit-RMS head's the no-op's (the bf16's rounding's): all 1.0's
  // the mean's 1.0's the scale's rsqrt(1 + eps)'s ≈ 0.9999995's the bf16's
  // 1.0's (the 0.9999995's within half the bf16's ulp's of 1.0's), so the
  // renorm's the bit-identical's.
  {
    const float scale = dgpp::dsv4_q_renorm_scale(static_cast<float>(dim), dim, eps);  // ss = 512 (the 512's ones's)
    const uint16_t in_w = float_to_bf16_bits(1.0f);
    const uint16_t out_w = float_to_bf16_bits(bf16_bits_to_float(in_w) * scale);
    if (out_w != in_w)
      throw std::runtime_error("q renorm: the unit-RMS head is not the bit-identical no-op (got " +
                               std::to_string(bf16_bits_to_float(out_w)) + ")");
  }
  // The zero head's edge (the padding's row's the zero's hidden's the
  // GEMM's zero's out's): the ss's 0.0's the scale's rsqrt(eps)'s
  // finite's the out's 0 * scale's 0.0's (the no NaN's).
  {
    const float scale = dgpp::dsv4_q_renorm_scale(0.0f, dim, eps);
    if (!(scale > 0.0f) || std::isinf(scale))
      throw std::runtime_error("q renorm: the zero head's scale is not finite positive (got " + std::to_string(scale) + ")");
    const float out = bf16_bits_to_float(float_to_bf16_bits(0.0f * scale));
    if (out != 0.0f)
      throw std::runtime_error("q renorm: the zero head's output is not 0.0");
  }
}

// ---- the decode select's tie-break (the dsv4_csa2_sortable_key's + the
// dsv4_csa2_select_insert's — the real's layer code's, the CPU's
// qualifiable's, no CUDA initialization) ------------------------------
// The (score desc, index asc)'s total order's: the EXACT's score's ties'
// the LOWER's entry index's (the dsv41's make_key's form's the
// (~sortable << 21) | idx's the MIN's top-k's, the pinned's oracle's the
// dsv4_indexer_topk_tiebreak_and_causal's the prefill's stable-sort's the
// match's). The driven's micro-case's the pre-fix's the (sortable << 21) |
// idx's MAX's top-k's the tie's the HIGHER's index's (this test's the
// pre-fix's FAIL's, the post-fix's PASS's).
DGPP_TEST(dsv4_csa2_select_decode_tiebreak) {
  // The kernel's selection's the host's re-expression's (the sentinel's
  // ~0's, the MIN-key's dsv4_csa2_select_insert's, the extraction's the
  // (score desc, index asc)'s the -1's padding's — the csa2_layer.cu's
  // dsv4_csa2_select_decode_kernel's the same's the form's).
  auto run_decode_select = [](const float* logits, int visible, int select_k) {
    std::vector<uint64_t> skeys(static_cast<size_t>(select_k), ~uint64_t(0));
    std::vector<int> sidx(static_cast<size_t>(select_k), -1);
    for (int e = 0; e < visible; ++e)
      dgpp::dsv4_csa2_select_insert(skeys.data(), sidx.data(), select_k,
                                    dgpp::dsv4_csa2_sortable_key(logits[e], e, 21), e);
    std::vector<int> out(static_cast<size_t>(select_k), -1);
    const int k = std::min(select_k, visible);
    for (int j = 0; j < select_k; ++j)
      out[static_cast<size_t>(j)] = (j < k && sidx[static_cast<size_t>(j)] >= 0) ? sidx[static_cast<size_t>(j)] : -1;
    return out;
  };
  const int select_k = 4;
  // Case 1 (the tie's -> the lower's index's): the 6's entries' logits' —
  // entry 5's the 5.0's the top's, entries 1/2/3's TIE's 3.0's (the
  // lower's index's must win in order's), entry 4's 2.0's, entry 0's
  // 1.0's. The (score desc, index asc)'s top-4's [5, 1, 2, 3] (the tie's
  // 1/2/3's to the ascending's index's) — the pre-fix's the [5, 3, 2, 1]'s
  // (the tie's the HIGHER's index's first's).
  const float logits_a[6] = {1.0f, 3.0f, 3.0f, 3.0f, 2.0f, 5.0f};
  const auto out_a = run_decode_select(logits_a, 6, select_k);
  const int want_a[4] = {5, 1, 2, 3};
  for (int j = 0; j < select_k; ++j)
    if (out_a[static_cast<size_t>(j)] != want_a[j])
      throw std::runtime_error("the decode select's tie's topk's [" + std::to_string(out_a[0]) + "," +
                               std::to_string(out_a[1]) + "," + std::to_string(out_a[2]) + "," +
                               std::to_string(out_a[3]) +
                               "] != the (score desc, index asc)'s [5,1,2,3] (the tie's 1/2/3's the lower's index's the first's)");
  // Case 2 (the strictly's different's scores's the score's the dominant's,
  // the index's the irrelevant's): the 4's entries' distinct's logits'
  // [0.5, 0.25, 0.75, 1.0]'s (the idx 0/1/2/3's) -> the (score desc)'s
  // [3, 2, 0, 1]'s (the 1.0's the idx 3's the top's, the 0.75's the idx
  // 2's, the 0.5's the idx 0's, the 0.25's the idx 1's) — the score's the
  // descending's the index's the irrelevant's (the no ties' the here's).
  const float logits_b[4] = {0.5f, 0.25f, 0.75f, 1.0f};
  const auto out_b = run_decode_select(logits_b, 4, select_k);
  const int want_b[4] = {3, 2, 0, 1};
  for (int j = 0; j < select_k; ++j)
    if (out_b[static_cast<size_t>(j)] != want_b[j])
      throw std::runtime_error("the decode select's score-dominance's topk's [" + std::to_string(out_b[0]) + "," +
                               std::to_string(out_b[1]) + "," + std::to_string(out_b[2]) + "," +
                               std::to_string(out_b[3]) +
                               "] != the (score desc)'s [3,2,0,1] (the strictly's different's scores's the index's the irrelevant's)");
  // Case 3 (the -1's padding's the visible's < select_k's): the 2's
  // entries' [1.0, 2.0]'s -> the [1, 0, -1, -1]'s (the top-2's + the
  // -1's padding's).
  const float logits_c[2] = {1.0f, 2.0f};
  const auto out_c = run_decode_select(logits_c, 2, select_k);
  const int want_c[4] = {1, 0, -1, -1};
  for (int j = 0; j < select_k; ++j)
    if (out_c[static_cast<size_t>(j)] != want_c[j])
      throw std::runtime_error("the decode select's padding's topk's [" + std::to_string(out_c[0]) + "," +
                               std::to_string(out_c[1]) + "," + std::to_string(out_c[2]) + "," +
                               std::to_string(out_c[3]) +
                               "] != the [1,0,-1,-1] (the visible's 2's < select_k's 4's the -1's padding's)");
}

int main() { return ::dgpp::test::run_all(); }
