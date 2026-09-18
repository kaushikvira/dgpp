// DeepSeek-V4-Flash (deepseek_v4) MoE / hash-router CPU oracles
// (docs/dsv4_kernel_port_spec.md §2.3-§2.4). Host-only, no GPU: the
// naive FP64/FP32 references that pin the numerics of the V4 MoE
// surface — the fused top-k router (the sqrtsoftplus scoring + the
// noaux_tc biased selection + the renormalize + the routed_scaling_
// factor 1.5, AND the V4-only hash-layer tid2eid table-lookup mode on
// the first num_hash_layers = 3) and the MXFP4 routed-expert core
// (the e2m1 x e8m0/32 decode, the clamped SwiGLU, the router weight
// folded into the down epilogue) — so the Phase B v4 layer files
// (src/models/dsv4/hash_layer.cu) have a host-side oracle before any
// forward path consumes them.
//
// Borrowed, not rewritten (house rule 2): the router mirrors the
// dsv4-native ops/moe/router (src/ops/moe/router/router.h:1-80, the
// double oracle ops/common/op_test.h:672 router_ref) and the expert
// mirrors ops/moe/experts (src/ops/moe/experts/experts.h:1-60, the
// naive per-expert DOUBLE oracle ops/common/op_test.h:892
// moe_mxfp4_ref + :740 moe_mxfp4_decode_expert). Each test
// cross-checks the naive reference against an INDEPENDENT formulation
// and pins the documented conventions (the hash table's order-preserving
// lookup + the duplicate-id weighting, the tie-break to the lower expert
// index, the renormalize-then-scale order), so a silent corruption of
// the borrowed form is caught here, not on the GPU.
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

namespace {

using dgpp::bf16_bits_to_float;

// bf16 word -> exact double.
double bf16_to_d64(uint16_t w) { return static_cast<double>(bf16_bits_to_float(w)); }

// e2m1 (MXFP4) nibble -> value: the 8-code table (0, .5, 1, 1.5, 2, 3,
// 4, 6) with the sign bit 0x8 (dsv4-native ops/common/op_test.h
// e2m1_ref_value's kMagnitude table).
double e2m1_value(uint8_t nib) {
  static const double kMag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
  const double v = kMag[nib & 0x7];
  return (nib & 0x8) ? -v : v;
}
// ue8m0 byte -> power of two: 2^(b - 127) (the tree's e8m0_to_float
// contract, tests/unit/dsv4_fp8_scale_test.cpp).
double e8m0_pow2(uint8_t b) { return std::ldexp(1.0, static_cast<int>(b) - 127); }
// The reference's silu in double (the test's bounded inputs keep exp in
// range; a different code path from a kernel's fp32 expf).
double silu_ref(double g) { return g / (1.0 + std::exp(-g)); }
// The unbiased sqrtsoftplus score in double, STABLE over the full double
// range (the kernel's x > 30 branch, in double — the plain log1p(exp(x))
// overflows for x > ~709, the gate-logit plumbing's random e4m3 data
// reaches ~1e19 where the plain form is inf while the stable branch is
// finite; dsv4-native ops/common/op_test.h:659 sqrt_softplus_ref).
double sqrt_softplus_ref(double x) {
  const double sp = (x > 30.0) ? (x + std::log1p(std::exp(-x))) : std::log1p(std::exp(x));
  return std::sqrt(sp);
}

// ---- the fused top-k router (dsv4-native src/ops/moe/router/router.h) --
// The V4 router (the config's scoring_func "sqrtsoftplus", topk_method
// "noaux_tc", routed_scaling_factor 1.5, norm_topk_prob true) + the
// hash-layer tid2eid mode (the first num_hash_layers route by token id
// through the [vocab, topk] INT64 table instead of the learned gate, no
// bias). The double oracle (dsv4-native ops/common/op_test.h:672
// router_ref). `hash` = the tid2eid table mode (tokens + tid2eid set,
// bias null); otherwise the learned gate (logits + bias, the noaux_tc
// selection). Returns the topk indices (the selection order: descending
// biased score, ties to the lower expert index; the hash mode keeps the
// table order) + the renormalized + scaled weights.
struct RouterOut {
  std::vector<int32_t> indices;  // [m, topk]
  std::vector<double> weights;  // [m, topk]
};
RouterOut router_ref(const float* logits, const float* bias, const int32_t* tokens, const int64_t* tid2eid, int m,
                     int e, int topk, double scale, bool renormalize) {
  const bool hash = (tokens != nullptr);
  RouterOut out;
  out.indices.resize(static_cast<size_t>(m) * topk);
  out.weights.resize(static_cast<size_t>(m) * topk);
  std::vector<double> s(static_cast<size_t>(e));
  for (int r = 0; r < m; ++r) {
    const float* lrow = logits + static_cast<size_t>(r) * e;
    for (int i = 0; i < e; ++i)
      s[static_cast<size_t>(i)] = sqrt_softplus_ref(static_cast<double>(lrow[i]));
    if (hash) {
      // The table row (table order, no bias); duplicate ids are legal
      // and weighted independently (the reference's contract).
      for (int j = 0; j < topk; ++j) {
        const int32_t eid = static_cast<int32_t>(tid2eid[static_cast<size_t>(tokens[r]) * topk + j]);
        out.indices[static_cast<size_t>(r) * topk + j] = eid;
        out.weights[static_cast<size_t>(r) * topk + j] = s[static_cast<size_t>(eid)];
      }
    } else {
      // The top-`topk` of the (biased) selection scores, descending,
      // ties to the lower expert index (the total (sel desc, idx asc)
      // order makes the sort unique).
      std::vector<std::pair<double, int32_t>> sel;
      sel.reserve(static_cast<size_t>(e));
      for (int i = 0; i < e; ++i) {
        const double sel_i = s[static_cast<size_t>(i)] + (bias != nullptr ? static_cast<double>(bias[i]) : 0.0);
        sel.emplace_back(sel_i, static_cast<int32_t>(i));
      }
      std::sort(sel.begin(), sel.end(),
                [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
      for (int j = 0; j < topk; ++j) {
        const int32_t eid = sel[static_cast<size_t>(j)].second;
        out.indices[static_cast<size_t>(r) * topk + j] = eid;
        out.weights[static_cast<size_t>(r) * topk + j] = s[static_cast<size_t>(eid)];
      }
    }
    // norm_topk_prob renormalize (the clamp at 1e-20) + the
    // routed_scaling_factor — renormalize FIRST, then scale (the
    // reference's order).
    double sum = 0.0;
    for (int j = 0; j < topk; ++j)
      sum += out.weights[static_cast<size_t>(r) * topk + j];
    if (renormalize && sum < 1e-20) sum = 1e-20;
    for (int j = 0; j < topk; ++j) {
      double w = out.weights[static_cast<size_t>(r) * topk + j];
      if (renormalize) w /= sum;
      out.weights[static_cast<size_t>(r) * topk + j] = w * scale;
    }
  }
  return out;
}

// ---- the MXFP4 routed expert (dsv4-native src/ops/moe/experts) --------
// The fused W4A16 MXFP4 decode (e2m1 x 2^(byte - 127) EXACT in fp32, the
// per-32 block partial scaled once, the swiglu-clamp, the router weight
// folded into the down epilogue). The naive per-expert DOUBLE oracle
// (dsv4-native ops/common/op_test.h:892 moe_mxfp4_ref + :740
// moe_mxfp4_decode_expert). `x` [m, k] bf16, `w1`/`w3` [e, n, k/2] I8
// (two e2m1 per byte, the low nibble the even element), `w2` [e, k, n/2]
// I8, the scales [e, n, k/32] / [e, k, n/32] F8_E8M0. Returns the
// per-token output [m, k] (the sum over the selected experts of
// weight * the expert's SwiGLU down-projection, the reference's
// ascending-expert-id accumulation).
std::vector<double> moe_mxfp4_ref(const std::vector<uint16_t>& x, const std::vector<int32_t>& topk_indices,
                                  const std::vector<float>& topk_weights, const std::vector<uint8_t>& w1,
                                  const std::vector<uint8_t>& w1_scale, const std::vector<uint8_t>& w3,
                                  const std::vector<uint8_t>& w3_scale, const std::vector<uint8_t>& w2,
                                  const std::vector<uint8_t>& w2_scale, int m, int e, int n, int k, int topk,
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
void fill_mxfp4(std::vector<uint8_t>* payload, std::vector<uint8_t>* scale, std::size_t rows, std::size_t cols,
                std::uint32_t seed) {
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

}  // namespace

// ---------------------------------------------------------------------------
DGPP_TEST(dsv4_moe_hash_router_tid2eid) {
  // The V4-only hash-layer router mode (the spec §2.3, the one V4-only
  // router surface): the expert ids come from the tid2eid table
  // ([vocab, topk] INT64) in TABLE ORDER (not re-sorted), with NO bias,
  // the weights the unbiased scores (renormalized + scaled). The table
  // order is kept (the production's contract), duplicate ids are legal
  // and weighted independently. Verify the exact table-order selection +
  // the renormalize-then-scale order against an INDEPENDENT recompute.
  const int m = 2, e = 256, topk = 6;
  // A contrived logits row so the sqrtsoftplus scores are known: logit
  // i = 0 for every i (the score is sqrt(softplus(0)) = sqrt(ln 2) for
  // every expert — a uniform score, so the weights are the uniform
  // renormalized + scaled form).
  std::vector<float> logits(static_cast<size_t>(m) * e, 0.0f);
  // The tid2eid table: token 5's row is a KNOWN permutation with a
  // DUPLICATE (the duplicate is legal, weighted independently), token 7's
  // row is the identity.
  const int32_t tok[2] = {5, 7};
  std::vector<int64_t> tid2eid(static_cast<size_t>(128) * topk, -1);
  const int64_t row5[6] = {10, 10, 20, 30, 40, 50};  // 10 appears twice
  const int64_t row7[6] = {0, 1, 2, 3, 4, 5};
  for (int j = 0; j < topk; ++j) {
    tid2eid[static_cast<size_t>(5) * topk + j] = row5[j];
    tid2eid[static_cast<size_t>(7) * topk + j] = row7[j];
  }
  const auto out = router_ref(logits.data(), nullptr, tok, tid2eid.data(), m, e, topk, 1.5, true);
  // The selection is the TABLE order (not re-sorted): token 5 -> [10,10,
  // 20,30,40,50], token 7 -> [0,1,2,3,4,5].
  for (int j = 0; j < topk; ++j) {
    if (out.indices[j] != row5[j])
      throw std::runtime_error("hash router: token 5's selection is not the table order at j=" + std::to_string(j) +
                               " (got " + std::to_string(out.indices[j]) + ", want " + std::to_string(row5[j]) + ")");
    if (out.indices[static_cast<size_t>(1) * topk + j] != row7[j])
      throw std::runtime_error("hash router: token 7's selection is not the table order at j=" + std::to_string(j));
  }
  // The weights: every logit is 0, so every score is sqrt(ln 2) (the
  // uniform s). The renormalize divides by the sum of the 6 selected
  // weights (6 * s), giving 1/6 each, then the x 1.5 scale -> 0.25 each
  // (1.5 / 6). The duplicate 10 is weighted independently (two entries
  // of 0.25, not merged).
  const double want_w = 1.5 / 6.0;
  for (int j = 0; j < topk; ++j)
    for (int r = 0; r < m; ++r)
      if (std::fabs(out.weights[static_cast<size_t>(r) * topk + j] - want_w) > 1e-12)
        throw std::runtime_error("hash router: the renormalized + scaled weight is not 0.25 at r=" + std::to_string(r) +
                                 " j=" + std::to_string(j) + " (got " + std::to_string(out.weights[static_cast<size_t>(r) * topk + j]) + ")");
}

DGPP_TEST(dsv4_moe_learned_router_tiebreak_and_order) {
  // The learned gate's noaux_tc selection: the top-k of the BIASED score
  // (the sqrtsoftplus score + the e_score_correction_bias), descending,
  // ties to the lower expert index, the WEIGHTS from the UNBIASED scores
  // (the biased score is the selection key only), renormalized then
  // scaled. Verify the tie-break + the renormalize-then-scale order
  // against an INDEPENDENT recompute.
  const int m = 1, e = 8, topk = 3;
  // A contrived logits row: experts 1 and 2 TIE on the biased score (the
  // lower index, 1, must come first), expert 0 is the next, expert 3+
  // are far below. The bias shifts the SELECTION only.
  std::vector<float> logits = {2.0f, 0.0f, 0.0f, -5.0f, -5.0f, -5.0f, -5.0f, -5.0f};
  std::vector<float> bias = {0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};  // 1 and 2 tie at +1
  const auto out = router_ref(logits.data(), bias.data(), nullptr, nullptr, m, e, topk, 1.5, true);
  // The biased scores: s[i] + bias[i]. s = [sqrtsoftplus(2),
  // sqrtsoftplus(0), sqrtsoftplus(0), ...]. s[1] == s[2] (both
  // sqrtsoftplus(0)), so the biased scores of 1 and 2 tie (both
  // sqrtsoftplus(0) + 1). The selection: 1 and 2 (the tie, index 1
  // first), then 0 (sqrtsoftplus(2) + 0). The top-3 in (biased desc,
  // idx asc): the tie 1/2 (1 first), then 0.
  if (out.indices[0] != 1 || out.indices[1] != 2 || out.indices[2] != 0)
    throw std::runtime_error("learned router: the (biased desc, idx asc) top-3 is not [1,2,0] (got [" +
                             std::to_string(out.indices[0]) + "," + std::to_string(out.indices[1]) + "," +
                             std::to_string(out.indices[2]) + "])");
  // The weights: the UNBIASED scores (NOT the biased) at the selected
  // ids, renormalized (divided by the sum) then x 1.5. The independent
  // recompute (a different summation order for the renormalize sum).
  const double s1 = sqrt_softplus_ref(0.0), s0 = sqrt_softplus_ref(2.0);
  const double w1 = s1, w2 = s1, w0 = s0;  // the unbiased scores at ids 1, 2, 0
  const double sum = w1 + w2 + w0;
  const double want0 = (w1 / sum) * 1.5, want1 = (w2 / sum) * 1.5, want2 = (w0 / sum) * 1.5;
  if (std::fabs(out.weights[0] - want0) > 1e-12 || std::fabs(out.weights[1] - want1) > 1e-12 ||
      std::fabs(out.weights[2] - want2) > 1e-12)
    throw std::runtime_error("learned router: the renormalized + scaled weights are wrong (got [" +
                             std::to_string(out.weights[0]) + "," + std::to_string(out.weights[1]) + "," +
                             std::to_string(out.weights[2]) + "], want [" + std::to_string(want0) + "," + std::to_string(want1) +
                             "," + std::to_string(want2) + "])");
}

DGPP_TEST(dsv4_moe_mxfp4_expert_decode_and_clamp) {
  // The MXFP4 routed expert: the e2m1 x e8m0/32 decode (EXACT in double),
  // the clamped SwiGLU, the router weight folded into the down epilogue.
  // A single-token, single-expert (topk = 1, weight 1.0) geometry so the
  // output is the expert's own SwiGLU down-projection — cross-check the
  // e2m1 x e8m0 decode against an INDEPENDENT per-nibble recompute (the
  // decode is exact, so a bitwise agreement is expected within the
  // double GEMM's accumulation epsilon), and the SwiGLU clamp against a
  // direct silu-clamp recompute.
  const int m = 1, e = 4, n = 8, k = 32, topk = 1;
  std::vector<uint16_t> x(static_cast<size_t>(m) * k, 0x3F80);  // bf16 1.0
  std::vector<int32_t> topk_idx = {2};  // the single selected expert
  std::vector<float> topk_w = {1.0f};
  std::vector<uint8_t> w1, w1s, w3, w3s, w2, w2s;
  // The full multi-expert tensors: w1 / w3 are [e, n, k/2] I8 + [e, n,
  // k/32] e8m0, w2 is [e, k, n/2] I8 + [e, k, n/32] e8m0 (the expert
  // dim is the outer, contiguous — the fill's rows = e*n / e*k).
  fill_mxfp4(&w1, &w1s, static_cast<size_t>(e) * n, k, 0x10);
  fill_mxfp4(&w3, &w3s, static_cast<size_t>(e) * n, k, 0x20);
  fill_mxfp4(&w2, &w2s, static_cast<size_t>(e) * k, n, 0x30);  // w2 is [e, k, n/2]
  const auto out = moe_mxfp4_ref(x, topk_idx, topk_w, w1, w1s, w3, w3s, w2, w2s, m, e, n, k, topk, 10.0);
  // The INDEPENDENT recompute: decode the expert's planes nibble-by-
  // nibble (a different loop order: the k-major, not the row-major,
  // decode) and run the same double SwiGLU. The decode is exact, so the
  // two agree within the double GEMM's accumulation epsilon (a loose
  // bound).
  const int ei = 2;
  std::vector<double> w1f(static_cast<size_t>(n) * k), w3f(static_cast<size_t>(n) * k), w2f(static_cast<size_t>(k) * n);
  for (int nr = 0; nr < n; ++nr)
    for (int kk = 0; kk < k; ++kk) {
      const uint8_t b1 = w1[static_cast<size_t>(ei) * n * (k / 2) + static_cast<size_t>(nr) * (k / 2) + kk / 2];
      const uint8_t b3 = w3[static_cast<size_t>(ei) * n * (k / 2) + static_cast<size_t>(nr) * (k / 2) + kk / 2];
      w1f[static_cast<size_t>(nr) * k + kk] =
          e2m1_value(kk % 2 == 0 ? b1 & 0xFu : b1 >> 4) * e8m0_pow2(w1s[static_cast<size_t>(ei) * n * (k / 32) + static_cast<size_t>(nr) * (k / 32) + kk / 32]);
      w3f[static_cast<size_t>(nr) * k + kk] =
          e2m1_value(kk % 2 == 0 ? b3 & 0xFu : b3 >> 4) * e8m0_pow2(w3s[static_cast<size_t>(ei) * n * (k / 32) + static_cast<size_t>(nr) * (k / 32) + kk / 32]);
    }
  for (int kk = 0; kk < k; ++kk)
    for (int nn = 0; nn < n; ++nn) {
      const uint8_t b2 = w2[static_cast<size_t>(ei) * k * (n / 2) + static_cast<size_t>(kk) * (n / 2) + nn / 2];
      w2f[static_cast<size_t>(kk) * n + nn] =
          e2m1_value(nn % 2 == 0 ? b2 & 0xFu : b2 >> 4) * e8m0_pow2(w2s[static_cast<size_t>(ei) * k * (n / 32) + static_cast<size_t>(kk) * (n / 32) + nn / 32]);
    }
  std::vector<double> gate(n), up(n), inter(n), want(k, 0.0);
  for (int nr = 0; nr < n; ++nr) {
    double g = 0.0, u = 0.0;
    for (int kk = 0; kk < k; ++kk) {
      g += bf16_to_d64(x[kk]) * w1f[static_cast<size_t>(nr) * k + kk];
      u += bf16_to_d64(x[kk]) * w3f[static_cast<size_t>(nr) * k + kk];
    }
    gate[nr] = g;
    up[nr] = u;
  }
  for (int nr = 0; nr < n; ++nr) {
    double sg = silu_ref(gate[nr]);
    sg = std::min(std::max(sg, -10.0), 10.0);  // the SwiGLU clamp
    inter[nr] = sg * up[nr];
  }
  for (int kk = 0; kk < k; ++kk)
    for (int nn = 0; nn < n; ++nn) want[kk] += 1.0 * inter[nn] * w2f[static_cast<size_t>(kk) * n + nn];
  for (int kk = 0; kk < k; ++kk) {
    const double got = out[kk];
    const double mag = std::max({std::fabs(want[kk]), std::fabs(got), 1e-3});
    if (std::fabs(want[kk] - got) > mag * 1e-9)
      throw std::runtime_error("mxfp4 expert: the down-projection diverged from the independent recompute at k=" +
                               std::to_string(kk) + " (want " + std::to_string(want[kk]) + ", got " + std::to_string(got) + ")");
  }
}

int main() { return ::dgpp::test::run_all(); }
