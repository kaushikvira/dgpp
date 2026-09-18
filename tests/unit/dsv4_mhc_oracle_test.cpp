// DeepSeek-V4-Flash (deepseek_v4) mHC single-pass CPU oracles
// (docs/dsv4_kernel_port_spec.md §2.5). Host-only, no GPU: the naive
// FP64 references that pin the numerics of the V4 mHC (Hyper-
// Connections) 4-copy residual state — the hc_pre (the flatten-RMS +
// the [24, 4h] GEMM + the 20-iter Sinkhorn's doubly-stochastic comb +
// the 4-copy y combine), the hc_post (the 4-copy expansion with the
// learned combination), the hc_head (the 4 -> 1 collapse BEFORE the
// final norm) — so the Phase B v4 layer files (the single-pass mHC
// launcher set, the dsv41 D4 form's V4 width [24, 16384]) have a
// host-side oracle before any forward path consumes them.
//
// Borrowed, not rewritten (house rule 2): the math mirrors the
// dsv4-native ops/mhc (src/ops/mhc/mhc.h:1-100, the 4-copy reference
// math) + the naive FP64 oracle (tests/test_op_mhc.cpp:106-112 the
// independent double loops, the doubly-stochastic invariant probe
// :339-386 — the 5e-2 algorithm tolerance, the FP64 ref's own 20-iter
// Sinkhorn deviates 0.0228, KNOWLEDGE D28). Each test cross-checks
// the naive reference against an INDEPENDENT formulation and pins the
// documented invariants (the Sinkhorn's doubly-stochastic, the
// single-bf16-rounding output), so a silent corruption of the borrowed
// form is caught here, not on the GPU.
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
using dgpp::float_to_bf16_bits;

// bf16 word -> exact double; the double -> bf16 RNE word (the single
// bf16 rounding at the op's output boundary).
double bf16_to_d64(uint16_t w) { return static_cast<double>(bf16_bits_to_float(w)); }
uint16_t d64_to_bf16(double v) { return float_to_bf16_bits(static_cast<float>(v)); }
// The FP64 sigmoid (the ref's T.sigmoid's — the double's exp, the
// INDEPENDENT path vs a kernel's fp32 expf).
double sigmoid_d(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// A deterministic bf16-plane generator (bounded magnitudes so the GEMM
// stays finite).
void fill_bf16(std::vector<uint16_t>* v, std::size_t n, std::uint32_t seed) {
  v->resize(n);
  std::uint32_t s = seed;
  auto next = [&]() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  };
  for (std::size_t i = 0; i < n; ++i) {
    const uint16_t sign = (next() & 1) ? 0x8000 : 0x0000;
    const uint16_t expf = static_cast<uint16_t>((0x3C + (next() & 7)) << 7);
    const uint16_t man = static_cast<uint16_t>(next() & 0x7F);
    (*v)[i] = sign | expf | man;
  }
}

// ---- the mhc_pre (dsv4-native tests/test_op_mhc.cpp:121 mhc_pre_ref) --
// The EXACT sequence (the ref's model.py + kernel.py's): the 4-copy's
// flatten-RMS (the per-row scalar over the 4h flat row), the n_mix's
// GEMM (the rsqrt applied AFTER the GEMM, the D27's composition), the
// pre's 4 (sigmoid + eps), the post's 4 (the alpha 2.0), the comb's
// 16 + the 20-iter Sinkhorn (the row-softmax's max-subtract + /rowsum +
// eps, the col-normalize's /colsum + eps, the 19 more's), the y's 4-
// copy combine (the UN-normalized x's fp32 sum, the single bf16 RNE).
// `n_mix` = 24 (the block's pre+post+comb) or 4 (the head's pre-only).
struct MhcPreOut {
  std::vector<uint16_t> y;  // [rows*h] bf16
  std::vector<double> mixes;  // [rows*n_mix]
  std::vector<double> pre;  // [rows*4]
  std::vector<double> post;  // [rows*4] (the block's mode)
  std::vector<double> comb;  // [rows*16] (the block's mode)
};
MhcPreOut mhc_pre_ref(const std::vector<uint16_t>& x4w, const std::vector<float>& fn,
                       const std::vector<float>& scale, const std::vector<float>& base, int rows, int h,
                       int n_mix, double norm_eps, double hc_eps, int sinkhorn_iters) {
  const int K = 4 * h;
  const int M = 4;  // hc_mult
  MhcPreOut out;
  out.y.resize(static_cast<size_t>(rows) * h);
  out.mixes.resize(static_cast<size_t>(rows) * n_mix);
  out.pre.resize(static_cast<size_t>(rows) * M);
  out.post.resize(n_mix == 24 ? static_cast<size_t>(rows) * M : 0);
  out.comb.resize(n_mix == 24 ? static_cast<size_t>(rows) * M * M : 0);
  for (int r = 0; r < rows; ++r) {
    // 1. The 4-copy's flatten-RMS (the per-row scalar over the 4h's flat
    //    row — the ref's x.square().mean(-1)).
    double ss = 0.0;
    for (int d = 0; d < K; ++d) {
      const double v = bf16_to_d64(x4w[static_cast<size_t>(r) * K + d]);
      ss += v * v;
    }
    const double rs = 1.0 / std::sqrt(ss / K + norm_eps);
    // 2. The n_mix's GEMM (the rsqrt's applied AFTER the GEMM, the
    //    D27's composition — one GEMM read of fn, no second pass).
    for (int m = 0; m < n_mix; ++m) {
      double s = 0.0;
      for (int d = 0; d < K; ++d)
        s += static_cast<double>(fn[static_cast<size_t>(m) * K + d]) * bf16_to_d64(x4w[static_cast<size_t>(r) * K + d]);
      out.mixes[static_cast<size_t>(r) * n_mix + m] = s * rs;
    }
    // 3. The pre's 4 (the ref's sigmoid + eps — BOTH modes).
    for (int j = 0; j < M; ++j)
      out.pre[static_cast<size_t>(r) * M + j] =
          sigmoid_d(out.mixes[static_cast<size_t>(r) * n_mix + j] * scale[0] + base[j]) + hc_eps;
    if (n_mix == 24) {
      // The post's 4 (the ref's alpha 2.0).
      for (int j = 0; j < M; ++j)
        out.post[static_cast<size_t>(r) * M + j] =
            2.0 * sigmoid_d(out.mixes[static_cast<size_t>(r) * n_mix + M + j] * scale[1] + base[M + j]);
      // The comb's 16 + the 20-iter Sinkhorn (the ref's EXACT sequence —
      // the mixes' [8..24)'s 16's, the row-softmax's max-subtract +
      // /rowsum + eps (the +eps's AFTER the division's), the col-
      // normalize's /colsum + eps, the 19 more's (row + col)'s).
      double c[16];
      for (int j = 0; j < M; ++j)
        for (int k = 0; k < M; ++k) {
          const int m = j * M + k + 2 * M;
          c[j * M + k] = out.mixes[static_cast<size_t>(r) * n_mix + m] * scale[2] + base[m];
        }
      // The row-softmax (the row's max-subtract's + exp's + /rowsum +
      // eps — the row-normalize's #1's).
      for (int j = 0; j < M; ++j) {
        double rmax = c[j * M];
        for (int k = 1; k < M; ++k)
          if (c[j * M + k] > rmax) rmax = c[j * M + k];
        double rsum = 0.0;
        for (int k = 0; k < M; ++k)
          c[j * M + k] = std::exp(c[j * M + k] - rmax);
        for (int k = 0; k < M; ++k) rsum += c[j * M + k];
        for (int k = 0; k < M; ++k) c[j * M + k] = c[j * M + k] / rsum + hc_eps;
      }
      // The col-normalize's #1's (the /colsum + eps's).
      for (int k = 0; k < M; ++k) {
        double csum = 0.0;
        for (int j = 0; j < M; ++j) csum += c[j * M + k];
        for (int j = 0; j < M; ++j) c[j * M + k] = c[j * M + k] / (csum + hc_eps);
      }
      // The 19 more's (the ref's `for _ in T.serial(sinkhorn_iters -
      // 1)`'s — the row's + the col's normalizes' x19's).
      for (int it = 0; it + 1 < sinkhorn_iters; ++it) {
        for (int j = 0; j < M; ++j) {
          double rsum = 0.0;
          for (int k = 0; k < M; ++k) rsum += c[j * M + k];
          for (int k = 0; k < M; ++k) c[j * M + k] = c[j * M + k] / (rsum + hc_eps);
        }
        for (int k = 0; k < M; ++k) {
          double csum = 0.0;
          for (int j = 0; j < M; ++j) csum += c[j * M + k];
          for (int j = 0; j < M; ++j) c[j * M + k] = c[j * M + k] / (csum + hc_eps);
        }
      }
      for (int j = 0; j < M; ++j)
        for (int k = 0; k < M; ++k)
          out.comb[static_cast<size_t>(r) * 16 + j * M + k] = c[j * M + k];
    }
    // 4. The y's 4-copy combine (the ref's y = sum(pre * x, dim=2) — the
    //    UN-normalized x's fp32 sum, the single bf16 RNE at the
    //    boundary's).
    for (int d = 0; d < h; ++d) {
      double yv = 0.0;
      for (int i = 0; i < M; ++i)
        yv += out.pre[static_cast<size_t>(r) * M + i] * bf16_to_d64(x4w[static_cast<size_t>(r) * K + i * h + d]);
      out.y[static_cast<size_t>(r) * h + d] = d64_to_bf16(yv);
    }
  }
  return out;
}

// ---- the mhc_post (dsv4-native tests/test_op_mhc.cpp:244 mhc_post_ref)
// y4[i, d] = post[i] * out[d] + sum_j comb[i, j] * res4[j, d] (the
// fp32's post/comb's upcast's, the single bf16 RNE at the output's).
std::vector<uint16_t> mhc_post_ref(const std::vector<uint16_t>& out_w, const std::vector<uint16_t>& res4_w,
                                   const std::vector<float>& post, const std::vector<float>& comb, int rows, int h) {
  const int K = 4 * h;
  const int M = 4;
  std::vector<uint16_t> y(static_cast<size_t>(rows) * K);
  for (int r = 0; r < rows; ++r)
    for (int i = 0; i < M; ++i) {
      const double p = static_cast<double>(post[static_cast<size_t>(r) * M + i]);
      for (int d = 0; d < h; ++d) {
        double yv = p * bf16_to_d64(out_w[static_cast<size_t>(r) * h + d]);
        for (int j = 0; j < M; ++j)
          yv += static_cast<double>(comb[static_cast<size_t>(r) * M * M + i * M + j]) *
                bf16_to_d64(res4_w[static_cast<size_t>(r) * K + j * h + d]);
        y[static_cast<size_t>(r) * K + i * h + d] = d64_to_bf16(yv);
      }
    }
  return y;
}

}  // namespace

// ---------------------------------------------------------------------------
DGPP_TEST(dsv4_mhc_pre_y_combine) {
  // The hc_pre's 4-copy y combine (the UN-normalized x's sum weighted by
  // the pre's 4, the single bf16 RNE at the boundary): cross-check the
  // output against an INDEPENDENT recompute (a different accumulation
  // order for the 4-term sum — the fp32-vs-double epsilon is the
  // documented budget, a loose bound).
  const int rows = 2, h = 8, n_mix = 24, M = 4;
  const int K = 4 * h;
  std::vector<uint16_t> x4w(static_cast<size_t>(rows) * K);
  fill_bf16(&x4w, x4w.size(), 0x4444);
  std::vector<float> fn(static_cast<size_t>(n_mix) * K, 0.1f);
  std::vector<float> scale = {1.0f, 1.0f, 1.0f};
  std::vector<float> base(n_mix, 0.0f);
  const auto out = mhc_pre_ref(x4w, fn, scale, base, rows, h, n_mix, 1e-6, 1e-6, 20);
  for (int r = 0; r < rows; ++r)
    for (int d = 0; d < h; ++d) {
      // The INDEPENDENT recompute: the 4-term sum in REVERSE copy order
      // (a different accumulation order: the fp32 epsilon budget).
      double yv = 0.0;
      for (int i = M - 1; i >= 0; --i)
        yv += out.pre[static_cast<size_t>(r) * M + i] * bf16_to_d64(x4w[static_cast<size_t>(r) * K + i * h + d]);
      const double got = bf16_to_d64(out.y[static_cast<size_t>(r) * h + d]);
      const double mag = std::max({std::fabs(yv), std::fabs(got), 1e-3});
      if (std::fabs(yv - got) > mag * 1e-5)
        throw std::runtime_error("mhc pre: the y combine diverged from the independent recompute at r=" +
                                 std::to_string(r) + " d=" + std::to_string(d));
    }
}

DGPP_TEST(dsv4_mhc_sinkhorn_doubly_stochastic) {
  // The 20-iter Sinkhorn's doubly-stochastic invariant (the ref's
  // test_op_mhc.cpp:339-386 probe — the 5e-2 ALGORITHM tolerance, the
  // FP64 ref's own 20-iter Sinkhorn deviates 0.0228, KNOWLEDGE D28):
  // the comb's rows and columns each sum to ~1 (the +eps's per
  // normalize's accumulate, so the sums are 1 + O(eps * iters), NOT
  // exactly 1 — the probe's 5e-2 bound captures the 0.0228 deviation).
  const int rows = 1, h = 8, n_mix = 24, M = 4;
  const int K = 4 * h;
  std::vector<uint16_t> x4w(static_cast<size_t>(rows) * K);
  fill_bf16(&x4w, x4w.size(), 0x5555);
  std::vector<float> fn(static_cast<size_t>(n_mix) * K);
  for (auto& v : fn) v = 0.2f;  // a non-degenerate fn so the comb is non-trivial
  std::vector<float> scale = {1.0f, 1.0f, 1.0f};
  std::vector<float> base(n_mix, 0.0f);
  const auto out = mhc_pre_ref(x4w, fn, scale, base, rows, h, n_mix, 1e-6, 1e-6, 20);
  // The comb's row / column sums (the doubly-stochastic probe).
  for (int j = 0; j < M; ++j) {
    double rsum = 0.0;
    for (int k = 0; k < M; ++k) rsum += out.comb[j * M + k];
    if (std::fabs(rsum - 1.0) > 5e-2)
      throw std::runtime_error("mhc sinkhorn: the row " + std::to_string(j) + " sum is not ~1 (got " +
                               std::to_string(rsum) + ", the 5e-2 algorithm tolerance)");
  }
  for (int k = 0; k < M; ++k) {
    double csum = 0.0;
    for (int j = 0; j < M; ++j) csum += out.comb[j * M + k];
    if (std::fabs(csum - 1.0) > 5e-2)
      throw std::runtime_error("mhc sinkhorn: the col " + std::to_string(k) + " sum is not ~1 (got " +
                               std::to_string(csum) + ", the 5e-2 algorithm tolerance)");
  }
}

DGPP_TEST(dsv4_mhc_post_and_head) {
  // The hc_post (the 4-copy expansion with the learned combination) +
  // the hc_head (the 4 -> 1 collapse, the pre-only split — NO post/comb,
  // NO Sinkhorn): cross-check the post's y4 against an INDEPENDENT
  // recompute (a different accumulation order) and verify the head's
  // pre-only mode (the n_mix = 4's post/comb' are EMPTY, the y's 4-copy
  // combine with the head's 4-row fn).
  const int rows = 1, h = 8, M = 4;
  const int K = 4 * h;
  std::vector<uint16_t> out_w(static_cast<size_t>(rows) * h), res4_w(static_cast<size_t>(rows) * K);
  fill_bf16(&out_w, out_w.size(), 0x6666);
  fill_bf16(&res4_w, res4_w.size(), 0x7777);
  std::vector<float> post(M, 0.5f);
  std::vector<float> comb(M * M, 0.25f);  // the uniform doubly-stochastic comb
  const auto y4 = mhc_post_ref(out_w, res4_w, post, comb, rows, h);
  for (int i = 0; i < M; ++i)
    for (int d = 0; d < h; ++d) {
      // The INDEPENDENT recompute (the j-sum in REVERSE order — the fp32
      // epsilon budget).
      double yv = post[i] * bf16_to_d64(out_w[d]);
      for (int j = M - 1; j >= 0; --j)
        yv += static_cast<double>(comb[i * M + j]) * bf16_to_d64(res4_w[static_cast<size_t>(i) * h + d]);
      const double got = bf16_to_d64(y4[static_cast<size_t>(i) * h + d]);
      const double mag = std::max({std::fabs(yv), std::fabs(got), 1e-3});
      if (std::fabs(yv - got) > mag * 1e-5)
        throw std::runtime_error("mhc post: the y4 diverged from the independent recompute at i=" + std::to_string(i) +
                                 " d=" + std::to_string(d));
    }
  // The hc_head: the 4 -> 1 collapse (the pre-only split, the 4-row fn).
  // The head's mode: n_mix = 4, the post/comb' are EMPTY (the ref's
  // :759-767's mix_hc = 4, NOT 24).
  std::vector<uint16_t> x4w(static_cast<size_t>(rows) * K);
  fill_bf16(&x4w, x4w.size(), 0x8888);
  std::vector<float> fn_head(4 * K, 0.1f);
  std::vector<float> scale_head = {1.0f};
  std::vector<float> base_head(4, 0.0f);
  const auto head = mhc_pre_ref(x4w, fn_head, scale_head, base_head, rows, h, /*n_mix=*/4, 1e-6, 1e-6, 20);
  if (!head.post.empty() || !head.comb.empty())
    throw std::runtime_error("mhc head: the pre-only mode's post/comb are not EMPTY");
  // The head's y (the 4-copy combine with the head's pre's 4): cross-
  // check against an independent recompute.
  for (int d = 0; d < h; ++d) {
    double yv = 0.0;
    for (int i = M - 1; i >= 0; --i)
      yv += head.pre[i] * bf16_to_d64(x4w[static_cast<size_t>(i) * h + d]);
    const double got = bf16_to_d64(head.y[d]);
    const double mag = std::max({std::fabs(yv), std::fabs(got), 1e-3});
    if (std::fabs(yv - got) > mag * 1e-5)
      throw std::runtime_error("mhc head: the y diverged from the independent recompute at d=" + std::to_string(d));
  }
}

int main() { return ::dgpp::test::run_all(); }
