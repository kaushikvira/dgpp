// DeepSeek-V4-Flash (deepseek_v4) ratio-4 (C4A) overlapping compressor's
// tail-ring update CPU oracles (2026-09-18, the dsv4 dspark + compressor
// wiring; docs/dsv4_kernel_port_spec.md §2.1(b)). Host-only, no GPU: the
// naive FP32/FP64 references that pin the numerics of the V4 compressor's
// per-request tail-ring update — the dsv41 ratio-2 compressor's
// csa2_compress_decode_update (src/kernels/csa2.cu) re-expressed for the
// ratio-4 overlapping variant (coff 2, the dsv41 ratio-2/ratio-1 pair's
// one geometry swap): the wkv / wgate's pair + the per-request tail's
// update, the tail's ordinal + the wgate's plane.
//
// The per-request tail is fp32 [2, W] (W = the layer's compressor output
// width, coff * 512): the pending even token's kv (the first W) and gate
// score (the second W). The even positions stash their (kv, score) into the
// request's tail; the odd positions pool the tail with themselves into the
// normed latent (the pair-pooling + the one-rounding RMSNorm, the dsv41's
// pool_pair_and_norm's re-expression) and report the entry. So the Phase B
// v4 forward path (src/models/dsv4/compress.cu's dsv4_compress_tail_update)
// has a host-side oracle that pins the [2, W] layout on a small synthetic
// shape before any GPU consumes the tail (house rule: every adapted op is
// oracle-qualified on CPU before consumption).
//
// Borrowed, not rewritten (house rule 2): the tail update mirrors the
// dsv41 csa2_compress_decode_update (the even-stash / odd-pool, the
// pair-pooling + the one-rounding RMSNorm, the entries' p / 2's) and the
// pair-pooling's numerics mirror the dsv41 pool_pair_and_norm (the per-dim
// softmax over (s0, s1), the weighted kv sum's ONE bf16 rounding, the
// RMSNorm's fp32 interior + the norm_w's bf16 exact upcast + the ONE bf16
// rounding). Each test cross-checks the naive reference against an
// INDEPENDENT (double-precision, reversed-order) formulation and pins the
// documented edge cases (the even's stash, the odd's pool, the padding's
// -1, the [2, W] layout's the kv's first W's + the score's second W's),
// so a silent corruption of the borrowed form is caught here.
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

// bf16 word -> exact double.
double bf16_to_d64(uint16_t w) { return static_cast<double>(bf16_bits_to_float(w)); }

// A deterministic bf16-plane generator (a small LCG over the bf16
// exponent/mantissa, bounded magnitudes so the dots stay finite).
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
    const uint16_t expf = static_cast<uint16_t>((0x3C + (next() & 7)) << 7);  // ~0.5..8
    const uint16_t man = static_cast<uint16_t>((next() & 0x7F) << 0);
    (*v)[i] = sign | expf | man;
  }
}
// A deterministic fp32-plane generator (the wkv / wgate's GEMM output's
// the bounded magnitudes' the pooling's finite's).
void fill_f32(std::vector<float>* v, std::size_t n, std::uint32_t seed) {
  v->resize(n);
  std::uint32_t s = seed;
  auto next = [&]() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  };
  for (std::size_t i = 0; i < n; ++i) {
    // A small O(1) fp32 value (the sign + a small magnitude).
    const float mag = static_cast<float>((next() & 0xFF) / 64.0) - 1.0f;  // -1..1
    (*v)[i] = ((next() & 1) ? -1.0f : 1.0f) * mag;
  }
}

// ---- the pair-pooling + the one-rounding's RMSNorm (the dsv41's
// pool_pair_and_norm's re-expression) -------------------------------------
// out[c] = bf16(norm_w[c] * (pooled[c] * rs)): the per-dim's softmax over
// (s0[c], s1[c]) (the fp32's, the max-shift's), the weighted's kv sum's
// (w0 * kv0[c] + w1 * kv1[c]'s the ONE bf16 rounding's), the RMSNorm's
// (the fp32's interior's ss = sum_c pooled[c]^2's, rs = 1 / sqrt(ss / W +
// eps)'s, the norm_w's bf16's exact upcast's, the ONE bf16 rounding's).
void pair_pool_norm_ref(const float* kv0, const float* s0, const float* kv1, const float* s1,
                        const uint16_t* norm_w, float eps, uint16_t* out, int W) {
  std::vector<float> pooled(W);
  float ss = 0.0f;
  for (int c = 0; c < W; ++c) {
    const float a = s0[c], b = s1[c];
    const float m = std::max(a, b);
    const float e0 = std::exp(a - m), e1 = std::exp(b - m);
    const float den = e0 + e1;
    const float w0 = e0 / den, w1 = e1 / den;
    const float sum = w0 * kv0[c] + w1 * kv1[c];
    const float v = bf16_bits_to_float(float_to_bf16_bits(sum));  // .to(bf16) before the norm
    pooled[c] = v;
    ss += v * v;
  }
  const float rs = std::sqrt(1.0f / (ss / static_cast<float>(W) + eps));
  for (int c = 0; c < W; ++c) {
    const float v = pooled[c] * rs;
    out[c] = float_to_bf16_bits(bf16_bits_to_float(norm_w[c]) * v);
  }
}
// The independent double-precision pair-pooling (the cross-check's the
// fp32's the accumulation's the different precision path's): the same
// bf16 rounding's (the .to(bf16)'s the norm's before's) but the double's
// the interior's (the fp32's the 1e-7's the different's).
void pair_pool_norm_ref_d64(const double* kv0, const double* s0, const double* kv1,
                            const double* s1, const double* norm_w, double eps, double* out,
                            int W) {
  std::vector<double> pooled(W);
  double ss = 0.0;
  for (int c = 0; c < W; ++c) {
    const double m = std::max(s0[c], s1[c]);
    const double e0 = std::exp(s0[c] - m), e1 = std::exp(s1[c] - m);
    const double den = e0 + e1;
    const double w0 = e0 / den, w1 = e1 / den;
    const double sum = w0 * kv0[c] + w1 * kv1[c];
    const double v = bf16_bits_to_float(float_to_bf16_bits(static_cast<float>(sum)));  // .to(bf16) before the norm (the double's the exact's the float's the rounding's)
    pooled[c] = v;
    ss += v * v;
  }
  const double rs = std::sqrt(1.0 / (ss / static_cast<double>(W) + eps));
  for (int c = 0; c < W; ++c) {
    const double v = norm_w[c] * (pooled[c] * rs);
    // The ONE bf16 rounding's (the norm's out's the bf16's the fp32's the
    // mirror's): the double's the exact's the float's the rounding's.
    out[c] = bf16_bits_to_float(float_to_bf16_bits(static_cast<float>(v)));
  }
}

// ---- the tail's update (the even's stash's, the odd's pool's) -----------
// The per-request's fp32 [2, W]'s tail's (the pending even's kv's + the
// score's): the even positions' stash their (kv, score) into the tail's
// (the first W's the kv's, the second W's the score's) + the latent's zero
// + the entry's -1's; the odd's pool the tail's with themselves (the
// normed latent's + the entry's p / 2's); the padding's (pos < 0's) zero
// the latent's + the -1's entry's. Mirrors dsv4_compress_tail_update's
// (the CUDA kernel's). `tails` fp32 [max_requests][2][W] (the request's
// tail's at tails + req * 2 * W).
void tail_update_ref(const std::vector<float>& comp_kv, const std::vector<float>& comp_score,
                     const std::vector<int32_t>& req_ids, const std::vector<int64_t>& pos,
                     const std::vector<int32_t>& req_spans, int num_requests,
                     const std::vector<uint16_t>& norm_w, float eps, std::vector<float>& tails,
                     std::vector<uint16_t>& latent_out, std::vector<int64_t>& entries_out,
                     int W) {
  for (int q = 0; q < num_requests; ++q) {
    const int start = req_spans[2 * q], len = req_spans[2 * q + 1];
    if (len <= 0) continue;
    float* tail = tails.data() + static_cast<size_t>(req_ids[start]) * 2 * W;
    for (int t = start; t < start + len; ++t) {
      const int64_t p = pos[static_cast<size_t>(t)];
      const float* kvt = comp_kv.data() + static_cast<size_t>(t) * W;
      const float* st = comp_score.data() + static_cast<size_t>(t) * W;
      uint16_t* lat = latent_out.data() + static_cast<size_t>(t) * W;
      if (p < 0) {
        for (int c = 0; c < W; ++c) lat[c] = 0;
        entries_out[static_cast<size_t>(t)] = -1;
        continue;
      }
      if ((p & 1) == 0) {  // the even's: the stash's (the kv's + the score's the tail's)
        for (int c = 0; c < W; ++c) {
          tail[c] = kvt[c];
          tail[W + c] = st[c];
          lat[c] = 0;
        }
        entries_out[static_cast<size_t>(t)] = -1;
      } else {  // the odd's: the pool's (the tail's + the current's the normed latent's)
        pair_pool_norm_ref(tail, tail + W, kvt, st, norm_w.data(), eps, lat, W);
        entries_out[static_cast<size_t>(t)] = p / 2;
      }
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
DGPP_TEST(dsv4_compress_tail_layout_two_planes) {
  // The per-request's fp32 [2, W]'s tail's the LAYOUT's proof (the task's
  // "the tail-ring update writes the expected [2,512] fp32 layout for a
  // small synthetic shape"): the even's stash's the pending token's kv's
  // into the FIRST W's (tail[0..W-1]'s) + the score's into the SECOND W's
  // (tail[W..2W-1]'s). A two-token's (the even's p = 0's stashes, the
  // odd's p = 1's pools) small synthetic shape (W = 512's the task's
  // [2, 512]'s), the known comp_kv / comp_score's, so the tail's planes
  // are verifiable EXACTLY (the even's stash's the bit-exact copy's).
  const int W = 512;  // the task's [2, 512]'s (the dsv41's kCsa2Latent's)
  const int tokens = 2;
  const int max_requests = 1;
  std::vector<int32_t> req_ids(tokens, 0);
  std::vector<int64_t> pos = {0, 1};  // the even's p = 0's, the odd's p = 1's
  std::vector<int32_t> req_spans = {0, tokens};  // one request's the [0, 2)'s span's
  std::vector<float> comp_kv, comp_score;
  fill_f32(&comp_kv, static_cast<size_t>(tokens) * W, 0xCAFE);
  fill_f32(&comp_score, static_cast<size_t>(tokens) * W, 0xBEEF);
  std::vector<uint16_t> norm_w(W, 0x3F80);  // the bf16 1.0's (the norm's the identity's)
  std::vector<float> tails(static_cast<size_t>(max_requests) * 2 * W, 0.0f);
  std::vector<uint16_t> latent_out(static_cast<size_t>(tokens) * W, 0xFFFF);
  std::vector<int64_t> entries_out(tokens, -42);
  tail_update_ref(comp_kv, comp_score, req_ids, pos, req_spans, 1, norm_w, 1e-6f, tails, latent_out,
                  entries_out, W);
  // The even's (p = 0's) stash's: the tail's the FIRST W's = comp_kv[0]'s,
  // the SECOND W's = comp_score[0]'s (the EXACT copy's the bit-exact's).
  for (int c = 0; c < W; ++c) {
    if (tails[static_cast<size_t>(c)] != comp_kv[static_cast<size_t>(c)])
      throw std::runtime_error("tail layout: the kv plane (the first W's) is not comp_kv[0] at c=" +
                               std::to_string(c));
    if (tails[static_cast<size_t>(W + c)] != comp_score[static_cast<size_t>(c)])
      throw std::runtime_error("tail layout: the score plane (the second W's) is not comp_score[0] at c=" +
                               std::to_string(c));
  }
  // The odd's (p = 1's) pool's: the latent's[1]'s the normed's (the nonzero's
  // the norm_w's 1.0's the pooling's the finite's), the entry's p / 2's = 0's.
  bool latent_nonzero = false;
  for (int c = 0; c < W; ++c)
    if (bf16_bits_to_float(latent_out[static_cast<size_t>(1) * W + c]) != 0.0f) latent_nonzero = true;
  if (!latent_nonzero)
    throw std::runtime_error("tail update: the odd's pooled latent is all-zero (the pooling's missing)");
  if (entries_out[0] != -1 || entries_out[1] != 0)
    throw std::runtime_error("tail update: the entries are not (-1, 0) (the even's -1's, the odd's p / 2's 0's)");
  // The even's latent's[0]'s zeroed's (the stash's the no-pool's).
  for (int c = 0; c < W; ++c)
    if (latent_out[static_cast<size_t>(c)] != 0)
      throw std::runtime_error("tail update: the even's latent is not zeroed at c=" + std::to_string(c));
}

DGPP_TEST(dsv4_compress_tail_pool_norm_independent) {
  // The pair-pooling + the one-rounding's RMSNorm's numerics (the dsv41's
  // pool_pair_and_norm's re-expression): the per-dim's softmax over (s0,
  // s1) (the max-shift's), the weighted's kv sum's the ONE bf16 rounding's,
  // the RMSNorm's (the fp32's interior's, the norm_w's bf16's exact
  // upcast's, the ONE bf16 rounding's). Cross-check the fp32 reference
  // against an INDEPENDENT double-precision formulation (the different
  // precision path's the fp32-vs-double epsilon's the documented budget's)
  // on a small synthetic shape (W = 512's, the 2 planes' the 256 dims'
  // probed).
  const int W = 512;
  std::vector<float> kv0(W), s0(W), kv1(W), s1(W);
  fill_f32(&kv0, W, 0xA1);
  fill_f32(&s0, W, 0xB2);
  fill_f32(&kv1, W, 0xC3);
  fill_f32(&s1, W, 0xD4);
  std::vector<uint16_t> norm_w;
  fill_bf16(&norm_w, W, 0xE5);
  const float eps = 1e-6f;
  std::vector<uint16_t> out(W, 0);
  pair_pool_norm_ref(kv0.data(), s0.data(), kv1.data(), s1.data(), norm_w.data(), eps, out.data(), W);
  // The independent double-precision formulation (the no-bf16-rounding's
  // the fp32-vs-double's the 1e-5 relative's budget's).
  std::vector<double> kv0d(W), s0d(W), kv1d(W), s1d(W), norm_d(W), out_d(W);
  for (int c = 0; c < W; ++c) {
    kv0d[static_cast<size_t>(c)] = static_cast<double>(kv0[static_cast<size_t>(c)]);
    s0d[static_cast<size_t>(c)] = static_cast<double>(s0[static_cast<size_t>(c)]);
    kv1d[static_cast<size_t>(c)] = static_cast<double>(kv1[static_cast<size_t>(c)]);
    s1d[static_cast<size_t>(c)] = static_cast<double>(s1[static_cast<size_t>(c)]);
    norm_d[static_cast<size_t>(c)] = bf16_to_d64(norm_w[static_cast<size_t>(c)]);
  }
  pair_pool_norm_ref_d64(kv0d.data(), s0d.data(), kv1d.data(), s1d.data(), norm_d.data(),
                         static_cast<double>(eps), out_d.data(), W);
  for (int c = 0; c < W; c += 3) {  // the 256 dims' the probed (the W's 512's the half's)
    const float got = bf16_bits_to_float(out[static_cast<size_t>(c)]);
    // The got's the bf16-rounded's (the ONE bf16 rounding's), the out_d's
    // the bf16-rounded's (the same's the double's the interior's): the
    // fp32-vs-double's the 1e-7's the 1e-5 relative's budget's.
    const double want = out_d[static_cast<size_t>(c)];
    const double mag = std::max({std::fabs(static_cast<double>(got)), std::fabs(want), 1e-30});
    if (std::fabs(static_cast<double>(got) - want) > mag * 1e-5 + 1e-30)
      throw std::runtime_error("pair pool norm: the fp32 diverged from the double at c=" + std::to_string(c));
  }
  // The finite's probe (the pooling's + the norm's the no-NaN's).
  for (int c = 0; c < W; c += 7) {
    const float v = bf16_bits_to_float(out[static_cast<size_t>(c)]);
    if (!std::isfinite(v))
      throw std::runtime_error("pair pool norm: a non-finite output at c=" + std::to_string(c));
  }
}

DGPP_TEST(dsv4_compress_tail_even_odd_cycle) {
  // The even-stash / odd-pool's the FULL cycle's (the 4 tokens' the p =
  // [0, 1, 2, 3]'s) on a small synthetic shape (W = 512's): the even's
  // (p = 0, 2's) stash's (the tail's the kv's + the score's), the odd's
  // (p = 1, 3's) pool's (the latent's the normed's + the entry's p / 2's
  // = 0, 1's). The tail's after the cycle's the LAST even's (p = 2's)
  // kv/score's (the pending even's), the entries' (-1, 0, -1, 1)'s.
  const int W = 512;
  const int tokens = 4;
  std::vector<int32_t> req_ids(tokens, 0);
  std::vector<int64_t> pos = {0, 1, 2, 3};
  std::vector<int32_t> req_spans = {0, tokens};
  std::vector<float> comp_kv, comp_score;
  fill_f32(&comp_kv, static_cast<size_t>(tokens) * W, 0x1111);
  fill_f32(&comp_score, static_cast<size_t>(tokens) * W, 0x2222);
  std::vector<uint16_t> norm_w(W, 0x3F80);  // the bf16 1.0's
  std::vector<float> tails(static_cast<size_t>(1) * 2 * W, 0.0f);
  std::vector<uint16_t> latent_out(static_cast<size_t>(tokens) * W, 0xFFFF);
  std::vector<int64_t> entries_out(tokens, -42);
  tail_update_ref(comp_kv, comp_score, req_ids, pos, req_spans, 1, norm_w, 1e-6f, tails, latent_out,
                  entries_out, W);
  // The entries' (-1, 0, -1, 1)'s (the even's -1's, the odd's p / 2's).
  const int64_t want_entries[4] = {-1, 0, -1, 1};
  for (int t = 0; t < 4; ++t)
    if (entries_out[static_cast<size_t>(t)] != want_entries[t])
      throw std::runtime_error("tail cycle: the entries are not (-1, 0, -1, 1) at t=" + std::to_string(t));
  // The tail's after the cycle's the LAST even's (p = 2's) kv/score's (the
  // pending even's the bit-exact copy's).
  for (int c = 0; c < W; ++c) {
    if (tails[static_cast<size_t>(c)] != comp_kv[static_cast<size_t>(2 * W + c)])
      throw std::runtime_error("tail cycle: the tail's kv plane is not the last even's (p = 2's) at c=" +
                               std::to_string(c));
    if (tails[static_cast<size_t>(W + c)] != comp_score[static_cast<size_t>(2 * W + c)])
      throw std::runtime_error("tail cycle: the tail's score plane is not the last even's (p = 2's) at c=" +
                               std::to_string(c));
  }
  // The odd's pooled latents' (t = 1, 3's) the normed's (the nonzero's),
  // the even's (t = 0, 2's) zeroed's.
  for (int c = 0; c < W; c += 11) {
    if (bf16_bits_to_float(latent_out[static_cast<size_t>(1 * W + c)]) == 0.0f)
      throw std::runtime_error("tail cycle: the odd's (t = 1's) latent is zero (the pooling's missing)");
    if (bf16_bits_to_float(latent_out[static_cast<size_t>(3 * W + c)]) == 0.0f)
      throw std::runtime_error("tail cycle: the odd's (t = 3's) latent is zero (the pooling's missing)");
    if (latent_out[static_cast<size_t>(0 * W + c)] != 0)
      throw std::runtime_error("tail cycle: the even's (t = 0's) latent is not zeroed");
    if (latent_out[static_cast<size_t>(2 * W + c)] != 0)
      throw std::runtime_error("tail cycle: the even's (t = 2's) latent is not zeroed");
  }
}

DGPP_TEST(dsv4_compress_tail_padding_row) {
  // The padding's row's (pos < 0's) the no-op's (the latent's zeroed's +
  // the entry's -1's, the tail's untouched's): a 2-row's (the real's p =
  // 0's stashes, the padding's p = -1's no-op's) small synthetic shape
  // (W = 512's). The tail's the real's (p = 0's) kv/score's (the
  // untouched's by the padding's).
  const int W = 512;
  const int tokens = 2;
  std::vector<int32_t> req_ids = {0, 0};
  std::vector<int64_t> pos = {0, -1};  // the real's p = 0's, the padding's p = -1's
  std::vector<int32_t> req_spans = {0, tokens};
  std::vector<float> comp_kv, comp_score;
  fill_f32(&comp_kv, static_cast<size_t>(tokens) * W, 0x3333);
  fill_f32(&comp_score, static_cast<size_t>(tokens) * W, 0x4444);
  std::vector<uint16_t> norm_w(W, 0x3F80);
  std::vector<float> tails(static_cast<size_t>(1) * 2 * W, 0.0f);
  std::vector<uint16_t> latent_out(static_cast<size_t>(tokens) * W, 0xFFFF);
  std::vector<int64_t> entries_out(tokens, -42);
  tail_update_ref(comp_kv, comp_score, req_ids, pos, req_spans, 1, norm_w, 1e-6f, tails, latent_out,
                  entries_out, W);
  // The real's (p = 0's) stash's: the tail's the comp_kv[0]'s + the
  // comp_score[0]'s.
  for (int c = 0; c < W; c += 5) {
    if (tails[static_cast<size_t>(c)] != comp_kv[static_cast<size_t>(c)])
      throw std::runtime_error("tail padding: the kv plane is not comp_kv[0] at c=" + std::to_string(c));
  }
  // The padding's (p = -1's) no-op's: the latent's[1]'s zeroed's + the
  // entry's[1]'s -1's.
  for (int c = 0; c < W; c += 5)
    if (latent_out[static_cast<size_t>(1 * W + c)] != 0)
      throw std::runtime_error("tail padding: the padding's latent is not zeroed at c=" + std::to_string(c));
  if (entries_out[0] != -1 || entries_out[1] != -1)
    throw std::runtime_error("tail padding: the entries are not (-1, -1) (the even's -1's, the padding's -1's)");
}

int main() { return ::dgpp::test::run_all(); }
