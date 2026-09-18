// DeepSeek-V4-Flash (deepseek_v4) DSpark draft/verify CPU oracles
// (docs/dsv4_kernel_port_spec.md §2.2). Host-only, no GPU: the naive
// FP64/FP32 references that pin the numerics of the V4 DSpark surface —
// the target-layers' stream mean, the block rows off the accepted verify
// rows, the Markov-biased head row, the confidence logit, and the DSpark
// union attention (the single softmax over the UNION of the three KV
// phases: the C4A compressed pool + the 128-slot projected-main-hidden
// ring + the in-memory bf16 draft/verify block KVs, non-causal, the
// attn_sink exactly-once) — so the Phase B v4 layer files
// (src/models/dsv4/dspark_layer.cu) have a host-side oracle before any
// forward path consumes them.
//
// Borrowed, not rewritten (house rule 2): the glue math mirrors the
// dsv41 DSpark kernels (src/kernels/dsv41_dspark.hpp:1-153, the
// `DSparkBlock` reference) and the union-attention contract mirrors the
// dsv4-native ops/dspark_attn (src/ops/dspark_attn/dspark_attn.h:1-120,
// the double oracle ops/common/op_test.h:1990 dspark_attn_ref). Each
// test cross-checks the naive reference against an INDEPENDENT
// formulation and pins the documented edge cases (the sink's
// exactly-once, the +inf degenerate limit, the all-zero no-tokens
// output), so a silent corruption of the borrowed form is caught here.
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
    // A bf16 word: sign + a small exponent + a few mantissa bits, so the
    // values are O(1) and finite.
    const uint16_t sign = (next() & 1) ? 0x8000 : 0x0000;
    const uint16_t expf = static_cast<uint16_t>((0x3C + (next() & 7)) << 7);  // ~0.5..8
    const uint16_t man = static_cast<uint16_t>((next() & 0x7F) << 0);
    (*v)[i] = sign | expf | man;
  }
}

// ---- the stream mean (dsv41_dspark.hpp dsv41_stream_mean_bf16) --------
// out[t, :] = bf16(mean over the hc_mult streams of streams[t, :, :]) —
// the fp32 sum in stream order, ONE rounding (torch's bf16 mean: h.mean
// (dim=2) of the attention input at a DSpark target layer).
std::vector<uint16_t> stream_mean(const std::vector<uint16_t>& streams, int rows, int hc_mult, int hidden) {
  std::vector<uint16_t> out(static_cast<size_t>(rows) * hidden);
  for (int t = 0; t < rows; ++t)
    for (int d = 0; d < hidden; ++d) {
      float acc = 0.0f;  // the fp32 sum in stream order (the reference's
                         // `h.mean(dim=2)` fp32 accumulation)
      for (int s = 0; s < hc_mult; ++s)
        acc += bf16_bits_to_float(streams[(static_cast<size_t>(t) * hc_mult + s) * hidden + d]);
      out[static_cast<size_t>(t) * hidden + d] = float_to_bf16_bits(acc / static_cast<float>(hc_mult));
    }
  return out;
}

// ---- the block rows (dsv41_dspark.hpp dsv41_dspark_block_rows) --------
// For each group g of rows_per_group rows: P = 1 + max(step_pos) over the
// group (-1 when no row is real: the group's block is padding) and `next`
// = that row's token; the block's rows g*block + k get pos P + k (or -1),
// tokens [next, noise, ...], the group's request id, one span per group.
struct BlockRows {
  std::vector<int64_t> pos;
  std::vector<int64_t> tok;
  std::vector<int32_t> req;
  std::vector<int32_t> spans;
};
BlockRows dspark_block_rows(const std::vector<int64_t>& step_pos, const std::vector<int64_t>& tokens,
                           const std::vector<int32_t>& req_ids, int groups, int rows_per_group, int block,
                           int64_t noise_id) {
  const int out_rows = groups * block;
  BlockRows out;
  out.pos.assign(out_rows, -1);
  out.tok.assign(out_rows, -1);
  out.req.assign(out_rows, 0);
  out.spans.assign(groups + 1, 0);
  for (int g = 0; g < groups; ++g) {
    int64_t P = -1;
    int next_row = -1;
    for (int i = 0; i < rows_per_group; ++i) {
      const int64_t p = step_pos[static_cast<size_t>(g) * rows_per_group + i];
      if (p < 0) continue;  // a padding row
      if (P < 0 || p > P) {
        P = p;
        next_row = static_cast<int>(g) * rows_per_group + i;
      }
    }
    out.spans[g] = g * block;
    if (P < 0) continue;  // an all-padding group: the block is padding
    P += 1;  // the block's first position (the next position past the max)
    for (int k = 0; k < block; ++k) {
      const int r = g * block + k;
      out.pos[r] = P + k;
      out.tok[r] = (k == 0) ? tokens[next_row] : noise_id;  // [next, noise, ...]
      out.req[r] = req_ids[static_cast<size_t>(g) * rows_per_group];  // the group's request id
    }
  }
  out.spans[groups] = groups * block;
  return out;
}

// ---- the Markov-biased head row (dsv41_dspark.hpp dsv41_dspark_markov_bias)
// out[g, j, v] = base[g*base_group_stride + block_row*count + v] +
//   sum_r embed[tok_g][r] * head[vocab_begin + v][r]
// the bf16 embedding row and the bf16 head rows, fp32 accumulation (one
// fp32 dot per vocab entry, in rank order).
void markov_bias(const std::vector<float>& base, int64_t base_group_stride, int block_row,
                 const std::vector<uint16_t>& markov_embed, const std::vector<uint16_t>& markov_head, int rank,
                 int vocab_begin, int count, const std::vector<int64_t>& tok, int tok_stride, int groups,
                 std::vector<float>& out, int64_t out_group_stride, int rows_out) {
  for (int g = 0; g < groups; ++g) {
    const int64_t tok_g = tok[static_cast<size_t>(g) * tok_stride];
    for (int j = 0; j < rows_out; ++j)
      for (int v = 0; v < count; ++v) {
        float acc = base[static_cast<size_t>(g) * base_group_stride + static_cast<size_t>(block_row) * count + v];
        for (int r = 0; r < rank; ++r)
          acc += bf16_bits_to_float(markov_embed[static_cast<size_t>(tok_g) * rank + r]) *
                 bf16_bits_to_float(markov_head[static_cast<size_t>(vocab_begin + v) * rank + r]);
        out[static_cast<size_t>(g) * out_group_stride + static_cast<size_t>(j) * count + v] = acc;
      }
  }
}

// ---- the confidence logit (dsv41_dspark.hpp dsv41_dspark_confidence) ----
// conf[g] = sum_d w[d] * x[g*x_group_stride + block_row*hidden + d] +
//   sum_r w[hidden + r] * embed[tok_g][r]
// the fp32 projection of [x_k | embed(tok_{k-1})], x the hc_pre'd hidden
// BEFORE the norm.
void dspark_confidence(const std::vector<uint16_t>& x, int64_t x_group_stride, int block_row, int hidden,
                       const std::vector<uint16_t>& markov_embed, int rank, const std::vector<int64_t>& tok,
                       int tok_stride, const std::vector<float>& w, int groups, std::vector<float>& conf) {
  for (int g = 0; g < groups; ++g) {
    float acc = 0.0f;
    for (int d = 0; d < hidden; ++d)
      acc += w[d] * bf16_bits_to_float(x[static_cast<size_t>(g) * x_group_stride + static_cast<size_t>(block_row) * hidden + d]);
    const int64_t tok_g = tok[static_cast<size_t>(g) * tok_stride];
    for (int r = 0; r < rank; ++r)
      acc += w[hidden + r] * bf16_bits_to_float(markov_embed[static_cast<size_t>(tok_g) * rank + r]);
    conf[g] = acc;
  }
}

// ---- the DSpark union attention (dsv4-native src/ops/dspark_attn/
// dspark_attn.h:1-120) ---------------------------------------------------
// The single max-shift softmax over the UNION of the three KV phases
// (the C4A compressed pool slots [n_comp] | the main-KV ring's LINEAR
// slots 0..raw_n-1 | the in-memory bf16 block KVs [n_block] SHARED — the
// non-causal all-queries-see-all-block-KVs), the D25 sink's exactly-once
// (the m' = max(max_t S[t], sink[h]) + the exp(sink - m') denominator
// term — the ops/common/op_test.h:1990 dspark_attn_ref closed form).
// `q` [64, 512] double (one row), `lats` [n_tokens, 512] double (the
// caller decodes the union's records in phase order: compressed -> ring
// -> block). Returns the [64, 512] double output.
std::vector<double> dspark_attn_ref(const double* q, const double* lats, int n_tokens, const float* attn_sink) {
  constexpr int kHeads = 64, kHeadDim = 512;
  const double softmax_scale = 0.04419417382415922;  // 1/sqrt(512)
  std::vector<double> out(static_cast<size_t>(kHeads) * kHeadDim, 0.0);
  std::vector<double> S(n_tokens);
  for (int h = 0; h < kHeads; ++h) {
    double mx = -std::numeric_limits<double>::infinity();
    for (int t = 0; t < n_tokens; ++t) {
      double s = 0.0;
      for (int d = 0; d < kHeadDim; ++d)
        s += q[static_cast<size_t>(h) * kHeadDim + d] * lats[static_cast<size_t>(t) * kHeadDim + d];
      s *= softmax_scale;
      S[t] = s;
      if (s > mx) mx = s;
    }
    const double sink = (attn_sink != nullptr) ? static_cast<double>(attn_sink[h])
                                               : -std::numeric_limits<double>::infinity();
    if (sink == std::numeric_limits<double>::infinity())
      continue;  // the +inf degenerate limit: the EXACTLY-zero output
    const double m = (sink > mx) ? sink : mx;  // m' = max(mx, sink)
    double sum = 0.0;
    for (int t = 0; t < n_tokens; ++t) sum += std::exp(S[t] - m);
    sum += std::exp(sink - m);  // the sink's mass (the exactly-once term)
    for (int d = 0; d < kHeadDim; ++d) {
      double acc = 0.0;
      for (int t = 0; t < n_tokens; ++t) acc += std::exp(S[t] - m) * lats[static_cast<size_t>(t) * kHeadDim + d];
      out[static_cast<size_t>(h) * kHeadDim + d] = (sum > 0.0) ? acc / sum : 0.0;
    }
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
DGPP_TEST(dsv4_dspark_stream_mean_single_rounding) {
  // The stream mean is the fp32 sum in stream order, ONE bf16 rounding:
  // cross-check against an INDEPENDENT (reversed stream order) fp32 sum —
  // the two may differ by the fp32 accumulation epsilon, so a loose
  // bf16-ulp budget (the reference's documented rounding class), and the
  // result must be the correctly-rounded bf16 of the mean.
  const int rows = 2, hc_mult = 4, hidden = 8;
  std::vector<uint16_t> streams(static_cast<size_t>(rows) * hc_mult * hidden);
  fill_bf16(&streams, streams.size(), 0xD5A1);
  const auto out = stream_mean(streams, rows, hc_mult, hidden);
  for (int t = 0; t < rows; ++t)
    for (int d = 0; d < hidden; ++d) {
      // The independent reversed-order fp32 sum (a different accumulation
      // order: the fp32 epsilon is the documented budget).
      float fwd = 0.0f, rev = 0.0f;
      for (int s = 0; s < hc_mult; ++s)
        fwd += bf16_bits_to_float(streams[(static_cast<size_t>(t) * hc_mult + s) * hidden + d]);
      for (int s = hc_mult - 1; s >= 0; --s)
        rev += bf16_bits_to_float(streams[(static_cast<size_t>(t) * hc_mult + s) * hidden + d]);
      const float mean_fwd = fwd / hc_mult, mean_rev = rev / hc_mult;
      // The output is the correctly-rounded bf16 of the forward mean.
      if (out[static_cast<size_t>(t) * hidden + d] != float_to_bf16_bits(mean_fwd))
        throw std::runtime_error("stream mean: the output is not the bf16 of the forward fp32 mean at t=" +
                                 std::to_string(t) + " d=" + std::to_string(d));
      // The forward / reversed means agree within the fp32 accumulation
      // epsilon (a few ulps of the magnitude).
      const double mag = std::max({std::fabs(static_cast<double>(mean_fwd)), std::fabs(static_cast<double>(mean_rev)), 1e-30});
      if (std::fabs(mean_fwd - mean_rev) > mag * 1e-5)
        throw std::runtime_error("stream mean: the forward / reversed fp32 means diverged beyond the fp32 epsilon");
    }
}

DGPP_TEST(dsv4_dspark_block_rows_layout) {
  // The block rows: P = 1 + max(step_pos) per group, the tokens [next,
  // noise, ...], the group's request id, one span per group, and the
  // all-padding group's -1 block. Verify the exact layout against the
  // dsv41_dspark.hpp contract with a contrived two-group input (group 0
  // real, group 1 all-padding).
  const int groups = 2, rpg = 3, block = 5;
  // Group 0: rows 0, 1, 2 with step_pos [10, 12, 11] (max 12 at row 1),
  // tokens [100, 101, 102]. P = 1 + 12 = 13, next = token[1] = 101.
  // Group 1: rows 3, 4, 5 all padding (step_pos -1). The block is -1.
  std::vector<int64_t> step_pos = {10, 12, 11, -1, -1, -1};
  std::vector<int64_t> tokens = {100, 101, 102, -1, -1, -1};
  std::vector<int32_t> req_ids = {7, 7, 7, 9, 9, 9};
  const auto out = dspark_block_rows(step_pos, tokens, req_ids, groups, rpg, block, 128799);
  // Group 0's block rows 0..4: pos [13, 14, 15, 16, 17], tokens
  // [101, 128799, 128799, 128799, 128799], req 7.
  const int64_t want_pos0[5] = {13, 14, 15, 16, 17};
  const int64_t want_tok0[5] = {101, 128799, 128799, 128799, 128799};
  for (int k = 0; k < block; ++k) {
    if (out.pos[k] != want_pos0[k] || out.tok[k] != want_tok0[k] || out.req[k] != 7)
      throw std::runtime_error("block rows: group 0's real block is wrong at k=" + std::to_string(k) + " (pos " +
                               std::to_string(out.pos[k]) + ", tok " + std::to_string(out.tok[k]) + ")");
  }
  // Group 1's block rows 5..9: all -1 (the all-padding group).
  for (int k = 0; k < block; ++k)
    if (out.pos[5 + k] != -1 || out.tok[5 + k] != -1)
      throw std::runtime_error("block rows: group 1's padding block is not -1 at k=" + std::to_string(k));
  // The spans: one per group (spans[g] = g*block, spans[groups] = groups*block).
  if (out.spans[0] != 0 || out.spans[1] != 5 || out.spans[2] != 10)
    throw std::runtime_error("block rows: the spans are not the per-group block offsets");
}

DGPP_TEST(dsv4_dspark_markov_bias_and_confidence) {
  // The Markov-biased head row + the confidence logit: the fp32 dot of
  // the bf16 embedding row and the bf16 head rows (one fp32 dot per
  // vocab entry, in rank order). Cross-check against an INDEPENDENT
  // double-precision dot (the fp32-vs-double epsilon is the documented
  // budget, a loose bound).
  const int groups = 2, rank = 8, count = 4, hidden = 6, rows_out = 2, block_row = 1;
  const int64_t base_gs = rows_out * count, out_gs = rows_out * count;
  std::vector<float> base(static_cast<size_t>(groups) * base_gs, 0.5f);
  std::vector<uint16_t> embed(static_cast<size_t>(64) * rank), head(static_cast<size_t>(count) * rank);
  fill_bf16(&embed, embed.size(), 0xEA11);
  fill_bf16(&head, head.size(), 0x9E37);
  std::vector<int64_t> tok = {3, 40};
  std::vector<float> out(static_cast<size_t>(groups) * out_gs, 0.0f);
  markov_bias(base, base_gs, block_row, embed, head, rank, /*vocab_begin=*/0, count, tok, /*tok_stride=*/1, groups,
              out, out_gs, rows_out);
  for (int g = 0; g < groups; ++g)
    for (int j = 0; j < rows_out; ++j)
      for (int v = 0; v < count; ++v) {
        // The independent double dot (a different precision path).
        double want = base[static_cast<size_t>(g) * base_gs + static_cast<size_t>(block_row) * count + v];
        for (int r = 0; r < rank; ++r)
          want += bf16_to_d64(embed[static_cast<size_t>(tok[g]) * rank + r]) *
                  bf16_to_d64(head[static_cast<size_t>(v) * rank + r]);
        const double got = out[static_cast<size_t>(g) * out_gs + static_cast<size_t>(j) * count + v];
        const double mag = std::max({std::fabs(want), std::fabs(got), 1e-3});
        if (std::fabs(want - got) > mag * 1e-5)
          throw std::runtime_error("markov bias: the fp32 dot diverged from the double dot at g=" + std::to_string(g) +
                                   " j=" + std::to_string(j) + " v=" + std::to_string(v));
      }
  // The confidence logit: the fp32 projection of [x_k | embed(tok)].
  std::vector<uint16_t> x(static_cast<size_t>(groups) * static_cast<size_t>(rows_out) * hidden);
  fill_bf16(&x, x.size(), 0xC0FF);
  std::vector<float> w(hidden + rank, 0.25f);
  std::vector<float> conf(groups, 0.0f);
  dspark_confidence(x, static_cast<int64_t>(rows_out) * hidden, block_row, hidden, embed, rank, tok, 1, w, groups, conf);
  for (int g = 0; g < groups; ++g) {
    double want = 0.0;
    for (int d = 0; d < hidden; ++d)
      want += w[d] * bf16_to_d64(x[static_cast<size_t>(g) * rows_out * hidden + static_cast<size_t>(block_row) * hidden + d]);
    for (int r = 0; r < rank; ++r) want += w[hidden + r] * bf16_to_d64(embed[static_cast<size_t>(tok[g]) * rank + r]);
    const double got = conf[g];
    const double mag = std::max({std::fabs(want), std::fabs(got), 1e-3});
    if (std::fabs(want - got) > mag * 1e-5)
      throw std::runtime_error("confidence: the fp32 projection diverged from the double dot at g=" + std::to_string(g));
  }
}

DGPP_TEST(dsv4_dspark_union_attn_sink_exactly_once) {
  // The DSpark union attention's single-softmax-over-the-union + the D25
  // sink's exactly-once: a three-phase union (n_comp = 2 compressed,
  // raw_n = 2 ring, n_block = 1 in-memory block = 5 tokens total) with a
  // single KV latent so the softmax is a known closed form. The null /
  // -inf sink is the no-op, the +inf sink is the EXACTLY-zero output, and
  // a finite sink adds exactly ONE exp(sink - m') to the merged
  // denominator (cross-checked against the closed form m' = max(mx,
  // sink) — the union's phases are ONE list, the 3-phase online-softmax
  // is math-equivalent within the fp32 budget).
  constexpr int kHeads = 64, kHeadDim = 512;
  const int n_tokens = 5;  // 2 compressed + 2 ring + 1 block
  std::vector<double> q(static_cast<size_t>(kHeads) * kHeadDim, 1.0);
  std::vector<double> lats(static_cast<size_t>(n_tokens) * kHeadDim, 2.0);  // every token's latent 2.0
  const double softmax_scale = 0.04419417382415922;
  const double s = 1.0 * 2.0 * kHeadDim * softmax_scale;  // every head's logit (q=1, latent=2)
  // The null / -inf sink: the 5-token softmax (every token equal, so
  // each gets 1/5 of the mass) -> the output is the mean latent (2.0).
  const auto out_null = dspark_attn_ref(q.data(), lats.data(), n_tokens, nullptr);
  for (int i = 0; i < kHeads * kHeadDim; ++i)
    if (std::fabs(out_null[i] - 2.0) > 1e-12)
      throw std::runtime_error("dspark union attn: the null-sink equal-token output is not the mean latent (got " +
                               std::to_string(out_null[i]) + ")");
  float neg_inf[kHeads];
  for (int h = 0; h < kHeads; ++h) neg_inf[h] = -std::numeric_limits<float>::infinity();
  const auto out_neginf = dspark_attn_ref(q.data(), lats.data(), n_tokens, neg_inf);
  for (int i = 0; i < kHeads * kHeadDim; ++i)
    if (out_neginf[i] != out_null[i])
      throw std::runtime_error("dspark union attn: the -inf sink is not bit-identical to the null sink (the no-op)");
  // The +inf sink: the EXACTLY-zero output (the degenerate limit).
  float pos_inf[kHeads];
  for (int h = 0; h < kHeads; ++h) pos_inf[h] = std::numeric_limits<float>::infinity();
  const auto out_posinf = dspark_attn_ref(q.data(), lats.data(), n_tokens, pos_inf);
  for (int i = 0; i < kHeads * kHeadDim; ++i)
    if (out_posinf[i] != 0.0)
      throw std::runtime_error("dspark union attn: the +inf sink is not the EXACTLY-zero output");
  // A finite sink: the exactly-once denominator (the sink's mass enters
  // the MERGED union denominator ONCE, not per-phase or per-head).
  const float sink = static_cast<float>(s - 1.0);  // sink < s
  float one_sink[kHeads];
  for (int h = 0; h < kHeads; ++h) one_sink[h] = sink;
  const auto out_sink = dspark_attn_ref(q.data(), lats.data(), n_tokens, one_sink);
  // The equal-token closed form: m' = max(mx = s, sink) = s (sink < s),
  // so exp(S[t] - m') = 1 for every token, the numerator is n_tokens *
  // latent (5 * 2.0 = 10, NOT just the single latent), the denominator
  // is n_tokens + exp(sink - s) (the exactly-once sink's mass).
  const double denom = static_cast<double>(n_tokens) + std::exp(static_cast<double>(sink) - s);
  const double want = (static_cast<double>(n_tokens) * 2.0) / denom;
  for (int d = 0; d < kHeadDim; d += 97)
    for (int h = 0; h < 4; ++h) {
      if (std::fabs(out_sink[static_cast<size_t>(h) * kHeadDim + d] - want) > 1e-12)
        throw std::runtime_error("dspark union attn: the finite sink's exactly-once denominator is wrong at head " +
                                 std::to_string(h) + " (want " + std::to_string(want) + ", got " +
                                 std::to_string(out_sink[static_cast<size_t>(h) * kHeadDim + d]) + ")");
    }
  // The all-zero no-tokens edge: n_tokens = 0 (the DSpark draft's no-
  // compressed-phase with no ring and no block) -> the l == 0's
  // no-tokens' 0.0 output (the documented edge, NOT an error).
  const auto out_zero = dspark_attn_ref(q.data(), lats.data(), 0, nullptr);
  for (int i = 0; i < kHeads * kHeadDim; ++i)
    if (out_zero[i] != 0.0)
      throw std::runtime_error("dspark union attn: the all-zero no-tokens output is not 0.0");
  (void)s;
}

int main() { return ::dgpp::test::run_all(); }
