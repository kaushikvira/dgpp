// DeepSeek-V4-Flash (deepseek_v4) 0731 checkpoint's Compressor's per-request
// tail-ring update CPU oracles (the 2026-09-21's the reference's form's,
// docs/dsv4_attention_spec.md §5: the G-tail-cadence / G-tail-pool /
// G-tail-ape / G-c128a-compressor's gaps' the closed's). Host-only, no GPU:
// the naive FP32/FP64 references that pin the numerics of the V4
// compressor's per-request tail-ring update — the reference's arithmetic
// (the checkpoint's inference/model.py's Compressor, 285-400's) re-
// expressed: the per-token's state's write (the kv_state / score_state's
// slot(p)'s, the APE's on the score's half's only's the p % ratio's), the
// EVERY ratio-th token's publish (the (coff * ratio)-entry's x 512-dim's
// pool's the coff x ratio's gather's, the one-rounding's RMSNorm's, the
// entry's p / ratio's at the rotation's p + 1 - ratio's), the C4A's
// window's shift's, the C128A's 128-entry's gated pool's.
//
// The per-request tail is fp32 [2][coff * ratio][coff * 512] (the
// checkpoint's kv_state / score_state's (b, coff * ratio, coff * head_dim)'s
// the kv's plane's the first coff * ratio * W's + the score's the second's):
// the C4A's (ratio 4's coff 2's) the 8 x 1024's (the 2 overlapping's
// windows' 4 tokens' the wkv / wgate's 1024-wide's the first's half's the
// overlap's plane's the second's the normal's), the C128A's (ratio 128's
// coff 1's) the 128 x 512's the 128-token's ring's. The state's init's the
// reference's (the kv's zero's + the score's the -inf's the 309-310's).
// So the Phase B v4 forward path (src/models/dsv4/compress.cu's
// dsv4_compress_tail_update) has a host-side oracle that pins the
// [2][coff * ratio][W] layout on a small synthetic shape before any GPU
// consumes the tail (house rule: every adapted op is oracle-qualified on
// CPU before consumption).
//
// Borrowed, not rewritten (house rule 2): the tail update mirrors the
// reference's Compressor's decode's form (the per-token's state's write's
// the APE's the publish's the pool's the shift's) and the pool's numerics
// mirror the reference's the score_state.softmax(dim=1)'s weighted sum's
// the fp32's interior's + the norm_w's bf16's exact upcast's + the ONE bf16
// rounding's. Each test cross-checks the naive fp32 reference against an
// INDEPENDENT (double-precision, the same's algorithm's the different's
// precision path's) formulation and pins the documented edge cases (the
// APE's the score's half's only's, the ratio's cadence's publish's, the
// C4A's window's shift's, the C128A's 128-entry's pool's, the padding's
// -1, the [2][coff * ratio][W] layout's), so a silent corruption of the
// borrowed form is caught here.
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
// A deterministic bf16-plane generator (the norm_w's the bounded's).
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

// The per-request's state's init (the reference's 309-310's the kv_state's
// zero's + the score_state's the -inf's the cold start's the publish's the
// 0's weight's the -inf's rows's): the fp32 [2][n_slots][W]'s (the kv's
// plane's the first n_slots * W's the zeroed's, the score's the next's the
// -inf's). Mirrors dsv4_compress_tail_init's (the CUDA kernel's).
std::vector<float> make_state(int n_requests, int n_slots, int W) {
  std::vector<float> state(static_cast<size_t>(n_requests) * 2 * n_slots * W, 0.0f);
  const size_t n_kv = static_cast<size_t>(n_slots) * W;
  for (int r = 0; r < n_requests; ++r)
    for (size_t c = 0; c < n_kv; ++c)  // the score's plane's (the kv's the 0's already's)
      state[static_cast<size_t>(r) * 2 * n_kv + n_kv + c] = -std::numeric_limits<float>::infinity();
  return state;
}
std::vector<double> make_state_d64(int n_requests, int n_slots, int W) {
  std::vector<double> state(static_cast<size_t>(n_requests) * 2 * n_slots * W, 0.0);
  const size_t n_kv = static_cast<size_t>(n_slots) * W;
  for (int r = 0; r < n_requests; ++r)
    for (size_t c = 0; c < n_kv; ++c)  // the score's plane's (the kv's the 0's already's)
      state[static_cast<size_t>(r) * 2 * n_kv + n_kv + c] = -std::numeric_limits<double>::infinity();
  return state;
}

// ---- the tail's update (the reference's decode's form's) -----------------
// The per-request's fp32 [2][n_slots][W]'s state's (the kv's plane's the
// first n_slots * W's + the score's the next's): every token's the state's
// write's (the slot(p)'s the p % ratio's the C4A's the ratio's offset's the
// second's plane's, the APE's on the score's half's only's the p % ratio's
// the reference's 351's), the EVERY ratio-th's (the (p + 1) % ratio == 0's
// the reference's 350's) the (coff * ratio)-entry's x 512's pool's the
// normed latent's + the entry's p / ratio's the ent_pos's p + 1 - ratio's
// the C4A's window's shift's (the reference's 366-367's), the other's rows'
// zero the latent's + the -1's entry's. Mirrors dsv4_compress_tail_update's
// (the CUDA kernel's). `state` fp32 [max_requests][2][n_slots][W].
void tail_update_ref(const std::vector<float>& comp_kv, const std::vector<float>& comp_score,
                     const std::vector<float>& ape, const std::vector<int32_t>& req_ids,
                     const std::vector<int64_t>& pos, const std::vector<int32_t>& req_spans,
                     int num_requests, const std::vector<uint16_t>& norm_w, float eps,
                     std::vector<float>& state, std::vector<uint16_t>& latent_out,
                     std::vector<int64_t>& entries_out, std::vector<int64_t>& ent_pos_out, int ratio,
                     int coff) {
  const int W = coff * 512;
  const int n_slots = coff * ratio;
  for (int q = 0; q < num_requests; ++q) {
    const int start = req_spans[2 * q], len = req_spans[2 * q + 1];
    if (len <= 0) continue;
    float* kv_state = state.data() + static_cast<size_t>(req_ids[start]) * 2 * n_slots * W;
    float* score_state = kv_state + static_cast<size_t>(n_slots) * W;
    for (int t = start; t < start + len; ++t) {
      const int64_t p = pos[static_cast<size_t>(t)];
      const float* kvt = comp_kv.data() + static_cast<size_t>(t) * W;
      const float* st = comp_score.data() + static_cast<size_t>(t) * W;
      uint16_t* lat = latent_out.data() + static_cast<size_t>(t) * 512;
      if (p < 0) {
        for (int c = 0; c < 512; ++c) lat[static_cast<size_t>(c)] = 0;
        entries_out[static_cast<size_t>(t)] = -1;
        ent_pos_out[static_cast<size_t>(t)] = -1;
        continue;
      }
      const float* arow = ape.data() + static_cast<size_t>(p % ratio) * W;
      const int slot = (coff == 2) ? (ratio + static_cast<int>(p % ratio)) : static_cast<int>(p % ratio);
      for (int c = 0; c < W; ++c) {
        kv_state[static_cast<size_t>(slot) * W + c] = kvt[c];
        score_state[static_cast<size_t>(slot) * W + c] = st[c] + arow[c];
      }
      if ((p + 1) % ratio == 0) {  // the EVERY ratio-th token's publish (the reference's 350's)
        std::vector<float> pooled(512);
        float ss = 0.0f;
        for (int d = 0; d < 512; ++d) {
          // The two-pass max-shift's softmax over the n_slots' entries'
          // (the reference's the score_state.softmax(dim=1)'s, the -inf's
          // rows' the 0's weight's).
          float M = -std::numeric_limits<float>::infinity();
          for (int e = 0; e < n_slots; ++e) {
            const int off = (coff == 2 && e >= ratio) ? 512 + d : d;  // the head_off's the gather's
            M = std::max(M, score_state[static_cast<size_t>(e) * W + off]);
          }
          float L = 0.0f, A = 0.0f;
          for (int e = 0; e < n_slots; ++e) {
            const int off = (coff == 2 && e >= ratio) ? 512 + d : d;
            const float w = std::exp(score_state[static_cast<size_t>(e) * W + off] - M);
            L += w;
            A += w * kv_state[static_cast<size_t>(e) * W + off];
          }
          const float v = bf16_bits_to_float(float_to_bf16_bits(A / L));  // .to(bf16) before the norm
          pooled[static_cast<size_t>(d)] = v;
          ss += v * v;
        }
        const float rs = std::sqrt(1.0f / (ss / 512.0f + eps));
        for (int d = 0; d < 512; ++d)
          lat[static_cast<size_t>(d)] =
              float_to_bf16_bits(bf16_bits_to_float(norm_w[static_cast<size_t>(d)]) *
                                 (pooled[static_cast<size_t>(d)] * rs));
        if (coff == 2) {  // the C4A's window's shift (the reference's 366-367's)
          for (int c = 0; c < ratio * W; ++c) {
            kv_state[static_cast<size_t>(c)] = kv_state[static_cast<size_t>(c + ratio * W)];
            score_state[static_cast<size_t>(c)] = score_state[static_cast<size_t>(c + ratio * W)];
          }
        }
        entries_out[static_cast<size_t>(t)] = p / ratio;  // the entry's ordinal's (the reference's 379's)
        ent_pos_out[static_cast<size_t>(t)] = p + 1 - ratio;  // the rotation's position's (the reference's 372's)
      } else {
        for (int c = 0; c < 512; ++c) lat[static_cast<size_t>(c)] = 0;
        entries_out[static_cast<size_t>(t)] = -1;
        ent_pos_out[static_cast<size_t>(t)] = -1;
      }
    }
  }
}

// The independent double-precision tail update (the cross-check's the
// fp32's the accumulation's the different precision path's): the same's
// algorithm's (the per-token's write's the APE's the publish's the pool's
// the shift's) the double's the interior's (the fp32's the 1e-7's the
// different's), the same's bf16 rounding's points's (the .to(bf16)'s the
// pool's before's the norm's + the norm's out's the ONE's). `state` the
// double's [2][n_slots][W]'s, the `latent_out' the bf16's [512]'s (the
// double's the interior's the bf16's the out's the same's the fp32's the
// rounding's point's).
void tail_update_ref_d64(const std::vector<float>& comp_kv, const std::vector<float>& comp_score,
                         const std::vector<float>& ape, const std::vector<int32_t>& req_ids,
                         const std::vector<int64_t>& pos, const std::vector<int32_t>& req_spans,
                         int num_requests, const std::vector<uint16_t>& norm_w, double eps,
                         std::vector<double>& state, std::vector<uint16_t>& latent_out, int ratio,
                         int coff) {
  const int W = coff * 512;
  const int n_slots = coff * ratio;
  std::vector<double> norm_d(512);
  for (int c = 0; c < 512; ++c) norm_d[static_cast<size_t>(c)] = bf16_to_d64(norm_w[static_cast<size_t>(c)]);
  for (int q = 0; q < num_requests; ++q) {
    const int start = req_spans[2 * q], len = req_spans[2 * q + 1];
    if (len <= 0) continue;
    double* kv_state = state.data() + static_cast<size_t>(req_ids[start]) * 2 * n_slots * W;
    double* score_state = kv_state + static_cast<size_t>(n_slots) * W;
    for (int t = start; t < start + len; ++t) {
      const int64_t p = pos[static_cast<size_t>(t)];
      const float* kvt = comp_kv.data() + static_cast<size_t>(t) * W;
      const float* st = comp_score.data() + static_cast<size_t>(t) * W;
      uint16_t* lat = latent_out.data() + static_cast<size_t>(t) * 512;
      if (p < 0) {
        for (int c = 0; c < 512; ++c) lat[static_cast<size_t>(c)] = 0;
        continue;
      }
      const float* arow = ape.data() + static_cast<size_t>(p % ratio) * W;
      const int slot = (coff == 2) ? (ratio + static_cast<int>(p % ratio)) : static_cast<int>(p % ratio);
      for (int c = 0; c < W; ++c) {
        kv_state[static_cast<size_t>(slot) * W + c] = static_cast<double>(kvt[c]);
        score_state[static_cast<size_t>(slot) * W + c] = static_cast<double>(st[c]) + static_cast<double>(arow[c]);
      }
      if ((p + 1) % ratio == 0) {  // the EVERY ratio-th token's publish
        std::vector<double> pooled(512);
        double ss = 0.0;
        for (int d = 0; d < 512; ++d) {
          double M = -std::numeric_limits<double>::infinity();
          for (int e = 0; e < n_slots; ++e) {
            const int off = (coff == 2 && e >= ratio) ? 512 + d : d;  // the head_off's the gather's
            M = std::max(M, score_state[static_cast<size_t>(e) * W + off]);
          }
          double L = 0.0, A = 0.0;
          for (int e = 0; e < n_slots; ++e) {
            const int off = (coff == 2 && e >= ratio) ? 512 + d : d;
            const double w = std::exp(score_state[static_cast<size_t>(e) * W + off] - M);
            L += w;
            A += w * kv_state[static_cast<size_t>(e) * W + off];
          }
          // The ONE bf16 rounding's (the .to(bf16)'s the pool's before's
          // the norm's the double's the exact's the float's the rounding's
          // the mirror's).
          pooled[static_cast<size_t>(d)] = bf16_to_d64(float_to_bf16_bits(static_cast<float>(A / L)));
          ss += pooled[static_cast<size_t>(d)] * pooled[static_cast<size_t>(d)];
        }
        const double rs = std::sqrt(1.0 / (ss / 512.0 + eps));
        for (int d = 0; d < 512; ++d) {
          const double v = norm_d[static_cast<size_t>(d)] * (pooled[static_cast<size_t>(d)] * rs);
          // The ONE bf16 rounding's (the norm's out's the bf16's the fp32's
          // the mirror's).
          lat[static_cast<size_t>(d)] = float_to_bf16_bits(static_cast<float>(v));
        }
        if (coff == 2) {  // the C4A's window's shift
          for (int c = 0; c < ratio * W; ++c) {
            kv_state[static_cast<size_t>(c)] = kv_state[static_cast<size_t>(c + ratio * W)];
            score_state[static_cast<size_t>(c)] = score_state[static_cast<size_t>(c + ratio * W)];
          }
        }
      } else {
        for (int c = 0; c < 512; ++c) lat[static_cast<size_t>(c)] = 0;
      }
    }
  }
}

// The fp32-vs-double's relative's budget check (the 1e-5's the fp32's the
// accumulation's the different's precision path's the documented's).
void check_within(const float got, const double want, const char* what, int c) {
  const double mag = std::max({std::fabs(static_cast<double>(got)), std::fabs(want), 1e-30});
  if (std::fabs(static_cast<double>(got) - want) > mag * 1e-5 + 1e-30)
    throw std::runtime_error(std::string(what) + " diverged from the double at c=" + std::to_string(c));
}

}  // namespace

// ---------------------------------------------------------------------------
DGPP_TEST(dsv4_compress_tail_c4a_layout_and_cadence) {
  // The C4A's (ratio 4's coff 2's W 1024's) the LAYOUT's + the CADENCE's
  // proof (the G-tail-cadence's the 2-token's the reference's ratio's
  // cadence's the closed's + the G-tail-pool's the 2-entry's pair's the
  // reference's 8-entry's x 512's the closed's): a 16-token's span (p =
  // 0..15's, the 4's publishes' p = 3, 7, 11, 15's the entries' 0, 1, 2,
  // 3's). The even's / the odd's parity's GONE's — the publish's the
  // EVERY 4th token's (the (p + 1) % 4 == 0's), the entry's p / 4's, the
  // ent_pos's p + 1 - 4's. The state's after the span's: the C4A's window's
  // shift's (the reference's 366-367's) the slots' 0..3's (the previous's
  // window's) the LAST publish's window's (p = 12..15's) kv/score's the
  // full's 1024's the bit-exact's, the slots' 4..7's (the current's
  // window's) the same's (the shift's copy's). The APE's on the score's
  // half's only's (the G-tail-ape's the closed's): the score's plane's the
  // GEMM's + the ape[p % 4]'s the bit-exact's, the kv's plane's the
  // untouched's GEMM's.
  const int ratio = 4, coff = 2, W = 1024;
  const int tokens = 16;
  std::vector<int32_t> req_ids(tokens, 0);
  std::vector<int64_t> pos(tokens);
  for (int t = 0; t < tokens; ++t) pos[static_cast<size_t>(t)] = t;
  std::vector<int32_t> req_spans = {0, tokens};
  std::vector<float> comp_kv, comp_score, ape;
  fill_f32(&comp_kv, static_cast<size_t>(tokens) * W, 0xCAFE);
  fill_f32(&comp_score, static_cast<size_t>(tokens) * W, 0xBEEF);
  fill_f32(&ape, static_cast<size_t>(ratio) * W, 0xA9E1);
  std::vector<uint16_t> norm_w(512, 0x3F80);  // the bf16 1.0's (the norm's the identity's)
  const int n_slots = coff * ratio;  // 8
  std::vector<float> state = make_state(1, n_slots, W);
  std::vector<uint16_t> latent_out(static_cast<size_t>(tokens) * 512, 0xFFFF);
  std::vector<int64_t> entries_out(tokens, -42), ent_pos_out(tokens, -42);
  tail_update_ref(comp_kv, comp_score, ape, req_ids, pos, req_spans, 1, norm_w, 1e-6f, state, latent_out,
                  entries_out, ent_pos_out, ratio, coff);
  // The CADENCE's: the entries' (-1, -1, -1, 0, -1, -1, -1, 1, -1, -1,
  // -1, 2, -1, -1, -1, 3)'s (the publish's p = 3, 7, 11, 15's the p / 4's),
  // the ent_pos's (-1, -1, -1, 0, -1, -1, -1, 4, -1, -1, -1, 8, -1, -1,
  // -1, 12)'s (the p + 1 - 4's).
  for (int t = 0; t < tokens; ++t) {
    const int64_t p = t;
    const bool publish = ((p + 1) % ratio == 0);
    const int64_t want_entry = publish ? (p / ratio) : -1;
    const int64_t want_entpos = publish ? (p + 1 - ratio) : -1;
    if (entries_out[static_cast<size_t>(t)] != want_entry)
      throw std::runtime_error("c4a cadence: the entry at t=" + std::to_string(t) + " is not " +
                               std::to_string(want_entry) + " (got " + std::to_string(entries_out[static_cast<size_t>(t)]) +
                               ")");
    if (ent_pos_out[static_cast<size_t>(t)] != want_entpos)
      throw std::runtime_error("c4a cadence: the ent_pos at t=" + std::to_string(t) + " is not " +
                               std::to_string(want_entpos));
    // The publish's row's latent's the normed's (the nonzero's the norm_w's
    // 1.0's the pooling's the finite's), the non-publish's zeroed's.
    bool nonzero = false;
    for (int c = 0; c < 512; c += 7)
      nonzero = nonzero || (bf16_bits_to_float(latent_out[static_cast<size_t>(t) * 512 + c]) != 0.0f);
    if (publish && !nonzero)
      throw std::runtime_error("c4a cadence: the publish's (t=" + std::to_string(t) + ") latent is all-zero");
    if (!publish)
      for (int c = 0; c < 512; c += 11)
        if (latent_out[static_cast<size_t>(t) * 512 + c] != 0)
          throw std::runtime_error("c4a cadence: the non-publish's (t=" + std::to_string(t) + ") latent is not zeroed");
  }
  // The state's after the span's: the C4A's window's shift's the slots'
  // 0..3's (the previous's window's) the LAST publish's window's (p = 12..
  // 15's) the bit-exact's, the slots' 4..7's (the current's window's) the
  // same's. The kv's plane's the GEMM's (the p = 12..15's), the score's
  // plane's the GEMM's + the ape[p % 4]'s (the APE's the score's half's
  // only's the G-tail-ape's the closed's).
  const size_t kv_base = 0;  // the req 0's
  const size_t sc_base = static_cast<size_t>(n_slots) * W;  // the score's plane's offset
  for (int e = 0; e < 4; ++e) {  // the slots' 0..7's (the 0..3's the 4..7's the same's after the shift's)
    const int p = 12 + e;  // the last publish's window's (p = 12..15's)
    for (int c = 0; c < W; c += 5) {
      const float want_kv = comp_kv[static_cast<size_t>(p) * W + c];
      const float want_sc = comp_score[static_cast<size_t>(p) * W + c] + ape[static_cast<size_t>(p % ratio) * W + c];
      if (state[static_cast<size_t>(kv_base + e * W + c)] != want_kv)
        throw std::runtime_error("c4a layout: the kv plane's slot " + std::to_string(e) + " c=" + std::to_string(c) +
                                 " is not the last window's (p=" + std::to_string(p) + ") kv");
      if (state[static_cast<size_t>(sc_base + e * W + c)] != want_sc)
        throw std::runtime_error("c4a layout: the score plane's slot " + std::to_string(e) + " c=" + std::to_string(c) +
                                 " is not the last window's (p=" + std::to_string(p) + ") score + ape[" +
                                 std::to_string(p % ratio) + "]");
    }
  }
}

DGPP_TEST(dsv4_compress_tail_pool_independent) {
  // The (coff * ratio)-entry's x 512's pool + the one-rounding's RMSNorm's
  // numerics (the reference's 356-361's the torch.cat's the plane-split's
  // gather's + the softmax's weighted sum's + the norm's the kv.to(dtype)'s
  // the G-tail-pool's the closed's): cross-check the fp32 reference's tail
  // update against an INDEPENDENT double-precision's (the same's
  // algorithm's the different's precision path's the fp32-vs-double's the
  // 1e-5's the documented budget's) on a small synthetic shape (the C4A's
  // ratio 4's coff 2's W 1024's, the 8-entry's the 512-dim's probed).
  const int ratio = 4, coff = 2, W = 1024;
  const int tokens = 8;  // the 2's publishes' (p = 3, 7's)
  std::vector<int32_t> req_ids(tokens, 0);
  std::vector<int64_t> pos(tokens);
  for (int t = 0; t < tokens; ++t) pos[static_cast<size_t>(t)] = t;
  std::vector<int32_t> req_spans = {0, tokens};
  std::vector<float> comp_kv, comp_score, ape;
  fill_f32(&comp_kv, static_cast<size_t>(tokens) * W, 0x11);
  fill_f32(&comp_score, static_cast<size_t>(tokens) * W, 0x22);
  fill_f32(&ape, static_cast<size_t>(ratio) * W, 0x33);
  std::vector<uint16_t> norm_w;
  fill_bf16(&norm_w, 512, 0x44);
  const float eps = 1e-6f;
  const int n_slots = coff * ratio;
  std::vector<float> state = make_state(1, n_slots, W);
  std::vector<uint16_t> latent_out(static_cast<size_t>(tokens) * 512, 0);
  std::vector<int64_t> entries_out(tokens, -42), ent_pos_out(tokens, -42);
  tail_update_ref(comp_kv, comp_score, ape, req_ids, pos, req_spans, 1, norm_w, eps, state, latent_out,
                  entries_out, ent_pos_out, ratio, coff);
  // The independent double-precision's (the same's input's the different's
  // precision path's).
  std::vector<double> state_d = make_state_d64(1, n_slots, W);
  std::vector<uint16_t> latent_out_d(static_cast<size_t>(tokens) * 512, 0);
  tail_update_ref_d64(comp_kv, comp_score, ape, req_ids, pos, req_spans, 1, norm_w, static_cast<double>(eps),
                      state_d, latent_out_d, ratio, coff);
  // The publish's rows' (p = 3, 7's) latent's the fp32's vs the double's
  // (the 512-dim's the probed's the 256's the half's the 1e-5's relative's
  // budget's).
  for (const int t : {3, 7})
    for (int c = 0; c < 512; c += 2)
      check_within(bf16_bits_to_float(latent_out[static_cast<size_t>(t) * 512 + c]),
                   bf16_to_d64(latent_out_d[static_cast<size_t>(t) * 512 + c]), "c4a pool", c);
  // The finite's probe (the pooling's + the norm's the no-NaN's).
  for (int c = 0; c < 512; c += 7) {
    const float v = bf16_bits_to_float(latent_out[static_cast<size_t>(7) * 512 + c]);
    if (!std::isfinite(v))
      throw std::runtime_error("c4a pool: a non-finite output at c=" + std::to_string(c));
  }
}

DGPP_TEST(dsv4_compress_tail_c128a_gated_pool) {
  // The C128A's (ratio 128's coff 1's W 512's) the 128-entry's gated pool's
  // (the G-c128a-compressor's the closed's the C++'s per-token's plain's
  // the reference's gated's pool's the replaced's): a 128-token's span (p =
  // 0..127's) the publish's the p = 127's only's (the (127 + 1) % 128 ==
  // 0's the entry's 0's the ent_pos's 0's), the 128-entry's x 512's pool's
  // the reference's 362-365's the plain's branch's. Cross-check the fp32's
  // pool's against an INDEPENDENT double-precision's (the 128-entry's the
  // 512-dim's the probed's the 1e-5's relative's budget's). The APE's on
  // the score's half's only's (the p % 128's the 128's the ring's the full's
  // coverage's the p = 0..127's the ape[0..127]'s).
  const int ratio = 128, coff = 1, W = 512;
  const int tokens = 128;
  std::vector<int32_t> req_ids(tokens, 0);
  std::vector<int64_t> pos(tokens);
  for (int t = 0; t < tokens; ++t) pos[static_cast<size_t>(t)] = t;
  std::vector<int32_t> req_spans = {0, tokens};
  std::vector<float> comp_kv, comp_score, ape;
  fill_f32(&comp_kv, static_cast<size_t>(tokens) * W, 0x55);
  fill_f32(&comp_score, static_cast<size_t>(tokens) * W, 0x66);
  fill_f32(&ape, static_cast<size_t>(ratio) * W, 0x77);
  std::vector<uint16_t> norm_w;
  fill_bf16(&norm_w, 512, 0x88);
  const int n_slots = coff * ratio;  // 128
  std::vector<float> state = make_state(1, n_slots, W);
  std::vector<uint16_t> latent_out(static_cast<size_t>(tokens) * 512, 0xFFFF);
  std::vector<int64_t> entries_out(tokens, -42), ent_pos_out(tokens, -42);
  tail_update_ref(comp_kv, comp_score, ape, req_ids, pos, req_spans, 1, norm_w, 1e-6f, state, latent_out,
                  entries_out, ent_pos_out, ratio, coff);
  // The CADENCE's: the publish's the p = 127's only's (the entry's 0's the
  // ent_pos's 0's), the other's 127's rows' the -1's.
  for (int t = 0; t < tokens; ++t) {
    const bool publish = (t == 127);
    if (entries_out[static_cast<size_t>(t)] != (publish ? 0 : -1))
      throw std::runtime_error("c128a cadence: the entry at t=" + std::to_string(t) + " is wrong");
    if (ent_pos_out[static_cast<size_t>(t)] != (publish ? 0 : -1))
      throw std::runtime_error("c128a cadence: the ent_pos at t=" + std::to_string(t) + " is wrong");
  }
  // The pool's the fp32's vs the double's (the 512-dim's the probed's the
  // 256's the half's the 1e-5's relative's budget's the norm_w's the bf16's
  // the exact's upcast's the double's the same's the fp32's).
  std::vector<double> state_d = make_state_d64(1, n_slots, W);
  std::vector<uint16_t> latent_out_d(static_cast<size_t>(tokens) * 512, 0);
  tail_update_ref_d64(comp_kv, comp_score, ape, req_ids, pos, req_spans, 1, norm_w, 1e-6, state_d, latent_out_d,
                      ratio, coff);
  for (int c = 0; c < 512; c += 2)
    check_within(bf16_bits_to_float(latent_out[static_cast<size_t>(127) * 512 + c]),
                 bf16_to_d64(latent_out_d[static_cast<size_t>(127) * 512 + c]), "c128a pool", c);
  // The APE's the score's half's only's (the G-tail-ape's the closed's):
  // the score's plane's slot p's the p % 128's = p's (the 128's the full's
  // coverage's) the comp_score[p]'s + the ape[p]'s the bit-exact's, the kv's
  // plane's the untouched's GEMM's.
  const size_t sc_base = static_cast<size_t>(n_slots) * W;
  for (const int p : {0, 1, 64, 127}) {
    for (int c = 0; c < W; c += 5) {
      const float want_kv = comp_kv[static_cast<size_t>(p) * W + c];
      const float want_sc = comp_score[static_cast<size_t>(p) * W + c] + ape[static_cast<size_t>(p) * W + c];
      if (state[static_cast<size_t>(p * W + c)] != want_kv)
        throw std::runtime_error("c128a ape: the kv plane's slot " + std::to_string(p) + " c=" + std::to_string(c) +
                                 " is not the GEMM's");
      if (state[static_cast<size_t>(sc_base + p * W + c)] != want_sc)
        throw std::runtime_error("c128a ape: the score plane's slot " + std::to_string(p) + " c=" + std::to_string(c) +
                                 " is not GEMM + ape[" + std::to_string(p) + "]");
    }
  }
}

DGPP_TEST(dsv4_compress_tail_padding_row) {
  // The padding's row's (pos < 0's) the no-op's (the latent's zeroed's +
  // the entry's -1's, the state's untouched's): a 4-row's (the real's p =
  // 0, 1, 2's + the padding's p = -1's) small synthetic shape (the C4A's
  // W 1024's). The state's the real's tokens' kv/score's (the untouched's
  // by the padding's).
  const int ratio = 4, coff = 2, W = 1024;
  const int tokens = 4;
  std::vector<int32_t> req_ids = {0, 0, 0, 0};
  std::vector<int64_t> pos = {0, 1, 2, -1};  // the 3's real's + the padding's
  std::vector<int32_t> req_spans = {0, tokens};
  std::vector<float> comp_kv, comp_score, ape;
  fill_f32(&comp_kv, static_cast<size_t>(tokens) * W, 0x3333);
  fill_f32(&comp_score, static_cast<size_t>(tokens) * W, 0x4444);
  fill_f32(&ape, static_cast<size_t>(ratio) * W, 0x5555);
  std::vector<uint16_t> norm_w(512, 0x3F80);
  const int n_slots = coff * ratio;
  std::vector<float> state = make_state(1, n_slots, W);
  std::vector<uint16_t> latent_out(static_cast<size_t>(tokens) * 512, 0xFFFF);
  std::vector<int64_t> entries_out(tokens, -42), ent_pos_out(tokens, -42);
  tail_update_ref(comp_kv, comp_score, ape, req_ids, pos, req_spans, 1, norm_w, 1e-6f, state, latent_out,
                  entries_out, ent_pos_out, ratio, coff);
  // The real's tokens' (p = 0, 1, 2's) the state's write's (the no
  // publish's the (2 + 1) % 4 != 0's): the slots' 4..6's (the C4A's the
  // ratio's offset's) the kv/score's the GEMM's + the APE's.
  const size_t sc_base = static_cast<size_t>(n_slots) * W;
  for (const int p : {0, 1, 2}) {
    const int slot = ratio + p;  // the C4A's the second's plane's
    for (int c = 0; c < W; c += 5) {
      if (state[static_cast<size_t>(slot * W + c)] != comp_kv[static_cast<size_t>(p) * W + c])
        throw std::runtime_error("tail padding: the kv plane's slot " + std::to_string(slot) + " c=" +
                                 std::to_string(c) + " is not the GEMM's");
      if (state[static_cast<size_t>(sc_base + slot * W + c)] !=
          comp_score[static_cast<size_t>(p) * W + c] + ape[static_cast<size_t>(p % ratio) * W + c])
        throw std::runtime_error("tail padding: the score plane's slot " + std::to_string(slot) + " c=" +
                                 std::to_string(c) + " is not GEMM + ape[" + std::to_string(p % ratio) + "]");
    }
  }
  // The padding's (p = -1's) no-op's: the latent's[3]'s zeroed's + the
  // entry's[3]'s -1's, the state's untouched's (the no's slot 4 + 3's the
  // p = 3's the write's).
  for (int c = 0; c < 512; c += 5)
    if (latent_out[static_cast<size_t>(3) * 512 + c] != 0)
      throw std::runtime_error("tail padding: the padding's latent is not zeroed at c=" + std::to_string(c));
  for (int t = 0; t < tokens; ++t)
    if (entries_out[static_cast<size_t>(t)] != -1)
      throw std::runtime_error("tail padding: the entries are not all -1 (the no publish's) at t=" + std::to_string(t));
}

int main() { return ::dgpp::test::run_all(); }
