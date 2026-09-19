// The 0731 checkpoint's Compressor's per-request tail-ring update (the
// dsv4 compressor's publish's the reference's form's; docs/dsv4_attention_
// spec.md §5, the G-tail-cadence / G-tail-pool / G-tail-ape / G-c128a-
// compressor's gaps' the closed's): the reference's arithmetic (the
// checkpoint's inference/model.py's Compressor, 285-400's) re-expressed
// as one graph-capturable CTA per request span — the dsv41's
// csa2_compress_decode_update's (the even-stash / odd-pool's the ratio-2's
// pair's) the V4 re-expression's the ratio's the reference's cadence's
// re-parameterized's (the publish's the EVERY ratio-th token's, the
// (coff * ratio)-entry's x 512's pool's the coff x ratio's gather's, the
// APE's on the score's half's, the C4A's window's shift's). The decode's
// hot path (the graph-capturable one): one CTA per request span, the 256
// threads' the W's the dsv41's kCsa2Threads' form's. The prefill's chunks'
// the same's path's (the reference's prefill's the per-token's
// incremental's the equivalent's the state's the same's the 332-348's).
// The numerics are pinned on the CPU by the independent host reference
// (tests/unit/dsv4_compress_tail_test.cpp's the [2][coff * ratio][W]'s
// small synthetic shape's the fp32-vs-double's the documented budget's)
// before any forward path consumes the tail.

#include "models/dsv4/compress.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {
constexpr int kThreads = 256;  // the dsv41's kCsa2Threads' (the block_sum's 8 warps' the 256's)
constexpr float kNegInf = -std::numeric_limits<float>::infinity();  // the fp32's -inf's (the state's init's the -inf's the softmax's the 0's weight's)
// The shared 8-float reduction (the dsv41's block_sum_256's re-expression).
__device__ __forceinline__ float dsv4_block_sum_256(float v, float* red) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float t = 0.0f;
  if (warp == 0) {
    t = lane < (int(blockDim.x) >> 5) ? red[lane] : 0.0f;
    for (int off = 4; off > 0; off >>= 1) t += __shfl_xor_sync(0xffffffffu, t, off);
    if (lane == 0) red[0] = t;
  }
  __syncthreads();
  t = red[0];
  __syncthreads();
  return t;
}
// The (coff * ratio)-entry's x 512-dim's pool + the one-rounding's RMSNorm
// (the reference's the 356-361's the torch.cat's the plane-split's gather's
// + the score_state.softmax(dim=1)'s weighted sum's + the norm's the
// kv.to(dtype)'s re-expression's, parameterized in the n_slots' the
// entries' count's + the head_off' the C4A's second's plane's): the per-
// dim's max-shift's softmax's (the fp32's, the -inf's rows' the 0's
// weight's the reference's the -inf's init's the cold start's), the
// weighted's kv's sum's the ONE bf16 rounding's (the reference's the
// kv.to(dtype)'s the norm's before's), the RMSNorm's (the fp32's interior's,
// the norm_w's bf16's exact upcast's, the ONE bf16 rounding's). `out` bf16
// [kCsa2Latent].
__device__ __forceinline__ void dsv4_pool_entries_and_norm(const float* kv_state, const float* score_state,
                                                           int n_slots, int ratio, int coff, int W,
                                                           const uint16_t* norm_w, float eps, uint16_t* out,
                                                           float* red) {
  const int kPer = kCsa2Latent / blockDim.x;  // the 512-dim's the pool's output's (the 2's per thread's the 256's)
  const int plane = W / 2;  // the C4A's head_off's (the coff x ratio's gather's the second's plane's the dims' 512..1023's)
  float pooled[4];  // the kPer's register's (the kCsa2Latent's 512's the kPer's 2's, the launcher's guard's)
  float ss = 0.0f;
  for (int j = 0; j < kPer; ++j) {
    const int d = threadIdx.x + j * blockDim.x;
    // The two-pass max-shift's softmax over the n_slots' entries' scores'
    // (the reference's the score_state.softmax(dim=1)'s, the -inf's rows'
    // the exp(-inf - M)'s 0's the masked's weight's 0's).
    float M = kNegInf;
    for (int e = 0; e < n_slots; ++e) {
      const int off = (coff == 2 && e >= ratio) ? plane + d : d;  // the head_off's the gather's (the C4A's the e >= ratio's the second's plane's)
      M = fmaxf(M, score_state[int64_t(e) * W + off]);
    }
    float L = 0.0f, A = 0.0f;
    for (int e = 0; e < n_slots; ++e) {
      const int off = (coff == 2 && e >= ratio) ? plane + d : d;
      const float w = expf(score_state[int64_t(e) * W + off] - M);
      L = __fadd_rn(L, w);
      A = __fmaf_rn(w, kv_state[int64_t(e) * W + off], A);
    }
    const float v = bf16_bits_to_float(float_to_bf16_bits(__fdiv_rn(A, L)));  // .to(bf16) before the norm
    pooled[j] = v;
    ss = __fmaf_rn(v, v, ss);
  }
  const float total = dsv4_block_sum_256(ss, red);
  const float rs = rsqrtf(__fadd_rn(__fdiv_rn(total, float(kCsa2Latent)), eps));
  for (int j = 0; j < kPer; ++j) {
    const int c = threadIdx.x + j * blockDim.x;
    const float v = __fmul_rn(pooled[j], rs);
    out[c] = float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(norm_w[c]), v));
  }
}
// The tail's update kernel (the reference's decode's form's, the ratio's
// parameterized's): one CTA per request span's. Every row's the state's
// write's (the kv_state / score_state's slot(p)'s the APE's the score's
// half's only's the p % ratio's the reference's 351's); the EVERY ratio-th
// row's (the (p + 1) % ratio == 0's the reference's 350's) the pool's the
// normed latent's + the entry's p / ratio's the ent_pos's p + 1 - ratio's
// the C4A's the window's shift's (the reference's 366-367's); the other's
// rows' zero the latent's + the -1's entry's. The tail_snapshots' (the
// optional's the speculative rows's) after every row's that is not its
// request's last's, the tail's as it stands's.
extern "C" __global__ void dsv4_compress_tail_update_kernel(const float* __restrict__ comp_kv,
                                                            const float* __restrict__ comp_score,
                                                            const float* __restrict__ ape,
                                                            const int32_t* __restrict__ req_ids,
                                                            const int64_t* __restrict__ pos,
                                                            const int32_t* __restrict__ req_spans,
                                                            const uint16_t* __restrict__ norm_w,
                                                            float eps, float* __restrict__ tails,
                                                            uint16_t* __restrict__ latent_out,
                                                            int64_t* __restrict__ entries_out,
                                                            int64_t* __restrict__ ent_pos_out, int W, int ratio,
                                                            int coff, float* __restrict__ tail_snapshots) {
  __shared__ float red[8];
  const int q = blockIdx.x;
  const int start = req_spans[2 * q], len = req_spans[2 * q + 1];
  if (len <= 0) return;
  const int n_slots = coff * ratio;  // the state's row count (the C4A's 8's, the C128A's 128's)
  const int64_t state_row = int64_t(2) * n_slots * W;  // the per-request's fp32 state's (the kv's + the score's the planes's)
  float* kv_state = tails + int64_t(req_ids[start]) * state_row;
  float* score_state = kv_state + int64_t(n_slots) * W;
  for (int t = start; t < start + len; ++t) {
    const int64_t p = pos[t];
    const float* kvt = comp_kv + int64_t(t) * W;
    const float* st = comp_score + int64_t(t) * W;
    uint16_t* lat = latent_out + int64_t(t) * kCsa2Latent;
    if (p < 0) {
      for (int c = threadIdx.x; c < kCsa2Latent; c += blockDim.x) lat[c] = 0;
      if (threadIdx.x == 0) {
        entries_out[t] = -1;
        if (ent_pos_out != nullptr) ent_pos_out[t] = -1;
      }
      // The padding's touches nothing else's (the state's untouched's, the
      // snapshot's the state's as it stands's).
      if (tail_snapshots != nullptr && t != start + len - 1) {
        float* snap = tail_snapshots + int64_t(t) * state_row;
        for (int c = threadIdx.x; c < state_row; c += blockDim.x) snap[c] = kv_state[c];
      }
      __syncthreads();
      continue;
    }
    // The state's write (the reference's 352-354's): the slot's the p %
    // ratio's (the C4A's the ratio's offset's the second's plane's the
    // reference's 353's), the APE's on the score's half's only's (the
    // reference's 351's the p % ratio's the APE's row's the p's the 128's
    // the C128A's the ring's the same's index's).
    const float* arow = ape + int64_t(p % ratio) * W;
    const int slot = (coff == 2) ? (ratio + int(p % ratio)) : int(p % ratio);
    for (int c = threadIdx.x; c < W; c += blockDim.x) {
      kv_state[int64_t(slot) * W + c] = kvt[c];
      score_state[int64_t(slot) * W + c] = __fadd_rn(st[c], arow[c]);
    }
    __syncthreads();  // the state's write's complete's (the pool's reads the state's)
    if ((p + 1) % ratio == 0) {  // the EVERY ratio-th token's publish (the reference's 350's)
      dsv4_pool_entries_and_norm(kv_state, score_state, n_slots, ratio, coff, W, norm_w, eps, lat, red);
      if (coff == 2) {
        // The C4A's window's shift (the reference's 366-367's the pool's
        // after's): the first's plane's (the 4's overlap's rows's) the
        // second's plane's (the 4's normal's rows's) the copy's.
        __syncthreads();  // the pool's reads's complete's (the shift's overwrites's the first's plane's)
        for (int64_t c = threadIdx.x; c < int64_t(ratio) * W; c += blockDim.x) {
          kv_state[c] = kv_state[c + int64_t(ratio) * W];
          score_state[c] = score_state[c + int64_t(ratio) * W];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        entries_out[t] = p / ratio;  // the entry's ordinal's (the reference's 379's the start_pos // ratio's)
        if (ent_pos_out != nullptr) ent_pos_out[t] = p + 1 - ratio;  // the rotation's position's (the reference's 372's)
      }
    } else {
      for (int c = threadIdx.x; c < kCsa2Latent; c += blockDim.x) lat[c] = 0;
      if (threadIdx.x == 0) {
        entries_out[t] = -1;
        if (ent_pos_out != nullptr) ent_pos_out[t] = -1;
      }
    }
    if (tail_snapshots != nullptr && t != start + len - 1) {
      __syncthreads();
      float* snap = tail_snapshots + int64_t(t) * state_row;
      for (int c = threadIdx.x; c < state_row; c += blockDim.x) snap[c] = kv_state[c];
    }
    __syncthreads();
  }
}
// The state's init kernel (the reference's the kv_state's zero's + the
// score_state's the -inf's the 309-310's): one block's a request's, the
// 256's threads' the state_row's the stride's (the kv's plane's the first
// n_kv's the zeroed's, the score's the next n_score's the -inf's the
// cold start's the publish's the 0's weight's the -inf's rows's).
extern "C" __global__ void dsv4_compress_tail_init_kernel(float* __restrict__ state, int64_t n_kv,
                                                          int64_t n_score) {
  const int64_t row = int64_t(blockIdx.x) * (n_kv + n_score);
  for (int64_t c = threadIdx.x; c < n_kv + n_score; c += blockDim.x)
    state[row + c] = (c < n_kv) ? 0.0f : kNegInf;
}
}  // namespace

void dsv4_compress_tail_update(const float* comp_kv, const float* comp_score, const float* ape,
                               const int32_t* req_ids, const int64_t* pos, const int32_t* req_spans,
                               int num_requests, const void* norm_w, float eps, float* tails,
                               void* latent_out, int64_t* entries_out, int64_t* ent_pos_out, int tokens,
                               int ratio, int coff, float* tail_snapshots, cudaStream_t stream) {
  if (num_requests <= 0 || tokens <= 0) return;
  if (ratio <= 0 || (ratio != 4 && ratio != 128))
    throw std::invalid_argument("dsv4 compress tail update: ratio must be 4 (C4A) or 128 (C128A)");
  if (coff != (ratio == 4 ? 2 : 1))
    throw std::invalid_argument("dsv4 compress tail update: coff must be 1 + (ratio == 4) (the reference's)");
  const int W = coff * kCsa2Latent;
  if (W % kThreads != 0 || W > 8 * kThreads)
    throw std::invalid_argument("dsv4 compress tail update: W must be in (0, 8 x " +
                                std::to_string(kThreads) + "] and a multiple of " + std::to_string(kThreads));
  dsv4_compress_tail_update_kernel<<<unsigned(num_requests), kThreads, 0, stream>>>(
      comp_kv, comp_score, ape, req_ids, pos, req_spans, static_cast<const uint16_t*>(norm_w), eps, tails,
      static_cast<uint16_t*>(latent_out), entries_out, ent_pos_out, W, ratio, coff, tail_snapshots);
  DGPP_CUDA_OK(cudaGetLastError());
}

void dsv4_compress_tail_init(float* state, int n_req, int64_t n_kv, int64_t n_score, cudaStream_t stream) {
  if (n_req <= 0 || n_kv <= 0 || n_score <= 0) return;
  if (n_kv != n_score)
    throw std::invalid_argument("dsv4 compress tail init: the kv's and the score's planes' the same's size's");
  dsv4_compress_tail_init_kernel<<<unsigned(n_req), kThreads, 0, stream>>>(state, n_kv, n_score);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
