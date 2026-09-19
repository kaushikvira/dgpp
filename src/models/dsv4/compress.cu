// The ratio-4 (C4A) overlapping compressor's tail-ring update (2026-09-18,
// the dsv4 dspark + compressor wiring; docs/dsv4_kernel_port_spec.md
// §2.1(b)): the dsv41 ratio-2 compressor's per-request tail (src/kernels/
// csa2.cu's csa2_compress_decode_update, the fp32 [2, kCsa2Latent] per-
// request ring) re-expressed for the V4 ratio-4 overlapping variant (coff
// 2, the dsv41 ratio-2/ratio-1 pair's one geometry swap) — the wkv /
// wgate's pair + the per-request tail's update, the tail's ordinal + the
// wgate's plane. The decode's hot path (the graph-capturable one): one CTA
// per request span, the 256 threads' the W's the dsv41's kCsa2Threads'
// form's. The numerics are pinned on the CPU by the independent host
// reference (tests/unit/dsv4_compress_tail_test.cpp's the [2, W] layout's
// small synthetic shape's) before any forward path consumes the tail.

#include "models/dsv4/compress.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

namespace dgpp {
namespace {
constexpr int kThreads = 256;  // the dsv41's kCsa2Threads' (the block_sum's 8 warps' the 256's)
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
// The pair-pooling + the one-rounding's RMSNorm (the dsv41's
// pool_pair_and_norm's re-expression, parameterized in W): the per-dim's
// softmax over (s0, s1) (the fp32's, the max-shift's), the weighted's kv
// sum's the ONE bf16 rounding's, the RMSNorm's (the fp32's interior's, the
// norm_w's bf16's exact upcast's, the ONE bf16 rounding's). `out` bf16 [W].
__device__ __forceinline__ void dsv4_pool_pair_and_norm(const float* kv0, const float* s0,
                                                        const float* kv1, const float* s1,
                                                        const uint16_t* norm_w, float eps,
                                                        uint16_t* out, int W, float* red) {
  const int kPer = W / blockDim.x;  // the W's the 256's (the dsv41's 512's 2's, the 1024's 4's)
  float pooled[8];  // the kPer's register's (the W's <= 2048's the kPer's <= 8's, the launcher's guard's)
  float ss = 0.0f;
  for (int j = 0; j < kPer; ++j) {
    const int c = threadIdx.x + j * blockDim.x;
    const float a = s0[c], b = s1[c];
    const float m = fmaxf(a, b);
    const float e0 = expf(a - m), e1 = expf(b - m);
    const float den = e0 + e1;
    const float w0 = e0 / den, w1 = e1 / den;
    const float sum = __fadd_rn(__fmul_rn(kv0[c], w0), __fmul_rn(kv1[c], w1));
    const float v = bf16_bits_to_float(float_to_bf16_bits(sum));  // .to(bf16) before the norm
    pooled[j] = v;
    ss = __fmaf_rn(v, v, ss);
  }
  const float total = dsv4_block_sum_256(ss, red);
  const float rs = rsqrtf(__fadd_rn(__fdiv_rn(total, float(W)), eps));
  for (int j = 0; j < kPer; ++j) {
    const int c = threadIdx.x + j * blockDim.x;
    const float v = __fmul_rn(pooled[j], rs);
    out[c] = float_to_bf16_bits(__fmul_rn(bf16_bits_to_float(norm_w[c]), v));
  }
}
// The tail's update kernel (the dsv41's compress_decode_update_kernel's V4
// re-expression, parameterized in W): one CTA per request span's, the even
// positions' stash their (kv, score) into the request's tail's (the fp32
// [2, W]'s the pending even's), the odd's pool the tail's with themselves
// (the normed latent's + the entry's p / 2's), the padding's (pos < 0's)
// zero the latent's + the -1's entry's. The tail_snapshots' (the optional's
// the speculative rows's) after every row's that is not its request's
// last's, the tail's as it stands's.
extern "C" __global__ void dsv4_compress_tail_update_kernel(const float* __restrict__ comp_kv,
                                                            const float* __restrict__ comp_score,
                                                            const int32_t* __restrict__ req_ids,
                                                            const int64_t* __restrict__ pos,
                                                            const int32_t* __restrict__ req_spans,
                                                            const uint16_t* __restrict__ norm_w,
                                                            float eps, float* __restrict__ tails,
                                                            uint16_t* __restrict__ latent_out,
                                                            int64_t* __restrict__ entries_out,
                                                            int64_t* __restrict__ ent_pos_out,
                                                            int W, int ratio,
                                                            float* __restrict__ tail_snapshots) {
  __shared__ float red[8];
  const int q = blockIdx.x;
  const int start = req_spans[2 * q], len = req_spans[2 * q + 1];
  if (len <= 0) return;
  float* tail = tails + int64_t(req_ids[start]) * 2 * W;
  for (int t = start; t < start + len; ++t) {
    const int64_t p = pos[t];
    const float* kvt = comp_kv + int64_t(t) * W;
    const float* st = comp_score + int64_t(t) * W;
    uint16_t* lat = latent_out + int64_t(t) * W;
    if (p < 0) {
      for (int c = threadIdx.x; c < W; c += blockDim.x) lat[c] = 0;
      if (threadIdx.x == 0) {
        entries_out[t] = -1;
        if (ent_pos_out != nullptr) ent_pos_out[t] = -1;
      }
      continue;
    }
    if ((p & 1) == 0) {  // the even's: the stash's (the kv's + the score's the tail's)
      for (int c = threadIdx.x; c < W; c += blockDim.x) {
        tail[c] = kvt[c];
        tail[W + c] = st[c];
        lat[c] = 0;
      }
      if (threadIdx.x == 0) {
        entries_out[t] = -1;
        if (ent_pos_out != nullptr) ent_pos_out[t] = -1;
      }
    } else {  // the odd's: the pool's (the tail's + the current's the normed latent's)
      __syncthreads();  // the tail's stash's (an earlier row of this block) is complete
      dsv4_pool_pair_and_norm(tail, tail + W, kvt, st, norm_w, eps, lat, W, red);
      if (threadIdx.x == 0) {
        entries_out[t] = p / 2;  // the entry's ordinal's (the dsv41's p / 2's)
        if (ent_pos_out != nullptr) ent_pos_out[t] = (p / 2) * ratio;  // the rotation's (the V4's mul = ratio's)
      }
    }
    if (tail_snapshots != nullptr && t != start + len - 1) {
      __syncthreads();
      float* snap = tail_snapshots + int64_t(t) * 2 * W;
      for (int c = threadIdx.x; c < 2 * W; c += blockDim.x) snap[c] = tail[c];
    }
    __syncthreads();
  }
}
}  // namespace

void dsv4_compress_tail_update(const float* comp_kv, const float* comp_score, const int32_t* req_ids,
                               const int64_t* pos, const int32_t* req_spans, int num_requests,
                               const void* norm_w, float eps, float* tails, void* latent_out,
                               int64_t* entries_out, int64_t* ent_pos_out, int tokens, int W,
                               int ratio, float* tail_snapshots, cudaStream_t stream) {
  if (num_requests <= 0 || tokens <= 0) return;
  if (W <= 0 || W % kThreads != 0 || W > 8 * kThreads)
    throw std::invalid_argument("dsv4 compress tail update: W must be in (0, 8 x " +
                                std::to_string(kThreads) + "] and a multiple of " + std::to_string(kThreads));
  if (ratio <= 0) throw std::invalid_argument("dsv4 compress tail update: ratio must be positive");
  dsv4_compress_tail_update_kernel<<<unsigned(num_requests), kThreads, 0, stream>>>(
      comp_kv, comp_score, req_ids, pos, req_spans, static_cast<const uint16_t*>(norm_w), eps, tails,
      static_cast<uint16_t*>(latent_out), entries_out, ent_pos_out, W, ratio, tail_snapshots);
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
