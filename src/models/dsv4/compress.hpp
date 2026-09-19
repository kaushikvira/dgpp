#pragma once
// The 0731 checkpoint's Compressor's per-request tail-ring update
// (docs/dsv4_attention_spec.md §5: the G-tail-cadence / G-tail-pool /
// G-tail-ape / G-c128a-compressor gaps' the closed's). The reference's
// arithmetic (the checkpoint's inference/model.py's Compressor, 285-400's)
// re-expressed as one graph-capturable CTA per request span:
//   * the per-token's state's write (the kv_state / score_state's the slot's,
//     the APE's on the score's half's only's, the reference's 351's);
//   * the EVERY ratio-th token's publish (the reference's 350's the
//     should_compress's (start_pos + 1) % ratio == 0's, the entry's ordinal's
//     the start_pos // ratio's 379's, the rotation's position's the
//     start_pos + 1 - ratio's 372's — the G-tail-cadence's the 2-token's
//     parity's the reference's ratio's cadence's the replaced's);
//   * the (coff * ratio)-entry's x 512-dim's pool (the coff x ratio's
//     gather's the reference's 356-358's the torch.cat's the
//     [kv_state[:, :ratio, :512], kv_state[:, ratio:, 512:]]'s the C4A's
//     plane-split's 8-entry's, the C128A's the 128-entry's the direct's,
//     the G-tail-pool's the 2-entry's x W's pair's the reference's 8-entry's
//     x 512's the replaced's) + the one-rounding's RMSNorm's;
//   * the C4A's (coff 2's) the window's shift's (the reference's 366-367's
//     the kv_state[:, :ratio]'s = the kv_state[:, ratio:]]'s the pool's
//     after's);
//   * the C128A's (ratio 128's coff 1's the G-c128a-compressor's) the
//     128-entry's gated pool's every 128 tokens's (the reference's 362-365's
//     the plain's branch's, the score_state's the -inf's init's 310's) —
//     the C++'s per-token's plain's wkv's + norm's the reference's gated's
//     pool's the replaced's.
//
// The per-request tail is fp32 [2][coff * ratio][coff * 512] (the
// checkpoint's kv_state / score_state's (b, coff * ratio, coff *
// head_dim)'s the kv's plane's the first coff * ratio * W's + the score's
// the second's): the C4A's (ratio 4's coff 2's) the 8 x 1024's (the 2
// overlapping windows' 4 tokens' the wkv / wgate's 1024-wide's the first's
// half's the overlap's plane's the second's the normal's), the C128A's
// (ratio 128's coff 1's) the 128 x 512's the 128-token's ring's. The
// state's init's the reference's (the kv_state's zero's + the score_state's
// the -inf's the torch.zeros's / torch.full(-inf)'s the 309-310's). Every
// row that is not its request's last snapshots the tail (the tail_snapshots'
// the optional's the speculative rows' the DSpark's verify's rollback's);
// the padding's (pos < 0's) touches nothing's (the latent's zeroed's + the
// -1's entry's).
//
// The CUDA kernel (compress.cu) is the graph-capturable decode form (the
// prefill's chunks' the same's path's — the reference's prefill's the
// per-token's incremental's the equivalent's the state's the same's); its
// numerics are pinned on the CPU by the independent host reference in
// tests/unit/dsv4_compress_tail_test.cpp (the [2][coff * ratio][W]'s
// layout on a small synthetic shape's the fp32-vs-double's the documented
// budget's) before any forward path consumes the tail.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "kernels/csa2.hpp"

namespace dgpp {

// The V4 compressor's tail-ring update (the 0731's reference's form's, the
// ratio-parameterized's the C4A's ratio 4's coff 2's + the C128A's ratio
// 128's coff 1's). `comp_kv` / `comp_score` are the wkv / wgate's GEMM's
// F32 outputs' (fp32 [tokens, W] each — the reference's the x.float()'s
// compression's the fp32's, the caller's the model's wkv / wgate's bf16
// weight's F32-out GEMM's); `ape` the compressor's fp32 [ratio, W]'s APE
// (the checkpoint's the f32 [ratio, coff * head_dim]'s census's, the
// reference's 303's); `norm_w` the compressor's bf16 [512] RMSNorm's
// weight's (the head_dim's the pool's output's width's); `tails` the per-
// request tail's base (fp32 [max_requests][2][coff * ratio][W] — the
// caller's, the model's d_tails_'s the tail's ordinal's plane's);
// `latent_out` the pooled, normed, unrotated's compressed latent's bf16
// [tokens, kCsa2Latent] (the 512-dim's the pool's output's the reference's
// the head_dim's); `entries_out` the int64 [tokens]'s entry's ordinal's
// (the p / ratio's the publish's rows's, -1 on the non-publish's / the
// padding's); `ent_pos_out` (optional) the entry's rotation's position's
// (the p + 1 - ratio's the reference's 372's, -1's the non-publish's);
// `tail_snapshots` (optional) the fp32 [tokens, 2, coff * ratio, W]'s
// per-row's. `W` must be coff * kCsa2Latent (the C4A's 1024's, the
// C128A's 512's) and a multiple of the thread count (the kernel's 256
// threads' the W's the dsv41's kCsa2Threads' form's).
void dsv4_compress_tail_update(const float* comp_kv, const float* comp_score, const float* ape,
                               const int32_t* req_ids, const int64_t* pos, const int32_t* req_spans,
                               int num_requests, const void* norm_w, float eps, float* tails,
                               void* latent_out, int64_t* entries_out, int64_t* ent_pos_out,
                               int tokens, int ratio, int coff, float* tail_snapshots,
                               cudaStream_t stream);

// The per-request's state's init (the reference's the kv_state's zero's +
// the score_state's the -inf's, the 309-310's): `state` the fp32
// [n_req][2][n_slots][W]'s the per-request's the kv's plane's the first
// n_slots * W's + the score's the next's (n_kv = n_score = n_slots * W's).
// One block's a request's (the grid's n_req's), the 256's threads' the
// state_row's the stride's.
void dsv4_compress_tail_init(float* state, int n_req, int64_t n_kv, int64_t n_score,
                             cudaStream_t stream);

}  // namespace dgpp
