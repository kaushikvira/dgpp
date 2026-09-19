#pragma once
// The ratio-4 (C4A) overlapping compressor's tail-ring update of
// DeepSeek-V4-Flash (deepseek_v4) (2026-09-18, the dsv4 dspark + compressor
// wiring; docs/dsv4_kernel_port_spec.md §2.1(b)). The dsv41 ratio-2
// compressor's per-request tail (src/kernels/csa2.cu's
// csa2_compress_decode_update, the fp32 [2, kCsa2Latent] per-request ring —
// the pending even token's kv + gate score, the one state family with
// per-row snapshots) re-expressed for the V4 ratio-4 overlapping variant
// (coff 2, the dsv41 ratio-2/ratio-1 pair's one geometry swap): the wkv /
// wgate's pair + the per-request tail's update, the tail's ordinal + the
// wgate's plane.
//
// The per-request tail is fp32 [2, W] (W = the layer's compressor output
// width, coff * 512): the pending even token's kv (the first W) and gate
// score (the second W). The even positions stash their (kv, score) into the
// request's tail; the odd positions pool the tail with themselves into the
// normed latent (the pair-pooling + the one-rounding RMSNorm, the dsv41's
// pool_pair_and_norm's re-expression) and report the entry. The
// tail_snapshots (optional, speculative rows): after every row that is not
// its request's last, the tail as it stands — rolling back to `a` accepted
// rows copies row start + a - 1 over the tail.
//
// The CUDA kernel (compress.cu) is the graph-capturable decode form; its
// numerics are pinned on the CPU by the independent host reference in
// tests/unit/dsv4_compress_tail_test.cpp (the [2, W] layout on a small
// synthetic shape) before any forward path consumes the tail.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// The V4 compressor's tail-ring update (the ratio-4's overlapping's, the
// dsv41's csa2_compress_decode_update's V4 re-expression). `comp_kv` /
// `comp_score` are the wkv / wgate's GEMM outputs (fp32 [tokens, W] each —
// the caller's, the model's wkv / wgate bf16 weight's F32-out GEMM's);
// `norm_w` the compressor's bf16 [W] RMSNorm weight; `tails` the per-
// request tail's base (fp32 [max_requests][2][W] — the caller's, the
// model's d_tails_'s the tail's ordinal's plane's); `latent_out` the
// normed, unrotated's compressed latent's bf16 [tokens, W]; `entries_out`
// the int64 [tokens]'s entry's ordinal's (p / 2 on the odd's, -1 on the
// even's / the padding's); `ent_pos_out` (optional) the entry's rotation
// position's (entries * ratio, the csa2_scaled_positions's mul = ratio's
// re-expression); `tail_snapshots` (optional) the fp32 [tokens, 2, W]'s
// per-row's. `W` must be a multiple of the thread count (the kernel's 256
// threads' the W's the dsv41's kCsa2Latent's 512's re-expression's).
void dsv4_compress_tail_update(const float* comp_kv, const float* comp_score,
                               const int32_t* req_ids, const int64_t* pos,
                               const int32_t* req_spans, int num_requests,
                               const void* norm_w, float eps, float* tails,
                               void* latent_out, int64_t* entries_out,
                               int64_t* ent_pos_out, int tokens, int W,
                               int ratio, float* tail_snapshots,
                               cudaStream_t stream);

}  // namespace dgpp
