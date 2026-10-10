// QSA sparse attention prefill, one warp per (query, KV group) -- the shape of
// SGLang's sparse GQA prefill kernel (BLOCK_N 16, one warp per program) on
// GB10's tensor cores. Default for long prefill walks; DGPP_QSA_WARP=0 falls
// back to the partial kernels (models/qwen/layers.cpp).
//
// The whole online softmax stays in one warp's registers: Q as m16
// A-fragments (the group's <= 16 query heads are the M rows), K/V tiles of 16
// selected tokens gathered into shared memory with cp.async (L1-allocating),
// S = Q K^T and O += P V on mma.sync m16n8k16, P reused from S's accumulator
// layout as the A-fragment (FlashAttention-2), no block barrier. It writes the
// normalized fp32 output straight into the caller's c_out -- no m/l/c
// partials, no combine pass. Lanes 0-15 resolve a tile's 16 physical rows
// (token -> block table) in one round; a serial chain starved the copies.
// One stage by default: ncu showed the two-stage form at 2 blocks/SM
// (shared-memory bound); one stage gives 5 warps/SM and residency hides the
// gather (DGPP_QSA_WARP_STAGES=2 at build time keeps the double buffer).
//
// Numerics: probabilities rounded to bf16 for P V, the denominator summing
// the unrounded values (the partial kernels' rule); the dots' summation order
// is the tensor cores'. Tolerance-equal to qsa_attn_prefill_partial +
// dsa_attn_combine (qsa_test), not bitwise -- the same class as SGLang's
// kernel, which also rounds P to bf16.
//
// FP8-MMA path (docs/qwen_fp8_mma_plan.md, Phase B): when the fp8 pool is
// active (k_scale != nullptr) and DGPP_QSA_FP8_MMA is on (default), the stored
// E4M3 K/V codes are fed straight to the tensor cores (mma m16n8k32 e4m3) with
// no bf16 dequant: Q is quantized per-row (scale alpha), the score epilogue
// applies alpha*beta^k (rank-1, plan §2.2), and the V-weighted probability
// P*beta^v is quantized per-row (power-of-two scale gamma, §2.3) for the PV
// mma, whose epilogue applies gamma. l still sums the unrounded P. The path is
// tolerance-equal (the plan §3 band, l2_rel <= 0.10) to the bf16 warp path,
// not bitwise. DGPP_QSA_FP8_MMA=0 keeps the Phase A in-kernel dequant.
#include "kernels/qsa.hpp"

#include <cuda_fp8.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/latent_format.hpp"

namespace dgpp {
namespace {

constexpr int kD = 256;
constexpr int kTileTok = 16;
constexpr int kRow = kD + 8;  // padded bf16 row: conflict-free ldmatrix
constexpr int kStageElems = 2 * kTileTok * kRow;  // K then V
#ifndef DGPP_QSA_WARP_STAGES
#define DGPP_QSA_WARP_STAGES 1
#endif
constexpr int kStages = DGPP_QSA_WARP_STAGES;
constexpr size_t kSmem = size_t(kStages) * kStageElems * 2;  // bytes
// The E4M3 tiles: raw 1 B/elem codes, no row padding (8-byte cp.async and the
// 4-byte B-fragment loads are aligned). K then V, 256 B/token each.
constexpr int kRow8 = kD;                 // bytes per e4m3 token row
constexpr size_t kStage8 = 2 * kTileTok * kRow8;  // bytes, one e4m3 stage

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const int bytes = valid ? 16 : 0;  // 0: zero-fill
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  return static_cast<uint32_t>(float_to_bf16_bits(lo)) | (static_cast<uint32_t>(float_to_bf16_bits(hi)) << 16);
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src, bool valid) {
  const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  const int bytes = valid ? 8 : 0;  // 0: zero-fill
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(d), "l"(src), "r"(bytes));
}
// The E4M3 mma (the fp8_gemm.cu / dsa.cu asset): m16n8k32, 4 e4m3 per A
// register, 4 e4m3 per B register.
__device__ __forceinline__ void mma_e4m3(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
               "{%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// The MXFP8 block-scaled mma (C.1a, docs/qwen_fp8_mxfp8_sf_layout.md): applies
// 2^(sfa-127) * 2^(sfb-127) to this k-block's dot inside the tensor core. The
// selector operands are PAIRS of .b16 registers interleaved after each scale
// register ({sfa} {sel,sel} {sfb} {sel,sel}, the mxf4nvf4 4X form in
// moe_w4a4.cu); all four are 0 for scale_vec::1X. Probe-verified map (the doc's
// "Probe evidence"): only byte 0 of each register is read. sfa byte0 = A row g
// (thread (g,0)) or row g+8 (thread (g,1)); sfb byte0 = B col g (thread (g,0)).
// Bytes 1-3 are ignored -- fill 0x7f (2^0 neutral).
__device__ __forceinline__ void mma_e4m3_sf(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                            uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa),
                 "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)), "r"(sfb),
                 "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)));
}
// Four floats -> four E4M3 codes, low to high (the mma A/B fragment order).
__device__ __forceinline__ uint32_t pack4_e4m3(float a, float b, float c, float d) {
  const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
  return lo | (hi << 16);
}
// The power-of-two P scale (plan §5.1): 2^ceil(log2(x)), the fp8_block recipe.
__device__ __forceinline__ float pow2_ceil(float x) {
  return dgpp::e8m0_byte_to_float(dgpp::e8m0_ceil_log2_byte(x));
}

// One instantiation per math mode so each branch gets its own register
// allocation and code footprint: MODE 0 = the bf16 path (pure bf16 cache, or the
// fp8 pool dequantized to bf16 in-kernel), MODE 1 = the E4M3 path (raw codes on
// the tensor cores, row-scale or the C.1a block-scale sub-variant). The unified
// single kernel was register-capped at 255 by the E4M3 branch's appetite, which
// cost the bf16 branch occupancy/IPC; the split lets each mode compile lean.
template <int MODE>
__global__ __launch_bounds__(32) void qsa_attn_prefill_warp_kernel(
    const uint16_t* __restrict__ q, int64_t q_row_stride, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const float* __restrict__ k_scale,
    const float* __restrict__ v_scale, const int32_t* __restrict__ req_ids, const int32_t* __restrict__ topk,
    int topk_stride, const int32_t* __restrict__ counts, int local_heads, int kv_heads, int block_tokens,
    const int32_t* __restrict__ block_tables, int blocks_per_request, float scale, float* __restrict__ out,
    int fp8_mma, const uint8_t* __restrict__ k_bscale, int mx) {
  extern __shared__ __align__(16) uint16_t sm[];
  const int64_t r = blockIdx.x;
  const int kvh = blockIdx.y;
  const int group = local_heads / kv_heads;
  const int h0 = kvh * group;
  const int lane = threadIdx.x;
  const int g = lane >> 2, t = lane & 3;
  const int width = kv_heads * kD;
  const int cnt = counts[r];
  const int32_t* toks = topk + r * topk_stride;
  const int32_t* bt = block_tables + static_cast<int64_t>(req_ids[r]) * blocks_per_request;
  const float sl2 = scale * 1.4426950408889634f;
  // MODE 1 = the FP8-MMA path (plan §5); MODE 0 = the bf16 path (pure or fp8
  // dequant). C.1a (e8mx) is the block-scaled QK^T sub-variant, MODE 1 only.
  const bool e8mx = (MODE == 1) && mx && k_bscale != nullptr;

  // Q A-fragments: bf16 16 k-steps x 4 registers; the E4M3 path quantizes the
  // same rows to 8 k-steps x 4 registers (4 e4m3 each, the m16n8k32 layout)
  // with the per-row scale alpha (plan §2.1). Rows past the group are zero.
  uint32_t qa[16][4];
  float alpha0 = 1.f, alpha1 = 1.f;  // E4M3 Q scales, rows g, g + 8
  uint16_t qsf[8];  // C.1a: per-k-step Q block scales (byte0 row g, byte1 row g+8)
  {
    const uint16_t* q0 = q + r * q_row_stride + static_cast<int64_t>(h0) * kD;
    const bool v0 = g < group, v1 = g + 8 < group;
    if constexpr (MODE == 1) {
    if (e8mx) {
      // C.1a: 8 per-32-dim-block absmaxes (one per k-step) for rows g, g+8. The
      // lane's 8 dims in k-step ks all fall in block ks, so a 2-level quad
      // reduce per ks gives the block absmax; the e8m0 byte is a power of two
      // (the fp8_block recipe) and the mma applies 2^(sfa+sfb) per k-block.
      float am0[8], am1[8];
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        am0[ks] = 0.f;
        am1[ks] = 0.f;
        const int base = ks * 32;
        const int dims[8] = {base + 4 * t, base + 4 * t + 1, base + 4 * t + 2, base + 4 * t + 3,
                             base + 16 + 4 * t, base + 16 + 4 * t + 1, base + 16 + 4 * t + 2,
                             base + 16 + 4 * t + 3};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (v0) am0[ks] = fmaxf(am0[ks], fabsf(bf16_bits_to_float(q0[g * kD + dims[j]])));
          if (v1) am1[ks] = fmaxf(am1[ks], fabsf(bf16_bits_to_float(q0[(g + 8) * kD + dims[j]])));
        }
      }
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
        for (int o = 1; o <= 2; o <<= 1) {
          am0[ks] = fmaxf(am0[ks], __shfl_xor_sync(0xffffffffu, am0[ks], o));
          am1[ks] = fmaxf(am1[ks], __shfl_xor_sync(0xffffffffu, am1[ks], o));
        }
        const uint8_t b0 = am0[ks] > 0.f ? e8m0_ceil_log2_byte(am0[ks] / 448.f) : 127;
        const uint8_t b1 = am1[ks] > 0.f ? e8m0_ceil_log2_byte(am1[ks] / 448.f) : 127;
        qsf[ks] = static_cast<uint16_t>(b0) | (static_cast<uint16_t>(b1) << 8);
      }
      // Pack e4m3(q / block_scale) into the m16n8k32 A-fragment (the same
      // 4-e4m3-per-register layout as the row-scale path, per-k-step scale).
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int base = ks * 32;
        const float s0 = am0[ks] > 0.f ? e8m0_byte_to_float(qsf[ks] & 0xFF) : 1.f;
        const float s1 = am1[ks] > 0.f ? e8m0_byte_to_float((qsf[ks] >> 8) & 0xFF) : 1.f;
        float a[8], b[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int d = (j < 4) ? base + 4 * t + j : base + 16 + 4 * t + (j - 4);
          a[j] = v0 ? bf16_bits_to_float(q0[g * kD + d]) / s0 : 0.f;
          b[j] = v1 ? bf16_bits_to_float(q0[(g + 8) * kD + d]) / s1 : 0.f;
        }
        qa[ks][0] = pack4_e4m3(a[0], a[1], a[2], a[3]);
        qa[ks][1] = pack4_e4m3(b[0], b[1], b[2], b[3]);
        qa[ks][2] = pack4_e4m3(a[4], a[5], a[6], a[7]);
        qa[ks][3] = pack4_e4m3(b[4], b[5], b[6], b[7]);
      }
    } else {
      // alpha = absmax/448 over the row's 256 dims (this lane's 64, then the
      // quad's), the fp8_quantize_rows_kernel recipe done in-warp.
      float am0 = 0.f, am1 = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int base = ks * 32;
        const int dims[8] = {base + 4 * t, base + 4 * t + 1, base + 4 * t + 2, base + 4 * t + 3,
                             base + 16 + 4 * t, base + 16 + 4 * t + 1, base + 16 + 4 * t + 2, base + 16 + 4 * t + 3};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          if (v0) am0 = fmaxf(am0, fabsf(bf16_bits_to_float(q0[g * kD + dims[j]])));
          if (v1) am1 = fmaxf(am1, fabsf(bf16_bits_to_float(q0[(g + 8) * kD + dims[j]])));
        }
      }
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1) {
        am0 = fmaxf(am0, __shfl_xor_sync(0xffffffffu, am0, o));
        am1 = fmaxf(am1, __shfl_xor_sync(0xffffffffu, am1, o));
      }
      alpha0 = am0 > 0.f ? am0 / 448.f : 1.f;
      alpha1 = am1 > 0.f ? am1 / 448.f : 1.f;
      // Pack e4m3(q/alpha) into the m16n8k32 A-fragment (4 e4m3 per register:
      // a0 = row g k 4t..4t+3, a1 = row g+8, a2/a3 = k +16; the oracle in
      // qwen_fp8_mma_attn_test pins this mapping, plan §11).
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int base = ks * 32;
        float a[8], b[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int d = (j < 4) ? base + 4 * t + j : base + 16 + 4 * t + (j - 4);
          a[j] = v0 ? bf16_bits_to_float(q0[g * kD + d]) / alpha0 : 0.f;
          b[j] = v1 ? bf16_bits_to_float(q0[(g + 8) * kD + d]) / alpha1 : 0.f;
        }
        qa[ks][0] = pack4_e4m3(a[0], a[1], a[2], a[3]);
        qa[ks][1] = pack4_e4m3(b[0], b[1], b[2], b[3]);
        qa[ks][2] = pack4_e4m3(a[4], a[5], a[6], a[7]);
        qa[ks][3] = pack4_e4m3(b[4], b[5], b[6], b[7]);
      }
      }
    } else {
#pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        const int c = ks * 16 + 2 * t;
        qa[ks][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * kD + c) : 0u;
        qa[ks][1] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * kD + c) : 0u;
        qa[ks][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + g * kD + c + 8) : 0u;
        qa[ks][3] = v1 ? *reinterpret_cast<const uint32_t*>(q0 + (g + 8) * kD + c + 8) : 0u;
      }
    }
  }

  // The tile's per-token fp8 scales (lanes 0-15 resolve their token); the
  // E4M3 epilogues broadcast them with shfl. Set by stage().
  float my_ks = 0.f, my_vs = 0.f;
  // C.1a: the tile's 16 tokens' K block scales (8 e8m0 bytes each), set by
  // stage() for the block-scaled QK^T mma (SFB byte0 = token (j*8 + g) for
  // k-step ks; the probe-verified map reads only byte 0, per thread (g,0)).
  __shared__ uint8_t smem_ksf[16 * 8];

  // Stage a tile: 16 tokens x (K, V) x 32 16-byte chunks = 1024 chunks, 32 per lane.
  auto stage = [&](int tile, int buf) {
    uint16_t* kt = sm + buf * kStageElems;
    uint16_t* vt = kt + kTileTok * kRow;
    const int base = tile * kTileTok;
    // Lanes 0..15 resolve the tile's 16 physical rows in one round of loads
    // (a serial token -> block-table chain per issue step starved the copies).
    int64_t my_off = 0, my_phys = 0;
    if (lane < kTileTok && base + lane < cnt) {
      const int tok = toks[base + lane];
      const int64_t phys = static_cast<int64_t>(bt[tok / block_tokens]) * block_tokens + tok % block_tokens;
      my_phys = phys;
      my_off = phys * width + static_cast<int64_t>(kvh) * kD;
    }
    if constexpr (MODE == 1) {
      // FP8-MMA: copy the raw e4m3 codes (8 B/chunk, no dequant) into the
      // 256 B/token smem tiles; the mma below reads the codes directly.
      // Invalid tokens zero-fill (the PV padding lanes must stay 0).
      const uint8_t* k8 = reinterpret_cast<const uint8_t*>(k_cache);
      const uint8_t* v8 = reinterpret_cast<const uint8_t*>(v_cache);
      uint8_t* kt8 = reinterpret_cast<uint8_t*>(sm) + buf * kStage8;
      uint8_t* vt8 = kt8 + kTileTok * kRow8;
      my_ks = 0.f;
      my_vs = 0.f;
      if (lane < kTileTok && base + lane < cnt) {
        my_ks = k_scale[my_phys * kv_heads + kvh];
        my_vs = v_scale[my_phys * kv_heads + kvh];
      }
      if (e8mx && lane < kTileTok) {
        // The tile's K block scales (8 e8m0 bytes per token) into smem; invalid
        // tokens zero-fill (byte 0 -> 2^-127 -> 0 contribution, masked to -INF).
        if (base + lane < cnt && k_bscale != nullptr) {
          const uint8_t* src = k_bscale + (my_phys * kv_heads + kvh) * 8;
#pragma unroll
          for (int b = 0; b < 8; ++b) smem_ksf[lane * 8 + b] = src[b];
        } else {
#pragma unroll
          for (int b = 0; b < 8; ++b) smem_ksf[lane * 8 + b] = 0;
        }
      }
#pragma unroll
      for (int i = 0; i < 16; ++i) {  // token i, lane = its 8-byte chunk
        const int64_t off = __shfl_sync(0xffffffffu, my_off, i) + lane * 8;
        const bool valid = base + i < cnt;
        cp_async8(kt8 + i * kRow8 + lane * 8, k8 + off, valid);
        cp_async8(vt8 + i * kRow8 + lane * 8, v8 + off, valid);
      }
      asm volatile("cp.async.commit_group;\n" ::);
      return;
    } else {
    if (k_scale != nullptr) {
      // fp8 cache: vectorized gather (8 codes per lane, one 8-byte load — the
      // lane already covers 8 of a token's kD elements) of the e4m3 codes +
      // per-(row, kv-head) scale, dequantized to bf16 in the smem tiles; the
      // ldmatrix / mma below are unchanged (they read bf16). k_cache / v_cache
      // are the code planes (1 B/elem; `off` is an element index = byte offset).
      const uint8_t* k8 = reinterpret_cast<const uint8_t*>(k_cache);
      const uint8_t* v8 = reinterpret_cast<const uint8_t*>(v_cache);
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int64_t p = __shfl_sync(0xffffffffu, my_phys, i);
        const int64_t off = __shfl_sync(0xffffffffu, my_off, i) + lane * 8;
        const bool valid = base + i < cnt;
        const float ks = k_scale[p * kv_heads + kvh];
        const float vs = v_scale[p * kv_heads + kvh];
        // C.1a: K codes are block-quantized; dequant each lane's 8 codes with
        // its 32-block's e8m0 scale (block = lane/4: the lane's 8 codes span a
        // block's quarter). V stays row-quantized (the PV gamma epilogue).
        float ksc = ks;
        if (mx && k_bscale != nullptr)
          ksc = e8m0_byte_to_float(k_bscale[(p * kv_heads + kvh) * 8 + lane / 4]);
        uint16_t* kdst = kt + i * kRow + lane * 8;
        uint16_t* vdst = vt + i * kRow + lane * 8;
        if (valid) {
          const uint2 k2 = *reinterpret_cast<const uint2*>(k8 + off);
          const uint2 v2 = *reinterpret_cast<const uint2*>(v8 + off);
          const uint8_t* kb = reinterpret_cast<const uint8_t*>(&k2);
          const uint8_t* vb = reinterpret_cast<const uint8_t*>(&v2);
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            kdst[j] = latent_fp8_decode_bf16(kb[j], ksc);
            vdst[j] = latent_fp8_decode_bf16(vb[j], vs);
          }
        } else {
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            kdst[j] = 0;
            vdst[j] = 0;
          }
        }
      }
      return;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) {            // token i, lane = its 16-byte chunk
      const int64_t off = __shfl_sync(0xffffffffu, my_off, i) + lane * 8;
      const bool valid = base + i < cnt;
      cp_async16(kt + i * kRow + lane * 8, k_cache + off, valid);
      cp_async16(vt + i * kRow + lane * 8, v_cache + off, valid);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    }
  };

  float acc[32][4];
#pragma unroll
  for (int d = 0; d < 32; ++d) acc[d][0] = acc[d][1] = acc[d][2] = acc[d][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;  // rows g, g + 8 (lane-partial l)

  const int tiles = (cnt + kTileTok - 1) / kTileTok;
  if (kStages == 2 && tiles > 0) stage(0, 0);
  for (int it = 0; it < tiles; ++it) {
    const int buf = kStages == 2 ? (it & 1) : 0;
    if (kStages == 1) {
      // One stage: residency (more warps per SM) hides the gather instead.
      stage(it, 0);
      asm volatile("cp.async.wait_group 0;\n" ::);
    } else if (it + 1 < tiles) {
      stage(it + 1, buf ^ 1);
      asm volatile("cp.async.wait_group 1;\n" ::);
    } else {
      asm volatile("cp.async.wait_group 0;\n" ::);
    }
    __syncwarp();
    // S = Q K^T: two n8 tiles (tokens 0-7, 8-15).
    float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    // Mask tokens past the list, scale into the exp2 domain, row maxima over the quad.
    const int n = min(kTileTok, cnt - it * kTileTok);
    float mx0 = -INFINITY, mx1 = -INFINITY;
    if (e8mx) {
      // C.1a: block-scaled Q K^T. The mma applies 2^(sfa+sfb) per k-block
      // (Q block scale qsf[ks], K block scale from smem_ksf), so the score
      // epilogue is just the exp2-domain scale -- no alpha*beta^k rank-1.
      const uint8_t* kt8 = reinterpret_cast<const uint8_t*>(sm) + buf * kStage8;
      // Probe-verified SF map (docs/qwen_fp8_mxfp8_sf_layout.md): only byte 0 of
      // each register is read (sel=0). sfa byte0 = this thread's own A row's
      // block scale -- thread (g,0) -> row g (qsf byte0), thread (g,1) -> row
      // g+8 (qsf byte1); t in {2,3} carry no A scale (defensively row g / g+8).
      // sfb byte0 = B col g's block scale = K token (j*8 + g) for this k-step
      // (the token index is g, not 2t/2t+1). Bytes 1-3 = 0x7f (2^0 neutral).
      const uint32_t sfa_hi = 0x7F7F7F00u;  // bytes 1-3 neutral
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int kb = ks * 32;
        const uint32_t sfa_lo = qsf[ks] & 0xFF;  // row g's e8m0 byte
        const uint32_t sfa_hi_b = (qsf[ks] >> 8) & 0xFF;  // row g+8's e8m0 byte
        const uint32_t sfa = ((t == 1 || t == 3) ? sfa_hi_b : sfa_lo) | sfa_hi;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const uint8_t* krow = kt8 + (j * 8 + g) * kRow8 + kb + 4 * t;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(krow);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(krow + 16);
          // SFB byte0 = K block scale of token (j*8 + g) for this k-step.
          const uint32_t sfb = static_cast<uint32_t>(smem_ksf[(j * 8 + g) * 8 + ks]) | sfa_hi;
          mma_e4m3_sf(sc[j], qa[ks], b0, b1, sfa, sfb);
        }
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int col = j * 8 + 2 * t + e;
          sc[j][e] = col < n ? sc[j][e] * sl2 : -INFINITY;
          sc[j][2 + e] = col < n ? sc[j][2 + e] * sl2 : -INFINITY;
          mx0 = fmaxf(mx0, sc[j][e]);
          mx1 = fmaxf(mx1, sc[j][2 + e]);
        }
      }
    } else if (MODE == 1) {
      // E4M3 Q K^T: 8 k-steps x 2 n8 token tiles on the raw K codes; the score
      // epilogue applies alpha*beta^k (rank-1, plan §2.2) in the exp2 domain.
      const uint8_t* kt8 = reinterpret_cast<const uint8_t*>(sm) + buf * kStage8;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int kb = ks * 32;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const uint8_t* krow = kt8 + (j * 8 + g) * kRow8 + kb + 4 * t;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(krow);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(krow + 16);
          mma_e4m3(sc[j], qa[ks], b0, b1);
        }
      }
      const float bk0 = __shfl_sync(0xffffffffu, my_ks, 2 * t);
      const float bk1 = __shfl_sync(0xffffffffu, my_ks, 2 * t + 1);
      const float bk8 = __shfl_sync(0xffffffffu, my_ks, 8 + 2 * t);
      const float bk9 = __shfl_sync(0xffffffffu, my_ks, 8 + 2 * t + 1);
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float bka = j ? bk8 : bk0, bkb = j ? bk9 : bk1;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int col = j * 8 + 2 * t + e;
          const float s2a = sl2 * alpha0 * (e ? bkb : bka);
          const float s2b = sl2 * alpha1 * (e ? bkb : bka);
          sc[j][e] = col < n ? sc[j][e] * s2a : -INFINITY;
          sc[j][2 + e] = col < n ? sc[j][2 + e] * s2b : -INFINITY;
          mx0 = fmaxf(mx0, sc[j][e]);
          mx1 = fmaxf(mx1, sc[j][2 + e]);
        }
      }
    } else {
      const uint16_t* kt = sm + buf * kStageElems;
#pragma unroll
      for (int ks = 0; ks < 16; ++ks) {
        uint32_t b[4];
        ldsm_x4(b, kt + ((lane & 7) + ((lane >> 4) << 3)) * kRow + ks * 16 + ((lane >> 3) & 1) * 8);
        mma_bf16(sc[0], qa[ks], b[0], b[1]);
        mma_bf16(sc[1], qa[ks], b[2], b[3]);
      }
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const int col = j * 8 + 2 * t + e;
          sc[j][e] = col < n ? sc[j][e] * sl2 : -INFINITY;
          sc[j][2 + e] = col < n ? sc[j][2 + e] * sl2 : -INFINITY;
          mx0 = fmaxf(mx0, sc[j][e]);
          mx1 = fmaxf(mx1, sc[j][2 + e]);
        }
      }
    }
#pragma unroll
    for (int o = 1; o <= 2; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
    const float a0 = exp2f(m0 - n0), a1 = exp2f(m1 - n1);  // m = -inf on the first tile: 0
    m0 = n0;
    m1 = n1;
    float p[2][4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      p[j][0] = exp2f(sc[j][0] - n0);
      p[j][1] = exp2f(sc[j][1] - n0);
      p[j][2] = exp2f(sc[j][2] - n1);
      p[j][3] = exp2f(sc[j][3] - n1);
    }
    l0 = l0 * a0 + p[0][0] + p[0][1] + p[1][0] + p[1][1];
    l1 = l1 * a1 + p[0][2] + p[0][3] + p[1][2] + p[1][3];
    if (MODE == 1) {
      // V-weighted P (w = P*beta^v, plan §2.3) -> per-row power-of-two gamma,
      // e4m3 codes. l above already summed the unrounded P (the DSA pin).
      const float bv0 = __shfl_sync(0xffffffffu, my_vs, 2 * t);
      const float bv1 = __shfl_sync(0xffffffffu, my_vs, 2 * t + 1);
      const float bv2 = __shfl_sync(0xffffffffu, my_vs, 8 + 2 * t);
      const float bv3 = __shfl_sync(0xffffffffu, my_vs, 8 + 2 * t + 1);
      const float w0[4] = {p[0][0] * bv0, p[0][1] * bv1, p[1][0] * bv2, p[1][1] * bv3};
      const float w1[4] = {p[0][2] * bv0, p[0][3] * bv1, p[1][2] * bv2, p[1][3] * bv3};
      float am0 = fmaxf(fmaxf(fabsf(w0[0]), fabsf(w0[1])), fmaxf(fabsf(w0[2]), fabsf(w0[3])));
      float am1 = fmaxf(fmaxf(fabsf(w1[0]), fabsf(w1[1])), fmaxf(fabsf(w1[2]), fabsf(w1[3])));
#pragma unroll
      for (int o = 1; o <= 2; o <<= 1) {
        am0 = fmaxf(am0, __shfl_xor_sync(0xffffffffu, am0, o));
        am1 = fmaxf(am1, __shfl_xor_sync(0xffffffffu, am1, o));
      }
      const float gamma0 = am0 > 0.f ? pow2_ceil(am0 / 448.f) : 1.f;
      const float gamma1 = am1 > 0.f ? pow2_ceil(am1 / 448.f) : 1.f;
      // e4m3(w/gamma) gathered into the m16n8k32 A-fragment via a shared-memory
      // exchange (a shfl of the encoded codes gets its operand register clobbered
      // by the compiler before the pack, even at -O0). Each lane writes its 8
      // codes (4 row g + 4 row g+8); token k lives in lane (g<<2)|((k&7)>>1)
      // register (k>>3)*2+(k&1). Padding k 16..31 is 0 (§5.1).
      __shared__ uint8_t smem_p[32 * 8];
      {
        const uint8_t c0a = float_to_fp8_e4m3_bits(w0[0] / gamma0),
                     c0b = float_to_fp8_e4m3_bits(w0[1] / gamma0),
                     c0c = float_to_fp8_e4m3_bits(w0[2] / gamma0),
                     c0d = float_to_fp8_e4m3_bits(w0[3] / gamma0);
        const uint8_t c1a = float_to_fp8_e4m3_bits(w1[0] / gamma1),
                     c1b = float_to_fp8_e4m3_bits(w1[1] / gamma1),
                     c1c = float_to_fp8_e4m3_bits(w1[2] / gamma1),
                     c1d = float_to_fp8_e4m3_bits(w1[3] / gamma1);
        smem_p[lane * 8 + 0] = c0a; smem_p[lane * 8 + 1] = c0b;
        smem_p[lane * 8 + 2] = c0c; smem_p[lane * 8 + 3] = c0d;
        smem_p[lane * 8 + 4] = c1a; smem_p[lane * 8 + 5] = c1b;
        smem_p[lane * 8 + 6] = c1c; smem_p[lane * 8 + 7] = c1d;
      }
      __syncwarp();
      auto pcode = [&](int tok, int rowoff) -> uint32_t {
        const int ln = (tok & 7) >> 1, reg = (tok >> 3) * 2 + (tok & 1);
        return smem_p[((g << 2) | ln) * 8 + rowoff * 4 + reg];
      };
      const uint32_t pa[4] = {(static_cast<uint32_t>(pcode(4 * t, 0))) |
                              (static_cast<uint32_t>(pcode(4 * t + 1, 0)) << 8) |
                              (static_cast<uint32_t>(pcode(4 * t + 2, 0)) << 16) |
                              (static_cast<uint32_t>(pcode(4 * t + 3, 0)) << 24),
                              (static_cast<uint32_t>(pcode(4 * t, 1))) |
                              (static_cast<uint32_t>(pcode(4 * t + 1, 1)) << 8) |
                              (static_cast<uint32_t>(pcode(4 * t + 2, 1)) << 16) |
                              (static_cast<uint32_t>(pcode(4 * t + 3, 1)) << 24),
                              0u, 0u};
#pragma unroll
      for (int d = 0; d < 32; ++d) {
        acc[d][0] *= a0;
        acc[d][1] *= a0;
        acc[d][2] *= a1;
        acc[d][3] *= a1;
      }
      // O += P V: 32 n8 dim tiles. B = V (k=tok, n=dim): b0 = V[tok 4t..4t+3][dim]
      // (four strided 8-bit loads off the token-major tile), b1 = 0 (the
      // padding k 16..31, §5.1). The mma yields (1/gamma)*(P V); scale by
      // gamma on the add (§2.3: beta^v cancels, one copy in P, one in V).
      const uint8_t* vt8 =
          reinterpret_cast<const uint8_t*>(sm) + buf * kStage8 + kTileTok * kRow8;
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        float pv[4][4];
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          pv[d][0] = pv[d][1] = pv[d][2] = pv[d][3] = 0.f;
          const int dim = c * 4 + d;  // n8 dim tile (dim values dim*8..dim*8+7)
          const int ncol = dim * 8 + g;
          const uint32_t b0 = static_cast<uint32_t>(vt8[(4 * t) * kRow8 + ncol]) |
                              (static_cast<uint32_t>(vt8[(4 * t + 1) * kRow8 + ncol]) << 8) |
                              (static_cast<uint32_t>(vt8[(4 * t + 2) * kRow8 + ncol]) << 16) |
                              (static_cast<uint32_t>(vt8[(4 * t + 3) * kRow8 + ncol]) << 24);
          mma_e4m3(pv[d], pa, b0, 0u);
        }
#pragma unroll
        for (int d = 0; d < 4; ++d) {
          const int dim = c * 4 + d;
          acc[dim][0] += gamma0 * pv[d][0];
          acc[dim][1] += gamma0 * pv[d][1];
          acc[dim][2] += gamma1 * pv[d][2];
          acc[dim][3] += gamma1 * pv[d][3];
        }
      }
    } else {
      // P as the A-fragment over k = the tile's 16 tokens.
      const uint16_t* vt = sm + buf * kStageElems + kTileTok * kRow;
      const uint32_t pa[4] = {pack_bf16(p[0][0], p[0][1]), pack_bf16(p[0][2], p[0][3]),
                              pack_bf16(p[1][0], p[1][1]), pack_bf16(p[1][2], p[1][3])};
#pragma unroll
      for (int d = 0; d < 32; ++d) {
        acc[d][0] *= a0;
        acc[d][1] *= a0;
        acc[d][2] *= a1;
        acc[d][3] *= a1;
      }
      // O += P V: 16 x4.trans loads, two n8 dim tiles each.
#pragma unroll
      for (int dp = 0; dp < 16; ++dp) {
        uint32_t b[4];
        ldsm_x4_t(b, vt + ((lane & 7) + ((lane >> 3) & 1) * 8) * kRow + dp * 16 + (lane >> 4) * 8);
        mma_bf16(acc[dp * 2], pa, b[0], b[1]);
        mma_bf16(acc[dp * 2 + 1], pa, b[2], b[3]);
      }
    }
    __syncwarp();  // this buffer's reads done before the next stage() overwrites it
  }
#pragma unroll
  for (int o = 1; o <= 2; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
  float* o0 = out + (r * local_heads + h0 + g) * kD;
  float* o1 = out + (r * local_heads + h0 + g + 8) * kD;
#pragma unroll
  for (int d = 0; d < 32; ++d) {
    const int col = d * 8 + 2 * t;
    if (g < group) *reinterpret_cast<float2*>(o0 + col) = make_float2(acc[d][0] * i0, acc[d][1] * i0);
    if (g + 8 < group) *reinterpret_cast<float2*>(o1 + col) = make_float2(acc[d][2] * i1, acc[d][3] * i1);
  }
}

}  // namespace

bool qsa_warp_supported(int dim, int local_heads, int kv_heads) {
  return dim == kD && kv_heads > 0 && local_heads % kv_heads == 0 && local_heads / kv_heads <= 16;
}

void qsa_attn_prefill_warp(const uint16_t* q, int64_t q_row_stride, const uint16_t* k_cache,
                           const uint16_t* v_cache, const int32_t* req_ids, const int32_t* topk, int topk_stride,
                           const int32_t* counts, int rows, int local_heads, int kv_heads, int block_tokens,
                           const int32_t* block_tables, int blocks_per_request, float scale, float* out,
                           cudaStream_t stream, const float* k_scale, const float* v_scale,
                           const uint8_t* k_bscale) {
  if (rows <= 0) return;
  if (!qsa_warp_supported(kD, local_heads, kv_heads))
    throw std::invalid_argument("qsa_attn_prefill_warp: dim 256, <= 16 query heads per KV head");
  static const bool opted = [] {  // once per process, thread-safe (TP ranks may share one)
    // Both MODE instantiations get the smem + carveout (the bf16 and E4M3 paths
    // share the kSmem budget; the E4M3 raw-code tiles fit within it).
    DGPP_CUDA_OK(cudaFuncSetAttribute(qsa_attn_prefill_warp_kernel<0>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kSmem)));
    // All of L1 as shared memory: residency here is bounded by the 33.8 KB stages.
    DGPP_CUDA_OK(cudaFuncSetAttribute(qsa_attn_prefill_warp_kernel<0>,
                                      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    DGPP_CUDA_OK(cudaFuncSetAttribute(qsa_attn_prefill_warp_kernel<1>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(kSmem)));
    DGPP_CUDA_OK(cudaFuncSetAttribute(qsa_attn_prefill_warp_kernel<1>,
                                      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    return true;
  }();
  (void)opted;
  // The FP8-MMA toggle (plan §5): default ON. It decouples the E4M3 attention
  // from the fp8 pool -- =0 keeps the Phase A in-kernel dequant (and the pool
  // win) so short/long context can be A/B'd independently. It selects the MODE
  // template (1 = the E4M3 tensor-core path, 0 = the bf16 / dequant path), so
  // each branch compiles lean (its own register allocation and footprint).
  static const int fp8_mma = [] {
    const char* e = std::getenv("DGPP_QSA_FP8_MMA");
    return !(e != nullptr && e[0] == '0');
  }();
  // DGPP_QSA_FP8_MX (C.1a, default OFF): the block-scaled math, read per call
  // (not latched) so a test can setenv between runs; production sets it once.
  const int mx = qsa_fp8_mx();
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(kv_heads));
  const auto launch = [&](auto mode_tag) {
    constexpr int MODE = decltype(mode_tag)::value;
    qsa_attn_prefill_warp_kernel<MODE><<<grid, 32, kSmem, stream>>>(q, q_row_stride, k_cache, v_cache,
                                                                    k_scale, v_scale, req_ids, topk,
                                                                    topk_stride, counts, local_heads,
                                                                    kv_heads, block_tokens,
                                                                    block_tables, blocks_per_request,
                                                                    scale, out, fp8_mma, k_bscale, mx);
  };
  // MODE 1 needs the scale planes (the E4M3 path reads k_scale / v_scale per
  // token); a null k_scale is the bf16 call, so the pre-split guard was
  // `fp8_mma && k_scale != nullptr`. Keep it: launching MODE 1 with a null
  // scale plane is an illegal memory access.
  if (fp8_mma && k_scale != nullptr)
    launch(std::integral_constant<int, 1>{});
  else
    launch(std::integral_constant<int, 0>{});
  DGPP_CUDA_OK(cudaGetLastError());
}

}  // namespace dgpp
