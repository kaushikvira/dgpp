// bf12_wide_bench: the 12-bit packed GEMV at 5..8 activation rows, staged
// per 2048-column window (8 rows x 2048 x 2 B = 32 KB of smem), against
// (a) cuBLASLt at the same m (what the 5..8-row batch takes today), (b) two
// bf16 GEMV chunks, (c) two bf12 GEMV chunks. The wide kernel keeps the
// scalar chain, so it must be BITWISE the chunked GEMV.
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf12_gemv.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/gemm.hpp"
#include "kernels/gemv_common.cuh"

namespace {

using namespace dgpp;

template <int kRows>
__device__ __forceinline__ void consume_step(uint32_t lo, uint32_t hi, uint32_t ew,
                                             uint32_t base23,
                                             const uint16_t* __restrict__ sx, int wk, int c0w,
                                             int c0, const uint32_t* __restrict__ esc,
                                             uint32_t esc_b, uint32_t esc_e,
                                             float (&acc)[kRows]) {
  uint32_t wbits[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t b = ((j < 4 ? lo : hi) >> (8 * (j & 3))) & 0xFFu;
    const uint32_t code = (ew >> (4 * j)) & 0xFu;
    wbits[j] = ((b & 0x80u) << 24) | ((b & 0x7Fu) << 16) | (base23 + (code << 23));
  }
  const uint32_t any15 = ew & (ew >> 1) & (ew >> 2) & (ew >> 3) & 0x11111111u;
  if (any15 != 0u) {
    for (int j = 0; j < 8; ++j) {
      if (((ew >> (4 * j)) & 0xFu) != 0xFu) continue;
      const uint32_t col = static_cast<uint32_t>(c0 + j);
      for (uint32_t e = esc_b; e < esc_e; ++e) {
        const uint32_t ent = esc[e];
        if ((ent >> 16) == col) {
          wbits[j] = (ent & 0xFFFFu) << 16;
          break;
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const uint4 xv = *reinterpret_cast<const uint4*>(sx + static_cast<size_t>(r) * wk + c0w);
    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float w = std::bit_cast<float>(wbits[j]);
      const float x =
          bf16_bits_to_float(static_cast<uint16_t>((xw[j >> 1] >> (16 * (j & 1))) & 0xFFFFu));
      acc[r] = __fmaf_rn(w, x, acc[r]);
    }
  }
}

// kWin super-blocks (1024 columns each) staged and consumed per window.
template <int kRows, bool kOutF32, int kWin>
__global__ void bf12_wide_kernel(const uint16_t* __restrict__ act, size_t act_stride,
                                 const uint8_t* __restrict__ w,
                                 const uint32_t* __restrict__ rows,
                                 const uint32_t* __restrict__ esc, void* __restrict__ out, int n,
                                 int k) {
  extern __shared__ __align__(16) uint16_t sx[];
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * gemv::kWarps + warp;
  const bool live = row < n;
  const int nsb = k / kBf12Super;
  const int wk = kWin * kBf12Super;  // staged columns per window
  const uint8_t* wr = w + static_cast<size_t>(live ? row : 0) * nsb * kBf12SuperBytes;
  const uint32_t eb = live ? rows[2 * row] : 0u;
  const uint32_t base23 = live ? rows[2 * row + 1] : 0u;
  const uint32_t ee = live ? rows[2 * row + 2] : 0u;
  float acc[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) acc[r] = 0.f;
  for (int s0 = 0; s0 < nsb; s0 += kWin) {
    const int cols = (nsb - s0 < kWin ? nsb - s0 : kWin) * kBf12Super;
    // Stage columns [s0 * 1024, + cols) of every row at stride wk.
    {
      const uint16_t* x0 = act + static_cast<size_t>(s0) * kBf12Super;
      const int vecs = cols / 8;
      for (int i = threadIdx.x; i < kRows * vecs; i += blockDim.x) {
        const int r = i / vecs, c = i - r * vecs;
        reinterpret_cast<uint4*>(sx + static_cast<size_t>(r) * wk)[c] =
            reinterpret_cast<const uint4*>(x0 + static_cast<size_t>(r) * act_stride)[c];
      }
    }
    __syncthreads();
    const bool more = s0 + kWin < nsb;
    if (!live) {
      if (more) __syncthreads();
      continue;
    }
    uint4 a[kWin], b[kWin], e[kWin];
#pragma unroll
    for (int s = 0; s < kWin; ++s) {
      const int sb = s0 + s < nsb ? s0 + s : s0;
      const uint8_t* p = wr + static_cast<size_t>(sb) * kBf12SuperBytes + lane * 16;
      a[s] = *reinterpret_cast<const uint4*>(p);
      b[s] = *reinterpret_cast<const uint4*>(p + 512);
      e[s] = *reinterpret_cast<const uint4*>(p + 1024);
    }
#pragma unroll
    for (int s = 0; s < kWin; ++s) {
      if (s0 + s >= nsb) break;
      const int cw = s * kBf12Super + lane * 8;              // window-local column
      const int cb = (s0 + s) * kBf12Super + lane * 8;       // the row's column
      consume_step<kRows>(a[s].x, a[s].y, e[s].x, base23, sx, wk, cw, cb, esc, eb, ee, acc);
      consume_step<kRows>(a[s].z, a[s].w, e[s].y, base23, sx, wk, cw + 256, cb + 256, esc, eb, ee, acc);
      consume_step<kRows>(b[s].x, b[s].y, e[s].z, base23, sx, wk, cw + 512, cb + 512, esc, eb, ee, acc);
      consume_step<kRows>(b[s].z, b[s].w, e[s].w, base23, sx, wk, cw + 768, cb + 768, esc, eb, ee, acc);
    }
    if (more) __syncthreads();  // everyone done with sx before it is restaged
  }
  if (!live) return;
  gemv::warp_reduce<kRows>(acc);
  if (lane != 0) return;
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const size_t at = static_cast<size_t>(r) * n + row;
    if (kOutF32)
      static_cast<float*>(out)[at] = acc[r];
    else
      static_cast<uint16_t*>(out)[at] = float_to_bf16_bits(acc[r]);
  }
}

template <int kRows, int kWin>
void launch_wide(const uint16_t* act, size_t stride, const Bf12Matrix& m, void* out, bool f32) {
  const dim3 grid((m.n + gemv::kWarps - 1) / gemv::kWarps);
  const size_t smem = static_cast<size_t>(kRows) * kWin * kBf12Super * 2;
  if (f32)
    bf12_wide_kernel<kRows, true, kWin><<<grid, gemv::kThreads, smem, nullptr>>>(act, stride, m.packed, m.rows, m.esc, out, m.n, m.k);
  else
    bf12_wide_kernel<kRows, false, kWin><<<grid, gemv::kThreads, smem, nullptr>>>(act, stride, m.packed, m.rows, m.esc, out, m.n, m.k);
  DGPP_CUDA_OK(cudaGetLastError());
}

void run_wide(int rows, int win, const uint16_t* act, size_t stride, const Bf12Matrix& m, void* out, bool f32) {
#define W(R) if (win == 1) launch_wide<R, 1>(act, stride, m, out, f32); else if (win == 2) launch_wide<R, 2>(act, stride, m, out, f32); else launch_wide<R, 4>(act, stride, m, out, f32);
  switch (rows) {
    case 5: W(5) break;
    case 6: W(6) break;
    case 7: W(7) break;
    default: W(8) break;
  }
#undef W
}

template <typename F>
double time_us(F&& launch, int iters) {
  cudaEvent_t beg{}, end{};
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  for (int i = 0; i < 4; ++i) launch();
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  cudaEventRecord(beg, nullptr);
  for (int i = 0; i < iters; ++i) launch();
  cudaEventRecord(end, nullptr);
  cudaEventSynchronize(end);
  float ms = 0.f;
  cudaEventElapsedTime(&ms, beg, end);
  cudaEventDestroy(beg);
  cudaEventDestroy(end);
  return ms * 1e3 / iters;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: bf12_wide_bench FILE N K [rows=8] [copies=16] [iters=200]\n");
    return 2;
  }
  const std::string path = argv[1];
  const int n = std::atoi(argv[2]), k = std::atoi(argv[3]);
  const int rows = argc > 4 ? std::atoi(argv[4]) : 8;
  const int copies = argc > 5 ? std::atoi(argv[5]) : 16;
  const int iters = argc > 6 ? std::atoi(argv[6]) : 200;
  std::vector<uint16_t> w(static_cast<size_t>(n) * k);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f || std::fread(w.data(), 2, w.size(), f) != w.size()) return 1;
  std::fclose(f);
  const Bf12Host h = bf12_encode(w.data(), n, k);
  if (!h.ok) return 3;
  const size_t raw_bytes = w.size() * 2;
  std::vector<uint16_t*> raw(copies);
  std::vector<Bf12Matrix> pk(copies);
  for (int c = 0; c < copies; ++c) {
    DGPP_CUDA_OK(cudaMalloc(&raw[c], raw_bytes));
    DGPP_CUDA_OK(cudaMemcpy(raw[c], w.data(), raw_bytes, cudaMemcpyHostToDevice));
    void *p = nullptr, *r = nullptr, *e = nullptr, *rr = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&p, h.packed.size()));
    DGPP_CUDA_OK(cudaMalloc(&r, h.rows.size() * 4));
    DGPP_CUDA_OK(cudaMalloc(&e, h.esc.size() * 4));
    DGPP_CUDA_OK(cudaMalloc(&rr, h.raw.size() * 2));
    DGPP_CUDA_OK(cudaMemcpy(p, h.packed.data(), h.packed.size(), cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(r, h.rows.data(), h.rows.size() * 4, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(e, h.esc.data(), h.esc.size() * 4, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(rr, h.raw.data(), h.raw.size() * 2, cudaMemcpyHostToDevice));
    pk[c] = Bf12Matrix{static_cast<const uint8_t*>(p), static_cast<const uint32_t*>(r),
                       static_cast<const uint32_t*>(e), static_cast<const uint16_t*>(rr), n, k};
  }
  std::vector<uint16_t> hx(static_cast<size_t>(rows) * k);
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (auto& v : hx) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    v = float_to_bf16_bits((static_cast<int>(s % 2001) - 1000) / 250.0f);
  }
  uint16_t* x = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&x, hx.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  const size_t out_bytes = static_cast<size_t>(rows) * n * 4;
  void *o_ref = nullptr, *o_new = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&o_ref, out_bytes));
  DGPP_CUDA_OK(cudaMalloc(&o_new, out_bytes));
  void* ws = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&ws, 64u << 20));

  CublasLtGemm chunks;  // m <= 8 through the GEMV chunks (bf16)
  chunks.set_decode_rows(8);
  CublasLtGemm lt;      // m > 4 through cuBLASLt (today's 5..8-row lowering)
  lt.set_decode_rows(4);

  int row_esc_raw = h.raw_rows;
  std::printf("%s N%dxK%d rows %d: raw rows %d, escapes %zu\n", path.c_str(), n, k, rows, row_esc_raw, h.escapes);
  // Bitwise against the GEMV chunks (the scalar chain), bf16 and f32 out.
  for (bool f32 : {false, true}) {
    for (int win : {1, 2, 4}) {
      if (static_cast<size_t>(rows) * win * 1024 * 2 > 48 * 1024) continue;
      if (win * 1024 > k) continue;
      DGPP_CUDA_OK(cudaMemset(o_ref, 0, out_bytes));
      DGPP_CUDA_OK(cudaMemset(o_new, 0xFF, out_bytes));
      chunks.matmul(x, raw[0], o_ref, rows, n, k, DType::BF16, f32 ? GemmOut::F32 : GemmOut::BF16, k, ws, 64u << 20, nullptr);
      run_wide(rows, win, x, k, pk[0], o_new, f32);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const size_t ob = static_cast<size_t>(rows) * n * (f32 ? 4 : 2);
      std::vector<uint8_t> a(ob), b(ob);
      cudaMemcpy(a.data(), o_ref, ob, cudaMemcpyDeviceToHost);
      cudaMemcpy(b.data(), o_new, ob, cudaMemcpyDeviceToHost);
      std::printf("  bitwise vs GEMV chunks (%s, window %d sb): %s\n", f32 ? "f32" : "bf16", win,
                  std::memcmp(a.data(), b.data(), ob) == 0 ? "IDENTICAL" : "DIFFERENT");
    }
  }
  for (int rep = 0; rep < 3; ++rep) {
    int c = 0;
    const double t_lt = time_us([&] { lt.matmul(x, raw[c], o_ref, rows, n, k, DType::BF16, GemmOut::BF16, k, ws, 64u << 20, nullptr); c = (c + 1) % copies; }, iters);
    c = 0;
    const double t_ch = time_us([&] { chunks.matmul(x, raw[c], o_ref, rows, n, k, DType::BF16, GemmOut::BF16, k, ws, 64u << 20, nullptr); c = (c + 1) % copies; }, iters);
    double t_w[3] = {0, 0, 0};
    int wi = 0;
    for (int win : {1, 2, 4}) {
      if (static_cast<size_t>(rows) * win * 1024 * 2 > 48 * 1024 || win * 1024 > k) { ++wi; continue; }
      c = 0;
      t_w[wi++] = time_us([&] { run_wide(rows, win, x, k, pk[c], o_new, false); c = (c + 1) % copies; }, iters);
    }
    std::printf("  rows %d: cuBLASLt %.1f us (%.1f GB/s) | bf16 GEMV chunks %.1f us | bf12 wide win1 %.1f, win2 %.1f, win4 %.1f us\n",
                rows, t_lt, raw_bytes / t_lt / 1e3, t_ch, t_w[0], t_w[1], t_w[2]);
  }
  return 0;
}
