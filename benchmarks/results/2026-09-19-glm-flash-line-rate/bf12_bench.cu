// bf12_bench: a LOSSLESS 12-bit re-encoding of bf16 weights (sign+mantissa
// byte, 4-bit exponent code against a per-tensor base, code 15 = escape to
// an exact side table) against the production bf16 GEMV, on real weights.
// The lane/step ownership and FMA order are the production kernel's, so the
// outputs must match bit for bit.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/gemv_common.cuh"

namespace {

using dgpp::bf16_bits_to_float;
using dgpp::float_to_bf16_bits;
namespace gemv = dgpp::gemv;

constexpr int kSuper = 1024;        // elements per super-block
constexpr int kSuperBytes = 1536;   // 512 smA + 512 smB + 512 exp

// One step (8 elements of one lane): sm bytes in (lo, hi), eight exponent
// nibbles in ew, the row's staged activations at column c0.
template <int kRows, bool kCheck>
__device__ __forceinline__ void consume_step(uint32_t lo, uint32_t hi, uint32_t ew,
                                             uint32_t base23,
                                             const uint16_t* __restrict__ sx, int k,
                                             int c0, const uint32_t* __restrict__ esc,
                                             uint32_t esc_b, uint32_t esc_e,
                                             float (&acc)[kRows]) {
  uint32_t wbits[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t b = ((j < 4 ? lo : hi) >> (8 * (j & 3))) & 0xFFu;
    const uint32_t nib = (ew >> (4 * j)) & 0xFu;
    wbits[j] = ((b & 0x80u) << 24) | ((b & 0x7Fu) << 16) | (base23 + (nib << 23));
  }
  if (kCheck) {
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
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    const uint4 xv = *reinterpret_cast<const uint4*>(sx + static_cast<size_t>(r) * k + c0);
    const uint32_t xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float w = std::bit_cast<float>(wbits[j]);
      const float x = bf16_bits_to_float(static_cast<uint16_t>((xw[j >> 1] >> (16 * (j & 1))) & 0xFFFFu));
      acc[r] = __fmaf_rn(w, x, acc[r]);
    }
  }
}

template <int kRows, bool kOutF32, bool kCheck, int kSB>
__global__ void bf12_gemv_kernel(const uint16_t* __restrict__ act, size_t act_stride,
                                 const uint8_t* __restrict__ w,
                                 const uint32_t* __restrict__ esc_begin,
                                 const uint32_t* __restrict__ esc, void* __restrict__ out,
                                 int n, int k, uint32_t base23) {
  extern __shared__ __align__(16) uint16_t sx[];
  gemv::stage_activations<kRows>(act, act_stride, k, sx);
  __syncthreads();
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * gemv::kWarps + warp;
  if (row >= n) return;
  const int nsb = k / kSuper;
  const uint8_t* wr = w + static_cast<size_t>(row) * nsb * kSuperBytes;
  uint32_t eb = 0, ee = 0;
  if (kCheck) {
    eb = esc_begin[row];
    ee = esc_begin[row + 1];
  }
  float acc[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) acc[r] = 0.f;
  for (int s0 = 0; s0 < nsb; s0 += kSB) {
    uint4 a[kSB], b[kSB], e[kSB];
#pragma unroll
    for (int s = 0; s < kSB; ++s) {
      const bool ok = s0 + s < nsb;
      const uint8_t* p = wr + static_cast<size_t>(ok ? s0 + s : s0) * kSuperBytes + lane * 16;
      a[s] = *reinterpret_cast<const uint4*>(p);
      b[s] = *reinterpret_cast<const uint4*>(p + 512);
      e[s] = *reinterpret_cast<const uint4*>(p + 1024);
    }
#pragma unroll
    for (int s = 0; s < kSB; ++s) {
      if (s0 + s >= nsb) break;
      const int cb = (s0 + s) * kSuper + lane * 8;
      consume_step<kRows, kCheck>(a[s].x, a[s].y, e[s].x, base23, sx, k, cb, esc, eb, ee, acc);
      consume_step<kRows, kCheck>(a[s].z, a[s].w, e[s].y, base23, sx, k, cb + 256, esc, eb, ee, acc);
      consume_step<kRows, kCheck>(b[s].x, b[s].y, e[s].z, base23, sx, k, cb + 512, esc, eb, ee, acc);
      consume_step<kRows, kCheck>(b[s].z, b[s].w, e[s].w, base23, sx, k, cb + 768, esc, eb, ee, acc);
    }
  }
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

struct Packed {
  std::vector<uint8_t> bytes;
  std::vector<uint32_t> esc_begin, esc;
  int base = 0;
  size_t escapes = 0;
};

Packed encode(const std::vector<uint16_t>& w, int n, int k) {
  Packed p;
  // The 15-wide exponent window with the fewest escapes.
  std::vector<uint64_t> hist(256, 0);
  for (uint16_t v : w) ++hist[(v >> 7) & 0xFF];
  uint64_t best = 0;
  for (int b = 0; b + 15 <= 256; ++b) {
    uint64_t c = 0;
    for (int i = 0; i < 15; ++i) c += hist[b + i];
    if (c > best) { best = c; p.base = b; }
  }
  const int nsb = k / kSuper;
  p.bytes.assign(static_cast<size_t>(n) * nsb * kSuperBytes, 0);
  p.esc_begin.assign(n + 1, 0);
  for (int row = 0; row < n; ++row) {
    p.esc_begin[row] = static_cast<uint32_t>(p.esc.size());
    uint8_t* wr = p.bytes.data() + static_cast<size_t>(row) * nsb * kSuperBytes;
    for (int c = 0; c < k; ++c) {
      const uint16_t v = w[static_cast<size_t>(row) * k + c];
      const int sb = c / kSuper, in = c % kSuper;
      const int t = in / 256, lane = (in % 256) / 8, j = in % 8;
      const uint8_t sm = static_cast<uint8_t>(((v >> 8) & 0x80) | (v & 0x7F));
      const int ex = (v >> 7) & 0xFF;
      int code = ex - p.base;
      if (code < 0 || code > 14) {
        code = 15;
        p.esc.push_back((static_cast<uint32_t>(c) << 16) | v);
      }
      uint8_t* blk = wr + static_cast<size_t>(sb) * kSuperBytes;
      // sm: steps 0,1 in segment A, 2,3 in segment B; 8 bytes per step per lane.
      uint8_t* smp = blk + (t < 2 ? 0 : 512) + lane * 16 + (t & 1) * 8 + j;
      *smp = sm;
      uint8_t* ep = blk + 1024 + lane * 16 + t * 4 + j / 2;
      *ep = static_cast<uint8_t>(*ep | (code << (4 * (j & 1))));
    }
  }
  p.esc_begin[n] = static_cast<uint32_t>(p.esc.size());
  p.escapes = p.esc.size();
  return p;
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

template <int kRows, bool kOutF32, bool kCheck, int kSB>
void launch12(const uint16_t* act, const uint8_t* w, const uint32_t* eb, const uint32_t* es,
              void* out, int n, int k, uint32_t base23) {
  const dim3 grid((n + gemv::kWarps - 1) / gemv::kWarps);
  bf12_gemv_kernel<kRows, kOutF32, kCheck, kSB><<<grid, gemv::kThreads, gemv::smem_bytes(kRows, k), nullptr>>>(
      act, static_cast<size_t>(k), w, eb, es, out, n, k, base23);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: bf12_bench FILE N K [rows=2] [copies=16] [iters=200]\n");
    return 2;
  }
  const std::string path = argv[1];
  const int n = std::atoi(argv[2]), k = std::atoi(argv[3]);
  const int rows = argc > 4 ? std::atoi(argv[4]) : 2;
  int copies = argc > 5 ? std::atoi(argv[5]) : 16;
  const int iters = argc > 6 ? std::atoi(argv[6]) : 200;
  std::vector<uint16_t> w(static_cast<size_t>(n) * k);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f || std::fread(w.data(), 2, w.size(), f) != w.size()) {
    std::fprintf(stderr, "read failed\n");
    return 1;
  }
  std::fclose(f);
  const Packed p = encode(w, n, k);
  const size_t raw_bytes = w.size() * 2, pk_bytes = p.bytes.size();
  std::printf("%s N%dxK%d: raw %.2f MB, packed %.2f MB (%.4f), base %d, escapes %zu (%.2e), esc table %.1f KB\n",
              path.c_str(), n, k, raw_bytes / 1e6, pk_bytes / 1e6,
              static_cast<double>(pk_bytes + p.esc.size() * 4 + p.esc_begin.size() * 4) / raw_bytes, p.base,
              p.escapes, static_cast<double>(p.escapes) / w.size(),
              (p.esc.size() + p.esc_begin.size()) * 4 / 1e3);

  // Distinct device copies so no launch re-reads what L2 still holds.
  std::vector<uint16_t*> raw(copies);
  std::vector<uint8_t*> pk(copies);
  std::vector<uint32_t*> eb(copies), es(copies);
  for (int c = 0; c < copies; ++c) {
    DGPP_CUDA_OK(cudaMalloc(&raw[c], raw_bytes));
    DGPP_CUDA_OK(cudaMemcpy(raw[c], w.data(), raw_bytes, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&pk[c], pk_bytes));
    DGPP_CUDA_OK(cudaMemcpy(pk[c], p.bytes.data(), pk_bytes, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&eb[c], p.esc_begin.size() * 4));
    DGPP_CUDA_OK(cudaMemcpy(eb[c], p.esc_begin.data(), p.esc_begin.size() * 4, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMalloc(&es[c], std::max<size_t>(p.esc.size(), 1) * 4));
    DGPP_CUDA_OK(cudaMemcpy(es[c], p.esc.data(), p.esc.size() * 4, cudaMemcpyHostToDevice));
  }
  // Activations: bf16 values in a realistic range.
  std::vector<uint16_t> hx(static_cast<size_t>(rows) * k);
  uint64_t s = 0x9E3779B97F4A7C15ull;
  for (auto& v : hx) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    const float fv = (static_cast<int>(s % 2001) - 1000) / 250.0f;
    v = float_to_bf16_bits(fv);
  }
  uint16_t* x = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&x, hx.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  const size_t out_elems = static_cast<size_t>(rows) * n;
  uint16_t* out_ref = nullptr;
  uint16_t* out_new = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&out_ref, out_elems * 4));
  DGPP_CUDA_OK(cudaMalloc(&out_new, out_elems * 4));
  const uint32_t base23 = static_cast<uint32_t>(p.base) << 23;

  auto run12 = [&](int c, bool check, int sbatch, bool f32) {
#define L12(R, F, C, S) launch12<R, F, C, S>(x, pk[c], eb[c], es[c], out_new, n, k, base23)
#define L12R(R) \
      if (f32) { if (check) { if (sbatch == 2) L12(R, true, true, 2); else L12(R, true, true, 4); } else { if (sbatch == 2) L12(R, true, false, 2); else L12(R, true, false, 4); } } \
      else     { if (check) { if (sbatch == 2) L12(R, false, true, 2); else L12(R, false, true, 4); } else { if (sbatch == 2) L12(R, false, false, 2); else L12(R, false, false, 4); } }
    switch (rows) {
      case 1: L12R(1) break;
      case 2: L12R(2) break;
      case 3: L12R(3) break;
      default: L12R(4) break;
    }
#undef L12R
#undef L12
  };

  // Bitwise gate (bf16 and f32 outputs), checked kernel, both batch depths.
  for (bool f32 : {false, true}) {
    for (int sb : {2, 4}) {
      DGPP_CUDA_OK(cudaMemset(out_ref, 0, out_elems * 4));
      DGPP_CUDA_OK(cudaMemset(out_new, 0xFF, out_elems * 4));
      dgpp::launch_bf16_gemv(x, k, raw[0], out_ref, f32, rows, n, k, nullptr);
      run12(0, true, sb, f32);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      const size_t ob = out_elems * (f32 ? 4 : 2);
      std::vector<uint8_t> a(ob), b(ob);
      cudaMemcpy(a.data(), out_ref, ob, cudaMemcpyDeviceToHost);
      cudaMemcpy(b.data(), out_new, ob, cudaMemcpyDeviceToHost);
      std::printf("  bitwise vs bf16_gemv (out %s, sb %d): %s\n", f32 ? "f32" : "bf16", sb,
                  std::memcmp(a.data(), b.data(), ob) == 0 ? "IDENTICAL" : "DIFFERENT");
    }
  }

  for (int rep = 0; rep < 3; ++rep) {
    int c = 0;
    const double t16 = time_us([&] { dgpp::launch_bf16_gemv(x, k, raw[c], out_ref, false, rows, n, k, nullptr); c = (c + 1) % copies; }, iters);
    c = 0;
    const double t12c2 = time_us([&] { run12(c, true, 2, false); c = (c + 1) % copies; }, iters);
    c = 0;
    const double t12c4 = time_us([&] { run12(c, true, 4, false); c = (c + 1) % copies; }, iters);
    c = 0;
    const double t12n4 = time_us([&] { run12(c, false, 4, false); c = (c + 1) % copies; }, iters);
    std::printf("  rows %d: bf16 %.1f us (%.1f GB/s) | bf12 check sb2 %.1f us, sb4 %.1f us (%.1f GB/s of its bytes, x%.3f) | nocheck sb4 %.1f us\n",
                rows, t16, raw_bytes / t16 / 1e3, t12c2, t12c4, pk_bytes / t12c4 / 1e3, t16 / t12c4, t12n4);
  }
  return 0;
}
