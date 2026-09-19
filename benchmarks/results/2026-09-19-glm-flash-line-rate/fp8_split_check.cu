// FP8 block-scaled projections at the DSA shapes: T rows at once vs two row blocks.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/scale_gemm.hpp"
using namespace dgpp;
static uint32_t h32(uint64_t seed, uint64_t idx) { uint64_t x = seed * 0x9E3779B97F4A7C15ULL ^ idx; x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return uint32_t(x ^ (x >> 32)); }
int main() {
  struct S { int m, n, k; };
  for (const S s : {S{2048, 1536, 4096}, S{2048, 512, 4096}, S{2048, 4096, 1536}, S{2048, 4096, 4096}, S{1424, 4096, 4096}, S{2048, 3072, 4096}, S{2048, 4096, 3072}}) {
    std::vector<uint16_t> act(size_t(s.m) * s.k); std::vector<uint8_t> w(size_t(s.n) * s.k);
    for (size_t i = 0; i < act.size(); ++i) { const uint32_t h = h32(1, i); act[i] = uint16_t((h & 0x8000u) | ((125u + (h >> 8) % 4) << 7) | (h & 0x7Fu)); }
    for (size_t i = 0; i < w.size(); ++i) { uint8_t c = uint8_t(h32(2, i)); if ((c & 0x7F) == 0x7F) c ^= 1; w[i] = c; }
    const int sr = (s.n + 127) / 128, sc = (s.k + 127) / 128; std::vector<float> sc_h(size_t(sr) * sc);
    for (size_t i = 0; i < sc_h.size(); ++i) sc_h[i] = 0.001f * float(1 + h32(3, i) % 7);
    uint16_t *da = nullptr, *o1 = nullptr, *o2 = nullptr; uint8_t* dw = nullptr; float* ds = nullptr;
    cudaMalloc(&da, act.size() * 2); cudaMalloc(&dw, w.size()); cudaMalloc(&ds, sc_h.size() * 4); cudaMalloc(&o1, size_t(s.m) * s.n * 2); cudaMalloc(&o2, size_t(s.m) * s.n * 2);
    cudaMemcpy(da, act.data(), act.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice); cudaMemcpy(ds, sc_h.data(), sc_h.size() * 4, cudaMemcpyHostToDevice);
    launch_scale_gemm_bf16(da, s.k, dw, ds, o1, s.m, s.n, s.k, nullptr, 0, 5);
    const int h = (s.m / 2) / 16 * 16;
    launch_scale_gemm_bf16(da, s.k, dw, ds, o2, h, s.n, s.k, nullptr, 0, 5);
    launch_scale_gemm_bf16(da + size_t(h) * s.k, s.k, dw, ds, o2 + size_t(h) * s.n, s.m - h, s.n, s.k, nullptr, 0, 5);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint16_t> a(size_t(s.m) * s.n), b(a.size());
    cudaMemcpy(a.data(), o1, a.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(b.data(), o2, b.size() * 2, cudaMemcpyDeviceToHost);
    size_t d = 0; for (size_t i = 0; i < a.size(); ++i) d += a[i] != b[i];
    std::printf("fp8 m=%d n=%d k=%d split at %d: %zu of %zu differ\n", s.m, s.n, s.k, h, d, a.size());
    cudaFree(da); cudaFree(dw); cudaFree(ds); cudaFree(o1); cudaFree(o2);
  }
  return 0;
}
