// Does a row's cuBLASLt result depend on the rows sharing its call?
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/gemm.hpp"
using namespace dgpp;
int main() {
  struct Shape { int m, n, k; bool f32; };
  const Shape shapes[] = {{2048, 6416, 4096, false}, {2048, 4096, 2048, false}, {2048, 2048, 128, false},
                          {1989, 6416, 4096, false}, {2048, 4096, 1536, false}, {2048, 128, 4096, false},
                          {2048, 32, 4096, true}, {1024, 6416, 4096, false}};
  CublasLtGemm gemm;
  void* ws = nullptr; DGPP_CUDA_OK(cudaMalloc(&ws, 64u << 20));
  for (const Shape& s : shapes) {
    std::vector<uint16_t> ha(size_t(s.m) * s.k), hw(size_t(s.n) * s.k);
    uint64_t r = 0x9E3779B97F4A7C15ull;
    auto rnd = [&] { r ^= r << 13; r ^= r >> 7; r ^= r << 17; return (int(r % 2001) - 1000) / 500.0f; };
    for (auto& v : ha) v = float_to_bf16_bits(rnd());
    for (auto& v : hw) v = float_to_bf16_bits(rnd() * 0.02f);
    uint16_t *a = nullptr, *w = nullptr; void *o1 = nullptr, *o2 = nullptr;
    const size_t ob = size_t(s.m) * s.n * (s.f32 ? 4 : 2);
    DGPP_CUDA_OK(cudaMalloc(&a, ha.size() * 2)); DGPP_CUDA_OK(cudaMalloc(&w, hw.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&o1, ob)); DGPP_CUDA_OK(cudaMalloc(&o2, ob));
    cudaMemcpy(a, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    const GemmOut od = s.f32 ? GemmOut::F32 : GemmOut::BF16;
    gemm.matmul(a, w, o1, s.m, s.n, s.k, DType::BF16, od, s.k, ws, 64u << 20, nullptr);
    const int h = (s.m / 2) & ~3;
    gemm.set_plan_rows(s.m);
    gemm.matmul(a, w, o2, h, s.n, s.k, DType::BF16, od, s.k, ws, 64u << 20, nullptr);
    gemm.matmul(a + size_t(h) * s.k, w, static_cast<uint8_t*>(o2) + size_t(h) * s.n * (s.f32 ? 4 : 2), s.m - h, s.n, s.k,
                DType::BF16, od, s.k, ws, 64u << 20, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    gemm.set_plan_rows(0);
    std::vector<uint8_t> b1(ob), b2(ob);
    cudaMemcpy(b1.data(), o1, ob, cudaMemcpyDeviceToHost); cudaMemcpy(b2.data(), o2, ob, cudaMemcpyDeviceToHost);
    size_t diff = 0; const size_t es = s.f32 ? 4 : 2;
    for (size_t i = 0; i < ob; i += es) diff += std::memcmp(&b1[i], &b2[i], es) != 0;
    std::printf("m=%d n=%d k=%d %s: split at %d -> %zu of %zu elements differ\n", s.m, s.n, s.k, s.f32 ? "f32" : "bf16", h, diff, ob / es);
    cudaFree(a); cudaFree(w); cudaFree(o1); cudaFree(o2);
  }
  return 0;
}
