// bf12 vs bf16 narrow GEMV on Qwen3.8-Flash-Next's four-node shapes, real
// weights. COLD: enough copies of each matrix that a pass over them is far
// past the 24 MB L2 (a decode step walks ~100 distinct matrices); WARM: one
// copy re-read (what a kernel sees behind the L2 prefetcher). Bitwise check.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/bf12_gemv.hpp"
#include "kernels/bf16_gemv.hpp"
using namespace dgpp;
static std::vector<uint16_t> read_bin(const std::string& path, size_t elems) {
  std::vector<uint16_t> v(elems);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f || std::fread(v.data(), 2, elems, f) != elems) { std::perror(path.c_str()); std::exit(1); }
  std::fclose(f);
  return v;
}
struct Dev { uint16_t* raw; Bf12Matrix m; };
int main() {
  const std::string dir = "./qwen/"  /* the dumps of dump_qwen.py */;
  struct Shape { const char* name; const char* file; int n, k; };
  const Shape shapes[] = {{"GDN qkv", "w_qkv_2560x2560.bin", 2560, 2560}, {"GDN out", "w_out_2560x1536.bin", 2560, 1536},
                          {"GR down", "w_grdown_320x10240.bin", 320, 10240}, {"GR up", "w_grup_10240x320.bin", 10240, 320},
                          {"lm head", "w_head_62080x2560.bin", 62080, 2560}};
  cudaStream_t stream; cudaStreamCreate(&stream);
  for (const Shape& s : shapes) {
    const std::vector<uint16_t> w = read_bin(dir + s.file, size_t(s.n) * s.k);
    const Bf12Host h = bf12_encode(w.data(), s.n, s.k);
    if (!h.ok) { std::printf("%s: not packable\n", s.name); continue; }
    const size_t raw_bytes = w.size() * 2;
    const int copies = raw_bytes > (200u << 20) ? 2 : std::max<int>(2, int((300u << 20) / raw_bytes));
    auto r16 = [](size_t b) { return (b + 15) / 16 * 16; };
    const size_t pb = h.packed.size(), bb = r16(h.rows.size() * 4), eb = r16(h.esc.size() * 4), rb = r16(h.raw.size() * 2);
    std::vector<Dev> d(copies);
    for (int c = 0; c < copies; ++c) {
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d[c].raw), raw_bytes));
      cudaMemcpy(d[c].raw, w.data(), raw_bytes, cudaMemcpyHostToDevice);
      uint8_t* p = nullptr;
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&p), pb + bb + eb + rb));
      cudaMemcpy(p, h.packed.data(), pb, cudaMemcpyHostToDevice);
      cudaMemcpy(p + pb, h.rows.data(), h.rows.size() * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(p + pb + bb, h.esc.data(), h.esc.size() * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(p + pb + bb + eb, h.raw.data(), h.raw.size() * 2, cudaMemcpyHostToDevice);
      d[c].m.packed = p; d[c].m.rows = reinterpret_cast<const uint32_t*>(p + pb);
      d[c].m.esc = reinterpret_cast<const uint32_t*>(p + pb + bb);
      d[c].m.raw = reinterpret_cast<const uint16_t*>(p + pb + bb + eb);
      d[c].m.n = s.n; d[c].m.k = s.k;
    }
    std::printf("%-8s [%6d x %5d] %7.2f MB bf16 -> %7.2f MB packed (%.3f), %d copies\n", s.name, s.n, s.k,
                raw_bytes / 1e6, (pb + bb + eb + rb) / 1e6, double(pb + bb + eb + rb) / raw_bytes, copies);
    for (int m : {1, 2}) {
      if (size_t(m) * s.k * 2 > 48 * 1024) continue;
      std::vector<uint16_t> ha(size_t(m) * s.k);
      uint64_t r = 0x9E3779B97F4A7C15ull;
      for (auto& v : ha) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; v = float_to_bf16_bits((int(r % 2001) - 1000) / 500.0f); }
      uint16_t* a = nullptr; uint16_t *o16 = nullptr, *o12 = nullptr;
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&a), ha.size() * 2));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&o16), size_t(m) * s.n * 2));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&o12), size_t(m) * s.n * 2));
      cudaMemcpy(a, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice);
      auto time_us = [&](bool packed, bool cold) {
        std::vector<double> t;
        for (int it = 0; it < 7; ++it) {
          const int passes = cold ? copies : 8;
          const auto t0 = std::chrono::steady_clock::now();
          for (int c = 0; c < passes; ++c) {
            const Dev& x = d[cold ? c : 0];
            if (packed) launch_bf12_gemv(a, s.k, x.m, o12, false, m, stream);
            else launch_bf16_gemv(a, s.k, x.raw, o16, false, m, s.n, s.k, stream);
          }
          cudaStreamSynchronize(stream);
          t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / passes);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
      };
      const double c16 = time_us(false, true), c12 = time_us(true, true), w16 = time_us(false, false), w12 = time_us(true, false);
      std::vector<uint16_t> b16(size_t(m) * s.n), b12(size_t(m) * s.n);
      cudaMemcpy(b16.data(), o16, b16.size() * 2, cudaMemcpyDeviceToHost);
      cudaMemcpy(b12.data(), o12, b12.size() * 2, cudaMemcpyDeviceToHost);
      std::printf("   m=%d  cold: bf16 %8.1f us (%5.0f GB/s)  bf12 %8.1f us (%+5.1f %%)   warm: bf16 %8.1f us  bf12 %8.1f us (%+5.1f %%)  %s\n", m,
                  c16, raw_bytes / 1e9 / (c16 * 1e-6), c12, (c12 / c16 - 1) * 100, w16, w12, (w12 / w16 - 1) * 100,
                  b16 == b12 ? "BITWISE" : "MISMATCH");
      cudaFree(a); cudaFree(o16); cudaFree(o12);
    }
    for (Dev& x : d) { cudaFree(x.raw); cudaFree(const_cast<uint8_t*>(x.m.packed)); }
  }
  return 0;
}
