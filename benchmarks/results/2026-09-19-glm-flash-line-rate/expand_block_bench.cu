// Does a row-blocked expand + Lt (blocks that fit the 24 MB L2) beat the
// whole-matrix expand + Lt? Through the production seam: CublasLtGemm with a
// released companion and different scratch slot sizes, against the resident
// Lt call. Real KDA-class weights (the q slice tiled to in_proj's rows).
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
#include "kernels/gemm.hpp"
using namespace dgpp;
static std::vector<uint16_t> read_bin(const std::string& path, size_t elems) {
  std::vector<uint16_t> v(elems);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f || std::fread(v.data(), 2, elems, f) != elems) { std::perror(path.c_str()); std::exit(1); }
  std::fclose(f);
  return v;
}
int main() {
  const std::string dir = "./";
  const int k = 4096, n = 6416;
  const std::vector<uint16_t> q = read_bin(dir + "w_q_2048x4096.bin", size_t(2048) * k);
  std::vector<uint16_t> w(size_t(n) * k);
  for (int r = 0; r < n; ++r) std::memcpy(&w[size_t(r) * k], &q[size_t(r % 2048) * k], size_t(k) * 2);
  const Bf12Host h = bf12_encode(w.data(), n, k);
  auto r16 = [](size_t b) { return (b + 15) / 16 * 16; };
  const size_t pb = h.packed.size(), bb = r16(h.rows.size() * 4), eb = r16(h.esc.size() * 4), rb = r16(h.raw.size() * 2);
  // Several copies of the weights so every pass is cold (as a 34-layer walk is).
  constexpr int kCopies = 6;
  std::vector<uint8_t*> packs(kCopies);
  std::vector<uint16_t*> raws(kCopies);
  std::vector<Bf12Matrix> ms(kCopies);
  for (int c = 0; c < kCopies; ++c) {
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&packs[c]), pb + bb + eb + rb));
    cudaMemcpy(packs[c], h.packed.data(), pb, cudaMemcpyHostToDevice);
    cudaMemcpy(packs[c] + pb, h.rows.data(), h.rows.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(packs[c] + pb + bb, h.esc.data(), h.esc.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(packs[c] + pb + bb + eb, h.raw.data(), h.raw.size() * 2, cudaMemcpyHostToDevice);
    ms[c].packed = packs[c];
    ms[c].rows = reinterpret_cast<const uint32_t*>(packs[c] + pb);
    ms[c].esc = reinterpret_cast<const uint32_t*>(packs[c] + pb + bb);
    ms[c].raw = reinterpret_cast<const uint16_t*>(packs[c] + pb + bb + eb);
    ms[c].n = n; ms[c].k = k;
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&raws[c]), w.size() * 2));
    cudaMemcpy(raws[c], w.data(), w.size() * 2, cudaMemcpyHostToDevice);
  }
  void* ws = nullptr; DGPP_CUDA_OK(cudaMalloc(&ws, 64u << 20));
  cudaStream_t stream; cudaStreamCreate(&stream);
  for (int m : {8, 64, 512, 1024}) {
    std::vector<uint16_t> ha(size_t(m) * k);
    uint64_t r = 0x9E3779B97F4A7C15ull;
    for (auto& v : ha) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; v = float_to_bf16_bits((int(r % 2001) - 1000) / 500.0f); }
    uint16_t* a = nullptr; uint16_t* out = nullptr;
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&a), ha.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&out), size_t(m) * n * 2));
    cudaMemcpy(a, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice);
    auto bench = [&](CublasLtGemm& g, const char* label, std::vector<uint8_t>* keep) {
      for (int c = 0; c < kCopies; ++c) g.matmul(a, raws[c], out, m, n, k, DType::BF16, GemmOut::BF16, k, ws, 64u << 20, stream);
      cudaStreamSynchronize(stream);
      std::vector<double> t;
      for (int it = 0; it < 5; ++it) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int c = 0; c < kCopies; ++c) g.matmul(a, raws[c], out, m, n, k, DType::BF16, GemmOut::BF16, k, ws, 64u << 20, stream);
        cudaStreamSynchronize(stream);
        t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / kCopies);
      }
      std::sort(t.begin(), t.end());
      std::vector<uint8_t> got(size_t(m) * n * 2);
      cudaMemcpy(got.data(), out, got.size(), cudaMemcpyDeviceToHost);
      bool same = true;
      if (keep->empty()) *keep = got; else same = *keep == got;
      std::printf("  m=%4d %-28s %9.1f us per call %s\n", m, label, t[t.size() / 2], same ? "" : "(NOT bitwise!)");
    };
    std::vector<uint8_t> ref;
    { CublasLtGemm g; g.set_decode_rows(4); bench(g, "resident Lt (bf16 in place)", &ref); }
    for (size_t slot_mb : {64, 12, 8, 6}) {
      CublasLtGemm g; g.set_decode_rows(4);
      for (int c = 0; c < kCopies; ++c) { g.register_bf12(raws[c], ms[c]); g.bf12_release_raw(raws[c]); }
      g.bf12_reserve_expand(1, slot_mb << 20);
      const std::string label = "released, slot " + std::to_string(slot_mb) + " MiB";
      bench(g, label.c_str(), &ref);
    }
    cudaFree(a); cudaFree(out);
  }
  return 0;
}
