// mHC site + stream update at H=4096: T rows at once against two row blocks.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/glm_mhc_launch.hpp"
using namespace dgpp;
static uint32_t h32(uint64_t seed, uint64_t idx) { uint64_t x = seed * 0x9E3779B97F4A7C15ULL ^ idx; x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return uint32_t(x ^ (x >> 32)); }
static std::vector<uint16_t> rnd(uint64_t seed, int64_t n, int lo, int hi) { std::vector<uint16_t> v(n); const uint32_t span = hi - lo; for (int64_t i = 0; i < n; ++i) { const uint32_t h = h32(seed, i); const uint32_t e = uint32_t(lo + 127) + (h >> 8) % span; v[i] = uint16_t((h & 0x8000u) | (e << 7) | (h & 0x7Fu)); } return v; }
template <typename T> static T* dev(size_t n) { void* p = nullptr; DGPP_CUDA_OK(cudaMalloc(&p, n * sizeof(T))); DGPP_CUDA_OK(cudaMemset(p, 0, n * sizeof(T))); return static_cast<T*>(p); }
int main(int argc, char** argv) {
  const int T = argc > 1 ? atoi(argv[1]) : 2048;
  GlmMhcConfig cfg; const int H = cfg.hidden, n = cfg.hc_mult; const size_t H4 = size_t(n) * H, C = cfg.coeff_rows();
  auto fn = rnd(1, int64_t(C) * H4, -6, -3), ln = rnd(2, H, -1, 1), streams = rnd(3, int64_t(T) * H4, -2, 1), sub = rnd(4, int64_t(T) * H, -2, 1);
  std::vector<float> base(C, 0.1f), scale(3, 0.5f);
  auto* d_fn = dev<uint16_t>(fn.size()); auto* d_ln = dev<uint16_t>(H); auto* d_st = dev<uint16_t>(streams.size()); auto* d_sub = dev<uint16_t>(sub.size());
  auto* d_base = dev<float>(C); auto* d_scale = dev<float>(3);
  cudaMemcpy(d_fn, fn.data(), fn.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(d_ln, ln.data(), H * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_st, streams.data(), streams.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(d_sub, sub.data(), sub.size() * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_base, base.data(), C * 4, cudaMemcpyHostToDevice); cudaMemcpy(d_scale, scale.data(), 12, cudaMemcpyHostToDevice);
  GlmMhcWeights w; w.fn = d_fn; w.base = d_base; w.scale = d_scale;
  struct Out { uint16_t *post, *comb, *normed, *next; float* logits; int* counters; };
  auto mk = [&] { return Out{dev<uint16_t>(size_t(T) * n), dev<uint16_t>(size_t(T) * n * n), dev<uint16_t>(size_t(T) * H), dev<uint16_t>(size_t(T) * H4), dev<float>(size_t(T) * C), dev<int>(T)}; };
  Out a = mk(), b = mk();
  launch_mhc_compute_normed(d_st, w, cfg, nullptr, a.post, a.comb, a.logits, d_ln, a.normed, 1e-5f, T, nullptr, a.counters, false);
  launch_mhc_stream_update(a.post, a.comb, d_sub, d_st, a.next, cfg, T, nullptr);
  const int TA = (T / 2) / 16 * 16;
  for (auto [r0, rows] : {std::pair<int,int>{0, TA}, {TA, T - TA}}) {
    launch_mhc_compute_normed(d_st + size_t(r0) * H4, w, cfg, nullptr, b.post + size_t(r0) * n, b.comb + size_t(r0) * n * n, b.logits + size_t(r0) * C, d_ln, b.normed + size_t(r0) * H, 1e-5f, rows, nullptr, b.counters + r0, false);
    launch_mhc_stream_update(b.post + size_t(r0) * n, b.comb + size_t(r0) * n * n, d_sub + size_t(r0) * H, d_st + size_t(r0) * H4, b.next + size_t(r0) * H4, cfg, rows, nullptr);
  }
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  auto cmp = [&](const char* name, const void* x, const void* y, size_t bytes) { std::vector<uint8_t> hx(bytes), hy(bytes); cudaMemcpy(hx.data(), x, bytes, cudaMemcpyDeviceToHost); cudaMemcpy(hy.data(), y, bytes, cudaMemcpyDeviceToHost); size_t d = 0; for (size_t i = 0; i < bytes; ++i) d += hx[i] != hy[i]; std::printf("  %-7s %s (%zu bytes differ)\n", name, d ? "DIFFERENT" : "identical", d); };
  std::printf("mHC T=%d split at %d:\n", T, TA);
  cmp("post", a.post, b.post, size_t(T) * n * 2); cmp("comb", a.comb, b.comb, size_t(T) * n * n * 2); cmp("normed", a.normed, b.normed, size_t(T) * H * 2); cmp("streams", a.next, b.next, size_t(T) * H4 * 2);
  return 0;
}
