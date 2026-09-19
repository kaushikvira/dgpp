// KDA layer at the production TP=4 geometry: one T-row prefill call against
// two row blocks with carried state (the fold overlap's attention site).
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "core/arena.hpp"
#include "kernels/gemm.hpp"
#include "models/kda_geometry.hpp"
#include "models/kda_layer.hpp"
#include "models/kda_state.hpp"
using namespace dgpp;
static uint32_t h32(uint64_t seed, uint64_t idx) { uint64_t x = seed * 0x9E3779B97F4A7C15ULL ^ idx; x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return uint32_t(x ^ (x >> 32)); }
static std::vector<uint16_t> rnd(uint64_t seed, int64_t n, int lo, int hi) { std::vector<uint16_t> v(n); const uint32_t span = hi - lo; for (int64_t i = 0; i < n; ++i) { const uint32_t h = h32(seed, i); const uint32_t e = uint32_t(lo + 127) + (h >> 8) % span; v[i] = uint16_t((h & 0x8000u) | (e << 7) | (h & 0x7Fu)); } return v; }
struct Dev { void* p = nullptr; size_t n = 0; explicit Dev(size_t b) : n(b ? b : 16) { DGPP_CUDA_OK(cudaMalloc(&p, n)); } ~Dev() { cudaFree(p); } void up(const void* h, size_t b) { DGPP_CUDA_OK(cudaMemcpy(p, h, b, cudaMemcpyHostToDevice)); } };
int main(int argc, char** argv) {
  const int T = argc > 1 ? atoi(argv[1]) : 2048;
  const bool pin = argc > 2 ? atoi(argv[2]) != 0 : true;
  KdaConfig cfg; cfg.tp_size = 4;
  const KdaGeometry g = KdaGeometry::from_config(cfg);
  cudaStream_t s; DGPP_CUDA_OK(cudaStreamCreate(&s));
  auto in_proj = rnd(11, int64_t(g.in_proj_cols) * cfg.hidden, -6, -3);
  auto f_b = rnd(12, int64_t(g.local_proj) * cfg.head_dim, -3, 0), g_b = rnd(13, int64_t(g.local_proj) * cfg.head_dim, -3, 0);
  auto conv = rnd(14, int64_t(g.conv_channels) * cfg.conv_width, -3, 0), o_norm = rnd(15, cfg.head_dim, -2, 0);
  auto o_proj = rnd(16, int64_t(cfg.hidden) * g.local_proj, -6, -3);
  std::vector<float> a_log(g.local_heads, 0.f), dt_bias(g.local_proj, 0.f);
  auto hid = rnd(17, int64_t(T) * cfg.hidden, -2, 0);
  Dev d_in_proj(in_proj.size() * 2), d_f_b(f_b.size() * 2), d_g_b(g_b.size() * 2), d_conv(conv.size() * 2), d_o_norm(o_norm.size() * 2), d_o_proj(o_proj.size() * 2), d_a(a_log.size() * 4), d_dt(dt_bias.size() * 4), d_in(hid.size() * 2), d_out1(size_t(T) * cfg.hidden * 2), d_out2(size_t(T) * cfg.hidden * 2);
  d_in_proj.up(in_proj.data(), in_proj.size() * 2); d_f_b.up(f_b.data(), f_b.size() * 2); d_g_b.up(g_b.data(), g_b.size() * 2); d_conv.up(conv.data(), conv.size() * 2); d_o_norm.up(o_norm.data(), o_norm.size() * 2); d_o_proj.up(o_proj.data(), o_proj.size() * 2); d_a.up(a_log.data(), a_log.size() * 4); d_dt.up(dt_bias.data(), dt_bias.size() * 4); d_in.up(hid.data(), hid.size() * 2);
  Arena arena; Arena::Config ac; ac.persistent_hot = KdaLayer::persistent_hot_bytes(cfg, T); arena.init(ac);
  CublasLtGemm gemm; Dev ws(64ull << 20);
  KdaLayerWeights w; w.in_proj = d_in_proj.p; w.f_b = d_f_b.p; w.g_b = d_g_b.p; w.conv = d_conv.p; w.a_log = static_cast<float*>(d_a.p); w.dt_bias = static_cast<float*>(d_dt.p); w.o_norm = d_o_norm.p; w.o_proj = d_o_proj.p;
  KdaLayer layer(arena, gemm, w, cfg, T, ws.p, ws.n);
  Arena pa; Arena::Config pc; pc.persistent_hot = 2 * g.slot_bytes; pa.init(pc);
  KdaStatePool pool; pool.init(pa, cfg, 2); pool.zero_all(s);
  if (!layer.prepare(T)) return 2;
  layer.enqueue(d_in.p, pool.recurrent(0, 0), pool.conv(0, 0), g.conv_state_width, d_out1.p, T, s);
  const int TA = (T / 2) / 16 * 16, TB = T - TA;
  if (pin) gemm.set_plan_rows(T);
  if (!layer.prepare(TA)) return 2;
  layer.enqueue(d_in.p, pool.recurrent(0, 1), pool.conv(0, 1), g.conv_state_width, d_out2.p, TA, s);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  if (!layer.prepare(TB)) return 2;
  layer.enqueue(static_cast<uint16_t*>(d_in.p) + size_t(TA) * cfg.hidden, pool.recurrent(0, 1), pool.conv(0, 1), g.conv_state_width, static_cast<uint16_t*>(d_out2.p) + size_t(TA) * cfg.hidden, TB, s);
  gemm.set_plan_rows(0);
  DGPP_CUDA_OK(cudaStreamSynchronize(s));
  std::vector<uint16_t> o1(size_t(T) * cfg.hidden), o2(o1.size());
  cudaMemcpy(o1.data(), d_out1.p, o1.size() * 2, cudaMemcpyDeviceToHost); cudaMemcpy(o2.data(), d_out2.p, o2.size() * 2, cudaMemcpyDeviceToHost);
  size_t diff = 0, first = o1.size(); for (size_t i = 0; i < o1.size(); ++i) if (o1[i] != o2[i]) { if (first == o1.size()) first = i; ++diff; }
  std::vector<float> s1(g.recurrent_bytes / 4), s2(s1.size());
  cudaMemcpy(s1.data(), pool.recurrent(0, 0), s1.size() * 4, cudaMemcpyDeviceToHost); cudaMemcpy(s2.data(), pool.recurrent(0, 1), s2.size() * 4, cudaMemcpyDeviceToHost);
  const bool state_same = std::memcmp(s1.data(), s2.data(), s1.size() * 4) == 0;
  std::printf("KDA T=%d split at %d (plan rows %s): %zu of %zu outputs differ (first at row %zu), final state %s\n", T, TA, pin ? "pinned" : "own", diff, o1.size(), first / cfg.hidden, state_same ? "identical" : "DIFFERENT");
  return 0;
}
