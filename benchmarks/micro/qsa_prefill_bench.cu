// Paired listed-QSA prefill timings on idle hardware. Synthetic values with
// paged, request-specific sparse lists; compare every FP32 partial exactly.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "kernels/qsa.hpp"

namespace {
template <class T> struct Buffer {
  T* p = nullptr;
  size_t size;
  explicit Buffer(size_t n) : size(n) { DGPP_CUDA_OK(cudaMallocManaged(&p, n * sizeof(T))); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
double time_ms(const std::function<void()>& fn, int iters) {
  for (int i = 0; i < 3; ++i) fn();
  cudaEvent_t start, end;
  DGPP_CUDA_OK(cudaEventCreate(&start));
  DGPP_CUDA_OK(cudaEventCreate(&end));
  DGPP_CUDA_OK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) fn();
  DGPP_CUDA_OK(cudaEventRecord(end));
  DGPP_CUDA_OK(cudaEventSynchronize(end));
  float ms = 0;
  DGPP_CUDA_OK(cudaEventElapsedTime(&ms, start, end));
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return ms / iters;
}
}

int main(int argc, char** argv) {
  try {
    int rows = 256, context = 8192, heads = 12, kv = 1, splits = 8, iters = 10, trials = 6;
    bool graph = false, warp = false;
    int mma = 1, mx = 0, selected_opt = 2051;
    for (int i = 1; i < argc; ++i) {
      std::string a = argv[i];
      if (a == "--graph") { graph = true; continue; }
      if (a == "--warp") { warp = true; continue; }
      if (i + 1 == argc) throw std::invalid_argument("missing value");
      int v = std::stoi(argv[++i]);
      if (a == "--rows") rows = v;
      else if (a == "--context") context = v;
      else if (a == "--heads") heads = v;
      else if (a == "--kv-heads") kv = v;
      else if (a == "--splits") splits = v;
      else if (a == "--iters") iters = v;
      else if (a == "--trials") trials = v;
      else if (a == "--mma") mma = v;
      else if (a == "--mx") mx = v;
      else if (a == "--selected") selected_opt = v;
      else throw std::invalid_argument("unknown option: " + a);
    }
    if (rows < 1 || context < rows || heads < 1 || kv < 1 || heads % kv || splits < 1 || iters < 1 || trials < 1)
      throw std::invalid_argument("invalid geometry");
    constexpr int dim = 256, block = 256, requests = 2;
    const int selected = selected_opt;
    const int blocks = (context + block - 1) / block, slots = blocks * block * requests;
    const size_t qw = heads * dim + 8;  // Exercise a padded query stride.
    Buffer<uint16_t> q(rows * qw), k(static_cast<size_t>(slots) * kv * dim), v(k.size);
    Buffer<int32_t> table(blocks * requests), req(rows), lists(rows * selected), counts(rows);
    std::mt19937 rng(20260915);
    std::normal_distribution<float> normal;
    for (auto* b : {&q, &k, &v})
      for (size_t i = 0; i < b->size; ++i) b->p[i] = dgpp::float_to_bf16_bits(normal(rng));
    std::iota(table.p, table.p + table.size, 0);
    std::shuffle(table.p, table.p + table.size, rng);
    std::fill(lists.p, lists.p + lists.size, -1);
    for (int r = 0; r < rows; ++r) {
      req.p[r] = r % requests;
      const int visible = context - rows + r + 1;
      counts.p[r] = r % 37 == 36 ? 0 : std::min(selected, visible);
      std::vector<int> tokens(visible);
      std::iota(tokens.begin(), tokens.end(), 0);
      std::shuffle(tokens.begin(), tokens.end(), rng);
      tokens.resize(counts.p[r]);
      std::sort(tokens.begin(), tokens.end());
      std::copy(tokens.begin(), tokens.end(), lists.p + r * selected);
    }
    if (warp) {
      // The warp kernel (qsa_warp.cu): one warp per (query, KV group). The fp8
      // scale planes are always passed so mma=0 exercises the in-kernel dequant
      // path (the lane's MMA=0 shape) and mma=1 the raw E4M3 tensor-core path;
      // the DGPP_QSA_FP8_MMA toggle (latched per process by the wrapper) selects
      // between them, DGPP_QSA_FP8_MX (read per call) the C.1a block-scale.
      setenv("DGPP_QSA_FP8_MMA", mma ? "1" : "0", 1);
      setenv("DGPP_QSA_FP8_MX", mx ? "1" : "0", 1);
      Buffer<float> k_scale(static_cast<size_t>(slots) * kv), v_scale(k_scale.size);
      Buffer<uint8_t> k_bscale(static_cast<size_t>(slots) * kv * 8);
      for (size_t i = 0; i < k_scale.size; ++i) k_scale.p[i] = 1.0f;
      for (size_t i = 0; i < v_scale.size; ++i) v_scale.p[i] = 1.0f;
      for (size_t i = 0; i < k_bscale.size; ++i) k_bscale.p[i] = 127;  // 2^0 neutral
      Buffer<float> out(static_cast<size_t>(rows) * heads * dim), out2(out.size);
      auto wrun = [&] {
        dgpp::qsa_attn_prefill_warp(q.p, qw, k.p, v.p, req.p, lists.p, selected, counts.p, rows,
                                    heads, kv, block, table.p, blocks, 1.0f / 16, out.p, nullptr,
                                    k_scale.p, v_scale.p, mx ? k_bscale.p : nullptr);
      };
      wrun();
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      // Determinism: a second run must be bit-identical (the DSA pin).
      std::memcpy(out2.p, out.p, out.size * sizeof(float));
      wrun();
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      size_t mism = 0;
      for (size_t i = 0; i < out.size; ++i)
        mism += std::memcmp(&out.p[i], &out2.p[i], sizeof(float)) != 0;
      const double ms = time_ms(wrun, iters);
      std::printf("{\"mode\":\"warp\",\"mma\":%d,\"mx\":%d,\"rows\":%d,\"context\":%d,\"heads\":%d,"
                  "\"kv_heads\":%d,\"selected\":%d,\"iters\":%d,\"ms\":%.6f,"
                  "\"determinism_mismatches\":%zu}\n",
                  mma, mx, rows, context, heads, kv, selected, iters, ms, mism);
      if (mism) throw std::runtime_error("warp kernel is not deterministic");
      return 0;
    }
    const size_t part = static_cast<size_t>(rows) * splits * heads;
    Buffer<float> m0(part), l0(part), c0(part * dim), m1(part), l1(part), c1(part * dim);
    auto run = [&](bool candidate) {
      auto fn = candidate ? dgpp::qsa_attn_prefill_partial : dgpp::qsa_attn_partial;
      fn(q.p, qw, k.p, v.p, req.p, lists.p, selected, counts.p, rows, splits, heads, kv, dim,
         block, table.p, blocks, 1.0f / 16, candidate ? m1.p : m0.p, candidate ? l1.p : l0.p,
         candidate ? c1.p : c0.p, nullptr, nullptr, nullptr, nullptr);
    };
    cudaGraph_t graphs[2]{};
    cudaGraphExec_t execs[2]{};
    if (graph) {
      cudaStream_t stream;
      DGPP_CUDA_OK(cudaStreamCreate(&stream));
      for (int b = 0; b < 2; ++b) {
        DGPP_CUDA_OK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        auto fn = b ? dgpp::qsa_attn_prefill_partial : dgpp::qsa_attn_partial;
        fn(q.p, qw, k.p, v.p, req.p, lists.p, selected, counts.p, rows, splits, heads, kv, dim,
           block, table.p, blocks, 1.0f / 16, b ? m1.p : m0.p, b ? l1.p : l0.p, b ? c1.p : c0.p, stream,
           nullptr, nullptr, nullptr);
        DGPP_CUDA_OK(cudaStreamEndCapture(stream, &graphs[b]));
        DGPP_CUDA_OK(cudaGraphInstantiate(&execs[b], graphs[b], nullptr, nullptr, 0));
      }
      cudaStreamDestroy(stream);
    }
    auto launch = [&](bool b) {
      if (graph) DGPP_CUDA_OK(cudaGraphLaunch(execs[b], nullptr));
      else run(b);
    };
    for (int t = 0; t < trials; ++t) {
      double ms[2];
      for (int j = 0; j < 2; ++j) {
        const int b = (t + j) % 2;
        ms[b] = time_ms([&] { launch(b); }, iters);
      }
      size_t mismatches = 0;
      for (auto pair : {std::pair{&m0, &m1}, std::pair{&l0, &l1}, std::pair{&c0, &c1}})
        for (size_t i = 0; i < pair.first->size; ++i)
          mismatches += std::memcmp(pair.first->p + i, pair.second->p + i, sizeof(float)) != 0;
      std::printf("{\"rows\":%d,\"context\":%d,\"heads\":%d,\"kv_heads\":%d,\"splits\":%d,"
                  "\"graph\":%s,\"trial\":%d,\"baseline_ms\":%.6f,\"candidate_ms\":%.6f,\"mismatches\":%zu}\n",
                  rows, context, heads, kv, splits, graph ? "true" : "false", t, ms[0], ms[1], mismatches);
      if (mismatches) throw std::runtime_error("FP32 partials differ");
    }
    for (int b = 0; b < 2; ++b) {
      if (execs[b]) cudaGraphExecDestroy(execs[b]);
      if (graphs[b]) cudaGraphDestroy(graphs[b]);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
