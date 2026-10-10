// C.1a (d): the prefix-cache block copy must carry the MXFP8 block-scale
// plane (docs/qwen_fp8_phase_c_plan.md §3). A missed copy silently corrupts
// cached prefixes, whose K codes are block-quantized, so this writes distinct
// bytes to block 0's K/V block planes, copies block 0 -> block 1, and checks
// block 1's plane matches (a missed copy leaves it zeroed from reset_all and
// fails loudly).
//
// A .cpp (not .cu) test: QwenKvPool's header chain (models/qwen/layers.hpp ->
// loader.hpp -> minijson) does not compile under nvcc, so this runs the pool
// through the host compiler.
//
// Run:  ctest -R qwen_fp8_block_copy   (or the built binary directly)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/test.hpp"
#include "kernels/latent_format.hpp"
#include "models/qwen/kv_pool.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Minimal device buffer (this test is self-contained; the kda_test_helpers one
// drags in the KDA model headers).
struct DevBuf {
  void* p = nullptr;
  size_t bytes = 0;
  explicit DevBuf(size_t n) {
    if (n == 0) n = 16;
    if (cudaMalloc(&p, n) != cudaSuccess) throw std::runtime_error("test cudaMalloc failed");
    bytes = n;
  }
  DevBuf(DevBuf&&) = delete;
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  ~DevBuf() {
    if (p) cudaFree(p);
  }
  void upload(const void* h, size_t n) {
    if (n > bytes) throw std::runtime_error("upload overruns buffer");
    if (cudaMemcpy(p, h, n, cudaMemcpyHostToDevice) != cudaSuccess)
      throw std::runtime_error("upload failed");
  }
  void download(void* h, size_t n) const {
    if (n > bytes) throw std::runtime_error("download overruns buffer");
    if (cudaMemcpy(h, p, n, cudaMemcpyDeviceToHost) != cudaSuccess)
      throw std::runtime_error("download failed");
  }
  template <typename T>
  T* as() {
    return static_cast<T*>(p);
  }
};

}  // namespace

DGPP_TEST(qwen_fp8_block_copy_carries_block_scales) {
  dgpp::QwenKvPoolShape s;
  s.layers = 1;
  s.kv_heads = 2;
  s.dim = 256;
  s.idx_dim = 128;
  s.kpool = 4;
  s.block_tokens = 16;
  s.max_requests = 4;
  s.token_slots = 64;
  s.format = dgpp::LatentFormat::kFp8;
  dgpp::QwenKvPool pool;
  pool.init(s);
  cudaStream_t st = nullptr;
  const auto c = pool.view(0);
  require(c.k_bscale != nullptr && c.v_bscale != nullptr,
          "fp8 pool must expose the MXFP8 block-scale plane");
  const size_t bytes = static_cast<size_t>(s.block_tokens) * s.kv_heads * 8;
  std::vector<uint8_t> kb(bytes), vb(bytes);
  for (size_t i = 0; i < bytes; ++i) {
    kb[i] = static_cast<uint8_t>(0x40 + (i * 7) % 127);
    vb[i] = static_cast<uint8_t>(0x80 + (i * 5) % 60);
  }
  DGPP_CUDA_OK(cudaMemcpy(c.k_bscale, kb.data(), bytes, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(c.v_bscale, vb.data(), bytes, cudaMemcpyHostToDevice));
  pool.copy_block_contents(0, 1, st);
  DGPP_CUDA_OK(cudaStreamSynchronize(st));
  DevBuf kb1(bytes), vb1(bytes);
  DGPP_CUDA_OK(cudaMemcpy(kb1.p, c.k_bscale + bytes, bytes, cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(vb1.p, c.v_bscale + bytes, bytes, cudaMemcpyDeviceToHost));
  std::vector<uint8_t> kb1_h(bytes), vb1_h(bytes);
  kb1.download(kb1_h.data(), bytes);
  vb1.download(vb1_h.data(), bytes);
  if (std::memcmp(kb1_h.data(), kb.data(), bytes) != 0)
    throw std::runtime_error("block copy missed the K block-scale plane");
  if (std::memcmp(vb1_h.data(), vb.data(), bytes) != 0)
    throw std::runtime_error("block copy missed the V block-scale plane");
  std::printf("[fp8-block-copy] block copy carries the MXFP8 block-scale plane (K+V, %zu bytes/block)\n",
              bytes);
}

int main() { return dgpp::test::run_all(); }
