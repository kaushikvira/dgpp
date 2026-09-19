// bf12_expand_bench: packed (12-bit) matrix -> its bf16 bits in a scratch
// buffer, the prefill path of bf12-only residency. Must be BITWISE the
// original; timed against a device memcpy of the bf16 bytes (the bandwidth
// reference) on real weights.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "kernels/bf12_gemv.hpp"
#include "kernels/gemv_common.cuh"

namespace {
using namespace dgpp;

// One step of one lane: eight elements' bf16 bits as one 16-byte vector.
__device__ __forceinline__ uint4 expand_step(uint32_t lo, uint32_t hi, uint32_t ew, uint32_t base7,
                                             int c0, const uint32_t* __restrict__ esc,
                                             uint32_t esc_b, uint32_t esc_e) {
  uint32_t v[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const uint32_t b = ((j < 4 ? lo : hi) >> (8 * (j & 3))) & 0xFFu;
    const uint32_t code = (ew >> (4 * j)) & 0xFu;
    v[j] = ((b & 0x80u) << 8) | (b & 0x7Fu) | (base7 + (code << 7));
  }
  const uint32_t any15 = ew & (ew >> 1) & (ew >> 2) & (ew >> 3) & 0x11111111u;
  if (any15 != 0u) {
    for (int j = 0; j < 8; ++j) {
      if (((ew >> (4 * j)) & 0xFu) != 0xFu) continue;
      const uint32_t col = static_cast<uint32_t>(c0 + j);
      for (uint32_t e = esc_b; e < esc_e; ++e) {
        const uint32_t ent = esc[e];
        if ((ent >> 16) == col) {
          v[j] = ent & 0xFFFFu;
          break;
        }
      }
    }
  }
  uint4 o;
  o.x = v[0] | (v[1] << 16);
  o.y = v[2] | (v[3] << 16);
  o.z = v[4] | (v[5] << 16);
  o.w = v[6] | (v[7] << 16);
  return o;
}

template <int kSB>
__global__ void bf12_expand_kernel(const uint8_t* __restrict__ w, const uint32_t* __restrict__ rows,
                                   const uint32_t* __restrict__ esc,
                                   const uint16_t* __restrict__ raw, uint16_t* __restrict__ out,
                                   int n, int k) {
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * gemv::kWarps + warp;
  if (row >= n) return;
  const int nsb = k / kBf12Super;
  const uint8_t* wr = w + static_cast<size_t>(row) * nsb * kBf12SuperBytes;
  const uint32_t eb = rows[2 * row];
  const uint32_t word = rows[2 * row + 1];
  const uint32_t ee = rows[2 * row + 2];
  uint16_t* o = out + static_cast<size_t>(row) * k;
  if ((word & kBf12RawRow) != 0u) {
    const uint16_t* rw = raw + static_cast<size_t>(word & ~kBf12RawRow) * k;
    for (int sb = 0; sb < nsb; ++sb) {
      uint4 v[4];
#pragma unroll
      for (int t = 0; t < 4; ++t)
        v[t] = *reinterpret_cast<const uint4*>(rw + sb * kBf12Super + t * 256 + lane * 8);
#pragma unroll
      for (int t = 0; t < 4; ++t)
        *reinterpret_cast<uint4*>(o + sb * kBf12Super + t * 256 + lane * 8) = v[t];
    }
    return;
  }
  const uint32_t base7 = (word >> 23) << 7;
  for (int s0 = 0; s0 < nsb; s0 += kSB) {
    uint4 a[kSB], b[kSB], e[kSB];
#pragma unroll
    for (int s = 0; s < kSB; ++s) {
      const int sb = s0 + s < nsb ? s0 + s : s0;
      const uint8_t* p = wr + static_cast<size_t>(sb) * kBf12SuperBytes + lane * 16;
      a[s] = *reinterpret_cast<const uint4*>(p);
      b[s] = *reinterpret_cast<const uint4*>(p + 512);
      e[s] = *reinterpret_cast<const uint4*>(p + 1024);
    }
#pragma unroll
    for (int s = 0; s < kSB; ++s) {
      if (s0 + s >= nsb) break;
      const int cb = (s0 + s) * kBf12Super + lane * 8;
      const uint4 v0 = expand_step(a[s].x, a[s].y, e[s].x, base7, cb, esc, eb, ee);
      const uint4 v1 = expand_step(a[s].z, a[s].w, e[s].y, base7, cb + 256, esc, eb, ee);
      const uint4 v2 = expand_step(b[s].x, b[s].y, e[s].z, base7, cb + 512, esc, eb, ee);
      const uint4 v3 = expand_step(b[s].z, b[s].w, e[s].w, base7, cb + 768, esc, eb, ee);
      *reinterpret_cast<uint4*>(o + cb) = v0;
      *reinterpret_cast<uint4*>(o + cb + 256) = v1;
      *reinterpret_cast<uint4*>(o + cb + 512) = v2;
      *reinterpret_cast<uint4*>(o + cb + 768) = v3;
    }
  }
}

template <int kSB>
void launch_expand(const Bf12Matrix& w, uint16_t* out, cudaStream_t stream) {
  const dim3 grid((w.n + gemv::kWarps - 1) / gemv::kWarps);
  bf12_expand_kernel<kSB><<<grid, gemv::kThreads, 0, stream>>>(w.packed, w.rows, w.esc, w.raw, out,
                                                                w.n, w.k);
  DGPP_CUDA_OK(cudaGetLastError());
}

std::vector<uint16_t> read_bin(const std::string& path, size_t elems) {
  std::vector<uint16_t> v(elems);
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { std::perror(path.c_str()); std::exit(1); }
  if (std::fread(v.data(), 2, elems, f) != elems) { std::fprintf(stderr, "short read %s\n", path.c_str()); std::exit(1); }
  std::fclose(f);
  return v;
}

struct DevMatrix {
  Bf12Matrix m;
  void* mem = nullptr;
};
DevMatrix upload(const Bf12Host& h, int n, int k) {
  auto r16 = [](size_t b) { return (b + 15) / 16 * 16; };
  const size_t pb = h.packed.size(), bb = r16(h.rows.size() * 4), eb = r16(h.esc.size() * 4),
               rb = r16(h.raw.size() * 2);
  DevMatrix d;
  DGPP_CUDA_OK(cudaMalloc(&d.mem, pb + bb + eb + rb));
  auto* dev = static_cast<uint8_t*>(d.mem);
  cudaMemcpy(dev, h.packed.data(), pb, cudaMemcpyHostToDevice);
  cudaMemcpy(dev + pb, h.rows.data(), h.rows.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev + pb + bb, h.esc.data(), h.esc.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dev + pb + bb + eb, h.raw.data(), h.raw.size() * 2, cudaMemcpyHostToDevice);
  d.m.packed = dev;
  d.m.rows = reinterpret_cast<const uint32_t*>(dev + pb);
  d.m.esc = reinterpret_cast<const uint32_t*>(dev + pb + bb);
  d.m.raw = reinterpret_cast<const uint16_t*>(dev + pb + bb + eb);
  d.m.n = n;
  d.m.k = k;
  return d;
}

double time_us(int iters, const auto& fn, cudaStream_t stream) {
  fn();
  cudaStreamSynchronize(stream);
  std::vector<double> t;
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    cudaStreamSynchronize(stream);
    t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

void run(const char* name, const std::string& path, int n, int k, int force_raw_rows) {
  std::vector<uint16_t> w = read_bin(path, static_cast<size_t>(n) * k);
  if (force_raw_rows > 0) {
    // Make a few rows pathological (spread exponents) so the raw-row path runs.
    for (int r = 0; r < force_raw_rows; ++r) {
      uint16_t* row = w.data() + static_cast<size_t>(r * 37 % n) * k;
      for (int c = 0; c < k; c += 13) row[c] = static_cast<uint16_t>((row[c] & 0x807F) | (((c * 7) % 250) << 7));
    }
  }
  const Bf12Host h = bf12_encode(w.data(), n, k);
  if (!h.ok) { std::printf("%s: not packable\n", name); return; }
  DevMatrix d = upload(h, n, k);
  const size_t bytes = static_cast<size_t>(n) * k * 2;
  uint16_t* out = nullptr;
  uint16_t* ref = nullptr;
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&out), bytes));
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&ref), bytes));
  cudaMemcpy(ref, w.data(), bytes, cudaMemcpyHostToDevice);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  // Distinct cold buffers per iteration are not needed here: the matrices
  // are far past the 24 MB L2 (or the head: 317 MB), so every pass is cold.
  const double t2 = time_us(30, [&] { launch_expand<2>(d.m, out, stream); }, stream);
  const double t4 = time_us(30, [&] { launch_expand<4>(d.m, out, stream); }, stream);
  const double tc = time_us(30, [&] { cudaMemcpyAsync(out, ref, bytes, cudaMemcpyDeviceToDevice, stream); }, stream);
  launch_expand<4>(d.m, out, stream);
  cudaStreamSynchronize(stream);
  std::vector<uint16_t> back(static_cast<size_t>(n) * k);
  cudaMemcpy(back.data(), out, bytes, cudaMemcpyDeviceToHost);
  const bool same = std::memcmp(back.data(), w.data(), bytes) == 0;
  const double moved = (bytes * 0.75 + bytes) / 1e9;
  std::printf("%-6s [%6d x %5d] %7.1f MB bf16, esc %zu raw rows %d: expand kSB2 %8.1f us, kSB4 %8.1f us "
              "(%.0f GB/s moved), memcpy d2d %8.1f us (%.0f GB/s moved) — %s\n",
              name, n, k, bytes / 1e6, h.escapes, h.raw_rows, t2, t4, moved / (t4 * 1e-6),
              tc, 2.0 * bytes / 1e9 / (tc * 1e-6), same ? "BITWISE" : "MISMATCH");
  cudaFree(out);
  cudaFree(ref);
  cudaFree(d.mem);
  cudaStreamDestroy(stream);
}
}  // namespace

int main() {
  const std::string dir = "./";
  run("q", dir + "w_q_2048x4096.bin", 2048, 4096, 0);
  run("o", dir + "w_o_4096x2048.bin", 4096, 2048, 0);
  run("head", dir + "w_head_38720x4096.bin", 38720, 4096, 0);
  run("q+raw", dir + "w_q_2048x4096.bin", 2048, 4096, 5);
  return 0;
}
