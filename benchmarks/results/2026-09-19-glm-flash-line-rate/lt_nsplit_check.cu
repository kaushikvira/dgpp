// Does a cuBLASLt output element depend on the WEIGHT rows sharing its call?
// out[m, n] = act[m, k] x W[n, k]^T run whole, against the same product run
// over weight-row blocks [r0, r1) (W + r0 * k, out column offset r0, the
// output's leading dimension still n) with the whole call's algorithm pinned.
// Bitwise here lets a bf12-only matrix larger than the expansion scratch
// (the lm head) expand and multiply a row block at a time.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"

using namespace dgpp;

namespace {
cublasLtHandle_t lt;
void* ws = nullptr;
constexpr size_t kWs = 64u << 20;

struct Plan {
  cublasLtMatmulDesc_t desc{};
  cublasLtMatrixLayout_t la{}, lb{}, ld{};
  cublasLtMatmulAlgo_t algo{};
  bool ok = false;
};

// n_call weight rows, the output's leading dimension n_ld.
Plan make_plan(int m, int n_call, int k, int n_ld, bool f32, bool want_algo) {
  Plan p;
  cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
  cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
  cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, k, n_call, k);
  cublasLtMatrixLayoutCreate(&p.lb, CUDA_R_16BF, k, m, k);
  cublasLtMatrixLayoutCreate(&p.ld, f32 ? CUDA_R_32F : CUDA_R_16BF, n_call, m, n_ld);
  if (want_algo) {
    cublasLtMatmulPreference_t pref{};
    cublasLtMatmulPreferenceCreate(&pref);
    size_t wsb = kWs;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsb, sizeof(wsb));
    cublasLtMatmulHeuristicResult_t heur{};
    int nres = 0;
    const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, p.desc, p.la, p.lb, p.ld, p.ld, pref, 1, &heur, &nres);
    cublasLtMatmulPreferenceDestroy(pref);
    p.ok = st == CUBLAS_STATUS_SUCCESS && nres > 0;
    p.algo = heur.algo;
  } else {
    p.ok = true;
  }
  return p;
}
void destroy(Plan& p) {
  cublasLtMatmulDescDestroy(p.desc);
  cublasLtMatrixLayoutDestroy(p.la);
  cublasLtMatrixLayoutDestroy(p.lb);
  cublasLtMatrixLayoutDestroy(p.ld);
}
}  // namespace

int main(int argc, char** argv) {
  const char* head_path = argc > 1 ? argv[1] : nullptr;
  cublasLtCreate(&lt);
  DGPP_CUDA_OK(cudaMalloc(&ws, kWs));
  struct Shape { int m, n, k; bool f32; int block; };
  const Shape shapes[] = {
      {2048, 38720, 4096, true, 8192}, {1024, 38720, 4096, true, 8192}, {256, 38720, 4096, true, 8192},
      {64, 38720, 4096, true, 8192},   {20, 38720, 4096, true, 8192},   {9, 38720, 4096, true, 8192},
      {5, 38720, 4096, true, 8192},    {2048, 38720, 4096, true, 4096}, {1989, 38720, 4096, true, 6000},
      {2048, 38720, 6144, true, 5120}, {16, 38720, 6144, true, 5120},   {2048, 37888, 5120, true, 6400},
      {33, 37888, 5120, true, 6400},   {2048, 6416, 4096, false, 2048}, {20, 6416, 4096, false, 2048},
      {2048, 4096, 8192, false, 1024}, {12, 4096, 8192, false, 1024},
  };
  for (const Shape& s : shapes) {
    std::vector<uint16_t> ha(size_t(s.m) * s.k), hw(size_t(s.n) * s.k);
    uint64_t r = 0x9E3779B97F4A7C15ull;
    auto rnd = [&] { r ^= r << 13; r ^= r >> 7; r ^= r << 17; return (int(r % 2001) - 1000) / 500.0f; };
    for (auto& v : ha) v = float_to_bf16_bits(rnd());
    bool real = false;
    if (head_path && s.n == 38720 && s.k == 4096) {
      FILE* f = std::fopen(head_path, "rb");
      if (f) { real = std::fread(hw.data(), 2, hw.size(), f) == hw.size(); std::fclose(f); }
    }
    if (!real) for (auto& v : hw) v = float_to_bf16_bits(rnd() * 0.02f);
    uint16_t *a = nullptr, *w = nullptr;
    void *o1 = nullptr, *o2 = nullptr, *o3 = nullptr;
    const size_t es = s.f32 ? 4 : 2;
    const size_t ob = size_t(s.m) * s.n * es;
    DGPP_CUDA_OK(cudaMalloc(&a, ha.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&w, hw.size() * 2));
    DGPP_CUDA_OK(cudaMalloc(&o1, ob));
    DGPP_CUDA_OK(cudaMalloc(&o2, ob));
    DGPP_CUDA_OK(cudaMalloc(&o3, ob));
    cudaMemcpy(a, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(w, hw.data(), hw.size() * 2, cudaMemcpyHostToDevice);
    cudaMemset(o1, 0, ob); cudaMemset(o2, 0, ob); cudaMemset(o3, 0, ob);
    const float alpha = 1.f, beta = 0.f;
    Plan whole = make_plan(s.m, s.n, s.k, s.n, s.f32, true);
    if (!whole.ok) { std::printf("no heuristic for the whole call\n"); return 1; }
    cublasStatus_t st = cublasLtMatmul(lt, whole.desc, &alpha, w, whole.la, a, whole.lb, &beta, o1, whole.ld, o1,
                                       whole.ld, &whole.algo, ws, kWs, nullptr);
    if (st != CUBLAS_STATUS_SUCCESS) { std::printf("whole call failed %d\n", int(st)); return 1; }
    int pinned_fail = 0;
    for (int r0 = 0; r0 < s.n; r0 += s.block) {
      const int nb = std::min(s.block, s.n - r0);
      // (2) the whole call's algorithm on the block; (3) the block's own.
      Plan blk = make_plan(s.m, nb, s.k, s.n, s.f32, true);
      st = cublasLtMatmul(lt, blk.desc, &alpha, w + size_t(r0) * s.k, blk.la, a, blk.lb, &beta,
                          static_cast<uint8_t*>(o2) + size_t(r0) * es, blk.ld,
                          static_cast<uint8_t*>(o2) + size_t(r0) * es, blk.ld, &whole.algo, ws, kWs, nullptr);
      if (st != CUBLAS_STATUS_SUCCESS) ++pinned_fail;
      st = cublasLtMatmul(lt, blk.desc, &alpha, w + size_t(r0) * s.k, blk.la, a, blk.lb, &beta,
                          static_cast<uint8_t*>(o3) + size_t(r0) * es, blk.ld,
                          static_cast<uint8_t*>(o3) + size_t(r0) * es, blk.ld, &blk.algo, ws, kWs, nullptr);
      if (st != CUBLAS_STATUS_SUCCESS) { std::printf("block call failed %d\n", int(st)); return 1; }
      destroy(blk);
    }
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint8_t> b1(ob), b2(ob), b3(ob);
    cudaMemcpy(b1.data(), o1, ob, cudaMemcpyDeviceToHost);
    cudaMemcpy(b2.data(), o2, ob, cudaMemcpyDeviceToHost);
    cudaMemcpy(b3.data(), o3, ob, cudaMemcpyDeviceToHost);
    size_t d2 = 0, d3 = 0;
    for (size_t i = 0; i < ob; i += es) {
      d2 += std::memcmp(&b1[i], &b2[i], es) != 0;
      d3 += std::memcmp(&b1[i], &b3[i], es) != 0;
    }
    std::printf("m=%4d n=%5d k=%4d %s %s blocks of %4d: pinned algo %zu differ%s, own algo %zu differ (of %zu)\n", s.m,
                s.n, s.k, s.f32 ? "f32 " : "bf16", real ? "real" : "rand", s.block, d2,
                pinned_fail ? " [PINNED CALL REJECTED]" : "", d3, ob / es);
    destroy(whole);
    cudaFree(a); cudaFree(w); cudaFree(o1); cudaFree(o2); cudaFree(o3);
  }
  return 0;
}
