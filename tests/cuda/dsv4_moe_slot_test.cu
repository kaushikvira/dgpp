// The DeepSeek-V4-Flash (deepseek_v4) MXFP4 slot expert's GPU gate: the
// slot (decode's) composition (launch_moe_slot_gate_up_swiglu_fp4 +
// launch_moe_slot_down_fp4 + launch_moe_slot_accum, the fp8 shared
// expert's the sh_*'s arguments's, the fp4_group's 32's) vs the CPU
// oracle (tests/unit/dsv4_moe_expert_oracle.hpp's moe_mxfp4_ref — the
// e2m1 x e8m0/32 decode, the per-32 block partial scaled once, the
// SwiGLU clamp 10, the router weight folded into the down epilogue).
//
// Two deterministic small cases (the small-geometry's the fp4 set's:
// k = 64's the MXFP4's compiled's, the shared's fp8's k = 32's):
//   A: the shared expert's zeroed (the contribution's 0's) — the
//      output's the routed oracle's (the second slot's weight's 0.0's,
//      the first's 1.0's), the max error's the bf16's intermediate's
//      rounding's + the fp32's dot's the double's the delta's (the
//      clamp's the bounded's data's never fires's, so the kernel's
//      swiglu's and the oracle's swiglu's differ by the rounding's
//      only's).
//   B: the shared expert's live's (the e4m3 x f32's bounded's, the
//      1.0's scale's) + the two routed's slots' (the weights' 0.6f /
//      0.4f's) — the expected's the routed oracle's + the shared's
//      contribution's (the kernel's swiglu's semantics's, the double's
//      recompute's the bf16's rounding points's), the same's
//      threshold's.
//
// Each case prints PASS/FAIL + the max error (against the threshold's
// the 2%'s the expected's magnitude's scaled's). Exit 0 on pass, 1 on
// fail, 2 on no-GPU (the ctest's SKIP_RETURN_CODE's contract's).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "dsv4_moe_expert_oracle.hpp"
#include "kernels/gemm.hpp"
#include "models/dsv4/hash_layer.hpp"

using namespace dgpp;

namespace {

// The geometry (the small's, the fp4 set's the k = 64's the MXFP4's
// compiled's, the shared's fp8's k = 32's): the hidden's 64's, the
// inter's 32's (the routed's n's + the shared's), the 4's experts's,
// the top-k's 2's (the slot's layout's tokens*(K+1)'s the 3's slots's
// — the 2's routed's + the shared's).
constexpr int kH = 64;
constexpr int kI = 32;
constexpr int kE = 4;
constexpr int kK = 2;
constexpr float kClamp = 10.0f;

// The bounded MXFP4 filler (the e2m1 nibbles' the LCG's, the e8m0's
// 2^-6..2^-4's the 121..123's): the |w|'s bounded at 6 x 2^-4's =
// 0.375's, so with the x's bounded at 0.25's the dots' stay under the
// clamp's 10's (the 64 x 0.25 x 0.375's = 6's, the silu's < 5.8's) —
// the kernel's swiglu's (the clamp_max's gate's + the clamp's up's) and
// the oracle's swiglu's (the silu-clamp's x up's) agree to the
// rounding's only's.
void fill_mxfp4_bounded(std::vector<uint8_t>* payload, std::vector<uint8_t>* scale, std::size_t rows,
                        std::size_t cols, std::uint32_t seed) {
  payload->resize(rows * cols / 2);
  scale->resize(rows * ((cols + 31) / 32));
  std::uint32_t s = seed;
  auto next = [&]() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  };
  for (std::size_t i = 0; i < payload->size(); ++i)
    (*payload)[i] = static_cast<uint8_t>(next());  // two e2m1 nibbles per byte
  for (std::size_t i = 0; i < scale->size(); ++i)
    (*scale)[i] = static_cast<uint8_t>(121 + static_cast<int>(next() % 3));  // 2^-6..2^-4
}

// The x's row (the bf16's, the bounded's): ((k * 37) % 9 - 4) * 2^-4's
// — the -0.25..0.25's the dyadic's, the bf16's exact's.
uint16_t x_word(int k) {
  const float v = (static_cast<float>((k * 37) % 9) - 4.0f) * 0.0625f;
  return float_to_bf16_bits(v);
}
double x_d64(int k) { return dsv4_moe_oracle::bf16_to_d64(x_word(k)); }

// The shared expert's e4m3 value (the deterministic's the small's
// set's {0, ±0.125, ±0.25, ±0.375, ±0.5}'s the e4m3's exact's, the
// |w|'s bounded at 0.5's — the dots' bounded at 8's under the clamp's
// the 64 x 0.25 x 0.5's).
float shared_value(std::size_t i) {
  const int v = static_cast<int>((i * 5 + 7) % 9) - 4;  // -4..4
  return v * 0.125f;
}

// The kernel's swiglu (the GLM's asymmetric's, the bf16's rounding
// points's the two's) in double (the test's recompute's): the dots' the
// bf16's, the gate's the clamp_max's only's, the up's the clamp's both's,
// the silu's the bf16's, the product's the bf16's.
double kernel_swiglu(double gate, double up) {
  double g = static_cast<double>(bf16_bits_to_float(float_to_bf16_bits(static_cast<float>(gate))));
  double u = static_cast<double>(bf16_bits_to_float(float_to_bf16_bits(static_cast<float>(up))));
  if (g > kClamp) g = kClamp;
  u = std::min(std::max(u, -static_cast<double>(kClamp)), static_cast<double>(kClamp));
  const double t = static_cast<double>(bf16_bits_to_float(float_to_bf16_bits(
      static_cast<float>(g / (1.0 + std::exp(-g))))));
  return static_cast<double>(bf16_bits_to_float(float_to_bf16_bits(static_cast<float>(t * u))));
}

// One case's run: the layer's expert's (the slot's path's) vs the
// expected's (the routed's oracle's + the shared's contribution's — the
// caller's the computed's, the 0.0's the zeroed's shared's). Prints
// PASS/FAIL + the max error's; throws on fail's. The shared's e4m3's
// planes' (the 32 x 32's grid's, the 1.0's scale's the exact's) the
// host's the device's the copied's.
void run_case(const char* name, const std::vector<int32_t>& ids, const std::vector<float>& wts,
              const std::vector<uint8_t>& sh1, const std::vector<uint8_t>& sh3, const std::vector<uint8_t>& sh2,
              const std::vector<double>& expected) {
  // ---- the device's data (the deterministic's, the small's) --------
  std::vector<uint16_t> xh;
  for (int k = 0; k < kH; ++k) xh.push_back(x_word(k));
  std::vector<uint8_t> w1, w1s, w3, w3s, w2, w2s;
  // The full multi-expert tensors (the oracle's fill's layout's the
  // [e, n, k/2]'s the experts' outer's): the bounded's filler's (the
  // seeds' the oracle's test's 0x10/0x20/0x30's).
  fill_mxfp4_bounded(&w1, &w1s, static_cast<std::size_t>(kE) * kI, kH, 0x10);
  fill_mxfp4_bounded(&w3, &w3s, static_cast<std::size_t>(kE) * kI, kH, 0x20);
  fill_mxfp4_bounded(&w2, &w2s, static_cast<std::size_t>(kE) * kH, kI, 0x30);
  std::vector<uint16_t> router(static_cast<std::size_t>(kE) * kH, 0);
  std::vector<float> bias(kE, 0.0f);

  // ---- the layer's (the scratch's the layer's formula's, the gemm's
  // the unused's by expert's, the reference's the held's) ----------
  Dsv4HashConfig cfg;
  cfg.hidden = kH;
  cfg.inter = kI;
  cfg.n_experts = kE;
  cfg.top_k = kK;
  cfg.num_hash_layers = 0;  // the learned's mode's (the bias's the required's, the 0.0's)
  cfg.tp = 1;  // the unsliced's (the local's = the full's)
  cfg.swiglu_limit = kClamp;
  const size_t scratch_bytes = Dsv4HashLayer::scratch_bytes(cfg, 1);
  std::vector<char> scratch(scratch_bytes);
  CublasLtGemm gemm;  // the reference's (the expert's the no GEMM's)
  Dsv4HashLayer layer(gemm, cfg, 1, scratch.data(), scratch_bytes, nullptr, 0);

  // ---- the device's buffers' (the payload's the 16B's aligned's the
  // cudaMalloc's) -----------------------------------------------
  auto dev_bytes = [&]<typename T>(const std::vector<T>& h) -> T* {
    T* p = nullptr;
    DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(16, h.size()) * sizeof(T)));
    if (!h.empty()) DGPP_CUDA_OK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
  };
  uint16_t* d_x = dev_bytes(xh);
  int32_t* d_ids = dev_bytes(ids);
  float* d_wts = dev_bytes(wts);
  uint16_t* d_out = nullptr;
  DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&d_out), kH * 2));
  uint16_t* d_router = dev_bytes(router);
  float* d_bias = dev_bytes(bias);
  uint8_t* d_w1 = dev_bytes(w1);
  uint8_t* d_w1s = dev_bytes(w1s);
  uint8_t* d_w3 = dev_bytes(w3);
  uint8_t* d_w3s = dev_bytes(w3s);
  uint8_t* d_w2 = dev_bytes(w2);
  uint8_t* d_w2s = dev_bytes(w2s);
  uint8_t* d_sh1 = dev_bytes(sh1);
  uint8_t* d_sh3 = dev_bytes(sh3);
  uint8_t* d_sh2 = dev_bytes(sh2);
  // The 32 x 32's grid's scales (the 1.0's the exact's): the gate/up's
  // [1, 2]'s the 2's floats's each's, the down's [2, 1]'s the 2's — the
  // 6's total's (the per-matrix's the offset's).
  std::vector<float> sh_scales(6, 1.0f);
  float* d_sh_scales = dev_bytes(sh_scales);

  // ---- the views (the device's pointers' the loader's re-expression's)
  std::vector<GlmFp4Matrix> exps(static_cast<size_t>(kE) * 3);
  for (int e = 0; e < kE; ++e) {
    exps[static_cast<size_t>(e) * 3 + 0] =
        GlmFp4Matrix{d_w1 + static_cast<size_t>(e) * kI * (kH / 2),
                     d_w1s + static_cast<size_t>(e) * kI * (kH / 32), nullptr, kI, kH, 32};
    exps[static_cast<size_t>(e) * 3 + 1] =
        GlmFp4Matrix{d_w3 + static_cast<size_t>(e) * kI * (kH / 2),
                     d_w3s + static_cast<size_t>(e) * kI * (kH / 32), nullptr, kI, kH, 32};
    exps[static_cast<size_t>(e) * 3 + 2] =
        GlmFp4Matrix{d_w2 + static_cast<size_t>(e) * kH * (kI / 2),
                     d_w2s + static_cast<size_t>(e) * kH * (kI / 32), nullptr, kH, kI, 32};
  }
  std::vector<GlmQuantMatrix> sh(3);
  sh[0] = GlmQuantMatrix{d_sh1, d_sh_scales, kI, kH, 32, 32};
  sh[1] = GlmQuantMatrix{d_sh3, d_sh_scales + 2, kI, kH, 32, 32};
  sh[2] = GlmQuantMatrix{d_sh2, d_sh_scales + 4, kH, kI, 32, 32};

  Dsv4HashWeights w;
  w.router_gate = d_router;
  w.router_bias = d_bias;
  w.tid2eid = nullptr;  // the non-hash's (the bias's the required's)
  w.expert_payload = d_w1;
  w.expert_scales = d_w1s;
  w.experts = exps.data();
  w.shared = sh.data();
  w.n_experts = kE;
  w.local_inter = kI;
  w.local_shared_inter = kI;
  w.layer = 0;

  // ---- the slot's path (the rebind's the view table's upload's, the
  // expert's the kernel's the only's) -----------------------------
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  layer.rebind(w, 0, stream);
  layer.expert(d_x, 1, d_ids, d_wts, d_out, stream);
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  std::vector<uint16_t> out_h(kH);
  DGPP_CUDA_OK(cudaMemcpy(out_h.data(), d_out, kH * 2, cudaMemcpyDeviceToHost));

  // ---- the compare (the max error's the threshold's the 2%'s) ----
  double max_abs = 0.0, max_mag = 0.0;
  for (int k = 0; k < kH; ++k) {
    const double got = dsv4_moe_oracle::bf16_to_d64(out_h[static_cast<size_t>(k)]);
    const double want = expected[static_cast<size_t>(k)];
    max_abs = std::max(max_abs, std::fabs(got - want));
    max_mag = std::max(max_mag, std::max(std::fabs(got), std::fabs(want)));
  }
  const double threshold = 0.02 * std::max(max_mag, 1.0);
  const bool pass = max_abs <= threshold;
  std::printf("    %s: max error %.6g (threshold %.6g = the 2%%'s the magnitude %.6g's) — %s\n", name, max_abs,
              threshold, std::max(max_mag, 1.0), pass ? "PASS" : "FAIL");
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
  cudaFree(d_x);
  cudaFree(d_ids);
  cudaFree(d_wts);
  cudaFree(d_out);
  cudaFree(d_router);
  cudaFree(d_bias);
  cudaFree(d_w1);
  cudaFree(d_w1s);
  cudaFree(d_w3);
  cudaFree(d_w3s);
  cudaFree(d_w2);
  cudaFree(d_w2s);
  cudaFree(d_sh1);
  cudaFree(d_sh3);
  cudaFree(d_sh2);
  cudaFree(d_sh_scales);
  if (!pass)
    throw std::runtime_error(std::string(name) + " failed: max error " + std::to_string(max_abs) +
                             " exceeds the threshold " + std::to_string(threshold));
}

// The shared expert's contribution (the kernel's semantics's, the
// double's recompute's): the e4m3 x 1.0's decode's the exact's, the
// kernel's swiglu's (the bf16's rounding points's), the down's the
// double's over the bf16's act's (the fp32's the kernel's the delta's
// the threshold's absorbs's). 0.0's when the zeroed's (case A's).
std::vector<double> shared_expected(const std::vector<uint8_t>& sh1, const std::vector<uint8_t>& sh3,
                                    const std::vector<uint8_t>& sh2) {
  const bool live = std::any_of(sh1.begin(), sh1.end(), [](uint8_t b) { return b != 0; });
  if (!live) return std::vector<double>(kH, 0.0);
  auto e4m3 = [](uint8_t b) { return static_cast<double>(fp8_e4m3_bits_to_float(b)); };
  std::vector<double> gate(kI), up(kI);
  for (int n = 0; n < kI; ++n) {
    double g = 0.0, u = 0.0;
    for (int k = 0; k < kH; ++k) {
      const double xv = x_d64(k);
      g += xv * e4m3(sh1[static_cast<size_t>(n) * kH + k]);
      u += xv * e4m3(sh3[static_cast<size_t>(n) * kH + k]);
    }
    gate[static_cast<size_t>(n)] = g;
    up[static_cast<size_t>(n)] = u;
  }
  std::vector<double> act(kI);
  for (int n = 0; n < kI; ++n)
    act[static_cast<size_t>(n)] = kernel_swiglu(gate[static_cast<size_t>(n)], up[static_cast<size_t>(n)]);
  std::vector<double> y(kH, 0.0);
  for (int k = 0; k < kH; ++k)
    for (int n = 0; n < kI; ++n) y[static_cast<size_t>(k)] += act[static_cast<size_t>(n)] * e4m3(sh2[static_cast<size_t>(k) * kI + n]);
  return y;  // the weight's 1.0's (the accum's shared's fold's)
}

// The shared expert's e4m3's planes (the bounded's filler's, the
// 0.0's when zeroed's): the case B's host's source's.
std::vector<uint8_t> shared_plane(std::size_t rows, std::size_t cols, std::size_t shift, bool live) {
  std::vector<uint8_t> v(rows * cols, 0);
  if (live)
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = float_to_fp8_e4m3_bits(shared_value(i + shift));
  return v;
}

}  // namespace

// ---------------------------------------------------------------------------
DGPP_TEST(dsv4_moe_slot_expert_vs_oracle_zero_shared) {
  // Case A: the shared expert's zeroed (the contribution's 0's) — the
  // slot's path's output's the routed oracle's (the single's effective's
  // expert's the weight's 1.0's, the second's slot's the 0.0's weight's
  // the no-op's). The bounded's data's (the clamp's never fires's), the
  // max error's the bf16's rounding's + the fp32's the double's the
  // delta's (the 2%'s threshold's).
  std::vector<uint16_t> xh;
  for (int k = 0; k < kH; ++k) xh.push_back(x_word(k));
  const std::vector<int32_t> ids = {2, 3};
  const std::vector<float> wts = {1.0f, 0.0f};
  std::vector<uint8_t> w1, w1s, w3, w3s, w2, w2s;
  fill_mxfp4_bounded(&w1, &w1s, static_cast<std::size_t>(kE) * kI, kH, 0x10);
  fill_mxfp4_bounded(&w3, &w3s, static_cast<std::size_t>(kE) * kI, kH, 0x20);
  fill_mxfp4_bounded(&w2, &w2s, static_cast<std::size_t>(kE) * kH, kI, 0x30);
  const auto expected = dsv4_moe_oracle::moe_mxfp4_ref(xh, ids, wts, w1, w1s, w3, w3s, w2, w2s, 1, kE, kI, kH, kK,
                                                        static_cast<double>(kClamp));
  const std::vector<uint8_t> zero1(static_cast<std::size_t>(kI) * kH, 0), zero3(static_cast<std::size_t>(kI) * kH, 0),
      zero2(static_cast<std::size_t>(kH) * kI, 0);
  run_case("the slot's expert's vs the oracle's (the shared's zeroed's)", ids, wts, zero1, zero3, zero2, expected);
}

DGPP_TEST(dsv4_moe_slot_expert_vs_oracle_live_shared) {
  // Case B: the shared expert's live's (the e4m3's the bounded's, the
  // 1.0's scale's) + the two routed's slots' (the weights' 0.6f /
  // 0.4f's the f32's exact's the kernel's the read's) — the expected's
  // the routed oracle's + the shared's contribution's (the kernel's
  // swiglu's semantics's the double's recompute's). The same's 2%'s
  // threshold's (the fp32's vs double's the dots's the delta's the
  // within's).
  std::vector<uint16_t> xh;
  for (int k = 0; k < kH; ++k) xh.push_back(x_word(k));
  const std::vector<int32_t> ids = {1, 3};
  const std::vector<float> wts = {0.6f, 0.4f};
  std::vector<uint8_t> w1, w1s, w3, w3s, w2, w2s;
  fill_mxfp4_bounded(&w1, &w1s, static_cast<std::size_t>(kE) * kI, kH, 0x10);
  fill_mxfp4_bounded(&w3, &w3s, static_cast<std::size_t>(kE) * kI, kH, 0x20);
  fill_mxfp4_bounded(&w2, &w2s, static_cast<std::size_t>(kE) * kH, kI, 0x30);
  auto expected = dsv4_moe_oracle::moe_mxfp4_ref(xh, ids, wts, w1, w1s, w3, w3s, w2, w2s, 1, kE, kI, kH, kK,
                                                 static_cast<double>(kClamp));
  const auto sh1 = shared_plane(kI, kH, 0, true);
  const auto sh3 = shared_plane(kI, kH, 1, true);
  const auto sh2 = shared_plane(kH, kI, 2, true);
  const auto sh_exp = shared_expected(sh1, sh3, sh2);
  for (size_t k = 0; k < expected.size(); ++k) expected[k] += sh_exp[k];  // the shared's the weight's 1.0's
  run_case("the slot's expert's vs the oracle's + the shared's (the live's)", ids, wts, sh1, sh3, sh2, expected);
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) {
    std::printf("dsv4_moe_slot_test: no GPU (the parent's the gate's the run's)\n");
    return 2;  // ctest: skip, no GPU
  }
  return dgpp::test::run_all();
}
