// Gates for the lossless 12-bit decode form of bf16 weights
// (kernels/bf12_gemv.hpp): the packed form decodes back to the bf16 bits it
// came from; a registered companion's launches through the GEMM seam are
// BITWISE the bf16 GEMV's at every decode row count, both outputs and a
// strided activation view; escapes (zeros, subnormals, outliers, NaN/Inf
// exponents, a row full of them up to the bound) take the exact path; a
// pathological row keeps the matrix in its bf16 form; wide calls ignore the
// companion; the prefetch view names the bytes the launch streams. And
// bf12-ONLY residency: a weight whose bf16 bytes were released answers every
// call — prefill GEMMs through the expansion scratch, whole or in weight-row
// blocks, decode batches of any width — bitwise as the resident instance.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "kernels/bf12_gemv.hpp"
#include "kernels/bf16_gemv.hpp"
#include "kernels/gemm.hpp"

namespace {

using dgpp::Bf12Host;
using dgpp::Bf12Matrix;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// Trained-looking weights: gaussian, so the exponents carry the 2^-j tail
// the format is built around (about 1.5e-4 of them leave the window).
std::vector<uint16_t> gaussian_bf16(size_t elems, float sigma, uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> dist(0.f, sigma);
  std::vector<uint16_t> v(elems);
  for (auto& x : v) x = dgpp::float_to_bf16_bits(dist(rng));
  return v;
}

struct Packed {
  std::vector<void*> allocs;
  Bf12Matrix m;
  ~Packed() {
    for (void* p : allocs) cudaFree(p);
  }
};

void upload(const Bf12Host& h, int n, int k, Packed& out) {
  void* p = nullptr;
  void* b = nullptr;
  void* e = nullptr;
  void* r = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&r, h.raw.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(r, h.raw.data(), h.raw.size() * 2, cudaMemcpyHostToDevice));
  out.m.raw = static_cast<const uint16_t*>(r);
  DGPP_CUDA_OK(cudaMalloc(&p, h.packed.size()));
  DGPP_CUDA_OK(cudaMalloc(&b, h.rows.size() * 4));
  DGPP_CUDA_OK(cudaMalloc(&e, h.esc.size() * 4));
  out.allocs = {p, b, e, r};
  DGPP_CUDA_OK(cudaMemcpy(p, h.packed.data(), h.packed.size(), cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(b, h.rows.data(), h.rows.size() * 4, cudaMemcpyHostToDevice));
  DGPP_CUDA_OK(cudaMemcpy(e, h.esc.data(), h.esc.size() * 4, cudaMemcpyHostToDevice));
  out.m.packed = static_cast<const uint8_t*>(p);
  out.m.rows = static_cast<const uint32_t*>(b);
  out.m.esc = static_cast<const uint32_t*>(e);
  out.m.n = n;
  out.m.k = k;
}

// One matmul through the seam; the raw output bytes.
std::vector<uint8_t> run(dgpp::CublasLtGemm& gemm, const uint16_t* act, size_t stride,
                         const uint16_t* w, int m, int n, int k, bool f32) {
  const size_t bytes = static_cast<size_t>(m) * n * (f32 ? 4 : 2);
  void* out = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&out, bytes));
  DGPP_CUDA_OK(cudaMemset(out, 0xA5, bytes));
  gemm.matmul(act, w, out, m, n, k, dgpp::DType::BF16,
              f32 ? dgpp::GemmOut::F32 : dgpp::GemmOut::BF16, stride, nullptr, 0, nullptr);
  DGPP_CUDA_OK(cudaDeviceSynchronize());
  std::vector<uint8_t> got(bytes);
  DGPP_CUDA_OK(cudaMemcpy(got.data(), out, bytes, cudaMemcpyDeviceToHost));
  cudaFree(out);
  return got;
}

// The companion's launches against the plain instance's, every decode row
// count, both outputs, contiguous and strided activations.
void bitwise_gate(const std::vector<uint16_t>& w, int n, int k, const char* label,
                  size_t* escapes_out = nullptr) {
  const Bf12Host h = dgpp::bf12_encode(w.data(), n, k);
  require(h.ok, std::string(label) + ": encodes");
  std::vector<uint16_t> back(w.size());
  dgpp::bf12_decode(h, n, k, back.data());
  require(std::memcmp(back.data(), w.data(), w.size() * 2) == 0,
          std::string(label) + ": the packed form decodes to the same bf16 bits");
  if (escapes_out) *escapes_out = h.escapes;

  uint16_t* dw = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&dw, w.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
  Packed pk;
  upload(h, n, k, pk);
  // The reference chunks every row count (the scalar chain per row); the
  // packed instance keeps the model's four decode rows and widens by itself.
  dgpp::CublasLtGemm plain, packed;
  plain.set_decode_rows(8);
  packed.set_decode_rows(4);
  packed.register_bf12(dw, pk.m);
  require(packed.bf12_registered() == 1, "one companion registered");
  {
    // Until the caller says its rows are a decode batch, a five-row call
    // keeps the weight's own bytes (a short prefill chunk's Lt algorithm).
    const void* view = nullptr;
    size_t view_bytes = 0;
    packed.resident_view(dw, w.size() * 2, 5, &view, &view_bytes);
    require(view == dw, std::string(label) + ": five rows stay bf16 outside a decode batch");
  }
  packed.set_bf12_wide(true);

  const size_t stride = static_cast<size_t>(k) + 48;  // a fused-row view
  const std::vector<uint16_t> act = gaussian_bf16(8 * stride, 1.5f, 0xAC7 + n);
  uint16_t* dx = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&dx, act.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(dx, act.data(), act.size() * 2, cudaMemcpyHostToDevice));
  // One to four rows: the narrow kernel (the wide one past the smem bound,
  // k = 8192 at four rows); five to eight: the wide kernel. Every row is
  // bitwise its scalar chain.
  for (int m = 1; m <= dgpp::kBf12MaxRows; ++m) {
    for (bool f32 : {false, true}) {
      for (size_t st : {static_cast<size_t>(k), stride}) {
        const auto want = run(plain, dx, st, dw, m, n, k, f32);
        const auto got = run(packed, dx, st, dw, m, n, k, f32);
        require(want == got, std::string(label) + ": m=" + std::to_string(m) +
                                 (f32 ? " f32" : " bf16") + " bitwise the bf16 GEMV");
      }
    }
    const void* view = nullptr;
    size_t view_bytes = 0;
    packed.resident_view(dw, w.size() * 2, m, &view, &view_bytes);
    require(view == pk.m.packed && view_bytes == dgpp::bf12_packed_bytes(n, k),
            std::string(label) + ": the prefetch view is the packed bytes");
  }
  // Past a packed launch's rows the call keeps the weight's own bytes.
  const void* view = nullptr;
  size_t view_bytes = 0;
  packed.resident_view(dw, w.size() * 2, dgpp::kBf12MaxRows + 1, &view, &view_bytes);
  require(view == dw && view_bytes == w.size() * 2, "a wide call's view is the bf16 weight");
  plain.resident_view(dw, w.size() * 2, 2, &view, &view_bytes);
  require(view == dw && view_bytes == w.size() * 2, "no companion: the bf16 weight");
  cudaFree(dx);
  cudaFree(dw);
  std::printf("[ OK ] %s N%dxK%d: %zu escapes, widest row %d, bitwise at every row count\n",
              label, n, k, h.escapes, h.max_row_escapes);
}

// bf12-ONLY residency (kernels/gemm.hpp, bf12_release_raw): the resident
// instance answers first from the bf16 bytes; then those bytes are
// SCRIBBLED on the device (a released weight's are gone — a stale read must
// not pass by luck), and the released instance has to reproduce every answer
// from the companion: Lt calls through the expansion scratch — whole, and in
// weight-row blocks when `slot_bytes` is narrower than the matrix — decode
// launches at any width, the prefetch view.
void released_gate(const std::vector<uint16_t>& w, int n, int k, size_t slot_bytes, int slots,
                   const char* label) {
  const Bf12Host h = dgpp::bf12_encode(w.data(), n, k);
  require(h.ok, std::string(label) + ": encodes");
  uint16_t* dw = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&dw, w.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(dw, w.data(), w.size() * 2, cudaMemcpyHostToDevice));
  Packed pk;
  upload(h, n, k, pk);

  // The expansion alone: every row range is the original bits.
  {
    uint16_t* ex = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&ex, w.size() * 2));
    for (auto [r0, rows] : std::vector<std::pair<int, int>>{{0, n}, {0, 1}, {n - 1, 1}, {n / 3, n - n / 3},
                                                            {1, std::min(n - 1, 9)}}) {
      DGPP_CUDA_OK(cudaMemset(ex, 0x5A, w.size() * 2));
      dgpp::launch_bf12_expand(pk.m, r0, rows, ex, nullptr);
      DGPP_CUDA_OK(cudaDeviceSynchronize());
      std::vector<uint16_t> back(static_cast<size_t>(rows) * k);
      DGPP_CUDA_OK(cudaMemcpy(back.data(), ex, back.size() * 2, cudaMemcpyDeviceToHost));
      require(std::memcmp(back.data(), w.data() + static_cast<size_t>(r0) * k, back.size() * 2) == 0,
              std::string(label) + ": rows [" + std::to_string(r0) + ", +" + std::to_string(rows) +
                  ") expand to their bf16 bits");
    }
    cudaFree(ex);
  }

  dgpp::CublasLtGemm resident, released;
  for (dgpp::CublasLtGemm* g : {&resident, &released}) {
    g->set_decode_rows(4);
    g->register_bf12(dw, pk.m);
  }
  const std::vector<uint16_t> act = gaussian_bf16(static_cast<size_t>(300) * k, 1.5f, 0x5EED + n);
  uint16_t* dx = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&dx, act.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(dx, act.data(), act.size() * 2, cudaMemcpyHostToDevice));
  void* ws = nullptr;
  constexpr size_t kWs = size_t{64} << 20;
  DGPP_CUDA_OK(cudaMalloc(&ws, kWs));
  const auto call = [&](dgpp::CublasLtGemm& g, int m, bool f32) {
    const size_t bytes = static_cast<size_t>(m) * n * (f32 ? 4 : 2);
    void* out = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&out, bytes));
    DGPP_CUDA_OK(cudaMemset(out, 0xA5, bytes));
    g.matmul(dx, dw, out, m, n, k, dgpp::DType::BF16, f32 ? dgpp::GemmOut::F32 : dgpp::GemmOut::BF16,
             static_cast<size_t>(k), ws, kWs, nullptr);
    DGPP_CUDA_OK(cudaDeviceSynchronize());
    std::vector<uint8_t> got(bytes);
    DGPP_CUDA_OK(cudaMemcpy(got.data(), out, bytes, cudaMemcpyDeviceToHost));
    cudaFree(out);
    return got;
  };
  // The resident answers: prefill rows (Lt past four rows, the chain to
  // four), then a row block under the chunk's pinned algorithm.
  const std::vector<int> prefill_rows = {1, 3, 4, 5, 8, 9, 20, 64, 257};
  std::vector<std::vector<uint8_t>> want_bf16, want_f32;
  for (int m : prefill_rows) {
    want_bf16.push_back(call(resident, m, false));
    want_f32.push_back(call(resident, m, true));
  }
  resident.set_plan_rows(257);
  const auto want_block = call(resident, 96, false);
  resident.set_plan_rows(0);
  // Decode batches: the resident instance's launches of up to eight rows.
  resident.set_bf12_wide(true);
  std::vector<std::vector<uint8_t>> want_decode;
  for (int m = 1; m <= 8; ++m) want_decode.push_back(call(resident, m, false));

  // The bf16 bytes go away.
  DGPP_CUDA_OK(cudaMemset(dw, 0xEE, w.size() * 2));
  released.bf12_release_raw(dw);
  require(released.bf12_raw_released(dw) && !resident.bf12_raw_released(dw), "the release is per instance");
  bool threw = false;
  try {
    (void)call(released, 64, false);
  } catch (const std::logic_error&) {
    threw = true;
  }
  require(threw, "a released weight's Lt call without a scratch is refused");
  released.bf12_reserve_expand(slots, slot_bytes);
  require(released.bf12_expand_bytes() >= static_cast<size_t>(slots) * slot_bytes, "the scratch is reserved");

  for (size_t i = 0; i < prefill_rows.size(); ++i) {
    const int m = prefill_rows[i];
    require(call(released, m, false) == want_bf16[i],
            std::string(label) + ": released prefill m=" + std::to_string(m) + " bf16 bitwise the resident call");
    require(call(released, m, true) == want_f32[i],
            std::string(label) + ": released prefill m=" + std::to_string(m) + " f32 bitwise the resident call");
  }
  released.set_plan_rows(257);
  require(call(released, 96, false) == want_block,
          std::string(label) + ": a released row block under the chunk's algorithm is bitwise the resident one");
  released.set_plan_rows(0);
  const bool whole = slot_bytes >= w.size() * 2;
  if (whole) {
    // A prefill walk finds the matrix still expanded: no second expansion
    // (wide calls; a short call runs L2-sized row blocks instead).
    const uint64_t before = released.bf12_expansions();
    require(call(released, 257, false) == want_bf16.back(), std::string(label) + ": a reused expansion is bitwise");
    require(call(released, 257, true) == want_f32.back(), std::string(label) + ": a reused expansion is bitwise (f32)");
    require(released.bf12_expansions() == before, std::string(label) + ": an expanded matrix is reused in place");
    // A short call's row blocks run through the scratch without leaving a
    // stale key behind: wide, short, wide again — every answer the resident one.
    require(call(released, 20, false) == want_bf16[6], std::string(label) + ": a short call after a wide one");
    require(call(released, 257, false) == want_bf16.back(), std::string(label) + ": a wide call after a short one");
  }
  // Decode batches: packed launches at every width, never the scratch —
  // to eight rows bitwise the resident instance's, past them the same
  // scalar chain in eight-row launches.
  released.set_bf12_wide(true);
  const uint64_t before = released.bf12_expansions();
  for (int m = 1; m <= 8; ++m)
    require(call(released, m, false) == want_decode[static_cast<size_t>(m - 1)],
            std::string(label) + ": released decode m=" + std::to_string(m) + " bitwise the resident launch");
  {
    const auto wide = call(released, 13, false);  // rows 0..7 then 8..12
    const size_t row_bytes = static_cast<size_t>(n) * 2;
    require(std::memcmp(wide.data(), want_decode[7].data(), 8 * row_bytes) == 0,
            std::string(label) + ": a thirteen-row decode batch's first launch is the eight-row chain");
  }
  require(released.bf12_expansions() == before, std::string(label) + ": a decode batch never touches the scratch");
  const void* view = nullptr;
  size_t view_bytes = 0;
  released.resident_view(dw, w.size() * 2, 300, &view, &view_bytes);
  require(view == pk.m.packed && view_bytes == dgpp::bf12_packed_bytes(n, k),
          std::string(label) + ": a released weight's view is the packed bytes at any width");
  released.set_bf12_wide(false);
  cudaFree(ws);
  cudaFree(dx);
  cudaFree(dw);
  std::printf("[ OK ] %s N%dxK%d released through %d x %zu KiB (%s): bitwise the resident instance\n", label, n, k,
              slots, slot_bytes >> 10, whole ? "whole" : "row blocks");
}

}  // namespace

DGPP_TEST(bf12_released_weights_answer_bitwise_from_the_companion) {
  // Whole-matrix slots (the KDA in/o classes, two slots: the fold overlap's
  // reuse), then slots narrower than the matrix — the head's row blocks,
  // ragged last block included — and a matrix with raw rows and escapes.
  {
    const int n = 1611, k = 4096;
    released_gate(gaussian_bf16(static_cast<size_t>(n) * k, 0.02f, 0x71), n, k, static_cast<size_t>(n) * k * 2, 2,
                  "whole");
  }
  {
    const int n = 3000, k = 2048;
    released_gate(gaussian_bf16(static_cast<size_t>(n) * k, 0.02f, 0x72), n, k, static_cast<size_t>(700) * k * 2, 1,
                  "blocks");
  }
  {
    // Rows with a tail (Qwen's hidden; the GR up projection): expansion
    // whole and in weight-row blocks.
    const int n = 1283, k = 2560;
    released_gate(gaussian_bf16(static_cast<size_t>(n) * k, 0.02f, 0x74), n, k, static_cast<size_t>(n) * k * 2, 1,
                  "tail 2560, whole");
    const int n2 = 4099, k2 = 320;
    released_gate(gaussian_bf16(static_cast<size_t>(n2) * k2, 0.02f, 0x75), n2, k2,
                  static_cast<size_t>(1024) * k2 * 2, 1, "tail 320, blocks");
  }
  {
    const int n = 515, k = 8192;
    auto w = gaussian_bf16(static_cast<size_t>(n) * k, 0.02f, 0x73);
    for (int c = 0; c < k; c += 3) w[static_cast<size_t>(77) * k + c] = static_cast<uint16_t>(((c % 90) + 60) << 7);
    for (int c = 0; c < k; c += 911) w[static_cast<size_t>(5) * k + c] = 0x0000;
    released_gate(w, n, k, static_cast<size_t>(128) * k * 2, 1, "raw rows + escapes, blocks");
  }
}

DGPP_TEST(bf12_trained_shapes_are_bitwise_the_bf16_gemv) {
  // The KDA in_proj class (k = hidden, ragged n), the o_proj class
  // (k = 2048: the two-super-block depth), the eh_proj class (k = 8192:
  // two rows fit the smem bound, three do not).
  // Then the rows with a TAIL (format v2): Qwen's hidden 2560 and its
  // 1536-wide slices (a pair unit), the GR up projection's k = 320 (all
  // tail: a full step and an eight-lane partial one), 1280 (a lone full
  // step), 1800 (pair + full + a one-lane partial), 3392 (three
  // super-blocks + 320), 10240 (GR down: ten super-blocks, pairs in flight),
  // and the smallest row there is.
  for (auto [n, k] : std::vector<std::pair<int, int>>{{1611, 4096}, {1024, 2048}, {515, 8192},
                                                      {7, 1024}, {777, 2560}, {1027, 1536},
                                                      {2051, 320}, {300, 1280}, {301, 1800},
                                                      {203, 3392}, {163, 10240}, {40, 8},
                                                      {33, 264}}) {
    const auto w = gaussian_bf16(static_cast<size_t>(n) * k, 0.02f, 0xB12 + n);
    size_t escapes = 0;
    bitwise_gate(w, n, k, "gaussian", &escapes);
    if (static_cast<size_t>(n) * k > 1000000)
      require(escapes > 0, "a trained-looking matrix exercises the escape path");
  }
}

DGPP_TEST(bf12_multi_launch_is_bitwise_the_bf16_multi_launch) {
  // The GDN's four input projections in one launch (k = hidden 2560: two
  // super-blocks and the pair tail): every output bitwise
  // launch_bf16_gemv_multi's, at one to four rows, both outputs — with one
  // problem left unpacked (its blocks run the bf16 chain).
  const int k = 2560;
  const int ns[4] = {1027, 771, 12, 12};
  std::vector<std::vector<uint16_t>> w(4);
  std::vector<uint16_t*> dw(4);
  std::vector<Packed> pk(4);
  for (int i = 0; i < 4; ++i) {
    w[i] = gaussian_bf16(static_cast<size_t>(ns[i]) * k, i == 1 ? 3.0e-4f : 0.02f, 0x300 + i);
    DGPP_CUDA_OK(cudaMalloc(&dw[i], w[i].size() * 2));
    DGPP_CUDA_OK(cudaMemcpy(dw[i], w[i].data(), w[i].size() * 2, cudaMemcpyHostToDevice));
    const Bf12Host h = dgpp::bf12_encode(w[i].data(), ns[i], k);
    require(h.ok, "multi: encodes");
    upload(h, ns[i], k, pk[i]);
  }
  const size_t stride = static_cast<size_t>(k) + 16;
  const std::vector<uint16_t> act = gaussian_bf16(4 * stride, 1.5f, 0x3AC7);
  uint16_t* dx = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&dx, act.size() * 2));
  DGPP_CUDA_OK(cudaMemcpy(dx, act.data(), act.size() * 2, cudaMemcpyHostToDevice));
  for (int m = 1; m <= 4; ++m) {
    for (bool f32 : {false, true}) {
      for (int unpacked : {-1, 2}) {
        const size_t es = f32 ? 4 : 2;
        std::vector<void*> want(4), got(4);
        dgpp::Bf16GemvProblem p16[4];
        dgpp::Bf12GemvProblem p12[4];
        for (int i = 0; i < 4; ++i) {
          const size_t bytes = static_cast<size_t>(m) * ns[i] * es;
          DGPP_CUDA_OK(cudaMalloc(&want[i], bytes));
          DGPP_CUDA_OK(cudaMalloc(&got[i], bytes));
          DGPP_CUDA_OK(cudaMemset(want[i], 0xA5, bytes));
          DGPP_CUDA_OK(cudaMemset(got[i], 0x5A, bytes));
          p16[i] = dgpp::Bf16GemvProblem{dx, stride, dw[i], want[i], ns[i]};
          p12[i].act = dx;
          p12[i].act_row_stride = stride;
          if (i != unpacked) p12[i].packed = pk[i].m;
          p12[i].weight = dw[i];
          p12[i].out = got[i];
          p12[i].n = ns[i];
        }
        dgpp::launch_bf16_gemv_multi(p16, 4, f32, m, k, nullptr);
        dgpp::launch_bf12_gemv_multi(p12, 4, f32, m, k, nullptr);
        DGPP_CUDA_OK(cudaDeviceSynchronize());
        for (int i = 0; i < 4; ++i) {
          const size_t bytes = static_cast<size_t>(m) * ns[i] * es;
          std::vector<uint8_t> a(bytes), b(bytes);
          DGPP_CUDA_OK(cudaMemcpy(a.data(), want[i], bytes, cudaMemcpyDeviceToHost));
          DGPP_CUDA_OK(cudaMemcpy(b.data(), got[i], bytes, cudaMemcpyDeviceToHost));
          require(a == b, "multi: problem " + std::to_string(i) + " m=" + std::to_string(m) +
                              (f32 ? " f32" : " bf16") + " bitwise the bf16 multi launch");
          cudaFree(want[i]);
          cudaFree(got[i]);
        }
      }
    }
  }
  cudaFree(dx);
  for (uint16_t* d : dw) cudaFree(d);
}

DGPP_TEST(bf12_rows_of_different_scale_keep_their_own_window) {
  // The KDA in_proj stacks projections of different magnitude: a window
  // per tensor would escape whole rows (and refuse the matrix).
  const int n = 48, k = 4096;
  std::vector<uint16_t> w;
  for (float sigma : {0.02f, 3.0e-5f, 40.0f}) {
    const auto part = gaussian_bf16(static_cast<size_t>(n / 3) * k, sigma, 0x5CA1E + w.size());
    w.insert(w.end(), part.begin(), part.end());
  }
  const Bf12Host h = dgpp::bf12_encode(w.data(), n, k);
  require(h.ok && h.max_row_escapes < 16, "every row packs inside its own window");
  bitwise_gate(w, n, k, "mixed scales");
}

DGPP_TEST(bf12_escapes_are_exact) {
  const int n = 96, k = 2048;
  auto w = gaussian_bf16(static_cast<size_t>(n) * k, 0.02f, 0xE5C);
  std::mt19937_64 rng(0xE5CA9E);
  const auto at = [&](int row, int col) -> uint16_t& { return w[static_cast<size_t>(row) * k + col]; };
  // Zeros of both signs, subnormal-range and huge exponents, the first and
  // last columns of a row, neighbours inside one step, a lane's every step.
  at(0, 0) = 0x0000;
  at(0, 1) = 0x8000;
  at(0, k - 1) = 0x0001;
  at(1, 7) = 0x7F00;  // exponent 254
  at(1, 8) = 0xFF7F;
  at(2, 255) = 0x0080;  // exponent 1
  at(2, 256) = 0x0081;
  for (int c = 0; c < k; c += 257) at(3, c) = 0x0000;
  // A row at the escape bound, another with every element of one step, and
  // an outlier channel past the bound (kept raw: the production chain).
  for (int i = 0; i < dgpp::kBf12MaxRowEscapes; ++i) at(4, static_cast<int>(rng() % k)) = 0x0000;
  for (int c = 0; c < k; c += 5) at(9, c) = static_cast<uint16_t>(((c % 200) + 20) << 7 | (c & 0x7F));
  for (int c = 0; c < k; c += 3) at(95, c) = static_cast<uint16_t>(0x8000 | (((c % 90) + 60) << 7));
  for (int j = 0; j < 8; ++j) at(5, 1024 + 3 * 8 + j) = static_cast<uint16_t>(0x0100 + j);
  require(dgpp::bf12_encode(w.data(), n, k).raw_rows == 2, "the two outlier rows stay raw");
  bitwise_gate(w, n, k, "adversarial");
  // Inf and NaN weights ride the table like any other exponent (their
  // products are the bf16 kernel's: the gate above is bitwise, NaN included).
  at(6, 100) = 0x7F80;  // +inf
  at(7, 200) = 0x7FC0;  // NaN
  bitwise_gate(w, n, k, "non-finite");
}

DGPP_TEST(bf12_refuses_what_it_cannot_hold) {
  const int n = 8, k = 1024;
  auto w = gaussian_bf16(static_cast<size_t>(n) * k, 0.02f, 0xBAD);
  // Rows whose exponents sweep 200 values: no fifteen-wide window holds
  // them. One of eight may stay raw; two of eight and the packing does not pay.
  for (int row : {3, 6})
    for (int c = 0; c < k; ++c)
      w[static_cast<size_t>(row) * k + c] = static_cast<uint16_t>(((c % 200) + 20) << 7);
  const Bf12Host h = dgpp::bf12_encode(w.data(), n, k);
  require(!h.ok && h.raw_rows == 2, "too many raw rows keep the matrix bf16");
  // An all-zero row is one exponent: it packs inside its own window.
  for (int c = 0; c < k; ++c) w[static_cast<size_t>(3) * k + c] = 0;
  const Bf12Host z = dgpp::bf12_encode(w.data(), n, k);
  require(z.ok, "an all-zero row packs");
  require(!dgpp::bf12_encode(w.data(), n, 1004).ok, "k off the eight-column grid is refused");
  require(!dgpp::bf12_shape_ok(4, 131072), "a column past 16 bits is refused");
  dgpp::CublasLtGemm gemm;
  Bf12Matrix empty;
  bool threw = false;
  try {
    gemm.register_bf12(w.data(), empty);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw, "an empty companion is refused");
}

int main() {
  int devices = 0;
  const cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) return 2;  // ctest: skip, no GPU
  return dgpp::test::run_all();
}
