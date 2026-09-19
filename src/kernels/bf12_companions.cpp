#include "kernels/bf12_companions.hpp"

#include <string>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/bf12_gemv.hpp"

namespace dgpp {
namespace {

size_t round16(size_t bytes) { return (bytes + 15) / 16 * 16; }

}  // namespace

size_t Bf12Companions::planned_slot_bytes(std::initializer_list<size_t> matrix_bytes) {
  size_t largest = 0;
  bool any = false;
  for (const size_t b : matrix_bytes) {
    if (b == 0) continue;
    any = true;
    if (b <= kExpandSlotCap) largest = std::max(largest, b);
  }
  if (!any) return 0;
  // Every matrix past the cap: the blocks' slot is the cap itself.
  const size_t slot = largest != 0 ? largest : kExpandSlotCap;
  return (slot + 255) / 256 * 256;
}

void Bf12Companions::finish(CublasLtGemm& gemm, int slots) {
  // The read-back buffer held the largest matrix (the head: 0.3-0.6 GiB of
  // the unified pool) — give it back now that nothing more is packed.
  std::vector<uint16_t>().swap(host_);
  if (released_ == 0) return;
  // Every released matrix was past the cap: the blocks' slot is the cap.
  const size_t slot = largest_slot_ != 0 ? largest_slot_ : kExpandSlotCap;
  gemm.bf12_reserve_expand(slots, slot);
  scratch_bytes_ = gemm.bf12_expand_bytes();
}

size_t Bf12Companions::planned_bytes(int64_t n, int64_t k) {
  if (n <= 0 || k <= 0 || n > INT32_MAX || !bf12_shape_ok(static_cast<int>(n), static_cast<int>(k)))
    return 0;
  const size_t rows = static_cast<size_t>(n), cols = static_cast<size_t>(k);
  const size_t escapes = rows * cols / 1000 + 16;
  const size_t raw_rows = rows / 64 + 1;
  return bf12_packed_bytes(static_cast<int>(n), static_cast<int>(k)) +
         round16((2 * (rows + 1) + escapes) * 4) + round16(raw_rows * cols * 2);
}

Bf12Companions::~Bf12Companions() {
  for (void* p : allocs_) cudaFree(p);
}

bool Bf12Companions::pack(const uint16_t* weight, int64_t n, int64_t k, CublasLtGemm& gemm,
                          cudaStream_t stream) {
  if (weight == nullptr || n <= 0 || k <= 0 || n > INT32_MAX ||
      !bf12_shape_ok(static_cast<int>(n), static_cast<int>(k))) {
    ++kept_;
    return false;
  }
  const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(k);
  host_.resize(elems);
  DGPP_CUDA_OK(cudaMemcpyAsync(host_.data(), weight, elems * 2, cudaMemcpyDeviceToHost, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  const Bf12Host h = bf12_encode(host_.data(), static_cast<int>(n), static_cast<int>(k));
  if (!h.ok) {
    ++kept_;
    return false;
  }
  // One allocation per matrix: [packed | rows | esc | raw rows], 16-byte joints.
  const size_t pb = h.packed.size();
  const size_t bb = round16(h.rows.size() * 4);
  const size_t eb = round16(h.esc.size() * 4);
  const size_t rb = round16(h.raw.size() * 2);
  void* mem = nullptr;
  DGPP_CUDA_OK(cudaMalloc(&mem, pb + bb + eb + rb));
  allocs_.push_back(mem);
  auto* dev = static_cast<uint8_t*>(mem);
  DGPP_CUDA_OK(cudaMemcpyAsync(dev, h.packed.data(), pb, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(dev + pb, h.rows.data(), h.rows.size() * 4, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(dev + pb + bb, h.esc.data(), h.esc.size() * 4, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaMemcpyAsync(dev + pb + bb + eb, h.raw.data(), h.raw.size() * 2, cudaMemcpyHostToDevice, stream));
  DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  Bf12Matrix m;
  m.packed = dev;
  m.rows = reinterpret_cast<const uint32_t*>(dev + pb);
  m.esc = reinterpret_cast<const uint32_t*>(dev + pb + bb);
  m.raw = reinterpret_cast<const uint16_t*>(dev + pb + bb + eb);
  m.n = static_cast<int>(n);
  m.k = static_cast<int>(k);
  gemm.register_bf12(weight, m);
  ++matrices_;
  raw_bytes_ += elems * 2;
  packed_bytes_ += pb + bb + eb + rb;
  escapes_ += h.escapes;
  raw_rows_ += static_cast<size_t>(h.raw_rows);
  if (h.max_row_escapes > widest_) widest_ = h.max_row_escapes;
  return true;
}

void Bf12Companions::log_summary(int rank, double seconds) const {
  if (matrices_ + kept_ == 0) return;
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  DGPP_LOG_INFO("rank {} bf12: {} decode matrices packed ({:.2f} GiB of bf16 -> {:.2f} GiB, {} escapes, "
                "widest row {}, {} rows kept raw; {} matrices kept bf16) in {:.1f} s",
                rank, matrices_, static_cast<double>(raw_bytes_) / kGiB,
                static_cast<double>(packed_bytes_) / kGiB, escapes_, widest_, raw_rows_, kept_, seconds);
  if (packed_only()) {
    if (released_ != 0)
      DGPP_LOG_INFO("rank {} bf12: {} of them released their bf16 bytes ({:.2f} GiB returned; prefill expands "
                    "through a {:.0f} MiB scratch)",
                    rank, released_, static_cast<double>(released_bytes_) / kGiB,
                    static_cast<double>(scratch_bytes_) / (1024.0 * 1024.0));
    if (released_ == 0)
      DGPP_LOG_INFO("rank {} bf12: this family keeps the bf16 bytes beside the companions (its loader grants "
                    "nothing aside): engine.bf16_weights = bf12 serves as bf12+bf16 here",
                    rank);
    else if (released_ != matrices_)
      DGPP_LOG_WARN("rank {} bf12: {} packed matrices kept their bf16 bytes (not granted aside by the "
                    "loader) — the memory plan counted them released",
                    rank, matrices_ - released_);
  }
}

}  // namespace dgpp
