#pragma once
// ReleasableRange: a device allocation whose MEMORY can be returned while
// its ADDRESS stays reserved (2026-09-19, bf12-only residency).
//
// WHY. A bf16 weight that a lossless 12-bit companion replaces
// (kernels/bf12_gemv.hpp) is dead weight once the companion exists — but its
// device address is the key every call site already holds (the GEMM finds
// the companion by it), and a cudaFree would hand that address to the next
// cudaMalloc: a later buffer could alias a registered weight. The CUDA
// virtual-memory API separates the two: reserve an address range, back it
// with a physical allocation, and later unmap + release the backing alone.
// The range stays reserved until destruction, so the key stays unique, and a
// stray read of a released weight is a clean fault (Xid 31) instead of
// another tensor's bytes.
//
// The backing is ordinary device memory (the type cudaMalloc hands out); the
// granularity is 2 MiB on the GB10, so a range is for whole matrices, not
// small tensors. Measured on the part: release returns the full mapped size
// to the pool at once.
#include <cstddef>

namespace dgpp {

class ReleasableRange {
 public:
  ReleasableRange() = default;
  // Reserves and maps at least `bytes` (rounded up to the granularity),
  // readable and writable by the current device. Throws std::runtime_error.
  explicit ReleasableRange(size_t bytes);
  ~ReleasableRange();
  ReleasableRange(ReleasableRange&& o) noexcept;
  ReleasableRange& operator=(ReleasableRange&& o) noexcept;
  ReleasableRange(const ReleasableRange&) = delete;
  ReleasableRange& operator=(const ReleasableRange&) = delete;

  void* data() const { return reinterpret_cast<void*>(va_); }
  size_t bytes() const { return bytes_; }          // as requested
  size_t mapped_bytes() const { return mapped_; }  // 0 once released
  bool mapped() const { return mapped_ != 0; }

  // Unmaps and frees the backing; the address range stays reserved (and
  // unreadable) until destruction. The caller guarantees no work that reads
  // the range is outstanding. Idempotent.
  void release();

  // The mapping granularity of the current device's memory (bytes).
  static size_t granularity();

 private:
  void destroy() noexcept;
  unsigned long long va_ = 0;      // CUdeviceptr
  unsigned long long handle_ = 0;  // CUmemGenericAllocationHandle
  size_t bytes_ = 0;
  size_t reserved_ = 0;
  size_t mapped_ = 0;
};

}  // namespace dgpp
