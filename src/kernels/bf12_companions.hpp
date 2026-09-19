#pragma once
// The owner of a model's bf12 companions (bf12_gemv.hpp): packs resident
// bf16 decode matrices after the load, keeps their device memory, registers
// them with the model's GEMM, and sizes them for the memory plan. One per
// model; every family that reads bf16 weights through CublasLtGemm's GEMV
// lowering uses it the same way.
//
// The process-wide switch is the deployment's `engine.bf16_weights`
// (common/bf16_residency.hpp): "checkpoint" — the bf16 bytes alone, the
// default; "bf12" — the 12-bit companions ALONE: each packed matrix's bf16
// bytes go back to the device as soon as its companion exists
// (pack_and_release), and a prefill GEMM expands what it needs into a small
// scratch (finish); "bf12+bf16" — the companions beside the bf16 bytes. The
// serving app sets it before any model plans or builds; DGPP_BF12 overrides
// it (A/B runs — the launcher forwards DGPP_* to every rank).
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <cuda_runtime.h>

#include "common/bf16_residency.hpp"
#include "kernels/gemm.hpp"

namespace dgpp {

class Bf12Companions {
 public:
  static bool enabled() { return bf16_residency() != Bf16Residency::Checkpoint; }
  // The companions alone: packed matrices give their bf16 bytes back.
  static bool packed_only() { return bf16_residency() == Bf16Residency::Bf12; }
  // The device bytes planned for the companion of a bf16 [n, k] matrix: the
  // packed rows, the row table, and an allowance for escapes and raw rows
  // (measured 1.5e-4 of the weights and 35 rows of a model; 1e-3 and one
  // row in 64 budgeted). 0 for a shape outside the format.
  static size_t planned_bytes(int64_t n, int64_t k);
  // bf12-only residency's expansion scratch (CublasLtGemm::bf12_reserve_expand):
  // a slot holds the largest packed matrix that fits kExpandSlotCap whole; a
  // larger one (an lm head) runs through it in weight-row blocks.
  // `matrix_bytes`: the packed matrices' bf16 sizes (0 entries ignored).
  static constexpr size_t kExpandSlotCap = size_t{128} << 20;
  static size_t planned_slot_bytes(std::initializer_list<size_t> matrix_bytes);

  Bf12Companions() = default;
  ~Bf12Companions();
  Bf12Companions(const Bf12Companions&) = delete;
  Bf12Companions& operator=(const Bf12Companions&) = delete;

  // Packs the resident device matrix weight[n, k] — read back through
  // `stream`, encoded on the host, uploaded — and registers the companion
  // with `gemm`. False (and nothing registered) when the matrix keeps its
  // bf16 form: a null pointer, a shape outside the format, too many raw
  // rows. Synchronizes `stream`; call after the load, before any capture.
  bool pack(const uint16_t* weight, int64_t n, int64_t k, CublasLtGemm& gemm, cudaStream_t stream);

  // pack(), then — under bf12-only residency — the matrix's own bytes go
  // back through `release` (the loader's release_packed for this matrix:
  // const void* -> bytes freed, 0 when it was not granted aside) and the
  // GEMM learns that only the address is left. Call as each layer lands, so
  // no more than a layer's bf16 bytes sit beside their companions.
  template <class Release>
  bool pack_and_release(const uint16_t* weight, int64_t n, int64_t k, CublasLtGemm& gemm,
                        cudaStream_t stream, Release&& release) {
    if (!pack(weight, n, k, gemm, stream)) return false;
    if (!packed_only()) return true;
    const size_t freed = release(static_cast<const void*>(weight));
    if (freed == 0) return true;
    gemm.bf12_release_raw(weight);
    ++released_;
    released_bytes_ += freed;
    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(k) * 2;
    if (bytes <= kExpandSlotCap) largest_slot_ = std::max(largest_slot_, bytes);
    return true;
  }
  // After the last pack: reserves the GEMM's expansion scratch (`slots`
  // buffers) when any matrix was released.
  void finish(CublasLtGemm& gemm, int slots);

  size_t matrices() const { return matrices_; }
  size_t kept_bf16() const { return kept_; }
  size_t bf16_bytes() const { return raw_bytes_; }
  size_t packed_bytes() const { return packed_bytes_; }
  size_t released() const { return released_; }
  size_t released_bytes() const { return released_bytes_; }
  // One INFO line (nothing when no matrix was offered).
  void log_summary(int rank, double seconds) const;

 private:
  std::vector<void*> allocs_;
  std::vector<uint16_t> host_;
  size_t matrices_ = 0, kept_ = 0, raw_bytes_ = 0, packed_bytes_ = 0, escapes_ = 0, raw_rows_ = 0;
  size_t released_ = 0, released_bytes_ = 0, largest_slot_ = 0, scratch_bytes_ = 0;
  int widest_ = 0;
};

}  // namespace dgpp
