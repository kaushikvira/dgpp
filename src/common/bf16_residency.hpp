#pragma once
// The resident form of the checkpoint's native bf16 decode weights
// (engine.bf16_weights). Process-wide: set once, before the model family is
// constructed — the loaders lay their layers out by it, the memory plans
// count by it, the models pack by it.
//
//   Checkpoint   bf16 as stored.
//   Bf12         the lossless 12-bit form alone (kernels/bf12_gemv.hpp): each
//                packable matrix loads into a releasable range
//                (loaders/releasable_range.hpp), is packed, and its bf16
//                bytes are returned — 0.75 of the memory; a prefill GEMM
//                expands the rows it needs into a small scratch first.
//   Bf12AndBf16  both forms resident: decode streams the 12-bit form, prefill
//                reads the bf16 bytes in place — 1.75 of the memory, no
//                prefill cost.
//
// DGPP_BF12 = off | on (= bf12) | both overrides the setting (A/B arms).
#include <string>

namespace dgpp {

enum class Bf16Residency { Checkpoint, Bf12, Bf12AndBf16 };

void set_bf16_residency(Bf16Residency mode);
Bf16Residency bf16_residency();

// "checkpoint" | "bf12" | "bf12+bf16" (the cluster JSON's values).
const char* bf16_residency_name(Bf16Residency mode);
bool parse_bf16_residency(const std::string& name, Bf16Residency* mode);

// Packable matrices load into releasable ranges (the Bf12 mode).
inline bool bf16_side_grants() { return bf16_residency() == Bf16Residency::Bf12; }

}  // namespace dgpp
