#pragma once
// bf12: a LOSSLESS 12-bit resident form of a bf16 weight matrix for the
// decode GEMV (2026-09-19).
//
// WHY THIS EXISTS: the decode step is a bandwidth problem and the native
// bf16 tensors (the KDA projections, the lm head) are over half of the bytes
// a GLM-5.3-Flash step reads. A bf16 weight's sign and mantissa are
// incompressible, but its exponent is not: a trained matrix spends ~2.6 bits
// of entropy on it and 99.98 % of the weights sit inside a window of fifteen
// consecutive exponents. The packed form stores the sign+mantissa byte and a
// 4-bit exponent code against a per-ROW base (a fused projection's rows do
// not share a scale: the KDA in_proj stacks f_a|g_a|q|k|v|b) — 12 bits a
// weight, 0.75 of the bytes — and the kernel rebuilds the exact bf16 bits in
// registers.
// Code 15 is the escape: the weight's true bits live in a per-row side table
// (column-sorted, a few entries per thousand rows) and the lane that meets
// the code fetches them before the weight enters its chain.
//
// NUMERICS: none. Every weight value is bit-for-bit the bf16 weight, and the
// lane/step ownership and the FMA order are bf16_gemv's (bf16_gemv.cuh), so
// every output is bitwise launch_bf16_gemv's — the unit test pins it, and a
// served transcript cannot tell the two apart. This is a storage format, not
// a quantization.
//
// LAYOUT: a row is k / 1024 super-blocks of 1536 bytes: [sm A | sm B | exp],
// 512 bytes each. Lane l owns 16 bytes of each segment at offset 16 * l:
// A holds the sign+mantissa bytes of its eight elements of steps 0 and 1 (a
// step is the production kernel's 256-element warp step, a lane's share of
// it eight elements), B steps 2 and 3, exp the four steps' exponent nibbles
// (low nibble = even element). Every warp load is a contiguous 512 bytes —
// four whole lines, as the bf16 core's.
//
// RAW ROWS: the escape lookup is a linear scan, so a row with more than
// kBf12MaxRowEscapes escapes (an outlier channel: a handful of q/k rows in
// a few layers) keeps its bf16 bits in a side array, and its warp — a warp
// is one weight row, so the branch is uniform across its lanes — runs the
// production row chain on them (bf16_gemv::row_dots itself).
//
// THE TAIL (2026-09-19, format v2; bf12_gemv.cuh has the layout): the
// k % 1024 columns that do not fill a super-block follow the row's
// super-blocks in units cut by the 256-column steps they hold, without
// padding — Qwen's hidden 2560 and its 1536-wide slices leave 512 columns
// (768 bytes), the GR up projection's k = 320 is all tail (480 bytes). A
// matrix with k % 1024 == 0 has none and is laid out as before.
//
// CONTRACT: k % 8 == 0 (the bf16 GEMV's), k <= 65536 (an escape's column is
// 16 bits), at most one row in kBf12MaxRawFraction kept raw (past that the
// matrix keeps its bf16 form: the packing would not pay).
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace dgpp {

constexpr int kBf12Super = 1024;       // elements per super-block
constexpr int kBf12SuperBytes = 1536;  // their packed bytes
constexpr int kBf12Window = 15;        // exponent codes 0..14; 15 escapes
constexpr int kBf12MaxRowEscapes = 64;
constexpr int kBf12MaxRows = 8;        // activation rows per launch
constexpr int kBf12MaxRawFraction = 8;
constexpr uint32_t kBf12RawRow = 0x80000000u;  // a rows[] base word: raw row | index

// Device views of one packed matrix (the owner keeps the allocations).
struct Bf12Matrix {
  const uint8_t* packed = nullptr;  // [n][bf12_row_bytes(k)]
  // [n + 1] pairs {first escape of the row, the row's window base << 23 —
  // or kBf12RawRow | its index in raw}; the pair after the last row closes
  // its escape range.
  const uint32_t* rows = nullptr;
  const uint32_t* esc = nullptr;    // (column << 16) | bf16 bits
  const uint16_t* raw = nullptr;    // [raw rows][k] bf16, 16-byte aligned
  int n = 0, k = 0;
};

// The host form bf12_encode produces (upload the three arrays as they are).
struct Bf12Host {
  std::vector<uint8_t> packed;
  std::vector<uint32_t> rows, esc;  // Bf12Matrix's arrays
  std::vector<uint16_t> raw;        // the raw rows (never empty: one pad row)
  size_t escapes = 0;
  int raw_rows = 0;
  int max_row_escapes = 0;  // the widest packed row's escapes
  bool ok = false;          // false: the matrix keeps its bf16 form
};

inline bool bf12_shape_ok(int n, int k) { return n > 0 && k > 0 && (k % 8) == 0 && k <= 65536; }
// A packed row: its super-blocks, then the tail's units (bf12_gemv.cuh) —
// a multiple of 16 bytes, twelve bits a weight but for the partial step's
// rounding.
inline size_t bf12_row_bytes(int k) {
  const int nsb = k / kBf12Super, cols = k - nsb * kBf12Super;
  const int steps = cols / 256, lanes = (cols - steps * 256) / 8;
  size_t bytes = static_cast<size_t>(nsb) * kBf12SuperBytes + static_cast<size_t>(steps >> 1) * 768 +
                 static_cast<size_t>(steps & 1) * 384;
  if (lanes > 0) bytes += static_cast<size_t>((8 * lanes + 15) & ~15) + static_cast<size_t>((4 * lanes + 15) & ~15);
  return bytes;
}
inline size_t bf12_packed_bytes(int n, int k) { return static_cast<size_t>(n) * bf12_row_bytes(k); }

// Packs w[n, k] (bf16 bits). threads <= 0 picks the hardware's. ok == false
// when the shape is outside the contract or too many rows would stay raw.
Bf12Host bf12_encode(const uint16_t* w, int n, int k, int threads = 0);
// The inverse (tests): out[n, k] bf16 bits.
void bf12_decode(const Bf12Host& h, int n, int k, uint16_t* out);

// m activation rows this launch can take: 1..kBf12MaxRows. Up to four rows
// whose staged rows fit the smem bound run the narrow kernel (whole rows
// staged, the row's loads in flight at once); five to eight rows — and the
// narrow counts past the bound — the wide one (the activations staged a
// super-block at a time, 16 KB at eight rows). Both keep the scalar chain,
// so a row's bits do not depend on the rows sharing its launch.
bool bf12_gemv_accepts(const Bf12Matrix& w, int m);

// Up to four GEMVs of the same m, k and output type in one launch — the
// twin of launch_bf16_gemv_multi (the GDN's qkv / z / a / b projections at
// decode): blocks [B_i, B_{i+1}) run problem i, each warp's work exactly
// the single launch's, so every output is bitwise its own launch — and
// bitwise the bf16 launch's. A problem whose matrix did not pack
// (packed.packed == nullptr) runs the bf16 chain on `weight` in its blocks.
// m in 1..4 with the staged rows inside the smem bound (bf16_gemv_accepts).
struct Bf12GemvProblem {
  const uint16_t* act = nullptr;
  size_t act_row_stride = 0;
  Bf12Matrix packed;                 // its companion, or empty
  const uint16_t* weight = nullptr;  // the bf16 rows (read only when not packed)
  void* out = nullptr;
  int n = 0;
};
constexpr int kBf12GemvMaxProblems = 4;
void launch_bf12_gemv_multi(const Bf12GemvProblem* problems, int n_problems, bool out_f32, int m,
                            int k, cudaStream_t stream);

// out[m, n] = act[m, k] x W^T, bitwise launch_bf16_gemv's rows on the
// unpacked weights (in chunks, past four rows). act_row_stride in elements;
// out bf16 or f32 by out_f32.
void launch_bf12_gemv(const uint16_t* act, size_t act_row_stride, const Bf12Matrix& w,
                      void* out, bool out_f32, int m, cudaStream_t stream);

// The exact bf16 bits of weight rows [row0, row0 + rows) into out[rows, k]
// (16-byte aligned): what a prefill GEMM reads when the matrix's own bf16
// bytes are not resident (bf12-only residency). A bandwidth kernel — 0.75
// read, 1.0 written, at the device memcpy's rate (16.8 MB: 130 us).
void launch_bf12_expand(const Bf12Matrix& w, int row0, int rows, uint16_t* out,
                        cudaStream_t stream);

}  // namespace dgpp
