#pragma once
// Capture-time copy-site tracing (the dsv4's kernels-only graph blocker's
// diagnosis, 2026-09-20; temporary — remove once the enumeration is
// settled). When a cudaMemcpy / cudaMemset call is recorded into a
// captured stream span, name the call site: the gate is
// cudaStreamIsCapturing, so a trace fires exactly once per capture (the
// recorded calls do not execute at replay) and never on the eager or
// replay paths.
//
// The grep keys (the exact log format):
//
//   [cap-trace] memcpy <site> <DIR> src=<ptr> dst=<ptr> <bytes> B
//   [cap-trace] memcpy2d <site> <DIR> src=<ptr> (pitch <p> B) dst=<ptr> (pitch <p> B) <w> x <h> B
//   [cap-trace] memset <site> dst=<ptr> <bytes> B
//
// where <DIR> is the call's declared direction (H2D / D2H / D2D). The
// captured graph's own nodes carry the matching dump from
// check_decode_graph (engine/graph_check.hpp):
//
//   rank <r>: <what> node [<i>/<n>]: memcpy|memcpy2d|memset <DIR> src=<ptr> dst=<ptr> ...
//
// The two join on (src, dst, size); the site lines carry the call order
// (the walk's layer order) the dump's node order does not guarantee.
#include <cstddef>

#include <cuda_runtime.h>

#include "common/log.hpp"

namespace dgpp {

// True when `stream` is inside a cudaStreamBeginCapture span.
inline bool stream_is_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  return cudaStreamIsCapturing(stream, &st) == cudaSuccess && st != cudaStreamCaptureStatusNone;
}

// Name a 1-D copy call site while it is recorded into a capture. No-op
// off a capturing stream (the eager / replay paths print nothing).
inline void capture_trace_copy(const char* site, const char* direction, const void* src, void* dst, size_t bytes,
                               cudaStream_t stream) {
  if (!stream_is_capturing(stream)) return;
  DGPP_LOG_INFO("[cap-trace] memcpy {} {} src={:p} dst={:p} {} B", site, direction, src, dst, bytes);
}

// Name a 2-D copy call site (the pitches in bytes, the width in bytes,
// the height the row count).
inline void capture_trace_copy2d(const char* site, const char* direction, const void* src, size_t src_pitch,
                                 void* dst, size_t dst_pitch, size_t width_bytes, size_t height, cudaStream_t stream) {
  if (!stream_is_capturing(stream)) return;
  DGPP_LOG_INFO("[cap-trace] memcpy2d {} {} src={:p} (pitch {} B) dst={:p} (pitch {} B) {} x {} B", site, direction,
                src, src_pitch, dst, dst_pitch, width_bytes, height);
}

// Name a memset call site.
inline void capture_trace_memset(const char* site, void* dst, size_t bytes, cudaStream_t stream) {
  if (!stream_is_capturing(stream)) return;
  DGPP_LOG_INFO("[cap-trace] memset {} dst={:p} {} B", site, dst, bytes);
}

}  // namespace dgpp
