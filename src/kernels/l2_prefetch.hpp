#pragma once
// L2 weight prefetch. The decode step is one serial chain of
// kernels, and whenever the chain sits in a latency-bound phase — the bus
// all-reduce, the KDA recurrence, the DSA select/absorb, the router's
// top-k — the memory system idles. Weights, unlike routed experts, are
// known in advance, so a low-priority side stream reads the NEXT kernels'
// bytes into L2 (24 MB on the GB10) while the chain waits; the consuming
// GEMV then hits L2 for that prefix. Pure performance: nothing the model
// reads is written, so every result is bit-identical with it on or off.
//
// Capture-safe: the side stream forks from and joins the main stream
// through events, which under stream capture become graph edges.
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace dgpp {

// How hard a prefetch pushes the memory system. On the GB10 the GPU, the
// CPU and the NIC share one LPDDR5X controller: a prefetch with megabytes
// in flight lifts EVERYONE's memory latency (Little's law — 3 MB in flight
// at 240 GB/s is 12 us of queue), which stalls the bus engine's posts and
// the peers' handshakes during a collective. Full keeps enough in flight
// for line rate; Light caps the in-flight bytes so the collective's
// latency-sensitive path stays fast while DRAM still moves ~150 GB/s.
enum class PrefetchRate { Off, Light, Full };

// Streams [ptr, ptr + bytes) through L2 (cache-global loads, no L1) with a
// modest grid so a concurrently running chain kernel still finds SM slots.
// bytes == 0 or ptr == nullptr is a no-op. ptr need not be aligned.
void launch_l2_prefetch(const void* ptr, size_t bytes, PrefetchRate rate,
                        cudaStream_t stream);

// The prefetch scheduler. A WINDOW is opened at a point of the main stream
// (everything added to it starts once the main stream reaches that point)
// and carries a byte budget: a window's prefetch must fit in L2 beside the
// chain's own traffic, and must not outlast the latency it hides by much
// (past that it merely competes with the consumer for bandwidth). Layers
// take a pointer to this class and call prefetch_after() at the one point
// in their enqueue where the next weight is known and the chain is about
// to go latency-bound.
//
// Knobs: DGPP_L2_PREFETCH=off disables (every call is a no-op);
// DGPP_L2_PREFETCH_MB overrides the default window budget;
// DGPP_L2_PREFETCH_BOUNDARY=off|light|full sets the rate of the windows
// that overlap a collective (boundary_rate(); default light);
// DGPP_L2_PREFETCH_LAYER=off|light|full the rate of the windows inside the
// attention layers (layer_rate(); default light — the Full rate slowed the
// small kernels it ran beside by as much as it saved on the projection).
class WeightPrefetcher {
 public:
  static constexpr size_t kDefaultWindowBytes = size_t{12} << 20;

  WeightPrefetcher();
  ~WeightPrefetcher();
  WeightPrefetcher(const WeightPrefetcher&) = delete;
  WeightPrefetcher& operator=(const WeightPrefetcher&) = delete;

  bool enabled() const { return enabled_; }
  size_t default_window_bytes() const { return window_bytes_; }
  // The rate for windows that run beside a collective (see PrefetchRate).
  PrefetchRate boundary_rate() const { return boundary_rate_; }
  // The rate for windows inside a layer (beside the chain's small kernels).
  PrefetchRate layer_rate() const { return layer_rate_; }

  // Opens a window at `main`'s current position with the given budget
  // (0 = the default) and rate. Adds before the next open_window() share
  // it. Rate Off opens a window that ignores every add().
  void open_window(cudaStream_t main, size_t budget_bytes = 0,
                   PrefetchRate rate = PrefetchRate::Full);
  // Adds a range to the open window, clamped to its remaining budget.
  // Adjacent (or nearly adjacent) ranges coalesce into one launch
  //: a window's adds are a layer's tensors in consumption
  // order and they sit contiguously in the resident image, so what was
  // ~620 one-tensor kernels per decode step — 40 % of the graph's nodes,
  // and cudaGraphLaunch costs ~0.45 us per node on this host — becomes a
  // few dozen. The pending range launches at the next non-adjacent add,
  // at the next open_window(), or at join(). DGPP_L2_PREFETCH_MERGE=off
  // restores one launch per add.
  void add(const void* ptr, size_t bytes);
  // add() for a range that is its OWN allocation (a bf16 weight's packed
  // companion — IGemm::resident_view — not a span of the layer's image): it
  // never joins a pending range and nothing joins it. The coalescing bridges
  // holes of up to kMergeGap between adds, which is a read of whatever lies
  // between them — a layer's neighbouring tensors inside one image, but
  // between two allocations a hole can be unmapped (a released bf16 range,
  // loaders/releasable_range.hpp, stays reserved and unreadable; a
  // neighbouring image's edge is an out-of-bounds read, compute-sanitizer
  // 2026-09-09). One launch of its own.
  void add_isolated(const void* ptr, size_t bytes);
  // The matmul weight `weight` through its resident view: add() when the
  // view is the weight itself (its image's span), add_isolated() when it is
  // a companion.
  void add_view(const void* weight, const void* view, size_t view_bytes) {
    if (view == weight) add(view, view_bytes);
    else add_isolated(view, view_bytes);
  }
  // open_window + add in one call — the layers' idiom.
  void prefetch_after(cudaStream_t main, const void* ptr, size_t bytes,
                      size_t budget_bytes = 0,
                      PrefetchRate rate = PrefetchRate::Full);
  // `main` waits for every prefetch issued since the last join. Required
  // before a capture ends (a forked stream must rejoin the origin); a
  // no-op when nothing was forked.
  void join(cudaStream_t main);
  // Launches (and cumulative bytes) issued so far — the merge's evidence.
  size_t launches() const { return launches_; }

 private:
  // The coalescing range (see add): [pending_begin_, pending_end_) not
  // yet launched; merge_ is the knob; kMergeGap the largest hole bridged.
  static constexpr size_t kMergeGap = size_t{2} << 20;  // bridged gaps are charged to the budget
  void flush_pending();
  bool merge_ = true;
  uintptr_t pending_begin_ = 0, pending_end_ = 0;
  size_t launches_ = 0;
  bool enabled_ = true;
  size_t window_bytes_ = kDefaultWindowBytes;
  PrefetchRate boundary_rate_ = PrefetchRate::Light;
  PrefetchRate layer_rate_ = PrefetchRate::Light;
  PrefetchRate rate_ = PrefetchRate::Full;  // the open window's
  size_t remaining_ = 0;
  bool window_open_ = false;
  bool forked_ = false;  // a fork since the last join (join must follow)
  cudaStream_t side_ = nullptr;
  cudaEvent_t fork_ = nullptr;
  cudaEvent_t join_ = nullptr;
};

}  // namespace dgpp
