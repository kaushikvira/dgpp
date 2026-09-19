#pragma once
// Validate node types in captured decode graphs.
//
// Memcpy and memset nodes use a shared, ordered copy-engine queue. In an
// in-process multi-rank world, a node waiting on one rank's collective can
// block another rank's earlier copy and form a dependency cycle. Decode
// uploads and resets therefore use kernels. Event-record nodes are also
// rejected; the replay verdict is published by a kernel.
//
// Kernel and empty nodes are allowed. A family may declare a bounded
// number of host nodes, used by Qwen to gather mapped n-gram rows. Such
// callbacks must avoid cross-rank waits. See docs/batched_mtp_graph_stall.md
// for the trace, reproducer and original failure analysis.
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {

// Log the node-type histogram and throw std::runtime_error for unsupported
// nodes or more than host_nodes callbacks. The family supplies its host
// node allowance through Model::session_graph_host_nodes().
inline void check_decode_graph(cudaGraph_t graph, int rank,
                                   const std::string& what, size_t host_nodes = 0) {
  size_t n = 0, e = 0;
  DGPP_CUDA_OK(cudaGraphGetNodes(graph, nullptr, &n));
  std::vector<cudaGraphNode_t> nodes(n);
  DGPP_CUDA_OK(cudaGraphGetNodes(graph, nodes.data(), &n));
  DGPP_CUDA_OK(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &e));
  size_t kernels = 0, empties = 0, memcpys = 0, memsets = 0, hosts = 0,
         events = 0, others = 0, max_in = 0;
  for (size_t i = 0; i < n; ++i) {
    cudaGraphNode_t node = nodes[i];
    cudaGraphNodeType t;
    DGPP_CUDA_OK(cudaGraphNodeGetType(node, &t));
    switch (t) {
      case cudaGraphNodeTypeKernel: ++kernels; break;
      case cudaGraphNodeTypeEmpty: ++empties; break;
      case cudaGraphNodeTypeMemcpy: ++memcpys; break;
      case cudaGraphNodeTypeMemset: ++memsets; break;
      case cudaGraphNodeTypeHost: ++hosts; break;
      // An event record node is REJECTED with the rest (2026-09-06): an
      // external event record node in the replayed decode graph stalled
      // about one relaunch in five — the graph never started on one rank
      // while its peer spun in the first collective. The pipelined
      // replay's verdict is published by a kernel node instead
      // (glm_publish_seq).
      case cudaGraphNodeTypeEventRecord: ++events; break;
      default: ++others; break;
    }
    // The copy-site enumeration (the dsv4's kernels-only graph blocker's
    // diagnosis, 2026-09-20): every non-kernel node's parameters, so the
    // capture's [cap-trace] site lines (common/capture_trace.hpp) join
    // onto the graph's own nodes by (src, dst, size). Kernels and empty
    // nodes print nothing (the thousands of them are the graph's body).
    if (t == cudaGraphNodeTypeMemcpy) {
      cudaMemcpy3DParms p{};
      DGPP_CUDA_OK(cudaGraphMemcpyNodeGetParams(node, &p));
      std::string dir;
      switch (p.kind) {
        case cudaMemcpyHostToDevice: dir = "H2D"; break;
        case cudaMemcpyDeviceToHost: dir = "D2H"; break;
        case cudaMemcpyDeviceToDevice: dir = "D2D"; break;
        default: dir = "kind=" + std::to_string(static_cast<int>(p.kind)); break;
      }
      if (p.extent.height > 1)
        DGPP_LOG_INFO("rank {}: {} node [{}/{}]: memcpy2d {} src={:p} (pitch {} B) dst={:p} (pitch {} B) {} x {} B",
                      rank, what, i, n, dir, p.srcPtr.ptr, p.srcPtr.pitch, p.dstPtr.ptr, p.dstPtr.pitch,
                      p.extent.width, p.extent.height);
      else
        DGPP_LOG_INFO("rank {}: {} node [{}/{}]: memcpy {} src={:p} dst={:p} {} B", rank, what, i, n, dir,
                      p.srcPtr.ptr, p.dstPtr.ptr, p.extent.width);
    } else if (t == cudaGraphNodeTypeMemset) {
      cudaMemsetParams p{};
      DGPP_CUDA_OK(cudaGraphMemsetNodeGetParams(node, &p));
      DGPP_LOG_INFO("rank {}: {} node [{}/{}]: memset dst={:p} {} x {} x {} B (value {})", rank, what, i, n, p.dst,
                    p.width, p.height, p.elementSize, p.value);
    } else if (t != cudaGraphNodeTypeKernel && t != cudaGraphNodeTypeEmpty) {
      const char* tname = "other";
      switch (t) {
        case cudaGraphNodeTypeHost: tname = "host"; break;
        case cudaGraphNodeTypeEventRecord: tname = "event-record"; break;
        case cudaGraphNodeTypeWaitEvent: tname = "event-wait"; break;
        case cudaGraphNodeTypeGraph: tname = "child-graph"; break;
        default: break;
      }
      DGPP_LOG_INFO("rank {}: {} node [{}/{}]: {} node (type {})", rank, what, i, n, tname,
                    static_cast<int>(t));
    }
    size_t deps = 0;
    DGPP_CUDA_OK(cudaGraphNodeGetDependencies(node, nullptr, nullptr, &deps));
    max_in = std::max(max_in, deps);
  }
  DGPP_LOG_INFO(
      "rank {}: {} shape: {} nodes, {} edges: kernel {} empty {} memcpy {} "
      "memset {} host {} event {} other {}; max in-degree {}",
      rank, what, n, e, kernels, empties, memcpys, memsets, hosts, events,
      others, max_in);
  if (memcpys != 0 || memsets != 0 || hosts > host_nodes || events != 0 ||
      others != 0)
    throw std::runtime_error(
        what + " captured " + std::to_string(memcpys) + " memcpy, " +
        std::to_string(memsets) + " memset, " + std::to_string(hosts) +
        " host, " + std::to_string(events) + " event and " +
        std::to_string(others) +
        " other node(s); the decode graph must be kernels-only (" +
        std::to_string(host_nodes) + " host node(s) declared) — a "
        "copy-engine node can deadlock the in-process multi-rank world "
        "(docs/batched_mtp_graph_stall.md)");
}

}  // namespace dgpp
