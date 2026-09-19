#pragma once
// Debug-only per-layer hidden-state dump for the dsv4 forward pass (the
// DGPP_DSV4_DUMP_LAYERS environment variable; the degenerate-output
// localizer: the per-layer mHC collapse + the final logits of the prompt's
// last position, to compare against the reference lane's).
//
// Inert by default: with the variable unset (or naming a missing
// directory) the dump is off and the model's path is exactly the no-dump's
// (no extra kernel, no copy, no file). The serve app refuses to activate
// it when decode_graph is on: the dump's device->host copies must never
// land inside a captured decode graph (run with decode_graph: false).
//
// What lands where. For every prefill pass that ends the prompt (the
// last_chunk's run_rows' — the last 32-token chunk's, or a snapshot
// cut's that ends the span's), in a numbered subdirectory per rank
// (<dir>/req_<NNNN>/rank_<R>/; NNNN's the process-local pass's
// counter's, the first request's the req_0000's; a second request's
// request gets the next number's, no overwrite's):
//   layer_<NN>.f32  -- after layer NN's (the 0-based's index's, 2
//     digits') completes: the mHC collapse of the residual streams'
//     the layer's exit's (the exact quantity the next layer's attention
//     site's / the final norm's consumes — the head's collapse's the
//     same call's), the model's own bf16 hidden's widened losslessly
//     to little-endian float32, hidden_size's floats, the pass's last
//     row's (the last prompt position's).
//   logits_top.txt  -- after the lm head: the top-20's (token id,
//     logit)'s of this rank's final-logit slice's the last prompt's
//     position's, one per line (the id's the absolute's, offset by
//     this rank's lm_vocab_begin's).
//   meta.txt        -- the pass's prompt token ids (the chunk's; pos0
//     + T's the positions' the chunk's position's the pass's),
//     hidden_size, num_hidden_layers, the world's size's, the rank,
//     and the layer's file's mapping.
//
// Which rank dumps: every rank dumps (each rank's a separate process's
// on its own machine's, its own local <dir>'s). The layer_<NN>.f32's
// files' are bitwise identical across ranks (the collapse's the
// post-fold streams' + the replicated mHC coefficients' — the
// boundary's output's, every rank's bitwise's the other's), so the
// dumped quantity's the full's post-collective's hidden's (not a
// rank-local's slice's) — read rank 0's. Only the logits_top.txt's
// differ (the sharded lm head's the vocab's slice's): merge the
// ranks' files' for the full vocab's top-20's.
//
// Ugly-but-obvious's on purpose's: a diagnostic's, not a feature's.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {

class Dsv4LayerDump {
 public:
  Dsv4LayerDump() = default;
  bool active() const { return active_; }

  // Reads the environment: active only when DGPP_DSV4_DUMP_LAYERS names
  // an existing directory (a missing directory logs a WARN's and stays
  // off's). Called by the serve app after the model's construction's
  // (its decode_graph's off's gate's the call site's).
  static Dsv4LayerDump from_env(int rank, int world, int hidden, int layers, int lm_vocab_begin,
                                int lm_vocab_count) {
    Dsv4LayerDump d;
    const char* env = std::getenv("DGPP_DSV4_DUMP_LAYERS");
    if (env == nullptr || env[0] == '\0') return d;  // unset: inert
    d.dir_ = env;
    std::error_code ec;
    if (!std::filesystem::is_directory(d.dir_, ec)) {
      DGPP_LOG_WARN(
          "dsv4 layer dump: DGPP_DSV4_DUMP_LAYERS={} names no directory — the dump stays off (create it first)",
          d.dir_);
      return d;
    }
    d.rank_ = rank;
    d.world_ = world;
    d.hidden_ = hidden;
    d.layers_ = layers;
    d.lm_vocab_begin_ = lm_vocab_begin;
    d.lm_vocab_count_ = lm_vocab_count;
    d.active_ = true;
    DGPP_LOG_INFO("dsv4 layer dump: active — per-layer hidden states into {} (rank {}/{})", d.dir_, rank, world);
    return d;
  }

  // One dumped prefill pass (the prompt's last chunk's): opens
  // <dir>/req_<NNNN>/rank_<R>/ and records the pass's prompt ids.
  void begin_pass(const int64_t* prompt_ids, int T, int64_t pos0) {
    if (!active_) return;
    ++pass_;
    std::error_code ec;
    pass_dir_ = dir_ + "/req_" + pad4(pass_ - 1) + "/rank_" + std::to_string(rank_);
    std::filesystem::create_directories(pass_dir_, ec);
    if (ec) {
      DGPP_LOG_WARN("dsv4 layer dump: cannot create {} ({}) — the pass's dump is off", pass_dir_, ec.message());
      active_ = false;
      return;
    }
    pass_pos0_ = pos0;
    pass_tokens_.assign(prompt_ids, prompt_ids + T);
  }

  // After layer `layer` completes: `hidden_bf16` is the pass's last
  // row's mHC collapse (bf16, hidden_ floats) — the layer-exit hidden
  // state (the next layer's / the final norm's input's). Widens to f32
  // and writes layer_<NN>.f32 (raw little-endian).
  void write_layer(int layer, const uint16_t* hidden_bf16, cudaStream_t stream) {
    if (!active_) return;
    std::vector<uint16_t> row(static_cast<size_t>(hidden_));
    DGPP_CUDA_OK(cudaMemcpyAsync(row.data(), hidden_bf16, static_cast<size_t>(hidden_) * 2, cudaMemcpyDeviceToHost,
                                 stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    const std::string path = pass_dir_ + "/layer_" + pad2(layer) + ".f32";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
      DGPP_LOG_WARN("dsv4 layer dump: cannot write {} — the layer's file is missing", path);
      return;
    }
    std::vector<float> out(static_cast<size_t>(hidden_));
    for (int i = 0; i < hidden_; ++i) {
      // The bf16's bits's the f32's top's 16's: the widening's lossless's.
      const uint32_t bits = static_cast<uint32_t>(row[static_cast<size_t>(i)]) << 16;
      std::memcpy(&out[static_cast<size_t>(i)], &bits, sizeof(float));
    }
    if (std::fwrite(out.data(), sizeof(float), out.size(), f) != out.size())
      DGPP_LOG_WARN("dsv4 layer dump: short write of {}", path);
    std::fclose(f);
  }

  // A generic named f32 widening dump (the same bf16->f32 form as above):
  // for the sub-step bisection (e.g. the MoE site's output vs the reference's).
  void write_named(const char* name, const uint16_t* src_bf16, int n, cudaStream_t stream) {
    if (!active_) return;
    std::vector<uint16_t> row(static_cast<size_t>(n));
    DGPP_CUDA_OK(cudaMemcpyAsync(row.data(), src_bf16, static_cast<size_t>(n) * 2, cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    const std::string path = pass_dir_ + "/" + name + ".f32";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) { DGPP_LOG_WARN("dsv4 layer dump: cannot write {}", path); return; }
    std::vector<float> out(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      const uint32_t bits = static_cast<uint32_t>(row[static_cast<size_t>(i)]) << 16;
      std::memcpy(&out[static_cast<size_t>(i)], &bits, sizeof(float));
    }
    if (std::fwrite(out.data(), sizeof(float), out.size(), f) != out.size())
      DGPP_LOG_WARN("dsv4 layer dump: short write of {}", path);
    std::fclose(f);
  }

  // The RAW hc_mult-stream state of the pass's last row (the same row
  // write_layer collapses), widened to f32: `state_bf16` is
  // [hc_mult, hidden] contiguous (the model's cur_ row layout). This is the
  // quantity comparable with the reference's own `layer_NN_hc.f32` — the
  // collapsed file above carries the layer's WEIGHTED pre combination and
  // the norm, so the two sides are not the same vector.
  void write_layer_hc(int layer, const uint16_t* state_bf16, int hc, cudaStream_t stream) {
    if (!active_) return;
    const size_t n = static_cast<size_t>(hc) * static_cast<size_t>(hidden_);
    std::vector<uint16_t> row(n);
    DGPP_CUDA_OK(cudaMemcpyAsync(row.data(), state_bf16, n * 2, cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    const std::string path = pass_dir_ + "/layer_" + pad2(layer) + "_hc.f32";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
      DGPP_LOG_WARN("dsv4 layer dump: cannot write {} — the layer's file is missing", path);
      return;
    }
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
      const uint32_t bits = static_cast<uint32_t>(row[i]) << 16;
      std::memcpy(&out[i], &bits, sizeof(float));
    }
    if (std::fwrite(out.data(), sizeof(float), out.size(), f) != out.size())
      DGPP_LOG_WARN("dsv4 layer dump: short write of {}", path);
    std::fclose(f);
  }

  // After the lm head: `logits_row` is the pass's last row's final-logit
  // slice on the device (this rank's [lm_vocab_count] f32; the sharded
  // head's slice's). Writes the top-20's (the absolute's token id's,
  // logit)'s to logits_top.txt.
  void write_logits_top(const float* logits_row, cudaStream_t stream) {
    if (!active_) return;
    std::vector<float> row(static_cast<size_t>(lm_vocab_count_));
    DGPP_CUDA_OK(cudaMemcpyAsync(row.data(), logits_row, static_cast<size_t>(lm_vocab_count_) * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    // The top-20 by the logit's (descending's), the id's the absolute's.
    std::vector<int> idx(static_cast<size_t>(lm_vocab_count_));
    for (int i = 0; i < lm_vocab_count_; ++i) idx[static_cast<size_t>(i)] = i;
    const int n = std::min(20, lm_vocab_count_);
    std::partial_sort(idx.begin(), idx.begin() + n, idx.end(),
                      [&](int a, int b) { return row[static_cast<size_t>(a)] > row[static_cast<size_t>(b)]; });
    const std::string path = pass_dir_ + "/logits_top.txt";
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
      DGPP_LOG_WARN("dsv4 layer dump: cannot write {} — the logits' file is missing", path);
      return;
    }
    for (int i = 0; i < n; ++i)
      std::fprintf(f, "%d %.6f\n", lm_vocab_begin_ + idx[static_cast<size_t>(i)],
                   row[static_cast<size_t>(idx[static_cast<size_t>(i)])]);
    std::fclose(f);
  }

  // meta.txt: the pass's prompt ids, the geometry, the rank, and the
  // layer's file's mapping.
  void write_meta() {
    if (!active_) return;
    const std::string path = pass_dir_ + "/meta.txt";
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) {
      DGPP_LOG_WARN("dsv4 layer dump: cannot write {} — the meta's file is missing", path);
      return;
    }
    std::fprintf(f, "prompt_tokens (%zu, pos0 %ld): ", pass_tokens_.size(), static_cast<long>(pass_pos0_));
    for (const int64_t id : pass_tokens_) std::fprintf(f, "%lld ", static_cast<long long>(id));
    std::fprintf(f, "\n");
    std::fprintf(f, "hidden_size: %d\n", hidden_);
    std::fprintf(f, "num_hidden_layers: %d\n", layers_);
    std::fprintf(f, "world: %d\n", world_);
    std::fprintf(f, "rank: %d\n", rank_);
    std::fprintf(f, "lm_vocab_begin: %d\n", lm_vocab_begin_);
    std::fprintf(f, "lm_vocab_count: %d\n", lm_vocab_count_);
    std::fprintf(
        f, "layer_mapping: layer_<NN>.f32 = the mHC collapse of the residual streams after layer NN (0-based) "
           "completes, the pass's last row (the last prompt position), the bf16's widened to little-endian f32 "
           "(hidden_size floats)\n");
    std::fprintf(f,
                 "logits_top.txt: the top-20 (absolute token id, logit) of this rank's final-logit slice for the "
                 "last prompt position (the sharded head's slice's; merge across the ranks' for the full vocab's)\n");
    std::fclose(f);
  }

 private:
  static std::string pad2(int v) {
    char b[8];
    std::snprintf(b, sizeof(b), "%02d", v);
    return b;
  }
  static std::string pad4(int v) {
    char b[8];
    std::snprintf(b, sizeof(b), "%04d", v);
    return b;
  }

  bool active_ = false;
  std::string dir_;
  int rank_ = 0;
  int world_ = 1;
  int hidden_ = 0;
  int layers_ = 0;
  int lm_vocab_begin_ = 0;
  int lm_vocab_count_ = 0;
  int pass_ = 0;
  std::string pass_dir_;
  int64_t pass_pos0_ = 0;
  std::vector<int64_t> pass_tokens_;
};

}  // namespace dgpp
