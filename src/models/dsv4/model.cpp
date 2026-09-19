#include "models/dsv4/model.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "common/capture_trace.hpp"
#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "kernels/csa2.hpp"
#include "kernels/glm_norm.hpp"
#include "kernels/glm_spec.hpp"
#include "kernels/kernels.hpp"
#include "kernels/scale_gemm.hpp"
#include "models/dsv4/compress.hpp"

namespace dgpp {
using session_detail::dev_alloc;
using session_detail::pinned_alloc;

namespace {
// The 0731's Compressor's per-request tail's state geometry (the
// checkpoint's kv_state / score_state's (coff * ratio, coff * head_dim)'s
// the 0731's the reference's form's the G-c128a-compressor's the closed's):
// the coff's the 1 + (ratio == 4)'s (the reference's 304's), the state's
// row's the 2 x coff * ratio x (coff * kCsa2Latent)'s the fp32's (the
// kv's plane's + the score's, the C4A's 8 x 1024's the 2 overlapping's
// windows' the C128A's 128 x 512's the 128-token's ring's). Shared by
// the constructor's allocation's + the snapshot's bytes's formula's.
size_t dsv4_tail_state_floats(int ratio) {
  const int coff = (ratio == 4) ? 2 : 1;
  return static_cast<size_t>(2) * static_cast<size_t>(coff * ratio) * static_cast<size_t>(coff * kCsa2Latent);
}
// The compressor's output width W (the wkv / wgate's width's, the C4A's
// kCsa2TailW's 1024's the C128A's kCsa2Latent's 512's).
int dsv4_tail_w(int ratio) { return ((ratio == 4) ? 2 : 1) * kCsa2Latent; }
}  // namespace

// ---------------------------------------------------------------------------
// The layer surfaces' configs and scratch formulas (the memory plan's and
// the constructor's share them).
// ---------------------------------------------------------------------------
Dsv4Csa2Config Dsv4Model::csa2_config(const Dsv4TextConfig& cfg, int tp_world) {
  Dsv4Csa2Config c;
  c.hidden = cfg.hidden_size;
  c.q_lora = cfg.q_lora_rank;
  c.o_lora = cfg.o_lora_rank;
  c.num_heads = cfg.num_attention_heads;
  c.o_groups = cfg.o_groups;
  c.index_heads = cfg.index_n_heads;
  c.index_topk = cfg.index_topk;
  c.window = cfg.sliding_window;
  c.ring_slots = std::max(160, cfg.sliding_window + 32);
  c.block_tokens = kBlockTokens;
  c.eps = cfg.rms_norm_eps;
  c.tp = tp_world;
  Dsv4Csa2Config::validate(c);
  return c;
}

Dsv4HashConfig Dsv4Model::hash_config(const Dsv4TextConfig& cfg, int tp_world) {
  Dsv4HashConfig c;
  c.hidden = cfg.hidden_size;
  c.inter = cfg.moe_intermediate_size;
  c.n_experts = cfg.n_routed_experts;
  c.top_k = cfg.num_experts_per_tok;
  c.num_hash_layers = cfg.num_hash_layers;
  c.vocab_size = cfg.vocab_size;
  c.routed_scaling_factor = cfg.routed_scaling_factor;
  c.norm_topk_prob = cfg.norm_topk_prob;
  c.swiglu_limit = cfg.swiglu_limit;
  c.tp = tp_world;
  Dsv4HashConfig::validate(c);
  return c;
}

Dsv4DsparkConfig Dsv4Model::dspark_config(const Dsv4TextConfig& cfg, int targets, int lm_vocab_begin,
                                          int lm_vocab_count) {
  Dsv4DsparkConfig c;
  c.hidden = cfg.hidden_size;
  c.num_targets = targets;
  c.target_layer_ids = cfg.dspark_target_layer_ids;
  c.markov_rank = cfg.dspark_markov_rank;
  c.block_size = cfg.dspark_block_size;
  c.noise_token_id = static_cast<int>(cfg.dspark_noise_token_id);
  c.lm_vocab_begin = lm_vocab_begin;
  c.lm_vocab_count = lm_vocab_count;
  c.window = cfg.sliding_window;
  Dsv4DsparkConfig::validate(c);
  return c;
}

size_t Dsv4Model::hash_scratch_bytes(const Dsv4TextConfig& cfg, int tp_world, int max_tokens) {
  // Mirrors Dsv4HashLayer::layout: the fused router's per-row top-k staging
  // (ids + weights) and the MXFP4 expert's gate / up / down planes.
  const int64_t inter = cfg.moe_intermediate_size / tp_world;
  const size_t T = static_cast<size_t>(max_tokens);
  const size_t K = static_cast<size_t>(cfg.num_experts_per_tok);
  size_t off = 0;
  const auto alloc = [&](size_t bytes) {
    off = (off + 255) / 256 * 256;
    off += std::max<size_t>(bytes, 16);
  };
  alloc(T * K * 4);  // topk_ids
  alloc(T * K * 4);  // topk_w
  alloc(T * static_cast<size_t>(inter) * 2);  // gate
  alloc(T * static_cast<size_t>(inter) * 2);  // up
  alloc(T * static_cast<size_t>(cfg.hidden_size) * 2);  // down
  return (off + 255) / 256 * 256;
}

size_t Dsv4Model::dspark_scratch_bytes(const Dsv4TextConfig& cfg, int lm_vocab_count, int max_rows) {
  // Mirrors Dsv4DsparkLayer::layout: the stream mean's rows, the block rows'
  // (pos / tok / req / spans), the block's head rows (base logits), the
  // confidence logits.
  const size_t R = static_cast<size_t>(std::max(1, max_rows));
  const size_t B = static_cast<size_t>(cfg.dspark_block_size);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  size_t off = 0;
  const auto alloc = [&](size_t bytes) {
    off = (off + 255) / 256 * 256;
    off += std::max<size_t>(bytes, 16);
  };
  alloc(R * H * 2);  // stream_mean
  alloc(R * B * 8);  // blk_pos
  alloc(R * B * 8);  // blk_tok
  alloc(R * B * 4);  // blk_req
  alloc((R * B + 1) * 4);  // blk_spans
  alloc(R * static_cast<size_t>(lm_vocab_count) * 4);  // base_logits
  alloc(R * 4);  // conf
  return (off + 255) / 256 * 256;
}

size_t Dsv4Model::union_attn_out_bytes(int max_rows) {
  // The DSpark union attention's out latent's (the 2026-09-20's dsv4 S3's
  // union-attn's q-latent's + the block-kv's wiring's): the [max_rows, 64,
  // 512]'s the bf16's (the Dsv4DsparkConfig's kHeads's x kHeadDim's the
  // DSpark's 64-head's the 512-dim's output's). The q latent's + the block
  // kv's the csa2 projection's real output's (the csa2 layer's q_latent's /
  // the block_kv's the getter's), so the model's staging's only the out
  // latent's (the 2026-09-18's stand-in's scratch's the q / the block's /
  // the out's three's regions's the single's out region's the replaced's).
  const size_t R = static_cast<size_t>(std::max(1, max_rows));
  return R * static_cast<size_t>(Dsv4DsparkConfig::kHeads) * static_cast<size_t>(Dsv4DsparkConfig::kHeadDim) *
         sizeof(uint16_t);
}

// ---------------------------------------------------------------------------
// Construction.
// ---------------------------------------------------------------------------
Dsv4Model::Dsv4Model(const Dsv4TextConfig& cfg, const std::string& checkpoint_dir, int max_tokens,
                     int64_t max_cache_tokens, Dsv4Residency residency, BoundaryReducer* boundary, int tp_rank,
                     int tp_world, int max_requests, bool mtp, int decode_rows)
    : cfg_(cfg),
      loader_(cfg, checkpoint_dir, tp_rank, tp_world, residency,
              tp_world > 1 ? Dsv4HeadSharding::VocabSharded : Dsv4HeadSharding::Full,
              mtp && residency == Dsv4Residency::Resident),
      world_(tp_world) {
  if (max_tokens <= 0) throw std::invalid_argument("Dsv4Model: max_tokens must be positive");
  if (mtp) {
    if (cfg_.num_nextn_predict_layers < 1 || cfg_.dspark_target_layer_ids.empty())
      throw std::invalid_argument("Dsv4Model: mtp needs the DSpark draft stages and target layers");
    if (cfg_.dspark_block_size < 1 || cfg_.dspark_block_size > 8)
      throw std::invalid_argument("Dsv4Model: dspark_block_size must be in [1, max_block]");
    for (const int l : cfg_.dspark_target_layer_ids)
      if (l < 0 || l >= cfg_.num_hidden_layers)
        throw std::invalid_argument("Dsv4Model: a DSpark target layer is out of range");
    targets_ = static_cast<int>(cfg_.dspark_target_layer_ids.size());
  }
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("Dsv4Model: max_requests must be in [1, kPickMaxRequests]");
  if (max_requests > decode_rows_cap() || decode_rows > decode_rows_cap())
    throw std::invalid_argument("Dsv4Model: the decode batch is bounded at " + std::to_string(decode_rows_cap()) +
                                " rows: max_requests and decode_rows must not exceed it");
  if ((tp_world > 1) != (boundary != nullptr))
    throw std::invalid_argument("Dsv4Model: a boundary reducer is required exactly when tp_world > 1");
  init_stream();
  loader_.set_reader_stream(stream_);
  log_memory_ledger("dsv4: loader opened");
  const int H = cfg_.hidden_size;
  globals_ = loader_.load_globals();
  embed_sharded_ = globals_.embed_vocab_count < cfg_.vocab_size;
  if (embed_sharded_ && boundary == nullptr)
    throw std::invalid_argument("Dsv4Model: a vocab-sharded embedding needs the boundary reducer (world > 1)");
  if (residency == Dsv4Residency::Resident) {
    for (int l = 0; l < (mtp ? cfg_.max_layer() : cfg_.num_hidden_layers); ++l) (void)loader_.load_layer(l);
  }
  log_memory_ledger("dsv4: layers resident");
  {
    SessionParams sp;
    sp.max_tokens = max_tokens;
    sp.max_cache_tokens =
        ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
    sp.rank = tp_rank;
    sp.world = tp_world;
    sp.boundary = boundary;
    sp.max_requests = max_requests;
    sp.decode_rows = decode_rows;
    sp.mtp = mtp;
    sp.vocab_size = cfg_.vocab_size;
    sp.hidden = H;
    sp.lm_vocab_begin = globals_.lm_vocab_begin;
    sp.lm_vocab_count = globals_.lm_vocab_count > 0 ? globals_.lm_vocab_count : static_cast<int>(cfg_.vocab_size);
    sp.max_position_embeddings = cfg_.max_position_embeddings;
    sp.block_tokens = 0;  // no paged pool: the attention's state is positional
    sp.draft_width = mtp ? targets_ * H : H;  // the draft window holds [h_t1 | h_t2 | h_t3] per row
    sp.eos = static_cast<int32_t>(cfg_.eos_token_id);
    init_session(sp);
  }
  if (max_decode_rows_ > decode_rows_cap())
    throw std::logic_error("Dsv4Model: the session core widened the decode batch past the family's cap");
  gemm_ws_bytes_ =
      std::max<size_t>(64u << 20, gemm_.query_workspace_bytes(max_tokens_, lm_vocab_count_, H, DType::BF16));
  gemm_ws_ = dev_alloc<char>(gemm_ws_bytes_);
  gemm_.set_decode_rows(max_decode_rows_);
  mhc_cfg_.hc_mult = cfg_.hc_mult;
  mhc_cfg_.hidden = H;
  mhc_cfg_.sinkhorn_iters = cfg_.hc_sinkhorn_iters;
  mhc_cfg_.hc_eps = cfg_.hc_eps;
  mhc_cfg_.norm_eps = cfg_.rms_norm_eps;
  GlmMhcConfig::validate_config(mhc_cfg_);
  csa2_cfg_ = csa2_config(cfg_, tp_world);
  hash_cfg_ = hash_config(cfg_, tp_world);
  // The decode form of every dense site: the streaming tensor-core GEMM
  // unless the environment asks for the GEMV chunks (an A/B switch).
  dense_mma_ = std::getenv("DGPP_DSV4_DENSE_GEMV") == nullptr;
  if (boundary_) boundary_->bind_stream(stream_);  // the stream-ordered reducer's stream
  csa2_cfg_.dense_mma = dense_mma_;
  gemm_.set_decode_mma(dense_mma_);
  // The mHC dots take the tiled form at every prefill row count: a row's
  // collapse coefficients are then one chain whatever rows share the launch.
  mhc_set_tile_min_tokens(1);
  // The layer -> cache / tail ordinals.
  cache_ord_.assign(static_cast<size_t>(cfg_.max_layer()), -1);
  tail_ord_.assign(static_cast<size_t>(cfg_.max_layer()), -1);
  cache_ratio_.clear();
  {
    int c = 0, t = 0;
    for (int l = 0; l < cfg_.max_layer(); ++l) {
      const int ratio = cfg_.compress_ratio(l);
      if (ratio > 0) {
        cache_ord_[static_cast<size_t>(l)] = c;
        cache_ratio_.push_back(ratio);
        ++c;
        // The 0731's Compressor's per-request state (the reference's form's,
        // the G-c128a-compressor's the closed's): every compressing layer
        // (the C4A's ratio-4's + the C128A's ratio-128's) owns its tail's
        // (the checkpoint's kv_state / score_state's), not only the C4A's.
        tail_ord_[static_cast<size_t>(l)] = t++;
        tail_ratio_.push_back(ratio);
      }
    }
    tails_ = t;
  }
  tail_off_.assign(static_cast<size_t>(tails_), 0);
  spec_off_.assign(static_cast<size_t>(tails_), 0);
  // The planar main + index caches (2026-09-19, the dsv4 seam S1b: the no-
  // pool positional state, the dsv41 Csa2StatePool's planes' raw-pointer
  // re-expression): per cache ordinal (kv source), the kFp4Block main
  // cache (288 B/row, self-describing) + the planar index cache (e4m3
  // [entries, 128] + fp32 row-scale [entries]). The size's Dsv4Csa2Layer's
  // static's (the per-ordinal ratio's the cache_ratio_'s, the
  // max_cache_tokens_'s the rounded's the entry's count's the basis's, the
  // kBlockTokens' the 128-token block's). The zero's the cold start's
  // (the entry's the 0's the scale's the 0's the attention's the empty's
  // the -inf's the no-op's). The C4A's publish's the csa2 layer's
  // publish_entries's (the dsv41's re-expression's), the C128A's the
  // separate item's (the C128A compressor's the G-c128a-compressor's gap's).
  if (!cache_ratio_.empty()) {
    const int64_t ct = max_cache_tokens_;  // the rounded cache capacity (a multiple of kBlockTokens)
    const int n = static_cast<int>(cache_ratio_.size());
    main_cache_.assign(static_cast<size_t>(n), nullptr);
    index_cache_.assign(static_cast<size_t>(n), nullptr);
    index_scale_.assign(static_cast<size_t>(n), nullptr);
    main_cache_bytes_.assign(static_cast<size_t>(n), 0);
    index_cache_bytes_.assign(static_cast<size_t>(n), 0);
    index_scale_bytes_.assign(static_cast<size_t>(n), 0);
    for (int o = 0; o < n; ++o) {
      const int r = cache_ratio_[static_cast<size_t>(o)];
      main_cache_bytes_[static_cast<size_t>(o)] = Dsv4Csa2Layer::main_cache_bytes(r, ct, kBlockTokens);
      index_cache_bytes_[static_cast<size_t>(o)] = Dsv4Csa2Layer::index_cache_bytes(r, ct, kBlockTokens);
      index_scale_bytes_[static_cast<size_t>(o)] = Dsv4Csa2Layer::index_scale_bytes(r, ct, kBlockTokens);
      main_cache_[static_cast<size_t>(o)] =
          reinterpret_cast<uint8_t*>(dev_alloc<char>(main_cache_bytes_[static_cast<size_t>(o)]));
      index_cache_[static_cast<size_t>(o)] =
          reinterpret_cast<uint8_t*>(dev_alloc<char>(index_cache_bytes_[static_cast<size_t>(o)]));
      index_scale_[static_cast<size_t>(o)] =
          reinterpret_cast<float*>(dev_alloc<char>(index_scale_bytes_[static_cast<size_t>(o)]));
      // The zero's the cold start's (the entry's the 0's the scale's the
      // 0's the attention's the empty's the -inf's the no-op's).
      DGPP_CUDA_OK(cudaMemsetAsync(main_cache_[static_cast<size_t>(o)], 0, main_cache_bytes_[static_cast<size_t>(o)],
                                   stream_));
      DGPP_CUDA_OK(cudaMemsetAsync(index_cache_[static_cast<size_t>(o)], 0, index_cache_bytes_[static_cast<size_t>(o)],
                                   stream_));
      DGPP_CUDA_OK(cudaMemsetAsync(index_scale_[static_cast<size_t>(o)], 0, index_scale_bytes_[static_cast<size_t>(o)],
                                   stream_));
    }
    log_memory_ledger("dsv4: the planar main + index caches (the no-pool positional state)");
  }
  // The layer surfaces' scratch (the caller's contract: the model allocates
  // it, the layers carve it).
  csa2_scratch_bytes_ =
      Dsv4Csa2Layer::scratch_bytes(csa2_cfg_, max_tokens_, max_cache_tokens_,
                                  std::min(kDecodeRowsCap, max_tokens_), kDecodeSplit, kDotBudget);
  csa2_scratch_ = dev_alloc<char>(csa2_scratch_bytes_);
  // The 0731's Compressor's per-request tails (the 2026-09-18's dsv4 dspark
  // + compressor wiring's the 2026-09-21's the reference's form's, the
  // G-c128a-compressor's + the G-tail-cadence / G-tail-pool / G-tail-ape's
  // the closed's; docs/dsv4_attention_spec.md §5): per compressing layer
  // (the C4A's + the C128A's), the checkpoint's kv_state / score_state's
  // (b, coff * ratio, coff * head_dim)'s the fp32's (the kv's plane's the
  // zero's + the score's the -inf's the reference's 309-310's the cold
  // start's the publish's the 0's weight's the -inf's rows's) + the spec
  // rows' tails (the rollback's the table's the spec_rows_'s).
  if (tails_ > 0) {
    size_t off = 0, off_spec = 0;
    for (int t = 0; t < tails_; ++t) {
      const size_t state_row = dsv4_tail_state_floats(tail_ratio_[static_cast<size_t>(t)]);
      tail_off_[static_cast<size_t>(t)] = off;
      off += static_cast<size_t>(max_requests) * state_row;
      spec_off_[static_cast<size_t>(t)] = off_spec;
      off_spec += static_cast<size_t>(max_decode_rows_) * state_row;
    }
    d_tails_ = dev_alloc<float>(off);
    spec_tails_ = dev_alloc<float>(off_spec);
    // The reference's init (the kv_state's zero's + the score_state's the
    // -inf's, the 309-310's): one block's a request's (the live's + the
    // spec rows's the same's).
    for (int t = 0; t < tails_; ++t) {
      const int64_t n = static_cast<int64_t>(dsv4_tail_state_floats(tail_ratio_[static_cast<size_t>(t)])) / 2;
      dsv4_compress_tail_init(d_tails_ + tail_off_[static_cast<size_t>(t)], max_requests, n, n, stream_);
      dsv4_compress_tail_init(spec_tails_ + spec_off_[static_cast<size_t>(t)], max_decode_rows_, n, n, stream_);
    }
    log_memory_ledger("dsv4: the compressor's per-request tails (the reference's form's the C4A's 8 x 1024's + the C128A's 128 x 512's)");
  }
  hash_scratch_bytes_ = hash_scratch_bytes(cfg_, tp_world, max_tokens_);
  hash_scratch_ = dev_alloc<char>(hash_scratch_bytes_);
  log_memory_ledger("dsv4: scratch");
  // The rotary tables: the window's (theta, no YaRN) and the compressed
  // layers' (compress theta with YaRN).
  {
    std::vector<float> win(32), comp(32);
    csa2_rope_inv_freq_host(cfg_.qk_rope_head_dim, cfg_.rope_theta, 0, cfg_.rope_factor, cfg_.beta_fast, cfg_.beta_slow,
                            win.data());
    csa2_rope_inv_freq_host(cfg_.qk_rope_head_dim, cfg_.compress_rope_theta, cfg_.original_max_position_embeddings,
                            cfg_.rope_factor, cfg_.beta_fast, cfg_.beta_slow, comp.data());
    inv_freq_window_ = dev_alloc<float>(32);
    inv_freq_compressed_ = dev_alloc<float>(32);
    DGPP_CUDA_OK(cudaMemcpy(inv_freq_window_, win.data(), 32 * 4, cudaMemcpyHostToDevice));
    DGPP_CUDA_OK(cudaMemcpy(inv_freq_compressed_, comp.data(), 32 * 4, cudaMemcpyHostToDevice));
  }
  // The layer objects (the views rebind per layer in the walk).
  csa2_ = std::make_unique<Dsv4Csa2Layer>(gemm_, csa2_cfg_, max_tokens_, max_cache_tokens_, csa2_scratch_,
                                          csa2_scratch_bytes_, gemm_ws_, gemm_ws_bytes_,
                                          std::min(kDecodeRowsCap, max_tokens_), kDecodeSplit, kDotBudget);
  moe_ = std::make_unique<Dsv4HashLayer>(gemm_, hash_cfg_, max_tokens_, hash_scratch_, hash_scratch_bytes_,
                                         gemm_ws_, gemm_ws_bytes_);
  if (mtp) {
    dspark_cfg_ = dspark_config(cfg_, targets_, lm_vocab_begin_, lm_vocab_count_);
    dspark_scratch_bytes_ = dspark_scratch_bytes(cfg_, lm_vocab_count_, max_decode_rows_);
    dspark_scratch_ = dev_alloc<char>(dspark_scratch_bytes_);
    // The DSpark union attention's staging (2026-09-20, the dsv4 S3's
    // union-attn's q-latent's + the block-kv's wiring's): two's the
    // [max_decode_rows, 64, 512]'s the bf16's regions' — region 0's the q
    // fallback's (the zeroed's stand-in's the tp>1's the head count's the
    // DSpark's replicated's 64-head's vs. the csa2's sharded's local's
    // delta's the no-OOB's safe's interim's) + region 1's the out latent's
    // (the DSpark's 64-head's the 512-dim's output's). The q latent's + the
    // block kv's the csa2 projection's real output's (the csa2 layer's
    // q_latent's / the block_kv's the getter's the 2026-09-18's stand-in's
    // scratch's the csa2 seam's the fill's the removed's — the tp=1's the
    // real q's the csa2 q's the 64's head's the match's, the tp>1's the
    // head count's delta's the zeroed's fallback's the interim's).
    union_attn_scratch_bytes_ = 2 * union_attn_out_bytes(max_decode_rows_);
    union_attn_scratch_ = dev_alloc<char>(union_attn_scratch_bytes_);
    // The q fallback's (region 0's) the zeroed's (the out region's region 1's
    // the kernel's writes's it's, no zero's needed's).
    DGPP_CUDA_OK(cudaMemsetAsync(union_attn_scratch_, 0, union_attn_out_bytes(max_decode_rows_), stream_));
    dspark_ = std::make_unique<Dsv4DsparkLayer>(gemm_, dspark_cfg_, max_decode_rows_, dspark_scratch_,
                                                 dspark_scratch_bytes_, gemm_ws_, gemm_ws_bytes_);
  }
  // The hash gate tables' device copies (the num_hash_layers' tid2eid's
  // whole tables; the router kernel reads the full table by the token id).
  if (cfg_.num_hash_layers > 0) {
    const Dsv4HashTables& tables = loader_.load_hash_tables();
    d_tid2eid_.resize(static_cast<size_t>(cfg_.num_hash_layers));
    for (int l = 0; l < cfg_.num_hash_layers; ++l) {
      const size_t n = static_cast<size_t>(cfg_.vocab_size) * static_cast<size_t>(cfg_.num_experts_per_tok);
      d_tid2eid_[static_cast<size_t>(l)] = dev_alloc<int64_t>(n);
      std::vector<int32_t> ids(static_cast<size_t>(cfg_.vocab_size));
      for (int i = 0; i < cfg_.vocab_size; ++i) ids[static_cast<size_t>(i)] = i;
      std::vector<int64_t> staging(n);
      tables.tables[static_cast<size_t>(l)]->gather(ids.data(), cfg_.vocab_size, staging.data());
      DGPP_CUDA_OK(cudaMemcpyAsync(d_tid2eid_[static_cast<size_t>(l)], staging.data(), n * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, stream_));
    }
    log_memory_ledger("dsv4: hash tables on the device");
  }
  // The hash tables are mmap'd from the checkpoint shards, so the sources
  // are held until their rows are gathered onto the device (the loader's
  // contract: load_hash_tables() precedes release_sources()).
  if (residency == Dsv4Residency::Resident) {
    loader_.release_sources();
    log_memory_ledger("dsv4: sources released");
  }
  // The DSpark stage's view (the main projection's the stage 0's, the head
  // tensors' the last stage's — the layer's rebind's both's). Resident's the
  // draft stages' the stable's (mtp implies the resident stack).
  if (mtp) {
    const Dsv4LayerResident& r0 = loader_.load_layer(cfg_.num_hidden_layers);
    const Dsv4LayerResident& rl =
        loader_.load_layer(cfg_.num_hidden_layers + cfg_.num_draft_stages() - 1);
    dspark_w_ = dspark_view(r0, rl, cfg_.num_draft_stages() - 1);
    dspark_->rebind(dspark_w_, cfg_.num_draft_stages() - 1);
  }
  // The activation buffers.
  const size_t M = static_cast<size_t>(max_tokens_);
  const size_t R = static_cast<size_t>(max_requests_);
  streams_a_ = dev_alloc<uint16_t>(M * 4 * H);
  streams_b_ = dev_alloc<uint16_t>(M * 4 * H);
  x_ = dev_alloc<uint16_t>(M * H);
  y_ = dev_alloc<uint16_t>(M * H);
  collapsed_ = dev_alloc<uint16_t>(M * H);
  post_bf16_ = dev_alloc<uint16_t>(M * 4);
  comb_bf16_ = dev_alloc<uint16_t>(M * 16);
  post_f32_ = dev_alloc<float>(M * 4);
  comb_f32_ = dev_alloc<float>(M * 16);
  pre_a_ = dev_alloc<float>(M * 4);
  pre_b_ = dev_alloc<float>(M * 4);
  one_hot_ = dev_alloc<float>(M * 4);
  mhc_logits_ = dev_alloc<float>(M * static_cast<size_t>(mhc_cfg_.coeff_rows()));
  gate_logits_ = dev_alloc<uint16_t>(M * static_cast<size_t>(cfg_.n_routed_experts));
  topk_ids_ = dev_alloc<int32_t>(M * static_cast<size_t>(cfg_.num_experts_per_tok));
  topk_w_ = dev_alloc<float>(M * static_cast<size_t>(cfg_.num_experts_per_tok));
  {
    std::vector<float> oh(M * 4, 0.0f);
    for (size_t t = 0; t < M; ++t) oh[t * 4] = 1.0f;
    DGPP_CUDA_OK(cudaMemcpy(one_hot_, oh.data(), oh.size() * 4, cudaMemcpyHostToDevice));
  }
  if (mtp) {
    const size_t W = static_cast<size_t>(targets_) * H;
    const size_t D = static_cast<size_t>(max_decode_rows_);
    const size_t block = static_cast<size_t>(cfg_.dspark_block_size);
    main_hidden_ = dev_alloc<uint16_t>(M * W);
    main_gather_ = dev_alloc<uint16_t>(D * W);
    main_x_ = dev_alloc<uint16_t>(std::max(M, D) * H);
    blk_pos_ = dev_alloc<int64_t>(D);
    blk_tok_ = dev_alloc<int64_t>(D);
    blk_req_ = dev_alloc<int32_t>(D);
    blk_spans_ = dev_alloc<int32_t>(D + 1);
    base_logits_ = dev_alloc<float>(D * static_cast<size_t>(lm_vocab_count_));
    conf_ = dev_alloc<float>(R * block);
    d_draft_pos_ = dev_alloc<int64_t>(M);
    d_draft_req_ = dev_alloc<int32_t>(M);
    DGPP_CUDA_OK(cudaMemsetAsync(conf_, 0, R * block * 4, stream_));
  }
  DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  log_memory_ledger("dsv4: model ready");
}

Dsv4Model::~Dsv4Model() {
  cudaStreamSynchronize(stream_);
  csa2_.reset();
  moe_.reset();
  dspark_.reset();
  cudaFree(gemm_ws_);
  cudaFree(csa2_scratch_);
  cudaFree(hash_scratch_);
  cudaFree(dspark_scratch_);
  cudaFree(union_attn_scratch_);
  cudaFree(d_tails_);
  cudaFree(spec_tails_);
  for (uint8_t* p : main_cache_) cudaFree(p);
  for (uint8_t* p : index_cache_) cudaFree(p);
  for (float* p : index_scale_) cudaFree(p);
  cudaFree(inv_freq_window_);
  cudaFree(inv_freq_compressed_);
  cudaFree(streams_a_);
  cudaFree(streams_b_);
  cudaFree(x_);
  cudaFree(y_);
  cudaFree(collapsed_);
  cudaFree(post_bf16_);
  cudaFree(comb_bf16_);
  cudaFree(post_f32_);
  cudaFree(comb_f32_);
  cudaFree(pre_a_);
  cudaFree(pre_b_);
  cudaFree(one_hot_);
  cudaFree(mhc_logits_);
  cudaFree(gate_logits_);
  cudaFree(topk_ids_);
  cudaFree(topk_w_);
  for (void* t : d_tid2eid_) cudaFree(t);
  cudaFree(main_hidden_);
  cudaFree(main_gather_);
  cudaFree(main_x_);
  cudaFree(blk_pos_);
  cudaFree(blk_tok_);
  cudaFree(blk_req_);
  cudaFree(blk_spans_);
  cudaFree(base_logits_);
  cudaFree(conf_);
  cudaFree(d_draft_pos_);
  cudaFree(d_draft_req_);
}

// ---------------------------------------------------------------------------
// The memory plan.
// ---------------------------------------------------------------------------
Dsv4Model::MemoryPlan Dsv4Model::plan_memory(const Dsv4TextConfig& cfg, int max_tokens, int64_t max_cache_tokens,
                                             int tp_rank, int tp_world, Dsv4Residency residency, int max_requests,
                                             bool mtp, int decode_rows) {
  if (max_tokens <= 0) throw std::invalid_argument("plan_memory: max_tokens must be positive");
  if (mtp && (cfg.num_nextn_predict_layers < 1 || cfg.dspark_target_layer_ids.empty()))
    throw std::invalid_argument("plan_memory: mtp needs the DSpark draft stages and target layers");
  if (max_requests <= 0 || max_requests > kPickMaxRequests)
    throw std::invalid_argument("plan_memory: max_requests must be in [1, kPickMaxRequests]");
  if (max_requests > decode_rows_cap() || decode_rows > decode_rows_cap())
    throw std::invalid_argument("plan_memory: the decode batch is bounded at " + std::to_string(decode_rows_cap()) +
                                " rows");
  const int rows = std::max({kDecodeRows, decode_rows, max_requests});
  const Dsv4HeadSharding head = tp_world > 1 ? Dsv4HeadSharding::VocabSharded : Dsv4HeadSharding::Full;
  const Dsv4LocalGeometry geo = Dsv4LocalGeometry::from_config(cfg, tp_rank, tp_world, head);
  const int64_t cache_tokens =
      ((std::max<int64_t>(max_cache_tokens, max_tokens) + kBlockTokens - 1) / kBlockTokens) * kBlockTokens;
  MemoryPlan plan;
  plan.context_tokens = std::min<int64_t>(cache_tokens, cfg.max_position_embeddings);
  const size_t M = static_cast<size_t>(max_tokens);
  const size_t R = static_cast<size_t>(max_requests);
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t V = static_cast<size_t>(geo.lm_vocab_count);
  size_t staging = 0;
  if (residency == Dsv4Residency::Resident) {
    plan.add("model weights (resident)", Dsv4LayerStream::resident_bytes(cfg, tp_rank, tp_world, head, mtp));
    staging = Dsv4LayerStream::staging_plan_bytes(cfg, tp_rank, tp_world, head, mtp);
  } else {
    size_t largest = 0;
    for (int l = 0; l < (mtp ? cfg.max_layer() : cfg.num_hidden_layers); ++l)
      largest = std::max(largest, Dsv4LayerStream::layer_bytes(cfg, l, tp_rank, tp_world));
    plan.add("model weights (one streamed layer + globals)",
             largest + Dsv4LayerStream::globals_bytes(cfg, tp_rank, tp_world, head));
  }
  const size_t after_weights = plan.total_bytes();
  plan.add("gemm workspace (at least)", size_t{64} << 20);
  const Dsv4Csa2Config c2 = csa2_config(cfg, tp_world);
  plan.add("csa2 scratch (projections, selection, attention)",
           Dsv4Csa2Layer::scratch_bytes(c2, max_tokens, cache_tokens, std::min(kDecodeRowsCap, max_tokens),
                                        kDecodeSplit, kDotBudget));
  // The planar main + index caches (2026-09-19, the dsv4 seam S1b: the no-pool
  // positional state, the dsv41 Csa2StatePool's planes' raw-pointer re-
  // expression): per cache ordinal (kv source), the kFp4Block main cache
  // (288 B/row) + the planar index cache (e4m3 [entries, 128] + fp32 row-
  // scale), sized from the config's cache_tokens + the per-ordinal ratio's
  // (the Dsv4Csa2Layer's static's, the model's allocation's the same
  // formula's).
  {
    size_t cache_bytes = 0;
    int n_caches = 0;
    for (int l = 0; l < cfg.max_layer(); ++l) {
      const int r = cfg.compress_ratio(l);
      if (r > 0) {
        cache_bytes += Dsv4Csa2Layer::main_cache_bytes(r, cache_tokens, kBlockTokens);
        cache_bytes += Dsv4Csa2Layer::index_cache_bytes(r, cache_tokens, kBlockTokens);
        cache_bytes += Dsv4Csa2Layer::index_scale_bytes(r, cache_tokens, kBlockTokens);
        ++n_caches;
      }
    }
    if (n_caches > 0)
      plan.add("csa2 planar main + index caches (the no-pool positional state, the kFp4Block main's + the e4m3 index's)",
               cache_bytes);
  }
  plan.add("hash moe scratch (the fused router's top-k staging, the MXFP4 expert's planes)",
           hash_scratch_bytes(cfg, tp_world, max_tokens));
  if (mtp)
    plan.add("dspark scratch (the stream mean's, the block rows', the base logits', the confidence's)",
             dspark_scratch_bytes(cfg, static_cast<int>(V), rows));
  if (mtp)
    plan.add("dspark union attention's staging (the q fallback's + the out latent's the [rows, 64, 512]'s the bf16's x2's)",
             2 * union_attn_out_bytes(rows));
  {
    size_t core_dev = 0, core_pin = 0;
    const size_t targets = cfg.dspark_target_layer_ids.size();
    const int draft_width = mtp ? static_cast<int>(targets) * cfg.hidden_size : cfg.hidden_size;
    session_core_plan_bytes(max_tokens, max_requests, cfg.hidden_size, static_cast<int>(V), mtp, draft_width,
                            &core_dev, &core_pin, rows);
    const size_t K = static_cast<size_t>(cfg.num_experts_per_tok);
    size_t act = core_dev + 2 * M * 4 * H * 2 + 4 * M * H * 2 + M * (4 + 16) * 2 + M * (4 + 16 + 4 + 4 + 4) * 4 +
                 M * static_cast<size_t>(cfg.hc_coeff_rows()) * 4;
    act += M * static_cast<size_t>(cfg.n_routed_experts) * 2 + M * K * 4 + M * K * 4 + M * V * 4;
    if (mtp) {
      const size_t D = static_cast<size_t>(rows);
      act += M * targets * H * 2 + D * targets * H * 2 + std::max(M, D) * H * 2 + D * (8 + 8 + 4 + 4) +
             D * V * 4 + R * static_cast<size_t>(cfg.dspark_block_size) * 4;
    }
    plan.add("activations (session core, the four residual streams, block io, mHC coefficients, router staging)",
             act, core_pin);
  }
  if (staging > 0) {
    const size_t later = plan.total_bytes() - after_weights;
    plan.add("loader staging beyond the caches that replace it (pinned host, transient)", 0,
             staging > later ? staging - later : 0);
  }
  return plan;
}

size_t Dsv4Model::session_snapshot_bytes(const Dsv4TextConfig& cfg, int, bool) {
  // The 0731's Compressor's per-request tails (the 2026-09-18's dsv4 dspark
  // + compressor wiring's the 2026-09-21's the reference's form's, the
  // G-c128a-compressor's + the G-tail-cadence / G-tail-pool / G-tail-ape's
  // the closed's; docs/dsv4_attention_spec.md §5): per compressing layer
  // (the C4A's + the C128A's), the checkpoint's kv_state / score_state's
  // (coff * ratio, coff * head_dim)'s the fp32's — the C4A's 8 x 1024's the
  // 2 overlapping's windows' + the C128A's 128 x 512's the 128-token's
  // ring's. The positional rings (the layer scratch's, the slot's the
  // position's) need no snapshot (a rejected draft's slot is never read by
  // a later query), so the tails are the only per-request state.
  size_t bytes = 0;
  for (int l = 0; l < cfg.num_hidden_layers; ++l) {
    const int ratio = cfg.compress_ratio(l);
    if (ratio > 0) bytes += dsv4_tail_state_floats(ratio) * sizeof(float);
  }
  return bytes;
}

size_t Dsv4Model::snapshot_state_bytes() const {
  size_t bytes = 0;
  for (int t = 0; t < tails_; ++t) bytes += dsv4_tail_state_floats(tail_ratio_[static_cast<size_t>(t)]) * sizeof(float);
  return bytes;
}

void Dsv4Model::reset_slot_state(int req) {
  if (d_tails_ == nullptr) return;  // no compressing layers: no per-request tails
  // The reference's init (the kv_state's zero's + the score_state's the
  // -inf's, the reference's 309-310's): per ordinal (the C4A's 8 x 1024's
  // the C128A's 128 x 512's), the request's state's the dsv4_compress_
  // tail_init's (the kernel's the graph's node's; the reset's the session's
  // core's the outside's capture's).
  for (int t = 0; t < tails_; ++t) {
    const size_t off = tail_off_[static_cast<size_t>(t)];
    const int64_t n = static_cast<int64_t>(dsv4_tail_state_floats(tail_ratio_[static_cast<size_t>(t)])) / 2;
    dsv4_compress_tail_init(d_tails_ + off + static_cast<size_t>(req) * 2 * n, 1, n, n, stream_);
  }
}

GlmSpecSegments Dsv4Model::spec_segments(int req, int snapshot_row0) const {
  GlmSpecSegments segs;
  if (d_tails_ == nullptr) return segs;  // no compressing layers: no per-request tails
  const auto add = [&](void* dst, const void* snapshots, size_t row_stride, size_t bytes) {
    if (segs.count >= kSpecMaxSegments) throw std::logic_error("spec_segments: too many state families");
    segs.seg[segs.count++] = GlmSpecSegment{dst, snapshots, row_stride, bytes};
  };
  // The per-ordinal's state's (the C4A's 8 x 1024's the 2 overlapping's
  // windows' the C128A's 128 x 512's the 128-token's ring's the 0731's
  // reference's form's the G-c128a-compressor's the closed's): the per-
  // ordinal's per_req's the state_row's (the 2 x coff * ratio x (coff * 512)'s
  // the fp32's), the tail_off_'s / the spec_off_'s the per-ordinal's offset's.
  for (int t = 0; t < tails_; ++t) {
    const size_t per_req = dsv4_tail_state_floats(tail_ratio_[static_cast<size_t>(t)]);
    add(d_tails_ + tail_off_[static_cast<size_t>(t)] + static_cast<size_t>(req) * per_req,
        spec_tails_ + spec_off_[static_cast<size_t>(t)] + static_cast<size_t>(snapshot_row0) * per_req,
        per_req * sizeof(float), per_req * sizeof(float));
  }
  return segs;
}

void Dsv4Model::write_state_snapshot(int req, uint8_t* d, int spec_row) {
  if (d_tails_ == nullptr) return;
  const bool live = spec_row < 0;
  const size_t row = live ? 0 : static_cast<size_t>(spec_row);
  // The per-ordinal's state's copy (the C4A's 8 x 1024's the C128A's 128 x
  // 512's the 0731's reference's form's the per-ordinal's per_req's the
  // state_row's the tail_off_'s / the spec_off_'s the offset's).
  for (int t = 0; t < tails_; ++t) {
    const size_t per_req = dsv4_tail_state_floats(tail_ratio_[static_cast<size_t>(t)]);
    const float* src = live ? d_tails_ + tail_off_[static_cast<size_t>(t)] + static_cast<size_t>(req) * per_req
                            : spec_tails_ + spec_off_[static_cast<size_t>(t)] + row * per_req;
    DGPP_CUDA_OK(cudaMemcpyAsync(d, src, per_req * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    capture_trace_copy("dsv4 write_state_snapshot tail", "D2D", src, d, per_req * sizeof(float), stream_);
    d += per_req * sizeof(float);
  }
}

void Dsv4Model::read_state_snapshot(int req, const uint8_t* s) {
  if (d_tails_ == nullptr) return;
  // The per-ordinal's state's copy (the C4A's 8 x 1024's the C128A's 128 x
  // 512's the 0731's reference's form's the per-ordinal's per_req's the
  // state_row's the tail_off_'s the offset's).
  for (int t = 0; t < tails_; ++t) {
    const size_t per_req = dsv4_tail_state_floats(tail_ratio_[static_cast<size_t>(t)]);
    DGPP_CUDA_OK(cudaMemcpyAsync(d_tails_ + tail_off_[static_cast<size_t>(t)] + static_cast<size_t>(req) * per_req, s,
                                 per_req * sizeof(float), cudaMemcpyDeviceToDevice, stream_));
    capture_trace_copy("dsv4 read_state_snapshot tail", "D2D", s, d_tails_ + tail_off_[static_cast<size_t>(t)] +
                                                            static_cast<size_t>(req) * per_req,
                       per_req * sizeof(float), stream_);
    s += per_req * sizeof(float);
  }
}

// ---------------------------------------------------------------------------
// Views and layer objects.
// ---------------------------------------------------------------------------
Dsv4Csa2LayerWeights Dsv4Model::csa2_view(const Dsv4LayerResident& r, int layer) const {
  const Dsv4AttnResident& a = r.attn;
  Dsv4Csa2LayerWeights w;
  w.wq_a = a.wq_a;
  w.wkv = a.wkv;
  w.wq_b = a.wq_b;
  w.wo_a = a.wo_a;
  w.wo_b = a.wo_b;
  w.q_norm = a.q_norm;
  w.kv_norm = a.kv_norm;
  w.attn_sink = a.attn_sink;
  w.idx_wq_b = a.idx_wq_b;
  w.idx_wp = a.idx_wp;
  w.comp_wkv = a.comp_wkv;
  w.comp_wgate = a.comp_wgate;
  w.comp_norm = a.comp_norm;
  w.comp_ape = a.comp_ape;  // the APE's (the f32 [ratio, W]'s the score's half's the reference's 303's)
  // The index-key plane: the ratio-4 (C4A) layer's the indexer's own
  // 128-wide rotated compressor's; the ratio-128 (C128A) layer's the main
  // latent's (the compressor's wkv / norm double as the index key's).
  w.idx_wk = a.index_source() ? a.idx_comp_wkv : a.comp_wkv;
  w.idx_k_norm = a.index_source() ? a.idx_comp_norm : a.comp_norm;
  w.ratio = cfg_.compress_ratio(layer);
  w.inv_freq = w.ratio > 0 ? inv_freq_compressed_ : inv_freq_window_;
  w.cache_ord = cache_ord_[static_cast<size_t>(layer)];
  w.tail_ord = tail_ord_[static_cast<size_t>(layer)];
  w.kv_source = a.kv_source();
  w.index_source = a.index_source();
  return w;
}

Dsv4HashWeights Dsv4Model::hash_view(const Dsv4LayerResident& r, int layer) const {
  const Dsv4MoeResident& m = r.moe;
  Dsv4HashWeights w;
  w.router_gate = m.router;
  w.router_bias = m.router_bias;  // null on the hash layers (they route via tid2eid)
  w.tid2eid = (layer < cfg_.num_hash_layers && layer < static_cast<int>(d_tid2eid_.size()))
                  ? d_tid2eid_[static_cast<size_t>(layer)]
                  : nullptr;
  // The MXFP4 expert's payload + scales (the [n_experts * 3]'s gate / up /
  // down's). The loader's per-expert views (not one packed array): the
  // expert's body (GPU-gate pending) reads the first's for now — the
  // composition's the packed expert array's the completion's.
  w.expert_payload = m.experts.empty() ? nullptr : m.experts[0].payload;
  w.expert_scales = m.experts.empty() ? nullptr : m.experts[0].scales;
  w.layer = layer;
  return w;
}

Dsv4DsparkWeights Dsv4Model::dspark_view(const Dsv4LayerResident& first, const Dsv4LayerResident& last,
                                         int stage) const {
  Dsv4DsparkWeights w;
  w.main_proj = first.draft.main_proj;
  w.main_norm = first.draft.main_norm;
  w.norm = last.draft.norm;
  w.markov_embed = last.draft.markov_w1;
  w.markov_head = last.draft.markov_w2;
  w.confidence = last.draft.confidence;
  w.attn_sink = last.attn.attn_sink;
  w.stage = stage;
  // The shared lm head: the DSpark surface's fp8 grid's view's the
  // rebind's null check's; the head's GEMM's the model's the bf16's (the
  // ops the model calls — stream_mean / block_rows / markov_bias /
  // confidence — never dereference it). Point the view's payload at the
  // model's bf16 lm head.
  w.lm_head.payload = reinterpret_cast<const uint8_t*>(globals_.lm_head);
  w.lm_head.scales = nullptr;
  w.lm_head.rows = lm_vocab_count_;
  w.lm_head.cols = cfg_.hidden_size;
  return w;
}

void Dsv4Model::build_layer_objects(const Dsv4LayerResident& r, int layer) {
  csa2_->rebind(csa2_view(r, layer), layer);
  moe_->rebind(hash_view(r, layer), layer);
}

int Dsv4Model::target_ordinal(int layer) const {
  for (int i = 0; i < targets_; ++i)
    if (cfg_.dspark_target_layer_ids[static_cast<size_t>(i)] == layer) return i;
  return -1;
}

const Dsv4DraftResident& Dsv4Model::draft_stage(int stage) {
  return loader_.load_layer(cfg_.num_hidden_layers + stage).draft;
}

// ---------------------------------------------------------------------------
// The session core's state hooks (the ratio-4 (C4A) overlapping compressor's
// per-request tails' the snapshots' the 2026-09-18's the dsv4 dspark +
// compressor wiring's): the attention's state is positional (the window ring's
// the layer scratch's, the slot's the position's, a rejected draft's slot is
// never read by a later query), so the tails are the only per-request state
// (the reset_slot_state's zero's, the spec segments' the rollback's the
// table's, the write / read's the snapshot's the copy's).
// ---------------------------------------------------------------------------
void Dsv4Model::graph_prepare() {
  if (loader_.residency() != Dsv4Residency::Resident)
    throw std::logic_error("session_graph_prepare: the decode graph needs a resident stack");
  for (int layer = 0; layer < (mtp_ ? cfg_.max_layer() : cfg_.num_hidden_layers); ++layer) {
    const Dsv4LayerResident& r = loader_.load_layer(layer);
    build_layer_objects(r, layer);
    for (int rows = 1; rows <= max_decode_rows_; ++rows)
      if (!csa2_->prepare(rows)) throw std::runtime_error("session_graph_prepare: CSA2 GEMM plans unavailable");
  }
  if (mtp_)
    for (int rows = 1; rows <= max_decode_rows_; ++rows)
      if (!dspark_->prepare(rows)) throw std::runtime_error("session_graph_prepare: DSpark plans unavailable");
}

// ---------------------------------------------------------------------------
// The DSpark draft (the DSpark layer's ops, the draft stages' walk on the
// CSA2 decode enqueue, the head's the Markov-biased row + the confidence).
// ---------------------------------------------------------------------------
void Dsv4Model::mtp_run_rows(int req, const int64_t* tokens, int64_t first_pos, int T, bool decode_row, bool capture,
                             int head_rows, int batch_requests) {
  if (!mtp_) throw std::logic_error("mtp_run_rows: MTP is not enabled");
  if (T <= 0 || T > max_tokens_) throw std::invalid_argument("mtp_run_rows: rows");
  if (head_rows < 0 || head_rows > T) throw std::invalid_argument("mtp_run_rows: head_rows");
  const int groups = batch_requests > 0 ? batch_requests : 1;
  if (T % groups != 0 || (head_rows > 0 && head_rows % groups != 0))
    throw std::invalid_argument("mtp_run_rows: rows must divide into the batch's groups");
  // A chain row: every call after the step's first draft until the next
  // walk (the capture sequence is host-ordered; the eager path names the
  // row by its position past the session's).
  const bool chain = decode_row && head_rows > 0 && draft_row_ > 0 &&
                     (capture ? T == groups : first_pos >= session_pos_[static_cast<size_t>(req)]);
  if (chain) {
    draft_chain_row(req, tokens, capture, head_rows, batch_requests);
    return;
  }
  const int64_t* d_pos = d_step_pos_;
  const int32_t* d_req = d_req_ids_;
  if (!decode_row) {
    std::vector<int64_t> pos(static_cast<size_t>(T));
    std::vector<int32_t> ids(static_cast<size_t>(T), req);
    for (int i = 0; i < T; ++i) pos[static_cast<size_t>(i)] = first_pos + i;
    DGPP_CUDA_OK(cudaMemcpyAsync(d_draft_pos_, pos.data(), static_cast<size_t>(T) * 8, cudaMemcpyHostToDevice, stream_));
    capture_trace_copy("dsv4 mtp_run_rows draft positions (prefill form)", "H2D", pos.data(), d_draft_pos_,
                       static_cast<size_t>(T) * 8, stream_);
    DGPP_CUDA_OK(cudaMemcpyAsync(d_draft_req_, ids.data(), static_cast<size_t>(T) * 4, cudaMemcpyHostToDevice, stream_));
    capture_trace_copy("dsv4 mtp_run_rows draft request ids (prefill form)", "H2D", ids.data(), d_draft_req_,
                       static_cast<size_t>(T) * 4, stream_);
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    d_pos = d_draft_pos_;
    d_req = d_draft_req_;
  }
  draft_first(req, tokens, d_pos, d_req, T, decode_row, capture, head_rows, batch_requests);
}

void Dsv4Model::draft_first(int req, const int64_t* tokens, const int64_t* d_pos, const int32_t* d_req, int T,
                            bool decode_row, bool capture, int head_rows, int batch_requests) {
  const int H = cfg_.hidden_size;
  const int W = targets_ * H;
  const int block = cfg_.dspark_block_size;
  const int stages = cfg_.num_draft_stages();
  const float eps = cfg_.rms_norm_eps;
  if (head_rows == 0) return;  // the prefill fills the window (run_rows); no block
  // ---- the accepted rows' main hidden -> main_x ----------------------------
  const uint16_t* src = main_hidden_;
  if (decode_row) {
    gather_draft_hidden(d_req, d_pos, main_gather_, T);
    src = main_gather_;
  }
  launch_scale_gemm_grid_bf16(src, static_cast<size_t>(W), dspark_w_.main_proj.payload, dspark_w_.main_proj.scales,
                              main_x_, T, H, W, stream_, 0, 7, 7, dense_mma_);
  csa2_rmsnorm_bf16(main_x_, H, dspark_w_.main_norm, main_x_, H, T, H, eps, stream_);
  // ---- the block: [next, noise x (block - 1)] at the next block positions --
  const int rpg = T / (batch_requests > 0 ? batch_requests : 1);
  const int groups = T / rpg;
  const int rows = groups * block;
  if (rows > max_decode_rows_) throw std::invalid_argument("mtp_run_rows: the draft blocks exceed the decode batch");
  if (!dspark_->prepare(rows)) throw std::runtime_error("mtp_run_rows: DSpark plans unavailable");
  dspark_->block_rows(d_pos, tokens, d_req, groups, rpg, blk_pos_, blk_tok_, blk_req_, blk_spans_, stream_);
  cur_ = streams_a_;
  nxt_ = streams_b_;
  pre_cur_ = one_hot_;
  pre_nxt_ = pre_a_;
  gather_embedding(blk_tok_, rows, capture);
  WalkRows wr;
  wr.tokens = blk_tok_;
  wr.req_ids = blk_req_;
  wr.pos = blk_pos_;
  wr.spans = blk_spans_;
  wr.num_requests = groups;
  wr.req = req;
  wr.decode = true;
  wr.capture = capture;
  for (int s = 0; s < stages; ++s) {
    const Dsv4LayerResident& r = loader_.load_layer(cfg_.num_hidden_layers + s);
    if (!csa2_->prepare(rows)) throw std::runtime_error("mtp_run_rows: CSA2 GEMM plans unavailable");
    enqueue_layer(r, cfg_.num_hidden_layers + s, rows, wr);
  }
  // ---- the head over the block: the collapse with the last stage's pre, the
  // draft's norm, the shared lm head -> base_logits_ [rows, vocab slice] ----
  launch_mhc_collapse_normed(cur_, pre_cur_, nullptr, eps, collapsed_, nullptr, mhc_cfg_, rows, stream_);
  csa2_rmsnorm_bf16(collapsed_, H, dspark_w_.norm, h_, H, rows, H, eps, stream_);
  gemm_.matmul(h_, globals_.lm_head, base_logits_, rows, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  // ---- block row 0, biased by the Markov head of `next`, into the head rows
  const int rows_out = head_rows / groups;
  dspark_->markov_bias(base_logits_, 0, blk_tok_, block, groups, logits_,
                       static_cast<int64_t>(rows_out) * lm_vocab_count_, rows_out, stream_);
  dspark_->confidence(collapsed_, 0, blk_tok_, block, groups,
                      conf_ + static_cast<size_t>(batch_requests > 0 ? 0 : req) * block, block, stream_);
  draft_row_ = 1;
  if (decode_row && (!capture || decode_tail_mirrors_) && head_rows <= max_decode_rows_)
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_,
                                 static_cast<size_t>(head_rows) * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}

void Dsv4Model::draft_chain_row(int req, const int64_t* tokens, bool capture, int head_rows, int batch_requests) {
  const int block = cfg_.dspark_block_size;
  const int groups = batch_requests > 0 ? batch_requests : 1;
  if (draft_row_ >= block) throw std::logic_error("mtp_run_rows: more chain rows than the block holds");
  if (head_rows != groups) throw std::invalid_argument("mtp_run_rows: a chain call heads one row per request");
  const int row = draft_row_++;
  dspark_->markov_bias(base_logits_, row, tokens, 1, groups, logits_, lm_vocab_count_, 1, stream_);
  dspark_->confidence(collapsed_, row, tokens, 1, groups,
                      conf_ + static_cast<size_t>(batch_requests > 0 ? 0 : req) * block + row, block, stream_);
  if (!capture || decode_tail_mirrors_)
    DGPP_CUDA_OK(cudaMemcpyAsync(h_tail_logits_, logits_,
                                 static_cast<size_t>(head_rows) * lm_vocab_count_ * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream_));
}

// ---------------------------------------------------------------------------
// The walk.
// ---------------------------------------------------------------------------
uint16_t* Dsv4Model::stage(uint16_t* fallback, int T, int width, bool capture) {
  if (!boundary_) return fallback;
  uint16_t* s = boundary_->stage(T, width);
  if (s == nullptr && capture) throw std::runtime_error("run_rows: a capture fold does not fit the recorder's staged buffer");
  return s ? s : fallback;
}

void Dsv4Model::fold(uint16_t* buf, int T, int width, bool capture) {
  if (!boundary_) return;
  // The host-driven reducer folds after the producing kernels have
  // quiesced; the stream-ordered one launches the fold on this stream
  // behind them and needs no drain.
  if (!capture && !boundary_->stream_ordered()) DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  boundary_->reduce(buf, T, width);
}

void Dsv4Model::gather_embedding(const int64_t* tokens, int T, bool capture) {
  const int H = cfg_.hidden_size;
  if (!embed_sharded_) {
    glm_embed_bcast_streams(globals_.embed, tokens, cur_, T, H, stream_);
    return;
  }
  uint16_t* e = stage(x_, T, H, capture);
  embed_gather_sliced_bf16(globals_.embed, tokens, e, T, H, globals_.embed_vocab_begin, globals_.embed_vocab_count,
                           stream_);
  fold(e, T, H, capture);
  for (int s = 0; s < 4; ++s) {
    DGPP_CUDA_OK(cudaMemcpy2DAsync(cur_ + static_cast<size_t>(s) * H, static_cast<size_t>(4) * H * 2, e,
                                   static_cast<size_t>(H) * 2, static_cast<size_t>(H) * 2, static_cast<size_t>(T),
                                   cudaMemcpyDefault, stream_));
    capture_trace_copy2d(("dsv4 gather_embedding sharded (stream " + std::to_string(s) + ")").c_str(), "D2D", e,
                         static_cast<size_t>(H) * 2, cur_ + static_cast<size_t>(s) * H, static_cast<size_t>(4) * H * 2,
                         static_cast<size_t>(H) * 2, static_cast<size_t>(T), stream_);
  }
}

void Dsv4Model::mhc_site(const uint16_t* streams, const GlmMhcWeights& w, const uint16_t* ln, int T, bool decode) {
  MhcSinglePass sp;
  sp.pre_in = pre_cur_;
  sp.pre_out = pre_nxt_;
  sp.post_f32 = post_f32_;
  sp.comb_f32 = comb_f32_;
  // Decode rows stay on the per-coefficient form at any count (a batched
  // row bitwise the row alone; the >= 16-token prefill forms are not).
  (void)launch_mhc_compute_normed(streams, w, mhc_cfg_, collapsed_, post_bf16_, comb_bf16_, mhc_logits_, nullptr,
                                  nullptr, cfg_.rms_norm_eps, T, stream_, nullptr, false, &sp, decode);
  csa2_rmsnorm_bf16(collapsed_, cfg_.hidden_size, ln, x_, cfg_.hidden_size, T, cfg_.hidden_size, cfg_.rms_norm_eps,
                     stream_);
  pre_cur_ = pre_nxt_;
  pre_nxt_ = pre_cur_ == pre_a_ ? pre_b_ : pre_a_;
}

void Dsv4Model::stream_update(const uint16_t* sublayer_out, int T) {
  launch_mhc_stream_update_f32(post_f32_, comb_f32_, sublayer_out, cur_, nxt_, mhc_cfg_, T, stream_);
  std::swap(cur_, nxt_);
}

void Dsv4Model::enqueue_layer(const Dsv4LayerResident& r, int layer, int T, const WalkRows& rows) {
  const int H = cfg_.hidden_size;
  build_layer_objects(r, layer);
  // DSpark: the target layers' attention INPUT, its stream mean, per row
  // (the reference reads it before the layer).
  if (mtp_) {
    const size_t W = static_cast<size_t>(targets_) * H;  // the fused stream-mean width
    const int ord = target_ordinal(layer);
    if (ord >= 0)
      dspark_->stream_mean(cur_, T, main_hidden_ + static_cast<size_t>(ord) * H, static_cast<int64_t>(W),
                           stream_);
  }
  // ---- the attention site ----------------------------------------------------
  GlmMhcWeights aw;
  aw.fn = r.mhc.attn_fn;
  aw.base = r.mhc.attn_base;
  aw.scale = r.mhc.attn_scale;
  mhc_site(cur_, aw, r.attn_norm, T, rows.decode);
  uint16_t* attn_out = stage(y_, T, H, rows.capture);
  if (!csa2_->prepare(T)) throw std::runtime_error("run_rows: CSA2 GEMM plans unavailable");
  // The decode path (the prefill's chunks ride it too): the window ring's
  // the layer scratch's (always available), the main / index cache's the
  // model's planar positional state's (2026-09-19, the dsv4 seam S1b's the
  // real pointer's the dsv41 pool's planes' re-expression's): the main
  // cache's the kFp4Block planar's (the kv sources' the ratio > 0's the
  // real pointer's, the window-only's the SWA's the nullptr's the window
  // source's always runs), the index cache's the planar e4m3's + the fp32
  // row-scale's (the index sources' the C4A's the 64-head selection's the
  // real pointer's, the C128A's the no-index's the nullptr's). The csa2
  // layer's publish_entries's writes the compressor's entries' into them
  // (the C4A's + the C128A's the populated's the 0731's reference's form's
  // the G-c128a-compressor's the closed's).
  // The compressor's per-request tail (the 0731's reference's form's, the
  // model's d_tails_'s the tail's ordinal's plane's — the dsv41's
  // pool.tails(w_.tail_ord)'s no-pool's re-expression's): the live per-request
  // tails (the main walk's) at this layer's tail ordinal (tail_ord_'s the
  // -1's the SWA-only's / the draft stages' the no-tail's layers'), the
  // per-ordinal's tail_off_'s (the C4A's 8 x 1024's the C128A's 128 x 512's
  // the reference's state's) + the per-ordinal's W's (the dsv4_tail_w_'s the
  // C4A's 1024's the C128A's 512's). The spec rows' tails (spec_tails_'s
  // the DSpark verify's rollback's) ride the pending's verify's call site's.
  const int tord = tail_ord_[static_cast<size_t>(layer)];
  float* tails = (tord >= 0 && d_tails_ != nullptr) ? d_tails_ + tail_off_[static_cast<size_t>(tord)] : nullptr;
  const int tails_w = (tord >= 0) ? dsv4_tail_w(tail_ratio_[static_cast<size_t>(tord)]) : 0;
  // The planar main / index cache pointers (2026-09-19, the dsv4 seam S1b's):
  // the main cache's the kv sources' (the ratio > 0's the cord >= 0's) the
  // real pointer's (the main source's attention's the csa2 layer's the
  // main_cache != nullptr's gate's), the index cache's the index sources'
  // (the C4A's the is_index_layer's the 64-head selection's) the real
  // pointer's + the fp32 row-scale's, the C128A's the no-index's the
  // nullptr's, the window-only's the SWA's the nullptr's (the window
  // source's always runs).
  const int cord = cache_ord_[static_cast<size_t>(layer)];
  uint8_t* main_cache =
      (cord >= 0 && cord < static_cast<int>(main_cache_.size())) ? main_cache_[static_cast<size_t>(cord)] : nullptr;
  uint8_t* index_cache = (cord >= 0 && cfg_.is_index_layer(layer) && cord < static_cast<int>(index_cache_.size()))
                            ? index_cache_[static_cast<size_t>(cord)]
                            : nullptr;
  float* index_scale = (index_cache != nullptr) ? index_scale_[static_cast<size_t>(cord)] : nullptr;
  csa2_->enqueue_decode(x_, main_cache, index_cache, index_scale, rows.req_ids, rows.pos, rows.spans, rows.num_requests,
                       T, attn_out, stream_, nullptr, tails, tails_w);
  // The DSpark union attention (2026-09-20, the dsv4 S3's union-attn's
  // q-latent's + the block-kv's wiring's; the dsv4_dspark_union_attn's the
  // 3-phase's single softmax's over [compressed | raw ring | block]'s):
  // the draft stages' (43/44/45's) SWA-only's attention (the no-
  // compressed-phase's n_comp == 0's, the no-raw-ring's raw_n == 0's, the
  // in-memory block's). The q latent's + the block kv's the csa2
  // projection's real output's (the csa2 layer's q_latent's / the block_kv's
  // the getter's — the 2026-09-18's stand-in's scratch's the csa2 seam's the
  // fill's the replaced's): the q_latent's the csa2 q's (the wq_b's q
  // latent's the per-head's re-normalized's (the G-q-renorm's the project_q_kv's
  // closed's) + the RoPE'd's) + the
  // block_kv's the csa2 kv's (the wkv's block kv's the in-memory's
  // unquantized's the step's rows' own kv_latent's). The pool's (the C4A's
  // compressed's the n_comp > 0's the VERIFY's phase's) + the raw ring's
  // (the 584 B's projected-main-hidden's the ring's append's) stay null (the
  // C4A's prefill's compressor's + the pool's geometry's the G-union-
  // wiring's / the G-cache-format's the open's — the DRAFT's n_comp == 0's
  // the ratio-0's class's, the pool's / the ring's the separate's named's
  // gap's). The DSpark kernel's 64-head's form (the kHeads's) vs. the csa2
  // layer's local_heads()'s (the 64/tp's): the real q-latent's the csa2
  // q's only when they match (the tp=1's the 64's); the tp>1's the head
  // count's the DSpark's replicated's 64-head's vs. the csa2's sharded's
  // local's the geometry's delta's (the zeroed's q fallback's the no-OOB's
  // safe's interim's, the parity's gate's settles's the DSpark's sharding's
  // decision's). The block's the single-request's the groups == 1's the
  // n_block == block_size's the in-memory's the step's m rows' own's;
  // the groups > 1's the per-request's block's the kernel's shared's form's
  // the delta's (the parity's gate's the batched's block's wiring's).
  if (layer >= cfg_.num_hidden_layers && dspark_ != nullptr && union_attn_scratch_ != nullptr) {
    const size_t per_row = static_cast<size_t>(Dsv4DsparkConfig::kHeads) * static_cast<size_t>(Dsv4DsparkConfig::kHeadDim);
    uint16_t* q_fallback = static_cast<uint16_t*>(union_attn_scratch_);  // region 0's the zeroed's
    uint16_t* out_latent = q_fallback + static_cast<size_t>(max_decode_rows_) * per_row;  // region 1's
    const uint16_t* q_latent = (csa2_->config().local_heads() == Dsv4DsparkConfig::kHeads) ? csa2_->q_latent()
                                                                                           : q_fallback;
    const uint16_t* block_kv = csa2_->block_kv();
    dspark_->union_attn(q_latent, nullptr, nullptr, 0, nullptr, 0, block_kv, cfg_.dspark_block_size, out_latent, T,
                        stream_);
  }
  fold(attn_out, T, H, rows.capture);  // block boundary 1: wo_b's partial
  stream_update(attn_out, T);
  // ---- the MoE site ------------------------------------------------------------
  GlmMhcWeights fw;
  fw.fn = r.mhc.ffn_fn;
  fw.base = r.mhc.ffn_base;
  fw.scale = r.mhc.ffn_scale;
  mhc_site(cur_, fw, r.ffn_norm, T, rows.decode);
  uint16_t* ffn_out = stage(y_, T, H, rows.capture);
  // The router's gate scores (bf16), the fused top-k (the learned's the
  // noaux_tc's, the hash's the tid2eid table's), the MXFP4 expert's
  // (the GPU-gate pending's completion's body's).
  gemm_.matmul(x_, r.moe.router, gate_logits_, T, cfg_.n_routed_experts, H, DType::BF16, GemmOut::BF16,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  moe_->route(gate_logits_, rows.tokens, T, topk_ids_, topk_w_, stream_);
  moe_->expert(x_, T, topk_ids_, topk_w_, ffn_out, stream_);
  fold(ffn_out, T, H, rows.capture);  // block boundary 2: the sliced experts
  stream_update(ffn_out, T);
}

Dsv4Model::Outputs Dsv4Model::run_rows(const RowRun& run) {
  const int T = run.T, req = run.req;
  if (run.capture && loader_.residency() != Dsv4Residency::Resident)
    throw std::logic_error("run_rows: a capture needs a resident stack");
  const int H = cfg_.hidden_size;
  const RowInputs in = begin_run(run);
  cur_ = streams_a_;
  nxt_ = streams_b_;
  // Layer 0's collapse reads the constant one-hot (1, 0, 0, 0) directly:
  // the sites rotate through pre_a_ / pre_b_ from there (no copy node —
  // the decode graph is kernels-only by contract).
  pre_cur_ = one_hot_;
  pre_nxt_ = pre_a_;
  gather_embedding(in.tokens, T, run.capture);
  Outputs out;
  WalkRows rows;
  rows.tokens = in.tokens;
  rows.req_ids = in.req_ids;
  rows.pos = in.pos;
  rows.spans = in.spans;
  rows.num_requests = in.num_requests;
  rows.req = req;
  rows.pos0 = run.pos0;
  rows.decode = run.decode;
  rows.capture = run.capture;
  for (int layer = 0; layer < cfg_.num_hidden_layers; ++layer) {
    const Dsv4LayerResident& r = loader_.load_layer(layer);
    enqueue_layer(r, layer, T, rows);
  }
  // The head: the weighted collapse with the last site's pre, the final
  // norm, the lm head on every walked row.
  launch_mhc_collapse_normed(cur_, pre_cur_, nullptr, cfg_.rms_norm_eps, collapsed_, nullptr, mhc_cfg_, T, stream_);
  csa2_rmsnorm_bf16(collapsed_, H, globals_.final_norm, h_, H, T, H, cfg_.rms_norm_eps, stream_);
  gemm_.matmul(h_, globals_.lm_head, logits_, T, lm_vocab_count_, H, DType::BF16, GemmOut::F32,
               static_cast<size_t>(H), gemm_ws_, gemm_ws_bytes_, stream_);
  // DSpark: the last rows' target hidden into the slots' windows by
  // position (the draft gathers its accepted rows' from there); a
  // prefill's walked rows are the draft's rows (mtp_run_rows).
  if (mtp_) {
    const size_t W = static_cast<size_t>(targets_) * H;
    const int n = std::min(T, max_decode_rows_);
    store_draft_hidden(main_hidden_ + static_cast<size_t>(T - n) * W, in.req_ids + (T - n), in.pos + (T - n), n);
    draft_row_ = 0;
  }
  // The stream-ordered reducer's verdict for this pass's folds (a failed
  // collective is an error here, not a wrong number read later).
  if (!run.capture && boundary_) boundary_->settle();
  out = finish_run(run, std::move(out));
  return out;
}

Dsv4Model::Outputs Dsv4Model::forward(const std::vector<int64_t>& token_ids) {
  const int T = static_cast<int>(token_ids.size());
  if (T <= 0) throw std::invalid_argument("forward: empty token batch");
  if (T > max_tokens_) throw std::invalid_argument("forward: tokens exceed max_tokens");
  if (T > max_context_) throw std::invalid_argument("forward: tokens exceed the context bound");
  for (int64_t id : token_ids)
    if (id < 0 || id >= cfg_.vocab_size) throw std::invalid_argument("forward: token id out of range");
  if (session_pos_[0] != 0) throw std::logic_error("forward: slot 0 holds an open session (close it first)");
  open_slot(0);
  RowRun run;
  run.req = 0;
  run.ids = token_ids.data();
  run.T = T;
  run.pos0 = 0;
  run.decode = false;
  run.all_rows = true;
  Outputs out = run_rows(run);
  session_close(0);
  return out;
}

}  // namespace dgpp
