#include "models/kda_layer.hpp"

#include "kernels/bf16_gemv.hpp"

#include <cmath>
#include <stdexcept>

#include "kernels/kda.hpp"

namespace dgpp {

KdaLayer::KdaLayer(Arena& arena, IGemm& gemm, const KdaLayerWeights& weights,
                   const KdaConfig& cfg, int max_tokens, void* gemm_workspace,
                   size_t gemm_ws_bytes, float o_norm_eps)
    : gemm_(gemm),
      w_(weights),
      cfg_(cfg),
      geo_(KdaGeometry::from_config(cfg)),
      max_tokens_(max_tokens),
      o_norm_eps_(o_norm_eps),
      gemm_ws_(gemm_workspace),
      gemm_ws_bytes_(gemm_ws_bytes) {
  KdaConfig::validate_config(cfg);
  if (max_tokens <= 0)
    throw std::invalid_argument("kda layer: max_tokens must be positive");
  if (!gemm_workspace || gemm_ws_bytes == 0)
    throw std::invalid_argument("kda layer: GEMM workspace required");
  const auto require = [](const void* p, const char* what) {
    if (!p)
      throw std::invalid_argument(std::string("kda layer: missing ") + what);
  };
  require(w_.in_proj, "in_proj");
  require(w_.f_b, "f_b");
  require(w_.g_b, "g_b");
  require(w_.conv, "conv");
  require(w_.a_log, "a_log");
  require(w_.dt_bias, "dt_bias");
  require(w_.o_norm, "o_norm");
  require(w_.o_proj, "o_proj");

  // Reference computes scale = K ** -0.5 in python float64 and passes the
  // fp32-rounded result to the kernel; reproduce that exactly.
  scale_ = static_cast<float>(std::pow(static_cast<double>(cfg.head_dim), -0.5));

  const size_t rows = static_cast<size_t>(max_tokens);
  proj_ = static_cast<uint16_t*>(arena.alloc_persistent(
      MemClass::DeviceHot, rows * geo_.in_proj_cols * 2, 256));
  g1_ = static_cast<uint16_t*>(arena.alloc_persistent(
      MemClass::DeviceHot, rows * geo_.local_proj * 2, 256));
  g2_ = static_cast<uint16_t*>(arena.alloc_persistent(
      MemClass::DeviceHot, rows * geo_.local_proj * 2, 256));
  qkv_conv_ = static_cast<uint16_t*>(arena.alloc_persistent(
      MemClass::DeviceHot, rows * geo_.conv_channels * 2, 256));
  core_ = static_cast<uint16_t*>(arena.alloc_persistent(
      MemClass::DeviceHot, rows * geo_.local_proj * 2, 256));
  normed_ = static_cast<uint16_t*>(arena.alloc_persistent(
      MemClass::DeviceHot, rows * geo_.local_proj * 2, 256));
}

size_t KdaLayer::persistent_hot_bytes(const KdaConfig& cfg, int max_tokens) {
  const KdaGeometry g = KdaGeometry::from_config(cfg);
  const size_t rows = static_cast<size_t>(max_tokens);
  // Each scratch buffer is separately 256-byte aligned in the arena, so the
  // honest footprint rounds every region up.
  const auto padded = [](size_t b) { return (b + 255) / 256 * 256; };
  return padded(rows * 2 * static_cast<size_t>(g.in_proj_cols)) +
         4 * padded(rows * 2 * g.local_proj) +
         padded(rows * 2 * static_cast<size_t>(g.conv_channels));
}

bool KdaLayer::prepare(int tokens) {
  if (tokens <= 0 || tokens > max_tokens_)
    throw std::invalid_argument("kda layer: token count out of range");
  const int hid = cfg_.hidden;
  const int lp = geo_.local_proj;
  const bool ok =
      gemm_.ensure_plan(tokens, geo_.in_proj_cols, hid, DType::BF16,
                        GemmOut::BF16, static_cast<size_t>(hid)) &&
      gemm_.ensure_plan(tokens, lp, cfg_.head_dim, DType::BF16, GemmOut::BF16,
                        static_cast<size_t>(geo_.in_proj_cols)) &&
      gemm_.ensure_plan(tokens, hid, lp, DType::BF16, GemmOut::BF16,
                        static_cast<size_t>(lp));
  return ok;
}

void KdaLayer::enqueue(const void* hidden_in, float* recurrent_state,
                       uint16_t* conv_state, int conv_state_width, void* out,
                       int tokens, cudaStream_t stream,
                       WeightPrefetcher* prefetch,
                       const KdaSpeculativeSinks& spec,
                       const KdaLayerBatch& batch) {
  if (tokens <= 0 || tokens > max_tokens_)
    throw std::invalid_argument("kda layer: token count out of range");
  if (!hidden_in || !recurrent_state || !conv_state || !out)
    throw std::invalid_argument("kda layer: null buffer");
  if (conv_state_width < geo_.conv_hist)
    throw std::invalid_argument("kda layer: conv state narrower than history");
  if (batch.enabled()) {
    if (batch.recurrent_request_stride_elems <
        static_cast<int64_t>(geo_.recurrent_elems))
      throw std::invalid_argument(
          "kda layer: recurrent request stride smaller than a state");
    const int64_t conv_elems =
        static_cast<int64_t>(geo_.conv_channels) * conv_state_width;
    if (batch.conv_request_stride_elems < conv_elems)
      throw std::invalid_argument(
          "kda layer: conv request stride smaller than a state");
  }

  const int hid = cfg_.hidden;
  const int lp = geo_.local_proj;
  const int n_in = geo_.in_proj_cols;
  // Column layout of the fused projection row: [f_a | g_a | q | k | v | b].
  // f_a/g_a lead: their column offsets (0 and head_dim) are 16-byte aligned
  // whenever head_dim % 8 == 0, which the strided f_b/g_b GEMMs require —
  // cuBLASLt silently returns garbage for misaligned activation pointers.
  // q|k|v|b trail because their consumers (conv, recurrent kernels) do
  // scalar bf16 loads and don't care.
  const int off_q = 2 * cfg_.head_dim;
  const int off_k = off_q + lp;
  const int off_v = off_k + lp;
  const int off_b = off_v + lp;

  // 1) merged f_a|g_a|q|k|v|b projection
  gemm_.matmul(hidden_in, w_.in_proj, proj_, tokens, n_in, hid, DType::BF16,
               GemmOut::BF16, static_cast<size_t>(hid), gemm_ws_,
               gemm_ws_bytes_, stream);
  // Steps 2-5 below are latency-bound (~45 us at decode) and the output
  // projection is the next weight the chain reads: start it now.
  if (prefetch) {
    // The bytes the o_proj launch streams (its packed companion's when the
    // GEMM holds one for these rows).
    const void* view = nullptr;
    size_t view_bytes = 0;
    gemm_.resident_view(w_.o_proj, o_proj_bytes(), tokens, &view, &view_bytes);
    prefetch->prefetch_after(stream, view, view_bytes, 0, prefetch->layer_rate());
  }
  // 2) decay logits g1 = f_b(f_a) and o_norm gate g2 = g_b(g_a); both read
  //    K-column slices of the fused projection row (strided activations).
  //    At decode both are 5 us launches of mostly fixed cost: one dual
  //    GEMV launch (bitwise the two launches) when the GEMV takes them.
  if (bf16_gemv_accepts(w_.f_b, tokens, cfg_.head_dim) &&
      bf16_gemv_accepts(w_.g_b, tokens, cfg_.head_dim)) {
    Bf16GemvProblem fb, gb;
    fb.act = proj_;
    fb.act_row_stride = static_cast<size_t>(n_in);
    fb.weight = static_cast<const uint16_t*>(w_.f_b);
    fb.out = g1_;
    fb.n = lp;
    gb.act = proj_ + cfg_.head_dim;
    gb.act_row_stride = static_cast<size_t>(n_in);
    gb.weight = static_cast<const uint16_t*>(w_.g_b);
    gb.out = g2_;
    gb.n = lp;
    launch_bf16_gemv_dual(fb, gb, /*out_f32=*/false, tokens, cfg_.head_dim,
                          stream);
  } else {
    gemm_.matmul(proj_, w_.f_b, g1_, tokens, lp, cfg_.head_dim, DType::BF16,
                 GemmOut::BF16, static_cast<size_t>(n_in), gemm_ws_,
                 gemm_ws_bytes_, stream);
    gemm_.matmul(proj_ + cfg_.head_dim, w_.g_b, g2_, tokens, lp, cfg_.head_dim,
                 DType::BF16, GemmOut::BF16, static_cast<size_t>(n_in),
                 gemm_ws_, gemm_ws_bytes_, stream);
  }
  // 3) causal depthwise conv + silu over the merged q|k|v channels; rolls
  //    the committed conv history in place.
  if (batch.enabled()) {
    kda_causal_conv_silu_bf16_batched(
        proj_ + off_q, n_in, w_.conv, conv_state,
        batch.conv_request_stride_elems, conv_state_width, qkv_conv_, tokens,
        geo_.conv_channels, cfg_.conv_width, batch.rows, stream, spec.conv);
  } else {
    kda_causal_conv_silu_bf16(proj_ + off_q, n_in, w_.conv, conv_state,
                              conv_state_width, qkv_conv_, tokens,
                              geo_.conv_channels, cfg_.conv_width, stream,
                              spec.conv);
  }
  // 4) FP32 recurrent update; beta is the raw b-column slice of the fused
  //    projection (strided), sigmoid applied in-kernel like the reference.
  if (batch.enabled()) {
    kda_recurrent_fwd_batched(
        qkv_conv_, g1_, proj_ + off_b, n_in, w_.a_log, w_.dt_bias,
        recurrent_state, batch.recurrent_request_stride_elems, core_, tokens,
        geo_.local_heads, cfg_.head_dim, cfg_.head_dim, cfg_.lower_bound,
        scale_, batch.rows, stream, spec.recurrent);
  } else {
    kda_recurrent_fwd(qkv_conv_, g1_, proj_ + off_b, n_in, w_.a_log,
                      w_.dt_bias, recurrent_state, core_, tokens,
                      geo_.local_heads, cfg_.head_dim, cfg_.head_dim,
                      cfg_.lower_bound, scale_, stream, spec.recurrent);
  }
  // 5) gated RMSNorm (sigmoid) on the recurrent output: one row per
  // (token, head), each of width head_dim.
  kda_gated_rmsnorm_sigmoid_bf16(core_, g2_, w_.o_norm, normed_,
                                 static_cast<int64_t>(tokens) *
                                     geo_.local_heads,
                                 cfg_.head_dim, o_norm_eps_, stream);
  // 6) output projection
  gemm_.matmul(normed_, w_.o_proj, out, tokens, hid, lp, DType::BF16,
               GemmOut::BF16, static_cast<size_t>(lp), gemm_ws_,
               gemm_ws_bytes_, stream);
}

}  // namespace dgpp
