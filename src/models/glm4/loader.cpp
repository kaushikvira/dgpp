#include "models/glm4/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "loaders/nvfp4_quant.hpp"

namespace dgpp {
namespace {

bool ends_with(const std::string& s, const char* suffix) {
  const size_t n = std::strlen(suffix);
  return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// The replicated set (rank-invariant reads at world > 1): the layer
// norms, the router and its bias, the q/k norms, the draft head's own
// tensors, the embedding and the final norm, and every scalar (global
// scales). Everything else is a slice: the attention projections and
// biases, every NVFP4 matrix, the draft's BF16 experts, the lm head under
// VocabSharded. Unused roles are never read at all.
bool is_replicated(const Glm4ExpectedTensor& e) {
  if (e.role == Glm4TensorRole::Fp4Global) return true;
  if (e.unused()) return true;
  switch (e.cls) {
    case Glm4WeightClass::Embed:
    case Glm4WeightClass::FinalNorm:
    case Glm4WeightClass::LayerNorm:
    case Glm4WeightClass::Router:
    case Glm4WeightClass::MtpHead:
    case Glm4WeightClass::KvScale:
      return true;
    case Glm4WeightClass::Attention:
      return ends_with(e.name, "q_norm.weight") || ends_with(e.name, "k_norm.weight");
    case Glm4WeightClass::LmHead:
    case Glm4WeightClass::DenseMlp:
    case Glm4WeightClass::SharedExpert:
    case Glm4WeightClass::RoutedExpert:
      return false;
  }
  return false;
}

}  // namespace

// The per-class builders (loaders/weight_build.hpp's primitives).
struct Glm4LoaderFamily::Builder : WeightBuilder<Glm4ExpectedTensor> {
  const Glm4TextConfig& cfg;
  const Glm4LocalGeometry& geo;
  Glm4LayerResident& out;
  float* globals_ = nullptr;  // the layer's gathered fp4 global scales (device)
  int globals_used_ = 0;
  int globals_total_ = 0;

  Builder(const Glm4TextConfig& cfg_, const Glm4LocalGeometry& geo_,
           const std::vector<Glm4ExpectedTensor>& table_,
           const std::unordered_map<std::string, const Glm4ExpectedTensor*>& by_name_,
           LayerBump& bump_, Glm4LayerResident& out_,
           const std::unordered_map<std::string, const TensorInfo*>& tensors_,
           std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<Glm4ExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_,
                                          geo_.rank, geo_.world, "glm4 loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const Glm4ExpectedTensor& e) const override { return is_replicated(e); }
  bool fp4_global(const Glm4ExpectedTensor& e) const override {
    return e.role == Glm4TensorRole::Fp4Global;
  }

  // ---- the gathered global scales ----------------------------------------
  void reserve_globals(int n) {
    globals_total_ = n;
    globals_used_ = 0;
    globals_ = static_cast<float*>(bump.alloc(static_cast<size_t>(std::max(n, 1)) * 4));
  }
  // The next slot: the device address the view points at (a real address
  // whenever the bump is not counting — the image restore's layout pass
  // allocates without copying); `*host` its staging mirror (copy mode).
  float* next_global(float** host) {
    if (globals_used_ >= globals_total_) fail("global scale slots exhausted (builder bug)");
    const int i = globals_used_++;
    *host = copy ? bump.host(globals_) + i : nullptr;
    return globals_ ? globals_ + i : reinterpret_cast<float*>(static_cast<size_t>(i) * 4);
  }
  // The modelopt per-tensor scale of `base` into a slot as 1 / weight_scale_2.
  float* load_global_reciprocal(const std::string& base) {
    const Glm4ExpectedTensor& eg = expected(base + ".weight_scale_2");
    float* host = nullptr;
    float* slot = next_global(&host);
    if (copy) {
      const TensorInfo& t = source(eg.name);
      float ws2;
      std::memcpy(&ws2, t.data, 4);
      if (!(ws2 > 0.0f) || !std::isfinite(ws2))
        fail("'" + eg.name + "' is not a positive finite scale");
      *host = 1.0f / ws2;
      consumed(t);
    }
    note_read(eg, 4);
    return slot;
  }
  // A requantized matrix's global (1 / the recipe's weight_scale_2).
  float* store_global(float ws2) {
    float* host = nullptr;
    float* slot = next_global(&host);
    if (copy) *host = 1.0f / ws2;
    return slot;
  }

  // ---- NVFP4 slices in the modelopt layout --------------------------------
  // Rows [row_start, +rows) of the [N, K] matrix `base`: payload rows and
  // scale rows are contiguous.
  GlmFp4Matrix load_fp4_rows_mo(const std::string& base, int64_t row_start, int64_t rows) {
    const Glm4ExpectedTensor& ep = expected(base + ".weight");
    const Glm4ExpectedTensor& es = expected(base + ".weight_scale");
    const int64_t N = ep.shape[0];
    const int64_t cols = ep.shape[1] * 2;
    fp4_check_cols(cols, who.c_str());
    check_range(base, row_start, rows, N);
    if (es.shape[0] != N || es.shape[1] != cols / kFp4Group)
      fail("NVFP4 scale geometry mismatch on " + base);
    const size_t pc = static_cast<size_t>(cols / 2), sc = static_cast<size_t>(cols / kFp4Group);
    GlmFp4Matrix q;
    q.rows = rows;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * sc));
    if (copy) {
      const TensorInfo& tp = source(ep.name);
      const TensorInfo& ts = source(es.name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(tp.data) + static_cast<size_t>(row_start) * pc,
                  static_cast<size_t>(rows) * pc);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.scales)),
                  static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(row_start) * sc,
                  static_cast<size_t>(rows) * sc);
      consumed(tp);
      consumed(ts);
    }
    note_read(ep, static_cast<size_t>(rows) * pc);
    note_read(es, static_cast<size_t>(rows) * sc);
    q.global_scale = load_global_reciprocal(base);
    return q;
  }

  // Columns [col_start, +cols) of every row of the [N, K] matrix `base`,
  // packed: col_start on a 16-block boundary.
  GlmFp4Matrix load_fp4_cols_mo(const std::string& base, int64_t col_start, int64_t cols) {
    const Glm4ExpectedTensor& ep = expected(base + ".weight");
    const Glm4ExpectedTensor& es = expected(base + ".weight_scale");
    const int64_t N = ep.shape[0];
    const int64_t full = ep.shape[1] * 2;
    fp4_check_cols(full, who.c_str());
    fp4_check_cols(cols, who.c_str());
    if (col_start % kFp4Group != 0)
      fail("NVFP4 column slice of '" + base + "' must start on a 16-element block boundary");
    check_range(base, col_start, cols, full);
    if (es.shape[0] != N || es.shape[1] != full / kFp4Group)
      fail("NVFP4 scale geometry mismatch on " + base);
    const size_t pc = static_cast<size_t>(cols / 2), pc_full = static_cast<size_t>(full / 2);
    const size_t sc = static_cast<size_t>(cols / kFp4Group), sc_full = static_cast<size_t>(full / kFp4Group);
    GlmFp4Matrix q;
    q.rows = N;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(N) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(N) * sc));
    if (copy) {
      const TensorInfo& tp = source(ep.name);
      const TensorInfo& ts = source(es.name);
      const uint8_t* sp = static_cast<const uint8_t*>(tp.data);
      const uint8_t* ss = static_cast<const uint8_t*>(ts.data);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      uint8_t* hs = bump.host(const_cast<uint8_t*>(q.scales));
      if (cols == full) {
        std::memcpy(hp, sp, static_cast<size_t>(N) * pc);
        std::memcpy(hs, ss, static_cast<size_t>(N) * sc);
      } else {
        for (int64_t r = 0; r < N; ++r) {
          std::memcpy(hp + r * pc, sp + r * pc_full + col_start / 2, pc);
          std::memcpy(hs + r * sc, ss + r * sc_full + col_start / kFp4Group, sc);
        }
      }
      consumed(tp);
      consumed(ts);
    }
    note_read(ep, static_cast<size_t>(N) * pc);
    note_read(es, static_cast<size_t>(N) * sc);
    q.global_scale = load_global_reciprocal(base);
    return q;
  }

  // ---- the draft's BF16 experts, requantized (plan D1) --------------------
  // rows_slice: rows [start, +count) of every column; else columns
  // [start, +count) of every row. The slice is one tensor of the recipe
  // (its own amax and per-tensor scale).
  GlmFp4Matrix requant_bf16(const std::string& base, bool rows_slice, int64_t start, int64_t count) {
    const Glm4ExpectedTensor& e = expected(base + ".weight");
    if (e.role != Glm4TensorRole::Bf16Expert || e.shape.size() != 2)
      fail("'" + e.name + "' is not a BF16 expert matrix");
    const int64_t N = e.shape[0], K = e.shape[1];
    const int64_t rows = rows_slice ? count : N;
    const int64_t cols = rows_slice ? K : count;
    fp4_check_cols(cols, who.c_str());
    if (!rows_slice && start % kFp4Group != 0)
      fail("requant column slice of '" + base + "' must start on a 16-element block boundary");
    check_range(base, start, count, rows_slice ? N : K);
    const size_t pc = static_cast<size_t>(cols / 2), sc = static_cast<size_t>(cols / kFp4Group);
    GlmFp4Matrix q;
    q.rows = rows;
    q.cols = cols;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * pc));
    q.scales = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * sc));
    float ws2 = 1.0f;
    if (copy) {
      const TensorInfo& t = source(e.name);
      const uint16_t* src = static_cast<const uint16_t*>(t.data);
      const int64_t col0 = rows_slice ? 0 : start;
      auto row_ptr = [&](int64_t r) { return src + (rows_slice ? start + r : r) * K + col0; };
      float amax = 0.0f;
      for (int64_t r = 0; r < rows; ++r) amax = std::max(amax, nvfp4_amax_bf16(row_ptr(r), cols));
      ws2 = nvfp4_tensor_scale(amax);
      std::vector<float> row(static_cast<size_t>(cols));
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      uint8_t* hs = bump.host(const_cast<uint8_t*>(q.scales));
      for (int64_t r = 0; r < rows; ++r) {
        const uint16_t* s = row_ptr(r);
        for (int64_t c = 0; c < cols; ++c) row[static_cast<size_t>(c)] = bf16_bits_to_float(s[c]);
        nvfp4_encode_row(row.data(), cols, ws2, hp + r * pc, hs + r * sc);
      }
      consumed(t);
    }
    note_read(e, static_cast<size_t>(rows) * static_cast<size_t>(cols) * 2);
    q.global_scale = store_global(ws2);
    return q;
  }

  // ---- the classes ---------------------------------------------------------
  void build_attention(const std::string& p) {
    const int64_t d = cfg.head_dim;
    Glm4AttnResident& a = out.attn;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_kv_heads = geo.local_kv_heads;
    a.kv_head_begin = geo.kv_head_begin;
    const int64_t q0 = static_cast<int64_t>(geo.head_begin) * d, qn = static_cast<int64_t>(geo.local_heads) * d;
    const int64_t k0 = static_cast<int64_t>(geo.kv_head_begin) * d, kn = static_cast<int64_t>(geo.local_kv_heads) * d;
    // Packable: the model's 12-bit companions replace the four projections
    // (WeightBuilder::grant_bf16 — aside under bf12-only residency).
    a.q_proj = load_bf16_rows(p + "q_proj.weight", q0, qn, /*packable=*/true);
    a.k_proj = load_bf16_rows(p + "k_proj.weight", k0, kn, /*packable=*/true);
    a.v_proj = load_bf16_rows(p + "v_proj.weight", k0, kn, /*packable=*/true);
    a.o_proj = load_bf16_cols(p + "o_proj.weight", q0, qn, /*packable=*/true);
    if (cfg.attention_bias) {
      a.q_bias = load_bf16_rows(p + "q_proj.bias", q0, qn);
      a.k_bias = load_bf16_rows(p + "k_proj.bias", k0, kn);
      a.v_bias = load_bf16_rows(p + "v_proj.bias", k0, kn);
    }
    if (cfg.use_qk_norm) {
      a.q_norm = load_bf16(p + "q_norm.weight");
      a.k_norm = load_bf16(p + "k_norm.weight");
    }
  }

  void build_dense(const std::string& p) {
    const int64_t I = geo.local_dense_inter, r = rank;
    Glm4DenseMlpResident& m = out.dense;
    m.local_inter = I;
    m.gate = load_fp4_rows_mo(p + "gate_proj", r * I, I);
    m.up = load_fp4_rows_mo(p + "up_proj", r * I, I);
    m.down = load_fp4_cols_mo(p + "down_proj", r * I, I);
  }

  void build_moe(const std::string& p, bool bf16_experts) {
    Glm4MoeResident& m = out.moe_w;
    m.router = load_bf16(p + "gate.weight");
    m.router_bias = load_f32(p + "gate.e_score_correction_bias");
    const int64_t I = geo.local_inter, S = geo.local_shared_inter, r = rank;
    m.local_inter = I;
    m.local_shared_inter = S;
    const int E = cfg.n_routed_experts;
    m.experts.resize(static_cast<size_t>(E) * 3);
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      GlmFp4Matrix* t = m.experts.data() + static_cast<size_t>(e) * 3;
      if (bf16_experts) {
        t[0] = requant_bf16(ep + "gate_proj", true, r * I, I);
        t[1] = requant_bf16(ep + "up_proj", true, r * I, I);
        t[2] = requant_bf16(ep + "down_proj", false, r * I, I);
      } else {
        t[0] = load_fp4_rows_mo(ep + "gate_proj", r * I, I);
        t[1] = load_fp4_rows_mo(ep + "up_proj", r * I, I);
        t[2] = load_fp4_cols_mo(ep + "down_proj", r * I, I);
      }
    }
    const std::string sp = p + "shared_experts.";
    if (bf16_experts) {
      m.shared[0] = requant_bf16(sp + "gate_proj", true, r * S, S);
      m.shared[1] = requant_bf16(sp + "up_proj", true, r * S, S);
      m.shared[2] = requant_bf16(sp + "down_proj", false, r * S, S);
    } else {
      m.shared[0] = load_fp4_rows_mo(sp + "gate_proj", r * S, S);
      m.shared[1] = load_fp4_rows_mo(sp + "up_proj", r * S, S);
      m.shared[2] = load_fp4_cols_mo(sp + "down_proj", r * S, S);
    }
  }

  void build_layer(int layer) {
    const int max_layer = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    if (layer < 0 || layer >= max_layer) fail("layer index out of range: " + std::to_string(layer));
    const bool is_mtp = layer == cfg.mtp_layer();
    const std::string p = glm4_layer_prefix(cfg, layer);
    out.layer = layer;
    out.moe = cfg.is_moe_layer(layer);
    // The global scales first: one slot per NVFP4 matrix of the layer.
    reserve_globals(out.moe ? 3 * (cfg.n_routed_experts + 1) : 3);
    if (is_mtp) {
      out.enorm = load_bf16(p + "enorm.weight");
      out.hnorm = load_bf16(p + "hnorm.weight");
      out.eh_proj = load_bf16(p + "eh_proj.weight", /*packable=*/true);
      out.shared_head_norm = load_bf16(p + "shared_head.norm.weight");
    }
    out.input_norm = load_bf16(p + "input_layernorm.weight");
    out.post_norm = load_bf16(p + "post_attention_layernorm.weight");
    build_attention(p + "self_attn.");
    if (out.moe)
      build_moe(p + "mlp.", is_mtp);
    else
      build_dense(p + "mlp.");
    if (globals_used_ != globals_total_) fail("global scale slots left unused (builder bug)");
  }
};

namespace {

std::pair<int, int> lm_head_slice(const Glm4TextConfig& cfg, int rank, int world) {
  const int64_t V = cfg.vocab_size;
  const int64_t begin = V * rank / world;
  const int64_t end = V * (rank + 1) / world;
  return {static_cast<int>(begin), static_cast<int>(end - begin)};
}

std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}

}  // namespace

// ---------------------------------------------------------------------------

Glm4LocalGeometry Glm4LocalGeometry::from_config(const Glm4TextConfig& cfg, int rank, int world,
                                                 Glm4HeadSharding head) {
  if (world > 1) glm4_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("glm4 loader: rank/world out of range");
  Glm4LocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  if (cfg.num_key_value_heads >= world) {
    g.local_kv_heads = cfg.num_key_value_heads / world;
    g.kv_head_begin = g.local_kv_heads * rank;
  } else {
    g.local_kv_heads = 1;
    g.kv_head_begin = rank / (world / cfg.num_key_value_heads);
  }
  g.local_inter = cfg.moe_intermediate_size / world;
  g.local_shared_inter = cfg.shared_expert_inter() / world;
  g.local_dense_inter = cfg.intermediate_size / world;
  if (head == Glm4HeadSharding::VocabSharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.lm_vocab_begin = b;
    g.lm_vocab_count = n;
  } else {
    g.lm_vocab_begin = 0;
    g.lm_vocab_count = cfg.vocab_size;
  }
  return g;
}

// ---- the family hooks (loaders/resident_stream.hpp) -------------------------

void Glm4LoaderFamily::validate_binding(const Glm4TextConfig& cfg, const PresentMap& present) {
  const Glm4BindReport rep = glm4_validate_text_binding(cfg, present);
  if (rep.ok()) return;
  std::string msg = "glm4 loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

// The draft's embedding and head copies must be the globals' bytes (the
// draft shares them): sampled at three windows of 1 MiB, refused otherwise.
void Glm4LoaderFamily::check_sources(const Glm4TextConfig& cfg, const LoaderTensorMap& tensors) {
  if (cfg.mtp_layer() < 0) return;
  const std::string p = glm4_layer_prefix(cfg, cfg.mtp_layer());
  const auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("glm4 loader: tensor not in checkpoint: " + name);
    return *it->second;
  };
  for (const auto& [dup, orig] : {std::pair{p + "embed_tokens.weight", std::string("model.embed_tokens.weight")},
                                  std::pair{p + "shared_head.head.weight", std::string("lm_head.weight")}}) {
    const TensorInfo& a = lookup(dup);
    const TensorInfo& b = lookup(orig);
    const size_t bytes = a.numel() * dtype_size(a.dtype);
    if (bytes != b.numel() * dtype_size(b.dtype))
      throw std::runtime_error("glm4 loader: '" + dup + "' differs in size from '" + orig + "'");
    const size_t window = std::min<size_t>(bytes, size_t{1} << 20);
    for (const size_t off : {size_t{0}, (bytes - window) / 2, bytes - window})
      if (std::memcmp(static_cast<const uint8_t*>(a.data) + off, static_cast<const uint8_t*>(b.data) + off,
                      window) != 0)
        throw std::runtime_error("glm4 loader: '" + dup + "' is not a copy of '" + orig +
                                 "' (the draft shares the embedding and the head; a distinct copy is "
                                 "not implemented)");
  }
}

bool Glm4LoaderFamily::digest_included(const Glm4ExpectedTensor& e) {
  return is_replicated(e) && !e.unused();
}

// The packed column slice (o_proj) reads its source after the builder
// returns: the attention class's sources are dropped after the packs.
bool Glm4LoaderFamily::discard_after_pack(const Glm4ExpectedTensor& e) {
  return e.cls == Glm4WeightClass::Attention;
}

size_t Glm4LoaderFamily::globals_bytes(const Glm4TextConfig& cfg, int rank, int world,
                                       LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(cfg.vocab_size) * H * 2);  // embed
  b += align_up_256(H * 2);                                        // final norm
  b += align_up_256(static_cast<size_t>(Glm4LocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count) * H * 2);
  return b;
}

size_t Glm4LoaderFamily::globals_side_bytes(const Glm4TextConfig& cfg, int rank, int world,
                                            LoaderHeadSharding head) {
  const int count = Glm4LocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count;
  if (!bf12_shape_ok(count, cfg.hidden_size)) return 0;
  return align_up_256(static_cast<size_t>(count) * static_cast<size_t>(cfg.hidden_size) * 2);
}

void Glm4LoaderFamily::build_globals(const Glm4TextConfig& cfg, const Glm4LocalGeometry& geo,
                                     const LoaderTensorMap& tensors, LayerBump& bump,
                                     Glm4GlobalsResident& out, uint64_t& source_bytes,
                                     uint64_t& verbatim_bytes, LoaderHeadSharding head) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("glm4 loader: global tensor missing: " + name);
    return *it->second;
  };
  auto copy_global = [&](const std::string& name) -> uint16_t* {
    const TensorInfo& t = lookup(name);
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(t.nbytes()));
    std::memcpy(bump.host(dst), t.data, t.nbytes());
    source_bytes += t.nbytes();
    verbatim_bytes += t.nbytes();
    return dst;
  };
  out.embed = copy_global("model.embed_tokens.weight");
  out.final_norm = copy_global("model.norm.weight");
  const TensorInfo& t = lookup("lm_head.weight");
  const size_t row_bytes = static_cast<size_t>(cfg.hidden_size) * 2;
  const int begin = geo.lm_vocab_begin, count = geo.lm_vocab_count;
  // The head is packable (the model's 12-bit companion replaces it).
  uint16_t* dst = static_cast<uint16_t*>(bf12_shape_ok(count, cfg.hidden_size)
                                             ? bump.alloc_side(static_cast<size_t>(count) * row_bytes)
                                             : bump.alloc(static_cast<size_t>(count) * row_bytes));
  std::memcpy(bump.host(dst), static_cast<const uint8_t*>(t.data) + static_cast<size_t>(begin) * row_bytes,
              static_cast<size_t>(count) * row_bytes);
  source_bytes += static_cast<size_t>(count) * row_bytes;
  if (head == LoaderHeadSharding::Full) verbatim_bytes += static_cast<size_t>(count) * row_bytes;
  out.lm_head = dst;
  out.lm_vocab_begin = begin;
  out.lm_vocab_count = count;
}

template class ResidentLayerStream<Glm4LoaderFamily>;

// ---------------------------------------------------------------------------

Glm4LayerStream::Glm4LayerStream(const Glm4TextConfig& cfg, const std::string& checkpoint_dir,
                                 int rank, int world, Glm4Residency residency,
                                 Glm4HeadSharding head, bool resident_mtp)
    : ResidentLayerStream<Glm4LoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                            resident_mtp) {
  open_resident_image();
}

void Glm4LayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage() = dir;
}
const std::string& Glm4LayerStream::resident_image_dir() { return resident_image_dir_storage(); }

}  // namespace dgpp
