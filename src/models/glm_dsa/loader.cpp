#include "models/glm_dsa/loader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "common/cuda_check.hpp"
#include "common/log.hpp"
#include "loaders/packq_quant.hpp"

namespace dgpp {
namespace {

bool contains(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

// The replicated set (rank-invariant reads at world > 1): the norms, the
// router and its bias, the indexer, the draft head's own tensors, the
// embedding and the final norm, every shape record, and the q_a / kv_a
// projections (their rows feed every rank's latent). Everything else is a
// slice: q_b, kv_b, o_proj, the dense MLP, every expert, the lm head under
// VocabSharded.
bool is_replicated(const GlmDsaExpectedTensor& e) {
  if (e.role == GlmDsaTensorRole::IntShape) return true;
  switch (e.cls) {
    case GlmDsaWeightClass::Embed:
    case GlmDsaWeightClass::FinalNorm:
    case GlmDsaWeightClass::LayerNorm:
    case GlmDsaWeightClass::Indexer:
    case GlmDsaWeightClass::Router:
    case GlmDsaWeightClass::MtpHead:
      return true;
    case GlmDsaWeightClass::Attention:
      return contains(e.name, "q_a_proj") || contains(e.name, "kv_a_proj_with_mqa");
    case GlmDsaWeightClass::LmHead:
    case GlmDsaWeightClass::DenseMlp:
    case GlmDsaWeightClass::SharedExpert:
    case GlmDsaWeightClass::RoutedExpert:
      return false;
  }
  return false;
}

}  // namespace

// The per-class builders (loaders/weight_build.hpp's primitives).
struct GlmDsaLoaderFamily::Builder : WeightBuilder<GlmDsaExpectedTensor> {
  const GlmDsaTextConfig& cfg;
  const GlmDsaLocalGeometry& geo;
  GlmDsaLayerResident& out;

  Builder(const GlmDsaTextConfig& cfg_, const GlmDsaLocalGeometry& geo_,
          const std::vector<GlmDsaExpectedTensor>& table_,
          const std::unordered_map<std::string, const GlmDsaExpectedTensor*>& by_name_,
          LayerBump& bump_, GlmDsaLayerResident& out_,
          const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<GlmDsaExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_,
                                            geo_.rank, geo_.world, "glm_dsa loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const GlmDsaExpectedTensor& e) const override { return is_replicated(e); }

  // ---- the packed triple's geometry ----------------------------------------
  struct PackedSource {
    const GlmDsaExpectedTensor* words;
    const GlmDsaExpectedTensor* scales;
    const GlmDsaExpectedTensor* shape;
    int64_t N, K;
    int bits;
  };
  PackedSource packed_source(const std::string& base) {
    PackedSource s;
    s.words = &expected(base + ".weight_packed");
    s.scales = &expected(base + ".weight_scale");
    s.shape = &expected(base + ".weight_shape");
    s.bits = s.words->bits;
    if (s.bits != 4 && s.bits != 8) fail("'" + base + "' is not a packed matrix");
    s.N = s.words->shape[0];
    s.K = s.words->shape[1] * 32 / s.bits;
    packed_check_cols(s.K, s.bits, who.c_str());
    if (s.scales->shape[0] != s.N || s.scales->shape[1] != s.K / kPackedGroup)
      fail("packed scale geometry mismatch on " + base);
    // The shape record: read (16 bytes, replicated) and checked against
    // the packed geometry in copy mode; counted in both.
    if (copy) {
      const TensorInfo& t = source(s.shape->name);
      int64_t rec[2];
      std::memcpy(rec, t.data, 16);
      if (rec[0] != s.N || rec[1] != s.K)
        fail("'" + base + ".weight_shape' [" + std::to_string(rec[0]) + ", " + std::to_string(rec[1]) +
             "] disagrees with the packed geometry [" + std::to_string(s.N) + ", " + std::to_string(s.K) + "]");
      consumed(t);
    }
    note_read(*s.shape, 16);
    return s;
  }

  // Rows [row_start, +rows) of the packed matrix `base` into `q`'s words
  // and scales at row offset `dst_row` of a (possibly larger) allocation.
  void copy_packed_rows(const PackedSource& s, int64_t row_start, int64_t rows, uint32_t* words_dst,
                        uint16_t* scales_dst, int64_t dst_row) {
    const size_t wpr = static_cast<size_t>(s.K * s.bits / 32), spr = static_cast<size_t>(s.K / kPackedGroup);
    check_range(s.words->name, row_start, rows, s.N);
    if (copy) {
      const TensorInfo& tw = source(s.words->name);
      const TensorInfo& ts = source(s.scales->name);
      std::memcpy(bump.host(words_dst) + static_cast<size_t>(dst_row) * wpr,
                  static_cast<const uint32_t*>(tw.data) + static_cast<size_t>(row_start) * wpr,
                  static_cast<size_t>(rows) * wpr * 4);
      std::memcpy(bump.host(scales_dst) + static_cast<size_t>(dst_row) * spr,
                  static_cast<const uint16_t*>(ts.data) + static_cast<size_t>(row_start) * spr,
                  static_cast<size_t>(rows) * spr * 2);
      consumed(tw);
      consumed(ts);
    }
    note_read(*s.words, static_cast<size_t>(rows) * wpr * 4);
    note_read(*s.scales, static_cast<size_t>(rows) * spr * 2);
  }

  GlmPackedMatrix alloc_packed(int64_t rows, int64_t cols, int bits) {
    GlmPackedMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.bits = bits;
    q.packed = static_cast<const uint32_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols * bits / 32) * 4));
    q.scales = static_cast<const uint16_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols / kPackedGroup) * 2));
    return q;
  }

  // Rows [row_start, +rows) of the [N, K] packed matrix `base` (rows are
  // independent: no alignment).
  GlmPackedMatrix load_packq_rows(const std::string& base, int64_t row_start, int64_t rows) {
    const PackedSource s = packed_source(base);
    GlmPackedMatrix q = alloc_packed(rows, s.K, s.bits);
    copy_packed_rows(s, row_start, rows, const_cast<uint32_t*>(q.packed), const_cast<uint16_t*>(q.scales), 0);
    return q;
  }

  // Columns [col_start, +cols) of every row of the [N, K] packed matrix
  // `base`, packed: col_start and cols on 64-group boundaries (a group is
  // also whole words at either width).
  GlmPackedMatrix load_packq_cols(const std::string& base, int64_t col_start, int64_t cols) {
    const PackedSource s = packed_source(base);
    packed_check_cols(cols, s.bits, who.c_str());
    if (col_start % kPackedGroup != 0)
      fail("packed column slice of '" + base + "' must start on a 64-element group boundary");
    check_range(base, col_start, cols, s.K);
    const size_t wpr_full = static_cast<size_t>(s.K * s.bits / 32), wpr = static_cast<size_t>(cols * s.bits / 32);
    const size_t spr_full = static_cast<size_t>(s.K / kPackedGroup), spr = static_cast<size_t>(cols / kPackedGroup);
    const size_t w0 = static_cast<size_t>(col_start * s.bits / 32), s0 = static_cast<size_t>(col_start / kPackedGroup);
    GlmPackedMatrix q = alloc_packed(s.N, cols, s.bits);
    if (copy) {
      const TensorInfo& tw = source(s.words->name);
      const TensorInfo& ts = source(s.scales->name);
      const uint32_t* sw = static_cast<const uint32_t*>(tw.data);
      const uint16_t* ss = static_cast<const uint16_t*>(ts.data);
      uint32_t* hw = bump.host(const_cast<uint32_t*>(q.packed));
      uint16_t* hs = bump.host(const_cast<uint16_t*>(q.scales));
      if (cols == s.K) {
        std::memcpy(hw, sw, static_cast<size_t>(s.N) * wpr * 4);
        std::memcpy(hs, ss, static_cast<size_t>(s.N) * spr * 2);
      } else {
        for (int64_t r = 0; r < s.N; ++r) {
          std::memcpy(hw + r * wpr, sw + r * wpr_full + w0, wpr * 4);
          std::memcpy(hs + r * spr, ss + r * spr_full + s0, spr * 2);
        }
      }
      consumed(tw);
      consumed(ts);
    }
    note_read(*s.words, static_cast<size_t>(s.N) * wpr * 4);
    note_read(*s.scales, static_cast<size_t>(s.N) * spr * 2);
    return q;
  }

  // Rows [row_start, +rows) of the packed matrix `base`, dequantized on the
  // host into a BF16 [rows, K] buffer (plan D6: bf16(code x scale), the
  // one rounding the bridge adds; the reference's own decompression).
  uint16_t* load_packq_rows_dequant_bf16(const std::string& base, int64_t row_start, int64_t rows) {
    const PackedSource s = packed_source(base);
    check_range(s.words->name, row_start, rows, s.N);
    const size_t wpr = static_cast<size_t>(s.K * s.bits / 32), spr = static_cast<size_t>(s.K / kPackedGroup);
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(s.K) * 2));
    if (copy) {
      const TensorInfo& tw = source(s.words->name);
      const TensorInfo& ts = source(s.scales->name);
      const uint32_t* sw = static_cast<const uint32_t*>(tw.data) + static_cast<size_t>(row_start) * wpr;
      const uint16_t* ss = static_cast<const uint16_t*>(ts.data) + static_cast<size_t>(row_start) * spr;
      uint16_t* h = bump.host(dst);
      for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < s.K; ++c)
          h[static_cast<size_t>(r) * s.K + c] =
              float_to_bf16_bits(packq_decode(sw, ss, s.K, s.bits, r, c));
      consumed(tw);
      consumed(ts);
    }
    note_read(*s.words, static_cast<size_t>(rows) * wpr * 4);
    note_read(*s.scales, static_cast<size_t>(rows) * spr * 2);
    return dst;
  }

  // Two BF16 [rows_a, K] and [rows_b, K] matrices stacked into one
  // allocation (the fused [q_a | kv_a] projection). Both read whole.
  uint16_t* load_bf16_fused(const std::string& a, const std::string& b) {
    const GlmDsaExpectedTensor& ea = expected(a);
    const GlmDsaExpectedTensor& eb = expected(b);
    if (ea.shape.size() != 2 || eb.shape.size() != 2 || ea.shape[1] != eb.shape[1])
      fail("fused rows of '" + a + "' and '" + b + "' disagree on K");
    // Packable (the model's 12-bit companion replaces it): grant_bf16.
    uint16_t* dst = static_cast<uint16_t*>(grant_bf16(ea.shape[0] + eb.shape[0], ea.shape[1], /*packable=*/true));
    if (copy) {
      const TensorInfo& ta = source(a);
      const TensorInfo& tb = source(b);
      std::memcpy(bump.host(dst), ta.data, ea.nbytes());
      std::memcpy(reinterpret_cast<uint8_t*>(bump.host(dst)) + ea.nbytes(), tb.data, eb.nbytes());
      consumed(ta);
      consumed(tb);
    }
    note_read(ea, ea.nbytes());
    note_read(eb, eb.nbytes());
    return dst;
  }
  // The packed form of the same stack: q_a's rows then kv_a's, one matrix.
  GlmPackedMatrix load_packq_fused(const std::string& a, const std::string& b) {
    const PackedSource sa = packed_source(a);
    const PackedSource sb = packed_source(b);
    if (sa.K != sb.K || sa.bits != sb.bits) fail("fused packed rows of '" + a + "' and '" + b + "' disagree");
    GlmPackedMatrix q = alloc_packed(sa.N + sb.N, sa.K, sa.bits);
    copy_packed_rows(sa, 0, sa.N, const_cast<uint32_t*>(q.packed), const_cast<uint16_t*>(q.scales), 0);
    copy_packed_rows(sb, 0, sb.N, const_cast<uint32_t*>(q.packed), const_cast<uint16_t*>(q.scales), sa.N);
    return q;
  }

  // ---- the draft's BF16 experts, requantized (plan D5) --------------------
  // rows_slice: rows [start, +count) of every column; else columns
  // [start, +count) of every row; the slice encoded at `bits` with the
  // checkpoint's recipe (loaders/packq_quant.hpp).
  GlmPackedMatrix requant_bf16(const std::string& base, bool rows_slice, int64_t start, int64_t count,
                               int bits) {
    const GlmDsaExpectedTensor& e = expected(base + ".weight");
    if (e.role != GlmDsaTensorRole::Bf16Expert || e.shape.size() != 2)
      fail("'" + e.name + "' is not a BF16 expert matrix");
    const int64_t N = e.shape[0], K = e.shape[1];
    const int64_t rows = rows_slice ? count : N;
    const int64_t cols = rows_slice ? K : count;
    packed_check_cols(cols, bits, who.c_str());
    if (!rows_slice && start % kPackedGroup != 0)
      fail("requant column slice of '" + base + "' must start on a 64-element group boundary");
    check_range(base, start, count, rows_slice ? N : K);
    GlmPackedMatrix q = alloc_packed(rows, cols, bits);
    if (copy) {
      const TensorInfo& t = source(e.name);
      const uint16_t* src = static_cast<const uint16_t*>(t.data);
      const int64_t col0 = rows_slice ? 0 : start;
      std::vector<float> row(static_cast<size_t>(cols));
      uint32_t* hw = bump.host(const_cast<uint32_t*>(q.packed));
      uint16_t* hs = bump.host(const_cast<uint16_t*>(q.scales));
      const size_t wpr = static_cast<size_t>(cols * bits / 32), spr = static_cast<size_t>(cols / kPackedGroup);
      for (int64_t r = 0; r < rows; ++r) {
        const uint16_t* s = src + (rows_slice ? start + r : r) * K + col0;
        for (int64_t c = 0; c < cols; ++c) row[static_cast<size_t>(c)] = bf16_bits_to_float(s[c]);
        packq_encode_row(row.data(), cols, bits, hw + r * wpr, hs + r * spr);
      }
      consumed(t);
    }
    note_read(e, static_cast<size_t>(rows) * static_cast<size_t>(cols) * 2);
    return q;
  }

  // ---- the classes ---------------------------------------------------------
  void build_attention(const std::string& p, int layer) {
    GlmDsaAttnResident& a = out.attn;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    const int64_t nope = cfg.qk_nope_head_dim, rope = cfg.qk_rope_head_dim, v = cfg.v_head_dim;
    const int64_t qb0 = static_cast<int64_t>(geo.head_begin) * (nope + rope);
    const int64_t qbn = static_cast<int64_t>(geo.local_heads) * (nope + rope);
    const int64_t kb0 = static_cast<int64_t>(geo.head_begin) * (nope + v);
    const int64_t kbn = static_cast<int64_t>(geo.local_heads) * (nope + v);
    const int64_t o0 = static_cast<int64_t>(geo.head_begin) * v, on = static_cast<int64_t>(geo.local_heads) * v;
    a.q_aln = load_bf16(p + "q_a_layernorm.weight");
    a.kv_aln = load_bf16(p + "kv_a_layernorm.weight");
    if (cfg.attention_bits_of(layer) != 0) {
      a.qkv_a_packed = load_packq_fused(p + "q_a_proj", p + "kv_a_proj_with_mqa");
      a.q_b_packed = load_packq_rows(p + "q_b_proj", qb0, qbn);
      a.kv_b = load_packq_rows_dequant_bf16(p + "kv_b_proj", kb0, kbn);
      a.o_proj_packed = load_packq_cols(p + "o_proj", o0, on);
    } else {
      a.qkv_a = load_bf16_fused(p + "q_a_proj.weight", p + "kv_a_proj_with_mqa.weight");
      a.q_b = load_bf16_rows(p + "q_b_proj.weight", qb0, qbn, /*packable=*/true);
      a.kv_b = load_bf16_rows(p + "kv_b_proj.weight", kb0, kbn);
      a.o_proj = load_bf16_cols(p + "o_proj.weight", o0, on, /*packable=*/true);
    }
    if (cfg.owns_indexer(layer)) {
      const std::string ip = p + "indexer.";
      a.wq_b = load_bf16(ip + "wq_b.weight", /*packable=*/true);
      a.wk = load_bf16(ip + "wk.weight", /*packable=*/true);
      a.wp = load_bf16(ip + "weights_proj.weight", /*packable=*/true);
      a.k_norm_w = load_bf16(ip + "k_norm.weight");
      a.k_norm_b = load_bf16(ip + "k_norm.bias");
    }
  }

  void build_dense(const std::string& p) {
    const int64_t I = geo.local_dense_inter, r = rank;
    GlmDsaDenseMlpResident& m = out.dense;
    m.local_inter = I;
    m.gate = load_bf16_rows(p + "gate_proj.weight", r * I, I, /*packable=*/true);
    m.up = load_bf16_rows(p + "up_proj.weight", r * I, I, /*packable=*/true);
    m.down = load_bf16_cols(p + "down_proj.weight", r * I, I, /*packable=*/true);
  }

  void build_moe(const std::string& p, int layer, bool draft) {
    GlmDsaMoeResident& m = out.moe_w;
    m.router = load_bf16(p + "gate.weight");
    m.router_bias = load_f32(p + "gate.e_score_correction_bias");
    const int64_t I = geo.local_inter, S = geo.local_shared_inter, r = rank;
    m.local_inter = I;
    m.local_shared_inter = S;
    const int E = cfg.n_routed_experts;
    // The draft's BF16 experts take the packed layers' widths (routed at
    // expert_bits, the shared expert at int8).
    const int eb = draft ? cfg.expert_bits : cfg.expert_bits_of(layer);
    const int sb = draft ? 8 : cfg.shared_bits_of(layer);
    m.experts.resize(static_cast<size_t>(E) * 3);
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      GlmPackedMatrix* t = m.experts.data() + static_cast<size_t>(e) * 3;
      if (draft) {
        t[0] = requant_bf16(ep + "gate_proj", true, r * I, I, eb);
        t[1] = requant_bf16(ep + "up_proj", true, r * I, I, eb);
        t[2] = requant_bf16(ep + "down_proj", false, r * I, I, eb);
      } else {
        t[0] = load_packq_rows(ep + "gate_proj", r * I, I);
        t[1] = load_packq_rows(ep + "up_proj", r * I, I);
        t[2] = load_packq_cols(ep + "down_proj", r * I, I);
      }
    }
    const std::string sp = p + "shared_experts.";
    if (draft) {
      m.shared[0] = requant_bf16(sp + "gate_proj", true, r * S, S, sb);
      m.shared[1] = requant_bf16(sp + "up_proj", true, r * S, S, sb);
      m.shared[2] = requant_bf16(sp + "down_proj", false, r * S, S, sb);
    } else {
      m.shared[0] = load_packq_rows(sp + "gate_proj", r * S, S);
      m.shared[1] = load_packq_rows(sp + "up_proj", r * S, S);
      m.shared[2] = load_packq_cols(sp + "down_proj", r * S, S);
    }
  }

  void build_layer(int layer) {
    const int max_layer = cfg.num_hidden_layers + (cfg.mtp_layer() >= 0 ? 1 : 0);
    if (layer < 0 || layer >= max_layer) fail("layer index out of range: " + std::to_string(layer));
    const bool draft = layer == cfg.mtp_layer();
    const std::string p = glm_dsa_layer_prefix(cfg, layer);
    out.layer = layer;
    out.moe = cfg.is_moe_layer(layer);
    if (draft) {
      out.enorm = load_bf16(p + "enorm.weight");
      out.hnorm = load_bf16(p + "hnorm.weight");
      out.eh_proj = load_bf16(p + "eh_proj.weight", /*packable=*/true);
      out.shared_head_norm = load_bf16(p + "shared_head.norm.weight");
    }
    out.input_norm = load_bf16(p + "input_layernorm.weight");
    out.post_norm = load_bf16(p + "post_attention_layernorm.weight");
    build_attention(p + "self_attn.", layer);
    if (out.moe)
      build_moe(p + "mlp.", layer, draft);
    else
      build_dense(p + "mlp.");
  }
};

namespace {

std::pair<int, int> lm_head_slice(const GlmDsaTextConfig& cfg, int rank, int world) {
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

GlmDsaLocalGeometry GlmDsaLocalGeometry::from_config(const GlmDsaTextConfig& cfg, int rank, int world,
                                                     GlmDsaHeadSharding head) {
  if (world > 1) glm_dsa_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("glm_dsa loader: rank/world out of range");
  GlmDsaLocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  g.local_inter = cfg.moe_intermediate_size / world;
  g.local_shared_inter = cfg.shared_expert_inter() / world;
  g.local_dense_inter = cfg.intermediate_size / world;
  if (head == GlmDsaHeadSharding::VocabSharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.lm_vocab_begin = b;
    g.lm_vocab_count = n;
  } else {
    g.lm_vocab_begin = 0;
    g.lm_vocab_count = cfg.vocab_size;
  }
  if (world > 1 && GlmDsaLayerStream::embed_vocab_sharded()) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.embed_vocab_begin = b;
    g.embed_vocab_count = n;
  } else {
    g.embed_vocab_begin = 0;
    g.embed_vocab_count = cfg.vocab_size;
  }
  return g;
}

// ---- the family hooks (loaders/resident_stream.hpp) -------------------------

void GlmDsaLoaderFamily::validate_binding(const GlmDsaTextConfig& cfg, const PresentMap& present) {
  const GlmDsaBindReport rep = glm_dsa_validate_text_binding(cfg, present);
  if (rep.ok()) return;
  std::string msg = "glm_dsa loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

bool GlmDsaLoaderFamily::digest_included(const GlmDsaExpectedTensor& e) { return is_replicated(e); }

// The BF16 column packs (the dense layers' o_proj and down, the draft's
// o_proj) read their sources after the builder returns: those classes'
// sources are dropped after the packs.
bool GlmDsaLoaderFamily::discard_after_pack(const GlmDsaExpectedTensor& e) {
  return e.cls == GlmDsaWeightClass::Attention || e.cls == GlmDsaWeightClass::DenseMlp;
}

size_t GlmDsaLoaderFamily::globals_bytes(const GlmDsaTextConfig& cfg, int rank, int world,
                                         LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const GlmDsaLocalGeometry geo = GlmDsaLocalGeometry::from_config(cfg, rank, world, head);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(geo.embed_vocab_count) * H * 2);  // embed (the whole table or this rank's rows)
  b += align_up_256(H * 2);                                               // final norm
  b += align_up_256(static_cast<size_t>(geo.lm_vocab_count) * H * 2);
  return b;
}

size_t GlmDsaLoaderFamily::globals_side_bytes(const GlmDsaTextConfig& cfg, int rank, int world,
                                              LoaderHeadSharding head) {
  const int count = GlmDsaLocalGeometry::from_config(cfg, rank, world, head).lm_vocab_count;
  if (!bf12_shape_ok(count, cfg.hidden_size)) return 0;
  return align_up_256(static_cast<size_t>(count) * static_cast<size_t>(cfg.hidden_size) * 2);
}

void GlmDsaLoaderFamily::build_globals(const GlmDsaTextConfig& cfg, const GlmDsaLocalGeometry& geo,
                                       const LoaderTensorMap& tensors, LayerBump& bump,
                                       GlmDsaGlobalsResident& out, uint64_t& source_bytes,
                                       uint64_t& verbatim_bytes, LoaderHeadSharding head) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("glm_dsa loader: global tensor missing: " + name);
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
  const size_t row_bytes = static_cast<size_t>(cfg.hidden_size) * 2;
  {
    // The embedding: the whole table, or this rank's vocabulary rows
    // (the lm head's slice) under embed_vocab_sharded().
    const TensorInfo& e = lookup("model.embed_tokens.weight");
    const int ebegin = geo.embed_vocab_begin, ecount = geo.embed_vocab_count;
    if (ebegin < 0 || ecount <= 0 || static_cast<size_t>(ebegin + ecount) * row_bytes > e.nbytes())
      throw std::runtime_error("glm_dsa loader: the embedding slice does not fit the table");
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(ecount) * row_bytes));
    std::memcpy(bump.host(dst), static_cast<const uint8_t*>(e.data) + static_cast<size_t>(ebegin) * row_bytes,
                static_cast<size_t>(ecount) * row_bytes);
    source_bytes += static_cast<size_t>(ecount) * row_bytes;
    if (ecount == cfg.vocab_size) verbatim_bytes += e.nbytes();
    out.embed = dst;
    out.embed_vocab_begin = ebegin;
    out.embed_vocab_count = ecount;
  }
  out.final_norm = copy_global("model.norm.weight");
  const TensorInfo& t = lookup("lm_head.weight");
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

template class ResidentLayerStream<GlmDsaLoaderFamily>;

// ---------------------------------------------------------------------------

namespace {
bool g_embed_vocab_sharded = false;
}  // namespace
void GlmDsaLayerStream::set_embed_vocab_sharded(bool on) { g_embed_vocab_sharded = on; }
bool GlmDsaLayerStream::embed_vocab_sharded() { return g_embed_vocab_sharded; }

GlmDsaLayerStream::GlmDsaLayerStream(const GlmDsaTextConfig& cfg, const std::string& checkpoint_dir,
                                     int rank, int world, GlmDsaResidency residency,
                                     GlmDsaHeadSharding head, bool resident_mtp)
    : ResidentLayerStream<GlmDsaLoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                              resident_mtp) {
  open_resident_image();
}

void GlmDsaLayerStream::set_resident_image_dir(const std::string& dir) {
  resident_image_dir_storage() = dir;
}
const std::string& GlmDsaLayerStream::resident_image_dir() { return resident_image_dir_storage(); }

}  // namespace dgpp
