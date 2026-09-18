#include "models/dsv4/loader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>

#include "common/cuda_check.hpp"
#include "common/log.hpp"

namespace dgpp {
namespace {

bool contains(const std::string& s, const char* needle) { return s.find(needle) != std::string::npos; }

// The replicated set (rank-invariant reads at world > 1): the norms, the
// routers and their biases, the indexers, the compressors (the indexer's
// own included), the mHC coefficients, the draft's head tensors
// (main_proj, the Markov head, the confidence head, the stage hc_head),
// the embedding, the final norm, wq_a and wkv (every rank's latent).
// Everything else is a slice: wq_b, wo_a, wo_b, the sink, every expert,
// the shared expert, the lm head. The hash tables are mapped, never read
// whole, so they are outside this set and the digest.
bool dsv4_is_replicated(const Dsv4ExpectedTensor& e) {
  switch (e.cls) {
    case Dsv4WeightClass::Embed:
    case Dsv4WeightClass::FinalNorm:
    case Dsv4WeightClass::LayerNorm:
    case Dsv4WeightClass::Indexer:
    case Dsv4WeightClass::Compressor:
    case Dsv4WeightClass::Router:
    case Dsv4WeightClass::Mhc:
    case Dsv4WeightClass::Draft:
      return true;
    case Dsv4WeightClass::Attention:
      return contains(e.name, ".wq_a.") || contains(e.name, ".wkv.");
    case Dsv4WeightClass::LmHead:
    case Dsv4WeightClass::SharedExpert:
    case Dsv4WeightClass::RoutedExpert:
      return false;
  }
  return false;
}

std::pair<int, int> lm_head_slice(const Dsv4TextConfig& cfg, int rank, int world) {
  const int64_t V = cfg.vocab_size;
  const int64_t begin = V * rank / world;
  const int64_t end = V * (rank + 1) / world;
  return {static_cast<int>(begin), static_cast<int>(end - begin)};
}

std::string& resident_image_dir_storage() {
  static std::string dir;
  return dir;
}

bool g_embed_vocab_sharded = false;

}  // namespace

bool Dsv4LoaderFamily::is_replicated(const Dsv4ExpectedTensor& e) { return dsv4_is_replicated(e); }

float Dsv4LoaderFamily::e8m0_to_float(uint8_t b) {
  if (b == 255) return std::nanf("");
  return std::ldexp(1.0f, static_cast<int>(b) - 127);
}

void Dsv4LoaderFamily::check_mxfp4_slice(int64_t col_start, int64_t cols) {
  fp4_check_cols(cols, who(), kMxfp4Group);
  if (col_start < 0 || col_start % kMxfp4Group != 0)
    throw std::invalid_argument(std::string(who()) + ": an MXFP4 column slice must start on a 32-element "
                                         "block boundary (got " +
                                         std::to_string(col_start) + ")");
}

// The per-class builders (loaders/weight_build.hpp's primitives), the
// dsv41 family's pattern on the release's 128 x 128 fp8 grid.
struct Dsv4LoaderFamily::Builder : WeightBuilder<Dsv4ExpectedTensor> {
  const Dsv4TextConfig& cfg;
  const Dsv4LocalGeometry& geo;
  Dsv4LayerResident& out;

  Builder(const Dsv4TextConfig& cfg_, const Dsv4LocalGeometry& geo_,
          const std::vector<Dsv4ExpectedTensor>& table_,
          const std::unordered_map<std::string, const Dsv4ExpectedTensor*>& by_name_, LayerBump& bump_,
          Dsv4LayerResident& out_, const std::unordered_map<std::string, const TensorInfo*>& tensors_,
          std::vector<DequantJob>& jobs_, std::vector<PackJob>& packs_, bool copy_)
      : WeightBuilder<Dsv4ExpectedTensor>(table_, by_name_, bump_, tensors_, jobs_, packs_, copy_,
                                           geo_.rank, geo_.world, "dsv4 loader"),
        cfg(cfg_), geo(geo_), out(out_) {}

  bool replicated(const Dsv4ExpectedTensor& e) const override { return is_replicated(e); }

  // ---- per-class source accounting (the host-side byte reconcile) -------
  // The inherited note_read accumulates the layer total; every builder
  // method here also folds its bytes into class_source_ so
  // class_source_plan() can build the reconcile's per-class closed form.
  std::map<Dsv4WeightClass, uint64_t> class_source() const { return class_source_; }
  void note_class(const Dsv4ExpectedTensor& e, uint64_t bytes) { class_source_[e.cls] += bytes; }

  uint16_t* load_bf16(const std::string& name) {
    const Dsv4ExpectedTensor& e = expected(name);
    uint16_t* p = WeightBuilder::load_bf16(name);
    note_class(e, e.nbytes());
    return p;
  }
  float* load_f32(const std::string& name) {
    const Dsv4ExpectedTensor& e = expected(name);
    float* p = WeightBuilder::load_f32(name);
    note_class(e, e.nbytes());
    return p;
  }
  float* load_f32_range(const std::string& name, int64_t start, int64_t count) {
    const Dsv4ExpectedTensor& e = expected(name);
    float* p = WeightBuilder::load_f32_range(name, start, count);
    note_class(e, static_cast<uint64_t>(count) * 4);
    return p;
  }
  float* load_bf16_as_f32(const std::string& name, int64_t start = 0, int64_t count = -1) {
    const Dsv4ExpectedTensor& e = expected(name);
    float* p = WeightBuilder::load_bf16_as_f32(name, start, count);
    const int64_t total = static_cast<int64_t>(e.numel());
    if (count < 0) count = total - start;
    note_class(e, static_cast<uint64_t>(count) * 2);
    return p;
  }

  // ---- fp8 pairs on the 128 x 128 grid (e8m0 -> fp32 scales) -----------
  struct Fp8Source {
    const Dsv4ExpectedTensor* w;
    const Dsv4ExpectedTensor* s;
    int64_t N, K, SR, SC;
  };
  Fp8Source fp8_source(const std::string& base) {
    Fp8Source f;
    f.w = &expected(base + ".weight");
    f.s = &expected(base + ".scale");
    if (f.w->role != Dsv4TensorRole::Fp8Payload || f.s->role != Dsv4TensorRole::Fp8Scale ||
        f.w->shape.size() != 2 || f.s->shape.size() != 2)
      fail("'" + base + "' is not an fp8 pair");
    f.N = f.w->shape[0];
    f.K = f.w->shape[1];
    f.SR = f.s->shape[0];
    f.SC = f.s->shape[1];
    const int b = cfg.fp8_block_size;
    if (f.SR != (f.N + b - 1) / b || f.SC != (f.K + b - 1) / b)
      fail("fp8 scale geometry mismatch on " + base);
    return f;
  }
  GlmQuantMatrix alloc_fp8(int64_t rows, int64_t cols) {
    const int b = cfg.fp8_block_size;  // the 128 x 128 grid (rs = cs = 7)
    GlmQuantMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_block_rows = b;
    q.scale_block_cols = b;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols)));
    q.scales = static_cast<const float*>(
        bump.alloc(static_cast<size_t>((rows + b - 1) / b) * static_cast<size_t>((cols + b - 1) / b) * 4));
    return q;
  }
  // Scale entries [col0, col0 + count) of source scale row `src_row` into
  // `dst`. THE conversion point for the scale-format gap (docs/dsv4_
  // kernel_port_spec.md §3): the checkpoint's F8_E8M0 bytes become the
  // F32 grid the 128 x 128 fp8 kernels read; the e8m0 bytes themselves
  // never reach the device.
  void convert_scales(const TensorInfo& ts, const Fp8Source& f, int64_t src_row, int64_t col0, int64_t count,
                      float* dst) const {
    const uint8_t* s = static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(src_row) * f.SC + col0;
    for (int64_t i = 0; i < count; ++i) dst[i] = e8m0_to_float(s[i]);
  }
  // Rows [row_start, +rows): row_start on a 128-row block boundary (every
  // row's scale row is then its own block row).
  GlmQuantMatrix load_fp8_rows(const std::string& base, int64_t row_start, int64_t rows) {
    const Fp8Source f = fp8_source(base);
    const int b = cfg.fp8_block_size;
    if (row_start % b != 0) fail("fp8 row slice of '" + base + "' must start on a 128-row block");
    check_range(base, row_start, rows, f.N);
    GlmQuantMatrix q = alloc_fp8(rows, f.K);
    const int64_t sr = (rows + b - 1) / b;
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(tw.data) + static_cast<size_t>(row_start) * f.K,
                  static_cast<size_t>(rows) * f.K);
      float* hs = bump.host(const_cast<float*>(q.scales));
      for (int64_t r = 0; r < sr; ++r) convert_scales(ts, f, row_start / b + r, 0, f.SC, hs + r * f.SC);
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(rows) * f.K);
    note_read(*f.s, static_cast<size_t>(sr) * f.SC);
    note_class(*f.w, static_cast<size_t>(rows) * f.K);
    note_class(*f.s, static_cast<size_t>(sr) * f.SC);
    return q;
  }
  GlmQuantMatrix load_fp8(const std::string& base) {
    const Fp8Source f = fp8_source(base);
    return load_fp8_rows(base, 0, f.N);
  }
  // Column ranges [(start, count)] of every row, packed in order: every
  // range on 128-column blocks (the scale columns follow exactly).
  GlmQuantMatrix load_fp8_col_ranges(const std::string& base,
                                     const std::vector<std::pair<int64_t, int64_t>>& ranges) {
    const Fp8Source f = fp8_source(base);
    const int b = cfg.fp8_block_size;
    int64_t cols = 0;
    for (const auto& [start, count] : ranges) {
      if (start % b != 0 || count % b != 0 || count <= 0)
        fail("fp8 column slice of '" + base + "' must be whole 128-column blocks");
      check_range(base, start, count, f.K);
      cols += count;
    }
    GlmQuantMatrix q = alloc_fp8(f.N, cols);
    const int64_t sc = cols / b;
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      const uint8_t* sp = static_cast<const uint8_t*>(tw.data);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      float* hs = bump.host(const_cast<float*>(q.scales));
      if (ranges.size() == 1 && ranges[0].first == 0 && cols == f.K) {
        std::memcpy(hp, sp, static_cast<size_t>(f.N) * f.K);
      } else {
        for (int64_t r = 0; r < f.N; ++r) {
          int64_t at = 0;
          for (const auto& [start, count] : ranges) {
            std::memcpy(hp + r * cols + at, sp + r * f.K + start, static_cast<size_t>(count));
            at += count;
          }
        }
      }
      for (int64_t r = 0; r < f.SR; ++r) {
        int64_t at = 0;
        for (const auto& [start, count] : ranges) {
          convert_scales(ts, f, r, start / b, count / b, hs + r * sc + at);
          at += count / b;
        }
      }
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(f.N) * static_cast<size_t>(cols));
    note_read(*f.s, static_cast<size_t>(f.SR) * static_cast<size_t>(sc));
    note_class(*f.w, static_cast<size_t>(f.N) * static_cast<size_t>(cols));
    note_class(*f.s, static_cast<size_t>(f.SR) * static_cast<size_t>(sc));
    return q;
  }
  GlmQuantMatrix load_fp8_cols(const std::string& base, int64_t col_start, int64_t cols) {
    return load_fp8_col_ranges(base, {{col_start, cols}});
  }

  // ---- MXFP4 pairs (e2m1 pairs + e8m0 per 32, kept as they are) --------
  struct Fp4Source {
    const Dsv4ExpectedTensor* w;
    const Dsv4ExpectedTensor* s;
    int64_t N, K;
  };
  Fp4Source fp4_source(const std::string& base) {
    Fp4Source f;
    f.w = &expected(base + ".weight");
    f.s = &expected(base + ".scale");
    if (f.w->role != Dsv4TensorRole::Fp4Payload || f.s->role != Dsv4TensorRole::Fp4Scale ||
        f.w->shape.size() != 2 || f.s->shape.size() != 2)
      fail("'" + base + "' is not an MXFP4 pair");
    f.N = f.w->shape[0];
    f.K = f.w->shape[1] * 2;
    fp4_check_cols(f.K, who.c_str(), kMxfp4Group);
    if (f.s->shape[0] != f.N || f.s->shape[1] != f.K / kMxfp4Group)
      fail("MXFP4 scale geometry mismatch on " + base);
    return f;
  }
  GlmFp4Matrix alloc_mxfp4(int64_t rows, int64_t cols) {
    GlmFp4Matrix q;
    q.rows = rows;
    q.cols = cols;
    q.scale_group = kMxfp4Group;
    q.global_scale = nullptr;
    q.payload = static_cast<const uint8_t*>(bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols / 2)));
    q.scales = static_cast<const uint8_t*>(
        bump.alloc(static_cast<size_t>(rows) * static_cast<size_t>(cols / kMxfp4Group)));
    return q;
  }
  GlmFp4Matrix load_mxfp4_rows(const std::string& base, int64_t row_start, int64_t rows) {
    const Fp4Source f = fp4_source(base);
    check_range(base, row_start, rows, f.N);
    const size_t pc = static_cast<size_t>(f.K / 2), sc = static_cast<size_t>(f.K / kMxfp4Group);
    GlmFp4Matrix q = alloc_mxfp4(rows, f.K);
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.payload)),
                  static_cast<const uint8_t*>(tw.data) + static_cast<size_t>(row_start) * pc,
                  static_cast<size_t>(rows) * pc);
      std::memcpy(bump.host(const_cast<uint8_t*>(q.scales)),
                  static_cast<const uint8_t*>(ts.data) + static_cast<size_t>(row_start) * sc,
                  static_cast<size_t>(rows) * sc);
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(rows) * pc);
    note_read(*f.s, static_cast<size_t>(rows) * sc);
    note_class(*f.w, static_cast<size_t>(rows) * pc);
    note_class(*f.s, static_cast<size_t>(rows) * sc);
    return q;
  }
  GlmFp4Matrix load_mxfp4_cols(const std::string& base, int64_t col_start, int64_t cols) {
    const Fp4Source f = fp4_source(base);
    check_mxfp4_slice(col_start, cols);  // 32-block start + whole 32-blocks
    check_range(base, col_start, cols, f.K);
    const size_t pc = static_cast<size_t>(cols / 2), pc_full = static_cast<size_t>(f.K / 2);
    const size_t sc = static_cast<size_t>(cols / kMxfp4Group), sc_full = static_cast<size_t>(f.K / kMxfp4Group);
    GlmFp4Matrix q = alloc_mxfp4(f.N, cols);
    if (copy) {
      const TensorInfo& tw = source(f.w->name);
      const TensorInfo& ts = source(f.s->name);
      const uint8_t* sp = static_cast<const uint8_t*>(tw.data);
      const uint8_t* ss = static_cast<const uint8_t*>(ts.data);
      uint8_t* hp = bump.host(const_cast<uint8_t*>(q.payload));
      uint8_t* hs = bump.host(const_cast<uint8_t*>(q.scales));
      if (cols == f.K) {
        std::memcpy(hp, sp, static_cast<size_t>(f.N) * pc);
        std::memcpy(hs, ss, static_cast<size_t>(f.N) * sc);
      } else {
        for (int64_t r = 0; r < f.N; ++r) {
          std::memcpy(hp + r * pc, sp + r * pc_full + col_start / 2, pc);
          std::memcpy(hs + r * sc, ss + r * sc_full + col_start / kMxfp4Group, sc);
        }
      }
      consumed(tw);
      consumed(ts);
    }
    note_read(*f.w, static_cast<size_t>(f.N) * pc);
    note_read(*f.s, static_cast<size_t>(f.N) * sc);
    note_class(*f.w, static_cast<size_t>(f.N) * pc);
    note_class(*f.s, static_cast<size_t>(f.N) * sc);
    return q;
  }

  // An F32 tensor rounded to bf16 at load (the mHC coefficient matrices,
  // the dsv41 plan D10: RNE, the kernel's bf16 form).
  uint16_t* load_f32_as_bf16(const std::string& name) {
    const Dsv4ExpectedTensor& e = expected(name);
    if (e.dtype != DType::F32) fail("'" + name + "' is not F32");
    const size_t n = e.numel();
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(n * 2));
    if (copy) {
      const TensorInfo& t = source(name);
      const float* src = static_cast<const float*>(t.data);
      uint16_t* h = bump.host(dst);
      for (size_t i = 0; i < n; ++i) {
        float v;
        std::memcpy(&v, src + i, 4);
        h[i] = float_to_bf16_bits(v);
      }
      consumed(t);
    }
    note_read(e, n * 4);
    note_class(e, n * 4);
    return dst;
  }

  // ---- the classes ---------------------------------------------------------
  void build_attention(const std::string& p, int layer) {
    Dsv4AttnResident& a = out.attn;
    a.local_heads = geo.local_heads;
    a.head_begin = geo.head_begin;
    a.local_groups = geo.local_groups;
    a.group_begin = geo.group_begin;
    const int64_t hd = cfg.head_dim, ol = cfg.o_lora_rank;
    a.q_norm = load_bf16(p + "q_norm.weight");
    a.kv_norm = load_bf16(p + "kv_norm.weight");
    a.attn_sink = load_f32_range(p + "attn_sink", geo.head_begin, geo.local_heads);
    a.wq_a = load_fp8(p + "wq_a");
    a.wkv = load_fp8(p + "wkv");
    a.wq_b = load_fp8_rows(p + "wq_b", static_cast<int64_t>(geo.head_begin) * hd,
                           static_cast<int64_t>(geo.local_heads) * hd);
    a.wo_a = load_fp8_rows(p + "wo_a", static_cast<int64_t>(geo.group_begin) * ol,
                           static_cast<int64_t>(geo.local_groups) * ol);
    a.wo_b = load_fp8_cols(p + "wo_b", static_cast<int64_t>(geo.group_begin) * ol,
                           static_cast<int64_t>(geo.local_groups) * ol);
    if (cfg.is_index_layer(layer)) {
      const std::string ip = p + "indexer.";
      a.idx_wq_b = load_fp8(ip + "wq_b");
      a.idx_wp = load_bf16(ip + "weights_proj.weight");
      const std::string cp = ip + "compressor.";
      a.idx_comp_ape = load_f32(cp + "ape");
      a.idx_comp_wkv = load_bf16(cp + "wkv.weight");
      a.idx_comp_wgate = load_bf16(cp + "wgate.weight");
      a.idx_comp_norm = load_bf16(cp + "norm.weight");
    }
    if (cfg.compress_ratio(layer) > 0) {
      const std::string cp = p + "compressor.";
      a.comp_ape = load_f32(cp + "ape");
      a.comp_wkv = load_bf16(cp + "wkv.weight");
      a.comp_wgate = load_bf16(cp + "wgate.weight");
      a.comp_norm = load_bf16(cp + "norm.weight");
    }
  }

  void build_moe(const std::string& p, int layer) {
    Dsv4MoeResident& m = out.moe;
    m.router = load_bf16(p + "gate.weight");
    // The hash layers (0..num_hash_layers-1) route through the mapped
    // tid2eid table (load_hash_tables) instead of a learned bias.
    if (!cfg.is_hash_layer(layer)) m.router_bias = load_f32(p + "gate.bias");
    const int64_t I = geo.local_inter, S = geo.local_shared_inter, r = rank;
    m.local_inter = I;
    m.local_shared_inter = S;
    // The draft stages reuse the backbone's expert set (no draft variant).
    const int E = cfg.n_routed_experts;
    m.n_experts = E;
    m.experts.resize(static_cast<size_t>(E) * 3);
    for (int e = 0; e < E; ++e) {
      const std::string ep = p + "experts." + std::to_string(e) + ".";
      GlmFp4Matrix* t = m.experts.data() + static_cast<size_t>(e) * 3;
      t[0] = load_mxfp4_rows(ep + "w1", r * I, I);  // gate
      t[1] = load_mxfp4_rows(ep + "w3", r * I, I);  // up
      t[2] = load_mxfp4_cols(ep + "w2", r * I, I);  // down
    }
    const std::string sp = p + "shared_experts.";
    m.shared[0] = load_fp8_rows(sp + "w1", r * S, S);
    m.shared[1] = load_fp8_rows(sp + "w3", r * S, S);
    m.shared[2] = load_fp8_cols(sp + "w2", r * S, S);
  }

  void build_mhc(const std::string& p) {
    Dsv4MhcResident& h = out.mhc;
    h.attn_fn = load_f32_as_bf16(p + "hc_attn_fn");
    h.attn_base = load_f32(p + "hc_attn_base");
    h.attn_scale = load_f32(p + "hc_attn_scale");
    h.ffn_fn = load_f32_as_bf16(p + "hc_ffn_fn");
    h.ffn_base = load_f32(p + "hc_ffn_base");
    h.ffn_scale = load_f32(p + "hc_ffn_scale");
  }

  void build_draft(const std::string& p, int layer) {
    Dsv4DraftResident& d = out.draft;
    const int stage = cfg.draft_stage(layer);
    if (stage == 0) {
      d.main_proj = load_fp8(p + "main_proj");
      d.main_norm = load_bf16(p + "main_norm.weight");
    }
    if (stage == cfg.num_draft_stages() - 1) {
      d.norm = load_bf16(p + "norm.weight");
      d.markov_w1 = load_bf16(p + "markov_head.markov_w1.weight");
      d.markov_w2 = load_bf16(p + "markov_head.markov_w2.weight");
      d.confidence = load_bf16_as_f32(p + "confidence_head.proj.weight");
      d.hc_head_fn = load_f32_as_bf16(p + "hc_head_fn");
      d.hc_head_base = load_f32(p + "hc_head_base");
      d.hc_head_scale = load_f32(p + "hc_head_scale");
    }
  }

  void build_layer(int layer) {
    if (layer < 0 || layer >= cfg.max_layer()) fail("layer index out of range: " + std::to_string(layer));
    const bool draft = cfg.is_draft(layer);
    const std::string p = dsv4_layer_prefix(cfg, layer);
    out.layer = layer;
    out.attn_norm = load_bf16(p + "attn_norm.weight");
    out.ffn_norm = load_bf16(p + "ffn_norm.weight");
    build_mhc(p);
    build_attention(p + "attn.", layer);
    build_moe(p + "ffn.", layer);
    if (draft) build_draft(p, layer);
  }

 private:
  std::map<Dsv4WeightClass, uint64_t> class_source_;
};

// ---------------------------------------------------------------------------

Dsv4LocalGeometry Dsv4LocalGeometry::from_config(const Dsv4TextConfig& cfg, int rank, int world,
                                                  Dsv4HeadSharding head) {
  if (world > 1) dsv4_tp_validate_geometry(cfg, rank, world);
  if (world < 1 || rank < 0 || rank >= world)
    throw std::invalid_argument("dsv4 loader: rank/world out of range");
  Dsv4LocalGeometry g;
  g.world = world;
  g.rank = rank;
  g.local_heads = cfg.num_attention_heads / world;
  g.head_begin = g.local_heads * rank;
  g.local_groups = cfg.o_groups / world;
  g.group_begin = g.local_groups * rank;
  g.local_inter = cfg.moe_intermediate_size / world;
  g.local_shared_inter = cfg.shared_expert_inter() / world;
  if (head == Dsv4HeadSharding::VocabSharded) {
    const auto [b, n] = lm_head_slice(cfg, rank, world);
    g.lm_vocab_begin = b;
    g.lm_vocab_count = n;
  } else {
    g.lm_vocab_begin = 0;
    g.lm_vocab_count = cfg.vocab_size;
  }
  if (world > 1 && g_embed_vocab_sharded) {
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

void Dsv4LoaderFamily::validate_binding(const Dsv4TextConfig& cfg, const PresentMap& present) {
  const Dsv4BindReport rep = dsv4_validate_text_binding(cfg, present);
  if (rep.ok()) return;
  std::string msg = "dsv4 loader: checkpoint binding failed: ";
  for (size_t i = 0; i < rep.errors.size() && i < 8; ++i) {
    if (i) msg += "; ";
    msg += rep.errors[i];
  }
  throw std::runtime_error(msg);
}

// The digest covers what a rank actually reads verbatim: the replicated
// set minus the hash tables (mapped, never read whole).
bool Dsv4LoaderFamily::digest_included(const Dsv4ExpectedTensor& e) {
  if (e.role == Dsv4TensorRole::HashTable) return false;
  return is_replicated(e);
}

std::map<Dsv4WeightClass, uint64_t> Dsv4LoaderFamily::class_source_plan(const Dsv4TextConfig& cfg, int layer,
                                                                         int rank, int world) {
  const Dsv4LocalGeometry geo =
      Dsv4LocalGeometry::from_config(cfg, rank, world, LoaderHeadSharding::Full);
  LayerBump bump;
  bump.counting = true;
  bump.capacity = SIZE_MAX;
  const std::vector<Dsv4ExpectedTensor> table = dsv4_expected_layer_tensors(cfg, layer);
  std::unordered_map<std::string, const Dsv4ExpectedTensor*> by_name;
  for (const auto& e : table) by_name.emplace(e.name, &e);
  Dsv4LayerResident scratch{};
  std::vector<DequantJob> jobs;
  std::vector<PackJob> packs;
  LoaderTensorMap no_tensors;
  Builder ctx(cfg, geo, table, by_name, bump, scratch, no_tensors, jobs, packs, false);
  ctx.build_layer(layer);
  return ctx.class_source();
}

size_t Dsv4LoaderFamily::globals_bytes(const Dsv4TextConfig& cfg, int rank, int world,
                                       LoaderHeadSharding head) {
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const Dsv4LocalGeometry geo = Dsv4LocalGeometry::from_config(cfg, rank, world, head);
  size_t b = 0;
  b += align_up_256(static_cast<size_t>(geo.embed_vocab_count) * H * 2);
  b += align_up_256(H * 2);
  b += align_up_256(static_cast<size_t>(geo.lm_vocab_count) * H * 2);
  b += align_up_256(static_cast<size_t>(cfg.hc_mult) * static_cast<size_t>(cfg.hc_dim()) * 2);
  b += align_up_256(static_cast<size_t>(cfg.hc_mult) * 4);
  b += align_up_256(4);
  return b;
}

void Dsv4LoaderFamily::build_globals(const Dsv4TextConfig& cfg, const Dsv4LocalGeometry& geo,
                                     const LoaderTensorMap& tensors, LayerBump& bump,
                                     Dsv4GlobalsResident& out, uint64_t& source_bytes,
                                     uint64_t& verbatim_bytes, LoaderHeadSharding head) {
  auto lookup = [&](const std::string& name) -> const TensorInfo& {
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second)
      throw std::runtime_error("dsv4 loader: global tensor missing: " + name);
    return *it->second;
  };
  const size_t H = static_cast<size_t>(cfg.hidden_size);
  const size_t row_bytes = H * 2;
  {
    const TensorInfo& e = lookup("embed.weight");
    const int ebegin = geo.embed_vocab_begin, ecount = geo.embed_vocab_count;
    if (ebegin < 0 || ecount <= 0 || static_cast<size_t>(ebegin + ecount) * row_bytes > e.nbytes())
      throw std::runtime_error("dsv4 loader: the embedding slice does not fit the table");
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(ecount) * row_bytes));
    std::memcpy(bump.host(dst), static_cast<const uint8_t*>(e.data) + static_cast<size_t>(ebegin) * row_bytes,
                static_cast<size_t>(ecount) * row_bytes);
    source_bytes += static_cast<size_t>(ecount) * row_bytes;
    if (ecount == cfg.vocab_size) verbatim_bytes += e.nbytes();
    out.embed = dst;
    out.embed_vocab_begin = ebegin;
    out.embed_vocab_count = ecount;
  }
  {
    const TensorInfo& t = lookup("norm.weight");
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(t.nbytes()));
    std::memcpy(bump.host(dst), t.data, t.nbytes());
    source_bytes += t.nbytes();
    verbatim_bytes += t.nbytes();
    out.final_norm = dst;
  }
  {
    const TensorInfo& t = lookup("head.weight");
    const int begin = geo.lm_vocab_begin, count = geo.lm_vocab_count;
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(static_cast<size_t>(count) * row_bytes));
    std::memcpy(bump.host(dst), static_cast<const uint8_t*>(t.data) + static_cast<size_t>(begin) * row_bytes,
                static_cast<size_t>(count) * row_bytes);
    source_bytes += static_cast<size_t>(count) * row_bytes;
    if (head == LoaderHeadSharding::Full) verbatim_bytes += static_cast<size_t>(count) * row_bytes;
    out.lm_head = dst;
    out.lm_vocab_begin = begin;
    out.lm_vocab_count = count;
  }
  // The model-level mHC head: fn rounded fp32 -> bf16 at load (the
  // per-layer mHC pattern), base / scale fp32 verbatim.
  {
    const TensorInfo& t = lookup("hc_head_fn");
    const size_t n = t.nbytes() / 4;
    uint16_t* dst = static_cast<uint16_t*>(bump.alloc(n * 2));
    const float* src = static_cast<const float*>(t.data);
    uint16_t* h = bump.host(dst);
    for (size_t i = 0; i < n; ++i) h[i] = float_to_bf16_bits(src[i]);
    source_bytes += t.nbytes();
    verbatim_bytes += t.nbytes();
    out.hc_head_fn = dst;
  }
  {
    const TensorInfo& t = lookup("hc_head_base");
    float* dst = static_cast<float*>(bump.alloc(t.nbytes()));
    std::memcpy(bump.host(dst), t.data, t.nbytes());
    source_bytes += t.nbytes();
    verbatim_bytes += t.nbytes();
    out.hc_head_base = dst;
  }
  {
    const TensorInfo& t = lookup("hc_head_scale");
    float* dst = static_cast<float*>(bump.alloc(t.nbytes()));
    std::memcpy(bump.host(dst), t.data, t.nbytes());
    source_bytes += t.nbytes();
    verbatim_bytes += t.nbytes();
    out.hc_head_scale = dst;
  }
}

template class ResidentLayerStream<Dsv4LoaderFamily>;

// ---- the hash table mapping ---------------------------------------------------

Dsv4HashTableMmap::Dsv4HashTableMmap(const std::string& path, uint64_t data_begin, int64_t rows, int topk)
    : data_begin_(data_begin), rows_(rows), topk_(topk) {
  if (rows <= 0 || topk <= 0)
    throw std::invalid_argument("Dsv4HashTableMmap: bad geometry");
  fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd_ < 0) throw std::runtime_error("Dsv4HashTableMmap: cannot open " + path);
  struct stat st {};
  if (fstat(fd_, &st) != 0) {
    ::close(fd_);
    throw std::runtime_error("Dsv4HashTableMmap: fstat " + path);
  }
  len_ = static_cast<size_t>(st.st_size);
  const uint64_t pend = data_begin + static_cast<uint64_t>(rows) * static_cast<uint64_t>(topk) * 8;
  if (pend > len_) {
    ::close(fd_);
    throw std::runtime_error("Dsv4HashTableMmap: the table runs past the end of " + path);
  }
  void* map = mmap(nullptr, len_, PROT_READ, MAP_SHARED, fd_, 0);
  if (map == MAP_FAILED) {
    ::close(fd_);
    throw std::runtime_error("Dsv4HashTableMmap: mmap " + path);
  }
  base_ = static_cast<uint8_t*>(map);
  // Rows are read one at a time from anywhere in the table: no readahead.
  madvise(base_, len_, MADV_RANDOM);
}

Dsv4HashTableMmap::~Dsv4HashTableMmap() {
  if (base_) munmap(base_, len_);
  if (fd_ >= 0) ::close(fd_);
}

const int64_t* Dsv4HashTableMmap::row(int64_t token) const {
  if (token < 0 || token >= rows_)
    throw std::out_of_range("Dsv4HashTableMmap: token " + std::to_string(token) + " outside the table");
  return reinterpret_cast<const int64_t*>(base_ + data_begin_ + static_cast<size_t>(token) * static_cast<size_t>(topk_) * 8);
}

void Dsv4HashTableMmap::gather(const int32_t* ids, int n, int64_t* dst) const {
  if (n <= 0) return;
  auto copy_range = [&](int64_t lo, int64_t hi) {
    for (int64_t t = lo; t < hi; ++t) {
      const int64_t* src = row(ids[t]);
      std::memcpy(dst + t * topk_, src, static_cast<size_t>(topk_) * 8);
    }
  };
  // Every row's pages asked for up front (the faults in flight together),
  // then the copies: a decode step's few dozen rows in one thread, a
  // prefill chunk's thousands across a few (the dsv41 Engram gather's
  // pattern).
  for (int64_t t = 0; t < n; ++t) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(row(ids[t]));
    const uintptr_t page = reinterpret_cast<uintptr_t>(p) & ~uintptr_t{4095};
    madvise(reinterpret_cast<void*>(page), 4096 + static_cast<size_t>(topk_) * 8, MADV_WILLNEED);
  }
  constexpr int64_t kPerThread = 256;
  if (n <= kPerThread) {
    copy_range(0, n);
    return;
  }
  const int threads = static_cast<int>(std::min<int64_t>(16, (n + kPerThread - 1) / kPerThread));
  const int64_t span = (n + threads - 1) / threads;
  std::vector<std::thread> pool;
  for (int w = 0; w < threads; ++w) {
    const int64_t lo = w * span, hi = std::min(static_cast<int64_t>(n), lo + span);
    if (lo < hi) pool.emplace_back(copy_range, lo, hi);
  }
  for (auto& th : pool) th.join();
}

// ---------------------------------------------------------------------------

void Dsv4LayerStream::set_embed_vocab_sharded(bool on) { g_embed_vocab_sharded = on; }
bool Dsv4LayerStream::embed_vocab_sharded() { return g_embed_vocab_sharded; }

Dsv4LayerStream::Dsv4LayerStream(const Dsv4TextConfig& cfg, const std::string& checkpoint_dir, int rank,
                                 int world, Dsv4Residency residency, Dsv4HeadSharding head, bool resident_mtp)
    : ResidentLayerStream<Dsv4LoaderFamily>(cfg, checkpoint_dir, rank, world, residency, head,
                                             resident_mtp) {
  open_resident_image();
}

void Dsv4LayerStream::set_resident_image_dir(const std::string& dir) { resident_image_dir_storage() = dir; }
const std::string& Dsv4LayerStream::resident_image_dir() { return resident_image_dir_storage(); }

const Dsv4HashTables& Dsv4LayerStream::load_hash_tables() {
  if (hash_loaded_ || cfg_.num_hash_layers == 0) return hash_;
  if (sources_released_)
    throw std::runtime_error("dsv4 loader: load_hash_tables after the checkpoint sources were released");
  hash_.vocab = cfg_.vocab_size;
  hash_.topk = cfg_.num_experts_per_tok;
  hash_.tables.clear();
  for (int l = 0; l < cfg_.num_hash_layers; ++l) {
    const std::string name = dsv4_layer_prefix(cfg_, l) + "ffn.gate.tid2eid";
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !it->second || !it->second->owner)
      throw std::runtime_error("dsv4 loader: hash table tensor not in the checkpoint: " + name);
    const TensorInfo& t = *it->second;
    if (t.shape.size() != 2 || t.shape[0] != cfg_.vocab_size || t.shape[1] != cfg_.num_experts_per_tok)
      throw std::runtime_error("dsv4 loader: hash table geometry mismatch on " + t.name);
    hash_.tables.push_back(std::make_unique<Dsv4HashTableMmap>(t.owner->path(), t.data_begin, t.shape[0],
                                                                cfg_.num_experts_per_tok));
  }
  hash_loaded_ = true;
  DGPP_LOG_INFO("dsv4 loader: rank {} hash tables mmap'ed from the checkpoint ({} tables of {} x {} I64, "
                "{:.2f} GiB mapped; the rows are token ids — no head dimension, whole on every rank)",
                rank_, hash_.tables.size(), cfg_.vocab_size, cfg_.num_experts_per_tok,
                hash_.mapped_bytes() / (1024.0 * 1024.0 * 1024.0));
  return hash_;
}

}  // namespace dgpp
