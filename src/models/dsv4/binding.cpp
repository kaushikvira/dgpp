#include "models/dsv4/binding.hpp"

#include <format>
#include <stdexcept>

namespace dgpp {
namespace {

using TensorList = std::vector<Dsv4ExpectedTensor>;

void add(TensorList& out, std::string name, DType dtype, std::vector<int64_t> shape,
         Dsv4WeightClass cls, int layer, int expert = -1,
         Dsv4TensorRole role = Dsv4TensorRole::Plain) {
  out.push_back(Dsv4ExpectedTensor{std::move(name), dtype, std::move(shape), cls, layer, expert, role});
}

void add_bf16(TensorList& out, const std::string& name, std::vector<int64_t> shape,
              Dsv4WeightClass cls, int layer) {
  add(out, name, DType::BF16, std::move(shape), cls, layer);
}
void add_f32(TensorList& out, const std::string& name, std::vector<int64_t> shape,
             Dsv4WeightClass cls, int layer) {
  add(out, name, DType::F32, std::move(shape), cls, layer);
}

int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

// One fp8 [rows, cols] matrix: the e4m3 payload and the e8m0 block scales.
void add_fp8(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
             Dsv4WeightClass cls, int layer, int block, int expert = -1) {
  add(out, base + ".weight", DType::F8_E4M3, {rows, cols}, cls, layer, expert, Dsv4TensorRole::Fp8Payload);
  add(out, base + ".scale", DType::F8_E8M0, {ceil_div(rows, block), ceil_div(cols, block)}, cls, layer, expert,
      Dsv4TensorRole::Fp8Scale);
}

// One MXFP4 [rows, cols] matrix: the packed nibbles and the per-32 e8m0 scales.
void add_fp4(TensorList& out, const std::string& base, int64_t rows, int64_t cols,
             Dsv4WeightClass cls, int layer, int block, int expert) {
  if (cols % block != 0 || cols % 2 != 0)
    throw std::invalid_argument("dsv4 binding: MXFP4 K must be a multiple of the block on " + base);
  add(out, base + ".weight", DType::I8, {rows, cols / 2}, cls, layer, expert, Dsv4TensorRole::Fp4Payload);
  add(out, base + ".scale", DType::F8_E8M0, {rows, cols / block}, cls, layer, expert, Dsv4TensorRole::Fp4Scale);
}

// The main KV compressor (inference/model.py Compressor): the overlapping
// ratio-4 window carries coff 2 copies of the latent width, the plain
// ratio-128 one coff 1. wkv / wgate are BF16 in the checkpoint.
void expect_compressor(TensorList& out, const std::string& p, const Dsv4TextConfig& cfg, int layer) {
  const int r = cfg.compress_ratio(layer);
  if (r == 0) return;
  const int64_t H = cfg.hidden_size, hd = cfg.head_dim, coff = cfg.compressor_coff(layer);
  const std::string cp = p + "compressor.";
  add_f32(out, cp + "ape", {r, coff * hd}, Dsv4WeightClass::Compressor, layer);
  add_bf16(out, cp + "wkv.weight", {coff * hd, H}, Dsv4WeightClass::Compressor, layer);
  add_bf16(out, cp + "wgate.weight", {coff * hd, H}, Dsv4WeightClass::Compressor, layer);
  add_bf16(out, cp + "norm.weight", {hd}, Dsv4WeightClass::LayerNorm, layer);
}

// The sparse-selection indexer (inference/model.py Indexer), ratio 4 only:
// its own rotated compressor over the 128-wide index latent (coff 2), the
// fp8 query projection and the BF16 weights projection.
void expect_indexer(TensorList& out, const std::string& p, const Dsv4TextConfig& cfg, int layer) {
  if (!cfg.is_index_layer(layer)) return;
  const int64_t H = cfg.hidden_size, ql = cfg.q_lora_rank;
  const int64_t ih = cfg.index_n_heads, id = cfg.index_head_dim;
  const std::string ip = p + "indexer.";
  add_fp8(out, ip + "wq_b", ih * id, ql, Dsv4WeightClass::Indexer, layer, cfg.fp8_block_size);
  add_bf16(out, ip + "weights_proj.weight", {ih, H}, Dsv4WeightClass::Indexer, layer);
  const std::string cp = ip + "compressor.";
  add_f32(out, cp + "ape", {4, 2 * id}, Dsv4WeightClass::Compressor, layer);
  add_bf16(out, cp + "wkv.weight", {2 * id, H}, Dsv4WeightClass::Compressor, layer);
  add_bf16(out, cp + "wgate.weight", {2 * id, H}, Dsv4WeightClass::Compressor, layer);
  add_bf16(out, cp + "norm.weight", {id}, Dsv4WeightClass::LayerNorm, layer);
}

void expect_attention(TensorList& out, const std::string& p, const Dsv4TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size;
  const int64_t heads = cfg.num_attention_heads, hd = cfg.head_dim;
  const int64_t ql = cfg.q_lora_rank, ol = cfg.o_lora_rank, og = cfg.o_groups;
  const int b = cfg.fp8_block_size;
  const Dsv4WeightClass c = Dsv4WeightClass::Attention;
  add_bf16(out, p + "q_norm.weight", {ql}, Dsv4WeightClass::LayerNorm, layer);
  add_bf16(out, p + "kv_norm.weight", {hd}, Dsv4WeightClass::LayerNorm, layer);
  add_f32(out, p + "attn_sink", {heads}, c, layer);
  add_fp8(out, p + "wq_a", ql, H, c, layer, b);
  add_fp8(out, p + "wq_b", heads * hd, ql, c, layer, b);
  add_fp8(out, p + "wkv", hd, H, c, layer, b);
  // wo_a is block-diagonal over the output groups: group g maps its
  // heads' (heads/og x hd) outputs to ol rows — stored as one
  // [og * ol, heads * hd / og] matrix.
  add_fp8(out, p + "wo_a", og * ol, heads * hd / og, c, layer, b);
  add_fp8(out, p + "wo_b", H, og * ol, c, layer, b);
  expect_compressor(out, p, cfg, layer);
  expect_indexer(out, p, cfg, layer);
}

void expect_moe(TensorList& out, const std::string& p, const Dsv4TextConfig& cfg, int layer) {
  const int64_t H = cfg.hidden_size, I = cfg.moe_intermediate_size;
  const int64_t S = cfg.shared_expert_inter();
  const int64_t E = cfg.n_routed_experts;
  const int topk = cfg.num_experts_per_tok;
  add_bf16(out, p + "gate.weight", {E, H}, Dsv4WeightClass::Router, layer);
  if (cfg.is_hash_layer(layer)) {
    // Hash (engram) layers: the gate routes by token id through the
    // tid2eid table instead of a learned bias.
    add(out, p + "gate.tid2eid", DType::I64, {cfg.vocab_size, topk}, Dsv4WeightClass::Router, layer, -1,
        Dsv4TensorRole::HashTable);
  } else {
    add_f32(out, p + "gate.bias", {E}, Dsv4WeightClass::Router, layer);
  }
  for (int e = 0; e < E; ++e) {
    const std::string ep = p + "experts." + std::to_string(e) + ".";
    add_fp4(out, ep + "w1", I, H, Dsv4WeightClass::RoutedExpert, layer, cfg.fp4_block_size, e);
    add_fp4(out, ep + "w2", H, I, Dsv4WeightClass::RoutedExpert, layer, cfg.fp4_block_size, e);
    add_fp4(out, ep + "w3", I, H, Dsv4WeightClass::RoutedExpert, layer, cfg.fp4_block_size, e);
  }
  const std::string sp = p + "shared_experts.";
  add_fp8(out, sp + "w1", S, H, Dsv4WeightClass::SharedExpert, layer, cfg.fp8_block_size);
  add_fp8(out, sp + "w2", H, S, Dsv4WeightClass::SharedExpert, layer, cfg.fp8_block_size);
  add_fp8(out, sp + "w3", S, H, Dsv4WeightClass::SharedExpert, layer, cfg.fp8_block_size);
}

void expect_mhc(TensorList& out, const std::string& p, const Dsv4TextConfig& cfg, int layer) {
  const int64_t rows = cfg.hc_coeff_rows(), width = cfg.hc_dim();
  for (const char* site : {"attn", "ffn"}) {
    const std::string s = p + "hc_" + site + "_";
    add_f32(out, s + "fn", {rows, width}, Dsv4WeightClass::Mhc, layer);
    add_f32(out, s + "base", {rows}, Dsv4WeightClass::Mhc, layer);
    add_f32(out, s + "scale", {3}, Dsv4WeightClass::Mhc, layer);
  }
}

void expect_draft_extras(TensorList& out, const std::string& p, const Dsv4TextConfig& cfg, int layer) {
  const int stage = cfg.draft_stage(layer);
  const int64_t H = cfg.hidden_size, V = cfg.vocab_size, R = cfg.dspark_markov_rank;
  const Dsv4WeightClass c = Dsv4WeightClass::Draft;
  if (stage == 0) {
    add_fp8(out, p + "main_proj", H, H * static_cast<int64_t>(cfg.dspark_target_layer_ids.size()), c, layer,
            cfg.fp8_block_size);
    add_bf16(out, p + "main_norm.weight", {H}, Dsv4WeightClass::LayerNorm, layer);
  }
  if (stage == cfg.num_draft_stages() - 1) {
    add_bf16(out, p + "norm.weight", {H}, Dsv4WeightClass::LayerNorm, layer);
    add_bf16(out, p + "markov_head.markov_w1.weight", {V, R}, c, layer);
    add_bf16(out, p + "markov_head.markov_w2.weight", {V, R}, c, layer);
    add_bf16(out, p + "confidence_head.proj.weight", {1, H + R}, c, layer);
    add_f32(out, p + "hc_head_fn", {cfg.hc_mult, cfg.hc_dim()}, Dsv4WeightClass::Mhc, layer);
    add_f32(out, p + "hc_head_base", {cfg.hc_mult}, Dsv4WeightClass::Mhc, layer);
    add_f32(out, p + "hc_head_scale", {1}, Dsv4WeightClass::Mhc, layer);
  }
}

}  // namespace

std::string dsv4_layer_prefix(const Dsv4TextConfig& cfg, int layer) {
  if (cfg.is_draft(layer)) return "mtp." + std::to_string(cfg.draft_stage(layer)) + ".";
  return "layers." + std::to_string(layer) + ".";
}

std::vector<Dsv4ExpectedTensor> dsv4_expected_layer_tensors(const Dsv4TextConfig& cfg, int layer) {
  if (layer < 0 || layer >= cfg.max_layer())
    throw std::invalid_argument("dsv4_expected_layer_tensors: layer out of range");
  const std::string p = dsv4_layer_prefix(cfg, layer);
  const int64_t H = cfg.hidden_size;
  TensorList out;
  if (cfg.is_draft(layer)) expect_draft_extras(out, p, cfg, layer);
  add_bf16(out, p + "attn_norm.weight", {H}, Dsv4WeightClass::LayerNorm, layer);
  add_bf16(out, p + "ffn_norm.weight", {H}, Dsv4WeightClass::LayerNorm, layer);
  expect_mhc(out, p, cfg, layer);
  expect_attention(out, p + "attn.", cfg, layer);
  expect_moe(out, p + "ffn.", cfg, layer);
  return out;
}

std::vector<Dsv4ExpectedTensor> dsv4_expected_global_tensors(const Dsv4TextConfig& cfg) {
  TensorList out;
  const int64_t H = cfg.hidden_size;
  add_bf16(out, "embed.weight", {cfg.vocab_size, H}, Dsv4WeightClass::Embed, -1);
  add_bf16(out, "norm.weight", {H}, Dsv4WeightClass::FinalNorm, -1);
  add_bf16(out, "head.weight", {cfg.vocab_size, H}, Dsv4WeightClass::LmHead, -1);
  // The model-level mHC head (inference/model.py Transformer): the
  // backbone's final hc_mix over the 4-stream state.
  add_f32(out, "hc_head_fn", {cfg.hc_mult, cfg.hc_dim()}, Dsv4WeightClass::Mhc, -1);
  add_f32(out, "hc_head_base", {cfg.hc_mult}, Dsv4WeightClass::Mhc, -1);
  add_f32(out, "hc_head_scale", {1}, Dsv4WeightClass::Mhc, -1);
  return out;
}

std::vector<Dsv4ExpectedTensor> dsv4_expected_text_tensors(const Dsv4TextConfig& cfg) {
  TensorList out = dsv4_expected_global_tensors(cfg);
  for (int l = 0; l < cfg.max_layer(); ++l) {
    TensorList layer = dsv4_expected_layer_tensors(cfg, l);
    out.insert(out.end(), std::make_move_iterator(layer.begin()), std::make_move_iterator(layer.end()));
  }
  return out;
}

Dsv4BindReport dsv4_validate_text_binding(
    const Dsv4TextConfig& cfg, const std::unordered_map<std::string, Dsv4TensorDesc>& present,
    size_t max_errors) {
  Dsv4BindReport rep;
  const auto expected = dsv4_expected_text_tensors(cfg);
  rep.expected = expected.size();
  auto push_error = [&](std::string msg) {
    if (rep.errors.size() < max_errors) rep.errors.push_back(std::move(msg));
  };
  auto shape_str = [](const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(s[i]);
    }
    return out + "]";
  };
  std::unordered_map<std::string, int8_t> consumed;
  consumed.reserve(present.size());
  for (const auto& e : expected) {
    auto it = present.find(e.name);
    if (it == present.end()) {
      ++rep.missing;
      push_error(std::format("missing tensor '{}'", e.name));
      continue;
    }
    consumed.emplace(e.name, 1);
    if (it->second.dtype != e.dtype) {
      ++rep.dtype_mismatch;
      push_error(std::format("'{}' dtype {} != expected {}", e.name, dtype_name(it->second.dtype),
                             dtype_name(e.dtype)));
      continue;
    }
    if (it->second.shape != e.shape) {
      ++rep.shape_mismatch;
      push_error(std::format("'{}' shape {} != expected {}", e.name, shape_str(it->second.shape),
                             shape_str(e.shape)));
      continue;
    }
    ++rep.matched;
    switch (e.role) {
      case Dsv4TensorRole::Fp4Payload: ++rep.fp4_matrices; break;
      case Dsv4TensorRole::Fp8Payload: ++rep.fp8_matrices; break;
      case Dsv4TensorRole::HashTable: ++rep.hash_tables; break;
      default: break;
    }
  }
  // A layer past the config's stack belongs to a truncated diagnostic
  // stack, not to a binding error: backbone layers beyond the backbone,
  // or draft stages beyond the shipped count.
  const auto beyond_stack = [&](const std::string& name) {
    auto check = [&](std::string_view prefix, int limit) {
      if (name.compare(0, prefix.size(), prefix) != 0) return false;
      size_t i = prefix.size();
      int layer = 0;
      bool digits = false;
      while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        layer = layer * 10 + (name[i] - '0');
        ++i;
        digits = true;
      }
      return digits && i < name.size() && name[i] == '.' && layer >= limit;
    };
    return check("layers.", cfg.num_hidden_layers) || check("mtp.", cfg.num_draft_stages());
  };
  for (const auto& [name, desc] : present) {
    if (consumed.count(name)) continue;
    if (beyond_stack(name)) {
      ++rep.beyond_stack;
      continue;
    }
    ++rep.unexpected;
    push_error(std::format("unexpected tensor '{}'", name));
  }
  return rep;
}

void dsv4_tp_validate_geometry(const Dsv4TextConfig& cfg, int rank, int world) {
  auto fail = [](const std::string& what) { throw std::invalid_argument("dsv4 tp geometry: " + what); };
  if (world < 1 || rank < 0 || rank >= world) fail("rank/world out of range");
  if (cfg.o_groups % world != 0)
    fail("o_groups must divide by world (a rank holds whole output groups of wo_a)");
  if (cfg.num_attention_heads % world != 0) fail("num_attention_heads must divide by world");
  if (cfg.index_n_heads % world != 0) fail("index_n_heads must divide by world");
  if (cfg.vocab_size % world != 0) fail("vocab_size must divide by world (the head is vocab-sharded)");
  if (cfg.n_routed_experts % world != 0) fail("n_routed_experts must divide by world (experts are sharded)");
  for (const int inter : {cfg.moe_intermediate_size, cfg.shared_expert_inter()}) {
    if (inter % world != 0) fail("an intermediate size must divide by world");
    if ((inter / world) % 32 != 0)
      fail("an intermediate slice must be a multiple of 32 (the MXFP4 block of the expert's column slice)");
  }
  if (cfg.shared_expert_inter() / world % cfg.fp8_block_size != 0)
    fail("the shared expert's intermediate slice must be a multiple of the fp8 block");
  if (((static_cast<int64_t>(cfg.o_groups) * cfg.o_lora_rank) / world) % cfg.fp8_block_size != 0)
    fail("the wo_b input slice must be a multiple of the fp8 block");
}

}  // namespace dgpp
