// dsv4_load_check: loads a DeepSeek-V4-Flash checkpoint through the
// resident loader at one rank's TP geometry and reports per-layer bytes
// and times, the byte formulas against actual usage, the per-class
// source plan against the G0 census closed form (docs/checkpoint_
// budget_dsv4.md, tools/dsv4_census.py), the resident digest, the
// mapped tid2eid hash tables, and the refusal behavior on bad
// geometry / binding. No bus, no forward: one process per rank.
//
//   dsv4_load_check --model ORG/NAME | --checkpoint-dir DIR
//                   [--world W] [--rank R] [--streaming] [--mtp]
//                   [--layers N] [--from L] [--image-dir DIR|off]
//                   [--hash] [--refuse]
//
// The census gate (per-class read bytes vs the closed form) runs only
// on a full load of every layer and draft stage at world 1 or 2 with
// the census's geometry (43 + 3, 256 x 2048, hidden 4096, vocab
// 129280). Two documented regroupings vs the census's name-based
// classes: the loader holds the whole DSpark Markov head on every rank
// (a layer's bytes cannot depend on the head sharding — the stream's
// counting pass has none) where the census slices markov_w2 by
// vocabulary rows at world 2, and the loader counts the indexer's own
// compressor under Compressor where the census's name-based "indexer"
// class includes it. Both are reported below.
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <map>
#include <string>
#include <vector>

#include "common/log.hpp"
#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "models/dsv4/config.hpp"
#include "models/dsv4/loader.hpp"

namespace {

using dgpp::Dsv4LayerStream;
using dgpp::Dsv4TextConfig;
using dgpp::Dsv4WeightClass;

const char* class_name(Dsv4WeightClass c) {
  switch (c) {
    case Dsv4WeightClass::Embed: return "embed";
    case Dsv4WeightClass::LmHead: return "lm_head";
    case Dsv4WeightClass::FinalNorm: return "final_norm";
    case Dsv4WeightClass::LayerNorm: return "norm";
    case Dsv4WeightClass::Attention: return "attention";
    case Dsv4WeightClass::Indexer: return "indexer";
    case Dsv4WeightClass::Compressor: return "compressor";
    case Dsv4WeightClass::Router: return "router";
    case Dsv4WeightClass::SharedExpert: return "shared_expert";
    case Dsv4WeightClass::RoutedExpert: return "routed_expert";
    case Dsv4WeightClass::Mhc: return "mhc";
    case Dsv4WeightClass::Draft: return "draft";
  }
  return "?";
}

// The G0 census (docs/checkpoint_budget_dsv4.md): per-class GiB at
// world 1 / 2 for /data/models/DeepSeek-V4-Flash-0731.
struct CensusClass {
  const char* name;
  double w1, w2;
};
const CensusClass kCensus[] = {
    {"routed_expert", 137.062, 68.531},
    {"draft_expert", 9.562, 4.781},
    {"attention_sharded", 4.031, 2.016},
    {"shared_expert", 1.008, 0.504},
    {"embed", 0.986, 0.986},
    {"lm_head", 0.986, 0.493},
    {"compressor", 0.490, 0.490},
    {"draft_attention_sharded", 0.281, 0.141},
    {"indexer", 0.256, 0.256},
    {"attention_replicated", 0.252, 0.252},
    {"mhc", 0.126, 0.126},
    {"router", 0.084, 0.084},
    {"draft_shared_expert", 0.070, 0.035},
    {"draft_markov_embed", 0.062, 0.062},
    {"draft_head", 0.062, 0.031},
    {"draft_proj", 0.047, 0.047},
    {"draft_attention_replicated", 0.018, 0.018},
    {"hash_table", 0.017, 0.017},
    {"draft_mhc", 0.009, 0.009},
    {"draft_router", 0.006, 0.006},
    {"norm", 0.001, 0.001},
    {"draft", 0.000, 0.000},
    {"draft_norm", 0.000, 0.000},
};
double census_gib(const std::string& name, int world) {
  for (const auto& c : kCensus)
    if (name == c.name) return world == 1 ? c.w1 : c.w2;
  throw std::runtime_error("census: unknown class " + name);
}

// The census geometry the table was generated for (the G0 checks).
bool census_compatible(const Dsv4TextConfig& cfg) {
  return cfg.num_hidden_layers == 43 && cfg.num_draft_stages() == 3 && cfg.n_routed_experts == 256 &&
         cfg.moe_intermediate_size == 2048 && cfg.hidden_size == 4096 && cfg.vocab_size == 129280 &&
         cfg.num_attention_heads == 64 && cfg.o_groups == 8 && cfg.num_hash_layers == 3 &&
         cfg.fp8_block_size == 128 && cfg.fp4_block_size == 32 && cfg.hc_mult == 4;
}

// The per-class closed form the census implies for the loader's
// (coarser) class groups, at the given world: the census's class sums
// with the loader's documented deltas. Two regroupings: (1) the census
// slices the DSpark Markov head's markov_w2 by vocabulary rows at
// world 2, while the loader holds the whole Markov head on every rank
// (a layer's bytes cannot depend on the head sharding — the stream's
// counting pass has none); (2) the census's name-based "indexer" class
// includes the indexer's own compressor, which the loader counts under
// Compressor (its resident view holds it there) — replicated, so the
// move is world-independent.
std::map<Dsv4WeightClass, double> census_expected(const Dsv4TextConfig& cfg, int world, double kGiB) {
  std::map<Dsv4WeightClass, double> e;
  const auto add = [&e](Dsv4WeightClass c, double g) { e[c] += g; };
  add(Dsv4WeightClass::RoutedExpert, census_gib("routed_expert", world) + census_gib("draft_expert", world));
  add(Dsv4WeightClass::SharedExpert, census_gib("shared_expert", world) + census_gib("draft_shared_expert", world));
  add(Dsv4WeightClass::Attention,
      census_gib("attention_sharded", world) + census_gib("attention_replicated", world) +
          census_gib("draft_attention_sharded", world) + census_gib("draft_attention_replicated", world));
  add(Dsv4WeightClass::Compressor, census_gib("compressor", world));
  add(Dsv4WeightClass::Router, census_gib("router", world) + census_gib("draft_router", world));
  add(Dsv4WeightClass::Mhc, census_gib("mhc", world) + census_gib("draft_mhc", world));
  add(Dsv4WeightClass::LayerNorm,
      census_gib("norm", world) + census_gib("draft", world) + census_gib("draft_norm", world));
  add(Dsv4WeightClass::Embed, census_gib("embed", world));
  add(Dsv4WeightClass::LmHead, census_gib("lm_head", world));
  add(Dsv4WeightClass::FinalNorm, 0.0);  // 8 KiB, inside the census's "norm" row
  // The loader holds the whole Markov head on every rank (markov_w1 and
  // markov_w2, [vocab, rank] bf16): the census's draft_markov_embed
  // (w1, whole) plus a whole w2, not the census's vocab-sliced
  // draft_head.
  const double markov_w2_whole =
      static_cast<double>(cfg.vocab_size) * cfg.dspark_markov_rank * 2.0 / kGiB;
  add(Dsv4WeightClass::Draft,
      census_gib("draft_proj", world) + census_gib("draft_markov_embed", world) + markov_w2_whole);
  // The indexer's own rotated compressor (the census's "indexer" rows
  // indexer.compressor.{ape,wkv,wgate}; the norm is 256 B x 21, inside
  // any tolerance): the loader's Compressor class, not Indexer.
  int n_index = 0;
  for (int l = 0; l < cfg.num_hidden_layers; ++l)
    if (cfg.is_index_layer(l)) ++n_index;
  const double idx_comp =
      static_cast<double>(n_index) *
          (4.0 * (2 * cfg.index_head_dim) * 4.0 + 2.0 * (2 * cfg.index_head_dim) * cfg.hidden_size * 2.0) /
          kGiB;
  add(Dsv4WeightClass::Indexer, census_gib("indexer", world) - idx_comp);
  e[Dsv4WeightClass::Compressor] += idx_comp;
  return e;
}

// The refusal gates (the device test's, against the real checkpoint):
// a world that does not divide the geometry is refused by name, and a
// config that does not bind the checkpoint is refused by name. No
// layers are loaded and no GPU memory is allocated (both throw before
// the stream's allocation).
int run_refusals(const Dsv4TextConfig& cfg, const std::string& ckpt) {
  int failed = 0;
  {
    bool refused = false;
    try {
      Dsv4LayerStream s(cfg, ckpt, 0, 3, dgpp::Dsv4Residency::Streaming, dgpp::Dsv4HeadSharding::Full);
    } catch (const std::invalid_argument& e) {
      refused = std::string(e.what()).find("world") != std::string::npos;
      DGPP_LOG_INFO("dsv4_load_check: world-3 refusal: {}", e.what());
    }
    if (refused)
      DGPP_LOG_INFO("dsv4_load_check: refusal 1/2 PASS — a world that does not divide the geometry is refused by name");
    else {
      DGPP_LOG_ERROR("dsv4_load_check: refusal 1/2 FAIL — world 3 was accepted");
      ++failed;
    }
  }
  {
    Dsv4TextConfig bad = cfg;
    bad.n_routed_experts += 1;  // expert 256 is not in the shards
    bool refused = false;
    try {
      Dsv4LayerStream s(bad, ckpt, 0, 1, dgpp::Dsv4Residency::Streaming, dgpp::Dsv4HeadSharding::Full);
    } catch (const std::runtime_error& e) {
      refused = std::string(e.what()).find("binding failed") != std::string::npos;
      DGPP_LOG_INFO("dsv4_load_check: bad-binding refusal: {}", e.what());
    }
    if (refused)
      DGPP_LOG_INFO("dsv4_load_check: refusal 2/2 PASS — a checkpoint that does not bind is refused by name");
    else {
      DGPP_LOG_ERROR("dsv4_load_check: refusal 2/2 FAIL — the unbindable config was accepted");
      ++failed;
    }
  }
  return failed;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_id, ckpt, image_dir;
  int world = 1, rank = 0, layers = -1, from = 0;
  bool streaming = false, mtp = false, hash = false, refuse = false;
  auto next = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error("missing value after " + std::string(argv[i]));
    return argv[++i];
  };
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--model") model_id = next(i);
      else if (a == "--checkpoint-dir") ckpt = next(i);
      else if (a == "--world") world = std::stoi(next(i));
      else if (a == "--rank") rank = std::stoi(next(i));
      else if (a == "--streaming") streaming = true;
      else if (a == "--mtp") mtp = true;
      else if (a == "--layers") layers = std::stoi(next(i));
      else if (a == "--from") from = std::stoi(next(i));
      else if (a == "--image-dir") image_dir = next(i);
      else if (a == "--hash") hash = true;
      else if (a == "--refuse") refuse = true;
      else throw std::runtime_error("unknown argument " + a);
    }
    if (ckpt.empty()) {
      if (model_id.empty()) throw std::runtime_error("--model or --checkpoint-dir is required");
      std::string err;
      ckpt = dgpp::hf::model_dir(model_id, &err);
      if (ckpt.empty()) throw std::runtime_error("cannot resolve " + model_id + ": " + err);
    }
    const std::string cfg_path = (std::filesystem::path(ckpt) / "config.json").string();
    if (dgpp::detect_architecture_file(cfg_path) != dgpp::ModelArchitecture::DeepseekV4)
      throw std::runtime_error("not a DeepseekV4 checkpoint: " + ckpt);
    const Dsv4TextConfig cfg = Dsv4TextConfig::from_json_file(cfg_path);
    if (refuse) {
      const int failed = run_refusals(cfg, ckpt);
      DGPP_LOG_INFO("dsv4_load_check: {} — refusals {}", failed ? "FAIL" : "PASS", failed ? "failed" : "ok");
      return failed ? 1 : 0;
    }
    if (!image_dir.empty()) Dsv4LayerStream::set_resident_image_dir(image_dir == "off" ? "" : image_dir);
    const dgpp::Dsv4HeadSharding head = world > 1 ? dgpp::Dsv4HeadSharding::VocabSharded : dgpp::Dsv4HeadSharding::Full;
    const auto geo = dgpp::Dsv4LocalGeometry::from_config(cfg, rank, world, head);
    const double kGiB = 1024.0 * 1024.0 * 1024.0;
    DGPP_LOG_INFO("dsv4_load_check: {} world {} rank {} {} — formulas: resident {:.2f} GiB (globals {:.2f}, staging {:.2f}); "
                  "{} layers + {} draft stages, {} experts x inter {} (shared {}), hash layers {}, "
                  "geometry: heads {} [{} +{}) groups {} [{} +{}) inter {} shared {} lm [{} +{}) embed [{} +{})",
                  ckpt, world, rank, streaming ? "streaming" : "resident",
                  Dsv4LayerStream::resident_bytes(cfg, rank, world, head, mtp) / kGiB,
                  Dsv4LayerStream::globals_bytes(cfg, rank, world, head) / kGiB,
                  Dsv4LayerStream::staging_plan_bytes(cfg, rank, world, head, mtp) / kGiB,
                  cfg.num_hidden_layers, cfg.num_draft_stages(), cfg.n_routed_experts, cfg.moe_intermediate_size,
                  cfg.shared_expert_inter(), cfg.num_hash_layers,
                  geo.local_heads, geo.head_begin, geo.local_heads, geo.local_groups, geo.group_begin,
                  geo.local_groups, geo.local_inter, geo.local_shared_inter, geo.lm_vocab_begin, geo.lm_vocab_count,
                  geo.embed_vocab_begin, geo.embed_vocab_count);
    const auto t0 = std::chrono::steady_clock::now();
    Dsv4LayerStream stream(cfg, ckpt, rank, world, streaming ? dgpp::Dsv4Residency::Streaming : dgpp::Dsv4Residency::Resident,
                            head, mtp);
    const auto t1 = std::chrono::steady_clock::now();
    DGPP_LOG_INFO("dsv4_load_check: opened in {:.1f} s; layer capacity {:.2f} GiB",
                  std::chrono::duration<double>(t1 - t0).count(), stream.layer_capacity() / kGiB);
    const dgpp::Dsv4ReplicatedDigest d = stream.hash_replicated();
    DGPP_LOG_INFO("dsv4_load_check: digest globals {:016x} tensors {} bytes {:.2f} GiB; layer 0 {:016x} layer 1 {:016x} last {:016x}",
                  d.globals, d.tensors, d.bytes / kGiB, d.layer[0], d.layer[1], d.layer.back());
    const auto& g = stream.load_globals();
    DGPP_LOG_INFO("dsv4_load_check: globals {:.2f} GiB, lm head rows [{}, +{}), embed rows [{}, +{})", g.bytes / kGiB,
                  g.lm_vocab_begin, g.lm_vocab_count, g.embed_vocab_begin, g.embed_vocab_count);
    const int last = mtp ? cfg.max_layer() : cfg.num_hidden_layers;
    const int n = layers < 0 ? last : std::min(last, from + layers);
    // The per-class source plan (the counting-pass closed form, the
    // host-side byte reconcile's per-class input).
    std::map<Dsv4WeightClass, uint64_t> plan;
    for (int l = from; l < n; ++l)
      for (const auto& [c, b] : dgpp::Dsv4LoaderFamily::class_source_plan(cfg, l, rank, world)) plan[c] += b;
    // The globals' closed form (the loader's build_globals accounting):
    // the embedding and lm-head slices, the final norm, the model-level
    // mHC head (source bytes, fp32 in the file).
    const int64_t H = cfg.hidden_size;
    plan[Dsv4WeightClass::Embed] += static_cast<uint64_t>(geo.embed_vocab_count) * H * 2;
    plan[Dsv4WeightClass::LmHead] += static_cast<uint64_t>(geo.lm_vocab_count) * H * 2;
    plan[Dsv4WeightClass::FinalNorm] += static_cast<uint64_t>(H) * 2;
    plan[Dsv4WeightClass::Mhc] +=
        static_cast<uint64_t>(cfg.hc_mult) * cfg.hc_dim() * 4 + static_cast<uint64_t>(cfg.hc_mult) * 4 + 4;
    size_t total = 0;
    int formula_fail = 0, plan_fail = 0;
    for (int l = from; l < n; ++l) {
      const auto tl = std::chrono::steady_clock::now();
      const uint64_t before = stream.source_bytes_read();
      const auto& r = stream.load_layer(l);
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tl).count();
      total += r.bytes;
      if (r.bytes != Dsv4LayerStream::layer_bytes(cfg, l, rank, world)) ++formula_fail;
      if (stream.source_bytes_read() - before != Dsv4LayerStream::planned_layer_source_bytes(cfg, l, rank, world))
        ++plan_fail;
      if (l < from + 4 || l == n - 1 || l % 8 == 0 || cfg.is_draft(l) || cfg.is_hash_layer(l))
        DGPP_LOG_INFO("dsv4_load_check: layer {} ({}/{}{}{}{}) {:.3f} GiB in {:.2f} s, read {:.3f} GiB; heads {} groups {}; experts {} x inter {} (shared {})",
                      l, cfg.compress_ratio(l), cfg.is_hash_layer(l) ? ", hash" : ", router",
                      cfg.is_index_layer(l) ? ", indexer" : "", cfg.compress_ratio(l) > 0 ? ", compressor" : "",
                      cfg.is_draft(l) ? std::format(", draft stage {}", cfg.draft_stage(l)).c_str() : "",
                      r.bytes / kGiB, s, (stream.source_bytes_read() - before) / kGiB, r.attn.local_heads,
                      r.attn.local_groups, r.moe.n_experts, r.moe.local_inter, r.moe.local_shared_inter);
      if (!streaming) continue;
      stream.release_layer();
    }
    if (hash) {
      const dgpp::Dsv4HashTables& ht = stream.load_hash_tables();
      uint64_t table_bytes = 0;
      for (const auto& t : ht.tables) table_bytes += static_cast<uint64_t>(t->rows()) * t->topk() * 8;
      // A small gather from the mapped table: the staged expert ids must
      // be valid router ids.
      std::vector<int32_t> ids(64);
      for (int i = 0; i < 64; ++i) ids[static_cast<size_t>(i)] = static_cast<int32_t>((i * 7919) % cfg.vocab_size);
      std::vector<int64_t> staged(static_cast<size_t>(64) * cfg.num_experts_per_tok);
      ht.tables[0]->gather(ids.data(), 64, staged.data());
      int64_t lo = INT64_MAX, hi = INT64_MIN;
      for (int64_t v : staged) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
      const bool ids_ok = lo >= 0 && hi < cfg.n_routed_experts;
      DGPP_LOG_INFO("dsv4_load_check: hash tables {} mapped ({:.2f} GiB of mapped shards; {:.3f} GiB of table bytes, "
                    "census {:.3f}); gather of 64 tokens staged expert ids in [{}, {}] {}",
                    ht.tables.size(), ht.mapped_bytes() / kGiB, table_bytes / kGiB,
                    census_gib("hash_table", world), lo, hi, ids_ok ? "ok" : "INVALID");
      if (!ids_ok) ++plan_fail;
    }
    if (!streaming) stream.release_sources();
    // The closed form (the per-class plan) against the actual read: the
    // stream enforces the per-layer plan == read (and throws on drift),
    // so the totals must agree byte-exact.
    uint64_t closed_form = 0;
    for (const auto& [c, b] : plan) closed_form += b;
    const uint64_t read = stream.source_bytes_read();
    DGPP_LOG_INFO("dsv4_load_check: per-class source plan (closed form) vs read {} GiB (verbatim {:.2f}):", read / kGiB,
                  stream.verbatim_source_bytes() / kGiB);
    for (const auto& [c, b] : plan)
      DGPP_LOG_INFO("  {:17s} {:9.3f} GiB", class_name(c), static_cast<double>(b) / kGiB);
    DGPP_LOG_INFO("  {:17s} {:9.3f} GiB (closed form; read {:9.3f} GiB, diff {:+.1f} bytes)", "TOTAL",
                  static_cast<double>(closed_form) / kGiB, static_cast<double>(read) / kGiB,
                  static_cast<double>(read) - static_cast<double>(closed_form));
    if (closed_form != read) ++plan_fail;
    // The census gate: the per-class closed form vs the G0 census, the
    // expert source share of the checkpoint, and the resident total.
    const bool census_gate = census_compatible(cfg) && from == 0 && n == cfg.max_layer() && (world == 1 || world == 2);
    if (census_gate) {
      const auto expected = census_expected(cfg, world, kGiB);
      int census_fail = 0;
      DGPP_LOG_INFO("dsv4_load_check: census closed form (docs/checkpoint_budget_dsv4.md, W={}):", world);
      for (const auto& [c, b] : plan) {
        const double got = static_cast<double>(b) / kGiB;
        const double want = expected.at(c);
        const double tol = std::max(0.01, 0.005 * want);
        const bool ok = std::abs(got - want) <= tol;
        if (!ok) ++census_fail;
        DGPP_LOG_INFO("  {:17s} plan {:8.3f} GiB | census {:8.3f} GiB | diff {:+.3f} GiB ({})", class_name(c), got, want,
                      got - want, ok ? "ok" : "MISMATCH");
      }
      const double expert_source =
          (static_cast<double>(plan[Dsv4WeightClass::RoutedExpert]) / kGiB) * world;
      const double census_expert = census_gib("routed_expert", 1) + census_gib("draft_expert", 1);
      const double total_want = [&] {
        double t = 0;
        for (const auto& [c, v] : expected) t += v;
        return t;
      }();
      const double total_got = static_cast<double>(read) / kGiB;
      if (std::abs(total_got - total_want) > 0.02) ++census_fail;
      DGPP_LOG_INFO("  expert source {:.2f} GiB ({} x {:.2f} GiB/rank) vs census {:.2f} GiB of the 166.9 GB checkpoint; "
                    "read total {:.3f} GiB vs census {:.3f} GiB ({})",
                    expert_source, world, plan[Dsv4WeightClass::RoutedExpert] / kGiB, census_expert, total_got,
                    total_want, std::abs(total_got - total_want) <= 0.02 ? "ok" : "MISMATCH");
      if (census_fail) {
        DGPP_LOG_ERROR("dsv4_load_check: census gate FAIL ({} mismatches)", census_fail);
        ++plan_fail;
      } else {
        DGPP_LOG_INFO("dsv4_load_check: census gate PASS — every class within tolerance, expert source {:.2f} GiB",
                      expert_source);
      }
    } else {
      DGPP_LOG_INFO("dsv4_load_check: census gate skipped (needs a full load of {} layers at world 1/2 with the "
                    "census geometry; loaded {} layers at world {})",
                    cfg.max_layer(), n - from, world);
    }
    DGPP_LOG_INFO("dsv4_load_check: {} layers {:.2f} GiB resident (+ globals {:.2f} = {:.2f} GiB); source bytes read {:.2f} GiB "
                  "(verbatim {:.2f}); image restored {} captured {}; total {:.1f} s",
                  n - from, total / kGiB, g.bytes / kGiB, (total + g.bytes) / kGiB, read / kGiB,
                  stream.verbatim_source_bytes() / kGiB, stream.image_layers_restored(), stream.image_layers_captured(),
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (formula_fail || plan_fail) {
      DGPP_LOG_ERROR("dsv4_load_check: FAIL — {} formula drift, {} plan/hash/census mismatches", formula_fail, plan_fail);
      return 1;
    }
    DGPP_LOG_INFO("dsv4_load_check: PASS — byte formulas, the source-byte plan and the census closed form all reconcile");
    return 0;
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("dsv4_load_check: {}", e.what());
    return 1;
  }
}
