// The DeepSeek-V4-Flash loader's HOST-side gates (no CUDA calls — runs on
// the CI box; the device-touching gates live in dsv4_loader_test.cpp,
// the GPU gate):
// - the synthetic fixture builds and re-binds: the shard re-opened through
//   the safetensors mapping satisfies the binding table, fixture and
//   table cannot disagree.
// - the byte reconcile at world 2: per weight class, the two ranks'
//   counting-pass source plans sum to the class's closed form (replicated
//   tensors read by both ranks, sharded ones read once in total, the
//   mapped hash tables read by no rank); the layer byte formulas are the
//   counting bump's usage and symmetric across ranks; the globals
//   formula.
// - the e8m0 -> fp32 scale decode (the scale-format gap's conversion,
//   docs/dsv4_kernel_port_spec.md §3) against the naive bit-level oracle,
//   all 256 bytes.
// - the MXFP4 column-slice contract: refused off a 32-element block
//   boundary, by name.
// - the tid2eid hash tables mapped from the shard: a sampled gather
//   bitwise the shard's own bytes, an out-of-range token refused.
// - a world that does not divide the geometry and a checkpoint that does
//   not bind are refused by name.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "dsv4_fixture.hpp"
#include "glm_rng.hpp"
#include "loaders/safetensors.hpp"
#include "models/dsv4/binding.hpp"
#include "models/dsv4/config.hpp"
#include "models/dsv4/loader.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::Dsv4ExpectedTensor;
using dgpp::Dsv4HashTableMmap;
using dgpp::Dsv4LayerStream;
using dgpp::Dsv4LoaderFamily;
using dgpp::Dsv4TextConfig;
using dgpp::Dsv4WeightClass;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  Dsv4TextConfig cfg;
  std::string dir;
  std::vector<Dsv4ExpectedTensor> table;
  const Dsv4ExpectedTensor& expected(const std::string& name) const {
    for (const auto& e : table)
      if (e.name == name) return e;
    throw std::runtime_error("fixture: no expected tensor " + name);
  }
  std::vector<uint8_t> bytes(const std::string& name) const {
    return dsv4fx::fixture_bytes(cfg, expected(name));
  }
};

Fixture write_fixture() {
  Fixture fx;
  fx.cfg = dsv4fx::tiny_config();
  fx.dir = (fs::current_path() / "dsv4_loader_cpu_fixture").string();
  dsv4fx::write_fixture(fx.cfg, fx.dir);
  fx.table = dgpp::dsv4_expected_text_tensors(fx.cfg);
  return fx;
}

const char* class_name(Dsv4WeightClass c) {
  switch (c) {
    case Dsv4WeightClass::Embed: return "Embed";
    case Dsv4WeightClass::LmHead: return "LmHead";
    case Dsv4WeightClass::FinalNorm: return "FinalNorm";
    case Dsv4WeightClass::LayerNorm: return "LayerNorm";
    case Dsv4WeightClass::Attention: return "Attention";
    case Dsv4WeightClass::Indexer: return "Indexer";
    case Dsv4WeightClass::Compressor: return "Compressor";
    case Dsv4WeightClass::Router: return "Router";
    case Dsv4WeightClass::SharedExpert: return "SharedExpert";
    case Dsv4WeightClass::RoutedExpert: return "RoutedExpert";
    case Dsv4WeightClass::Mhc: return "Mhc";
    case Dsv4WeightClass::Draft: return "Draft";
  }
  return "?";
}

// The bit-level oracle the CPU pin in tests/unit/dsv4_fp8_scale_test.cpp
// uses: 2^(b - 127) has the IEEE bits (b << 23) for 1..254, the denormal
// 2^-127 bits for 0, and 255 has no finite value.
float e8m0_oracle(uint8_t b) {
  if (b == 255) return std::nanf("");
  const uint32_t want = (b == 0) ? 0x00400000u : (static_cast<uint32_t>(b) << 23);
  float out;
  std::memcpy(&out, &want, 4);
  return out;
}

}  // namespace

DGPP_TEST(dsv4_loader_cpu_fixture_builds_and_binds) {
  const Fixture fx = write_fixture();
  auto shard = dgpp::SafetensorsFile::open((fs::path(fx.dir) / "model.safetensors").string());
  dgpp::Dsv4LoaderFamily::PresentMap present;
  shard->for_each([&](const dgpp::TensorInfo& t) {
    present.emplace(t.name, dgpp::Dsv4TensorDesc{t.dtype, t.shape});
  });
  require(shard->tensor_count() == fx.table.size(), "the shard holds every table tensor once");
  const dgpp::Dsv4BindReport rep = dgpp::dsv4_validate_text_binding(fx.cfg, present);
  require(rep.ok(), "the fixture checkpoint binds");
  require(rep.expected == fx.table.size() && rep.matched == rep.expected && rep.missing == 0 &&
              rep.unexpected == 0,
          "every table tensor present, nothing extra");
  std::printf("dsv4 fixture binding: %zu/%zu tensors (%zu fp8 matrices, %zu fp4 matrices, "
              "%zu hash tables)\n",
              rep.matched, rep.expected, rep.fp8_matrices, rep.fp4_matrices, rep.hash_tables);
}

DGPP_TEST(dsv4_loader_cpu_byte_reconcile_world_2) {
  const Fixture fx = write_fixture();
  const Dsv4TextConfig& cfg = fx.cfg;
  const int layers = cfg.max_layer();
  const int world = 2;  // the only deployment world
  uint64_t total_plan = 0, total_closed = 0;
  for (int l = 0; l < layers; ++l) {
    const auto plan0 = Dsv4LoaderFamily::class_source_plan(cfg, l, 0, world);
    const auto plan1 = Dsv4LoaderFamily::class_source_plan(cfg, l, 1, world);
    // The closed form: replicated tensors read by BOTH ranks, sharded ones
    // once in total, the mapped hash tables by no rank.
    std::map<Dsv4WeightClass, uint64_t> closed;
    for (const Dsv4ExpectedTensor& e : Dsv4LoaderFamily::layer_table(cfg, l)) {
      const uint64_t bytes = e.nbytes();
      uint64_t mult;
      if (e.role == dgpp::Dsv4TensorRole::HashTable)
        mult = 0;  // mapped, never read whole
      else if (Dsv4LoaderFamily::is_replicated(e))
        mult = static_cast<uint64_t>(world);
      else
        mult = 1;
      closed[e.cls] += bytes * mult;
    }
    uint64_t plan0_sum = 0, plan1_sum = 0, layer_closed = 0;
    for (const auto& [cls, want] : closed) {
      const auto it0 = plan0.find(cls), it1 = plan1.find(cls);
      const uint64_t v0 = it0 == plan0.end() ? 0 : it0->second;
      const uint64_t v1 = it1 == plan1.end() ? 0 : it1->second;
      require(v0 + v1 == want, "layer " + std::to_string(l) + " class " + class_name(cls) +
                                   " plan does not reconcile with the closed form");
      std::printf("  layer %2d %-13s plan r0=%-9llu r1=%-9llu closed=%-9llu OK\n", l, class_name(cls),
                  static_cast<unsigned long long>(v0), static_cast<unsigned long long>(v1),
                  static_cast<unsigned long long>(want));
      plan0_sum += v0;
      plan1_sum += v1;
      layer_closed += want;
    }
    require(Dsv4LayerStream::planned_layer_source_bytes(cfg, l, 0, world) == plan0_sum,
            "the layer source plan equals the per-class sum (rank 0)");
    require(Dsv4LayerStream::planned_layer_source_bytes(cfg, l, 1, world) == plan1_sum,
            "the layer source plan equals the per-class sum (rank 1)");
    // The byte formula (the counting bump's usage) is symmetric across
    // the ranks: every slice has the same size on both.
    const size_t fb0 = Dsv4LayerStream::layer_bytes(cfg, l, 0, world);
    const size_t fb1 = Dsv4LayerStream::layer_bytes(cfg, l, 1, world);
    require(fb0 == fb1 && fb0 > 0, "the layer byte formula is rank-symmetric and nonzero");
    total_plan += plan0_sum + plan1_sum;
    total_closed += layer_closed;
  }
  require(total_plan == total_closed, "the whole-stack source plan reconciles with the closed form");
  std::printf("dsv4 byte reconcile (world 2): %d layers, per-class plan == closed form, "
              "formula symmetric across ranks\n",
              layers);
  // The globals formula: the closed form over the globals' allocations.
  for (const auto head : {dgpp::Dsv4HeadSharding::Full, dgpp::Dsv4HeadSharding::VocabSharded}) {
    const size_t H = static_cast<size_t>(cfg.hidden_size);
    const size_t want = dgpp::align_up_256(static_cast<size_t>(cfg.vocab_size) * H * 2) +  // embed whole
                        dgpp::align_up_256(H * 2) +                                        // final norm
                        dgpp::align_up_256(static_cast<size_t>(
                                                head == dgpp::Dsv4HeadSharding::Full ? cfg.vocab_size
                                                                                      : cfg.vocab_size / world) *
                                            H * 2) +                                       // lm head
                        dgpp::align_up_256(static_cast<size_t>(cfg.hc_mult) *
                                           static_cast<size_t>(cfg.hc_dim()) * 2) +
                        dgpp::align_up_256(static_cast<size_t>(cfg.hc_mult) * 4) +
                        dgpp::align_up_256(4);
    for (int rank = 0; rank < world; ++rank) {
      require(Dsv4LayerStream::globals_bytes(cfg, rank, world, head) == want,
              "the globals formula does not match its closed form");
    }
    require(Dsv4LayerStream::lm_vocab_count(cfg, 0, world, head) ==
                (head == dgpp::Dsv4HeadSharding::Full ? cfg.vocab_size : cfg.vocab_size / world),
            "the lm head slice formula");
  }
}

DGPP_TEST(dsv4_loader_cpu_e8m0_decode_matches_the_oracle) {
  // All 256 e8m0 bytes against the bit-level oracle (the CPU pin's
  // contract, tests/unit/dsv4_fp8_scale_test.cpp): 2^(b - 127), the
  // denormal 2^-127 at 0, 255 -> NaN.
  for (uint32_t b = 0; b < 256; ++b) {
    const float got = Dsv4LoaderFamily::e8m0_to_float(static_cast<uint8_t>(b));
    const float want = e8m0_oracle(static_cast<uint8_t>(b));
    if (b == 255) {
      require(std::isnan(got), "e8m0 byte 255 must decode to NaN");
    } else {
      require(got == want, "e8m0 decode of byte " + std::to_string(b) + " disagrees with the oracle");
    }
  }
  // Strict monotonicity over the finite range (2^-127 .. 2^127; 255 is
  // NaN, excluded — a comparison against it is unordered).
  for (uint32_t b = 1; b + 1 < 255; ++b)
    require(Dsv4LoaderFamily::e8m0_to_float(static_cast<uint8_t>(b)) <
                Dsv4LoaderFamily::e8m0_to_float(static_cast<uint8_t>(b + 1)),
            "the e8m0 decode must be strictly increasing on 1..254");
}

DGPP_TEST(dsv4_loader_cpu_mxfp4_column_slices_refused_off_a_32_boundary) {
  // The builder's guard, exposed for the host test: whole 32-element
  // blocks, starting on a block boundary (the expert's column slice
  // contract; 2048 inter / 2 ranks = 1024 is a multiple of 32).
  Dsv4LoaderFamily::check_mxfp4_slice(0, 2048);
  Dsv4LoaderFamily::check_mxfp4_slice(1024, 1024);  // the rank-1 slice of the release's inter
  for (const int64_t bad : {16, 33, 1008, -32}) {
    bool refused = false;
    try {
      Dsv4LoaderFamily::check_mxfp4_slice(bad, 512);
    } catch (const std::invalid_argument& e) {
      refused = std::string(e.what()).find("32-element block boundary") != std::string::npos;
    }
    require(refused, "an off-boundary MXFP4 column start (" + std::to_string(bad) + ") is refused by name");
  }
  bool refused = false;
  try {
    Dsv4LoaderFamily::check_mxfp4_slice(64, 48);  // whole blocks, wrong size
  } catch (const std::invalid_argument& e) {
    refused = std::string(e.what()).find("multiple of 32") != std::string::npos;
  }
  require(refused, "an MXFP4 column span off a 32-multiple is refused by name");
}

DGPP_TEST(dsv4_loader_cpu_hash_tables_mapped_and_gathered) {
  const Fixture fx = write_fixture();
  const Dsv4TextConfig& cfg = fx.cfg;
  auto shard = dgpp::SafetensorsFile::open((fs::path(fx.dir) / "model.safetensors").string());
  const int topk = cfg.num_experts_per_tok;
  for (int l = 0; l < cfg.num_hash_layers; ++l) {
    const std::string name = dgpp::dsv4_layer_prefix(cfg, l) + "ffn.gate.tid2eid";
    const dgpp::TensorInfo& t = shard->at(name);
    require(t.shape.size() == 2 && t.shape[0] == cfg.vocab_size && t.shape[1] == topk,
            "the hash table geometry");
    Dsv4HashTableMmap m(t.owner->path(), t.data_begin, t.shape[0], topk);
    require(m.rows() == cfg.vocab_size && m.topk() == topk && m.mapped_bytes() > 0,
            "the mapping reports its geometry");
    // A sampled gather (the deterministic table's rows) bitwise the
    // shard's own bytes.
    const int n = 517;  // past the 256-row single-thread floor: the
    // pooled path's rows
    std::vector<int32_t> tokens(static_cast<size_t>(n));
    glmrng::Rng rng(0x7A5 + l);
    for (int i = 0; i < n; ++i) tokens[static_cast<size_t>(i)] = static_cast<int32_t>(rng.next() % static_cast<uint64_t>(cfg.vocab_size));
    std::vector<int64_t> staged(static_cast<size_t>(n) * topk);
    m.gather(tokens.data(), n, staged.data());
    for (int i = 0; i < n; ++i) {
      const uint8_t* want = static_cast<const uint8_t*>(t.data) +
                            static_cast<size_t>(tokens[static_cast<size_t>(i)]) * static_cast<size_t>(topk) * 8;
      require(std::memcmp(staged.data() + static_cast<size_t>(i) * topk, want,
                          static_cast<size_t>(topk) * 8) == 0,
              "the gathered hash row disagrees with the shard");
    }
    // The fixture's deterministic map: the gathered ids are the table's.
    const int64_t probe = tokens[0];
    require(staged[0] == probe * 7 % cfg.n_routed_experts &&
                staged[1] == (probe * 7 + 3) % cfg.n_routed_experts,
            "the fixture's token -> expert map");
    // A token past the table is refused by name.
    bool refused = false;
    try {
      (void)m.row(cfg.vocab_size);
    } catch (const std::out_of_range& e) {
      refused = std::string(e.what()).find("outside the table") != std::string::npos;
    }
    require(refused, "a token past the table is refused");
  }
}

DGPP_TEST(dsv4_loader_cpu_refuses_bad_geometry_and_binding_by_name) {
  const Fixture fx = write_fixture();
  {
    bool refused = false;
    try {
      (void)Dsv4LayerStream::layer_bytes(fx.cfg, 0, 0, 3);  // 2 output groups do not divide 3
    } catch (const std::invalid_argument& e) {
      refused = std::string(e.what()).find("world") != std::string::npos;
    }
    require(refused, "a world that does not divide the geometry is refused by name");
  }
  {
    auto shard = dgpp::SafetensorsFile::open((fs::path(fx.dir) / "model.safetensors").string());
    dgpp::Dsv4LoaderFamily::PresentMap present;
    shard->for_each([&](const dgpp::TensorInfo& t) {
      present.emplace(t.name, dgpp::Dsv4TensorDesc{t.dtype, t.shape});
    });
    Dsv4TextConfig cfg = fx.cfg;
    cfg.n_routed_experts = 16;  // experts 8..15 are not in the shard
    bool refused = false;
    try {
      Dsv4LoaderFamily::validate_binding(cfg, present);
    } catch (const std::runtime_error& e) {
      refused = std::string(e.what()).find("binding failed") != std::string::npos;
    }
    require(refused, "a checkpoint that does not bind is refused by name");
  }
}

int main() { return dgpp::test::run_all(); }
