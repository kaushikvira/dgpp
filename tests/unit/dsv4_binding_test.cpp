// The DeepSeek-V4-Flash expected-tensor table: its shape on the release's
// config (the per-layer counts the shard headers carry), the TP geometry
// acceptance at the deployment world (T=2), and — against the checkpoint's
// shard headers — the full binding: every expected tensor present with its
// dtype and shape, nothing unexpected. The checkpoint gate runs on the
// directory named by DGPP_DSV4_CHECKPOINT_DIR (a header-only mirror
// suffices: no payload is read).
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "common/test.hpp"
#include "dsv4_config_json.hpp"
#include "loaders/minijson.hpp"
#include "loaders/safetensors.hpp"
#include "models/dsv4/binding.hpp"
#include "models/dsv4/config.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

dgpp::Dsv4TextConfig release_config() {
  const std::string text = dsv4_test::config_json();
  const auto t = dgpp::minijson::parse(text);
  return dgpp::Dsv4TextConfig::parse(t.root);
}

constexpr size_t kTotalTensors = 72317;

std::filesystem::path landed_checkpoint() {
  namespace fs = std::filesystem;
  if (const char* dir = std::getenv("DGPP_DSV4_CHECKPOINT_DIR"); dir && *dir) {
    const fs::path p(dir);
    return (fs::is_directory(p) && fs::exists(p / "config.json")) ? p : fs::path{};
  }
  return {};
}

}  // namespace

DGPP_TEST(dsv4_binding_table_has_the_release_shape) {
  const dgpp::Dsv4TextConfig cfg = release_config();
  // A window hash layer (0, 1): 29 non-expert — 1536 experts + 29.
  require(dgpp::dsv4_expected_layer_tensors(cfg, 0).size() == 1565, "window hash layer");
  // A ratio-4 hash layer (2): the compressor + the indexer's 7 tensors, tid2eid instead of the bias.
  require(dgpp::dsv4_expected_layer_tensors(cfg, 2).size() == 1576, "ratio-4 hash layer");
  // A ratio-128 layer (3): the plain compressor only (no indexer at ratio 128).
  require(dgpp::dsv4_expected_layer_tensors(cfg, 3).size() == 1569, "ratio-128 layer");
  // A ratio-4 non-hash layer (4): the bias back.
  require(dgpp::dsv4_expected_layer_tensors(cfg, 4).size() == 1576, "ratio-4 layer");
  // The draft stages: stage 0 with main_proj/main_norm, stage 2 with the heads.
  require(dgpp::dsv4_expected_layer_tensors(cfg, 43).size() == 1568, "draft stage 0");
  require(dgpp::dsv4_expected_layer_tensors(cfg, 44).size() == 1565, "draft stage 1");
  require(dgpp::dsv4_expected_layer_tensors(cfg, 45).size() == 1572, "draft stage 2");
  require(dgpp::dsv4_expected_layer_tensors(cfg, 43)[0].name == "mtp.0.main_proj.weight", "draft prefix");
  require(dgpp::dsv4_expected_global_tensors(cfg).size() == 6, "globals");
  const auto all = dgpp::dsv4_expected_text_tensors(cfg);
  require(all.size() == kTotalTensors, "table size " + std::to_string(all.size()));
  size_t fp4 = 0, fp8 = 0, tables = 0;
  for (const auto& e : all) {
    switch (e.role) {
      case dgpp::Dsv4TensorRole::Fp4Payload: ++fp4; break;
      case dgpp::Dsv4TensorRole::Fp8Payload: ++fp8; break;
      case dgpp::Dsv4TensorRole::HashTable: ++tables; break;
      default: break;
    }
  }
  // fp4: 46 layers x 256 experts x 3 matrices.
  require(fp4 == 46 * 256 * 3, "fp4 matrices " + std::to_string(fp4));
  // fp8: 5 attention + 3 shared per layer (46), 21 indexer wq_b, 1 main_proj.
  require(fp8 == 46 * 8 + 21 + 1, "fp8 matrices " + std::to_string(fp8));
  require(tables == 3, "hash tables " + std::to_string(tables));
  // Shapes pinned to the shard headers.
  for (const auto& e : all) {
    if (e.name == "layers.3.attn.wq_b.weight")
      require(e.dtype == dgpp::DType::F8_E4M3 && e.shape == std::vector<int64_t>{32768, 1024}, "wq_b");
    if (e.name == "layers.3.attn.wq_b.scale") require(e.shape == std::vector<int64_t>{256, 8}, "wq_b scale");
    if (e.name == "layers.3.attn.wo_a.weight") require(e.shape == std::vector<int64_t>{8192, 4096}, "wo_a");
    if (e.name == "layers.3.attn.wo_a.scale") require(e.shape == std::vector<int64_t>{64, 32}, "wo_a scale");
    if (e.name == "layers.3.attn.wo_b.scale") require(e.shape == std::vector<int64_t>{32, 64}, "wo_b scale");
    if (e.name == "layers.3.ffn.experts.0.w1.weight")
      require(e.dtype == dgpp::DType::I8 && e.shape == std::vector<int64_t>{2048, 2048}, "w1 payload");
    if (e.name == "layers.3.ffn.experts.0.w1.scale")
      require(e.dtype == dgpp::DType::F8_E8M0 && e.shape == std::vector<int64_t>{2048, 128}, "w1 scale");
    if (e.name == "layers.3.ffn.experts.0.w2.weight") require(e.shape == std::vector<int64_t>{4096, 1024}, "w2 payload");
    if (e.name == "layers.3.ffn.experts.0.w2.scale") require(e.shape == std::vector<int64_t>{4096, 64}, "w2 scale");
    if (e.name == "layers.2.ffn.gate.tid2eid")
      require(e.dtype == dgpp::DType::I64 && e.shape == std::vector<int64_t>{129280, 6}, "tid2eid");
    if (e.name == "layers.3.attn.compressor.ape")
      require(e.dtype == dgpp::DType::F32 && e.shape == std::vector<int64_t>{128, 512}, "plain ape");
    if (e.name == "layers.4.attn.compressor.ape") require(e.shape == std::vector<int64_t>{4, 1024}, "overlap ape");
    if (e.name == "layers.4.attn.indexer.compressor.ape")
      require(e.shape == std::vector<int64_t>{4, 256}, "indexer ape");
    if (e.name == "layers.4.attn.indexer.wq_b.weight")
      require(e.shape == std::vector<int64_t>{8192, 1024}, "indexer wq_b");
    if (e.name == "layers.4.attn.indexer.wq_b.scale") require(e.shape == std::vector<int64_t>{64, 8}, "indexer scale");
    if (e.name == "layers.3.hc_attn_fn") require(e.dtype == dgpp::DType::F32 && e.shape == std::vector<int64_t>{24, 16384}, "hc fn");
    if (e.name == "mtp.0.main_proj.weight") require(e.shape == std::vector<int64_t>{4096, 12288}, "main_proj");
    if (e.name == "mtp.0.main_proj.scale") require(e.shape == std::vector<int64_t>{32, 96}, "main_proj scale");
    if (e.name == "mtp.2.confidence_head.proj.weight") require(e.shape == std::vector<int64_t>{1, 4352}, "confidence");
    if (e.name == "mtp.2.markov_head.markov_w1.weight")
      require(e.shape == std::vector<int64_t>{129280, 256}, "markov");
    if (e.name == "hc_head_fn") require(e.dtype == dgpp::DType::F32 && e.shape == std::vector<int64_t>{4, 16384}, "global hc head");
  }
}

DGPP_TEST(dsv4_tp_geometry_accepts_the_deployment_worlds) {
  const dgpp::Dsv4TextConfig cfg = release_config();
  for (const int w : {1, 2})
    for (int r = 0; r < w; ++r) dgpp::dsv4_tp_validate_geometry(cfg, r, w);
  auto refused = [&](int world) {
    try {
      dgpp::dsv4_tp_validate_geometry(cfg, 0, world);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  require(refused(3), "a world that does not divide the groups is refused");
  require(refused(16), "a world wider than the output groups is refused");
}

DGPP_TEST(dsv4_binding_matches_the_landed_checkpoint) {
  const auto dir = landed_checkpoint();
  if (dir.empty()) return;
  namespace fs = std::filesystem;
  const dgpp::Dsv4TextConfig cfg = dgpp::Dsv4TextConfig::from_json_file((dir / "config.json").string());
  std::unordered_map<std::string, dgpp::Dsv4TensorDesc> present;
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().extension() != ".safetensors") continue;
    auto f = dgpp::SafetensorsFile::open(entry.path().string());
    f->for_each([&](const dgpp::TensorInfo& t) { present.emplace(t.name, dgpp::Dsv4TensorDesc{t.dtype, t.shape}); });
  }
  const dgpp::Dsv4BindReport rep = dgpp::dsv4_validate_text_binding(cfg, present);
  std::string errs;
  for (const auto& e : rep.errors) errs += "\n  " + e;
  require(rep.ok(), "binding: missing " + std::to_string(rep.missing) + ", dtype " + std::to_string(rep.dtype_mismatch) +
                        ", shape " + std::to_string(rep.shape_mismatch) + ", unexpected " + std::to_string(rep.unexpected) + errs);
  require(rep.expected == kTotalTensors && rep.matched == kTotalTensors, "every tensor bound");
  require(rep.fp4_matrices == 46 * 256 * 3 && rep.fp8_matrices == 46 * 8 + 21 + 1 && rep.hash_tables == 3,
          "matrix counts");
}
