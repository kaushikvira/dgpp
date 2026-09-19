#include "common/bf16_residency.hpp"

#include <atomic>
#include <cstdlib>

#include "common/log.hpp"

namespace dgpp {
namespace {

std::atomic<int> g_mode{static_cast<int>(Bf16Residency::Checkpoint)};

// DGPP_BF12: -1 unset, else the mode.
int env_override() {
  static const int v = [] {
    const char* e = std::getenv("DGPP_BF12");
    if (e == nullptr) return -1;
    const std::string s(e);
    if (s == "off" || s == "0") return static_cast<int>(Bf16Residency::Checkpoint);
    if (s == "on" || s == "1" || s == "bf12") return static_cast<int>(Bf16Residency::Bf12);
    if (s == "both" || s == "bf12+bf16") return static_cast<int>(Bf16Residency::Bf12AndBf16);
    DGPP_LOG_WARN("DGPP_BF12={} ignored (off|on|both)", s);
    return -1;
  }();
  return v;
}

}  // namespace

void set_bf16_residency(Bf16Residency mode) { g_mode.store(static_cast<int>(mode)); }

Bf16Residency bf16_residency() {
  const int env = env_override();
  return static_cast<Bf16Residency>(env >= 0 ? env : g_mode.load());
}

const char* bf16_residency_name(Bf16Residency mode) {
  switch (mode) {
    case Bf16Residency::Bf12: return "bf12";
    case Bf16Residency::Bf12AndBf16: return "bf12+bf16";
    default: return "checkpoint";
  }
}

bool parse_bf16_residency(const std::string& name, Bf16Residency* mode) {
  for (const Bf16Residency m :
       {Bf16Residency::Checkpoint, Bf16Residency::Bf12, Bf16Residency::Bf12AndBf16})
    if (name == bf16_residency_name(m)) {
      if (mode != nullptr) *mode = m;
      return true;
    }
  return false;
}

}  // namespace dgpp
