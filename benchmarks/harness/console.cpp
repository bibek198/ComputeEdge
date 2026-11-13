#include "harness/console.h"

#include <iostream>

namespace computeedge::bench::harness {

Environment banner(const std::string& title, BannerStyle style) {
  const Environment env = record_environment();
  std::cout << title << "\n"
            << "  " << env.chip;
  if (style == BannerStyle::WithGpu) std::cout << " / " << env.gpu_name;
  std::cout << " (" << env.model << "), macOS " << env.os_version << " (" << env.os_build << ")\n"
            << "  power: " << env.power_source << (env.low_power_mode ? ", LOW POWER MODE" : "")
            << ", thermal: " << to_string(env.thermal_state) << ", build: " << env.build_type
            << (env.sanitizers ? " +sanitizers" : "") << ", git " << env.git_sha << "\n";
  if (env.sanitizers || env.build_type != "Release") {
    std::cout << "  WARNING: not a Release build without sanitizers; timings are not representative.\n";
  }
  if (env.power_source != "AC Power") std::cout << "  WARNING: not on AC power.\n";
  std::cout << '\n';
  return env;
}

}  // namespace computeedge::bench::harness
