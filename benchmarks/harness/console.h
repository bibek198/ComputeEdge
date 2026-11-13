#pragma once

// Console presentation shared by the benchmark binaries.

#include <string>

#include "computeedge/bench/environment.h"

namespace computeedge::bench::harness {

enum class BannerStyle {
  CpuOnly,  // "<chip> (<model>)": binaries that never touch the GPU
  WithGpu,  // "<chip> / <gpu> (<model>)"
};

// Records the environment and prints the run banner: title, machine, power,
// thermal state, build and git SHA, with a warning for non-Release or
// sanitizer builds and when not on AC power. Returns the recorded environment.
Environment banner(const std::string& title, BannerStyle style);

}  // namespace computeedge::bench::harness
