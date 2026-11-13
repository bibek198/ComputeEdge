#pragma once

// Measurement logic of the thermal_drift benchmark: the sustained-load loop.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "computeedge/bench/environment.h"
#include "computeedge/bench/runner.h"
#include "harness/results.h"
#include "options.h"

namespace computeedge::bench::thermal_drift {

// One measured call.
struct Sample {
  double t_s;
  uint64_t cycle;
  size_t workload;
  uint64_t ns;
  computeedge::bench::ThermalState thermal;
  bool on_ac;
};

struct RunOutput {
  std::vector<std::string> workloads;               // names, indexed by Sample::workload
  std::vector<Sample> samples;                      // every call, in order
  std::vector<computeedge::bench::Result> results;  // per-minute windows
  bool power_changed = false;                       // the run stopped because the power source changed
  int status = 0;                                   // process exit status: 0 ok, 3 power changed, 1 no samples
};

// Runs the sustained-load loop for opt.minutes and windows the samples per minute.
RunOutput run(const Options& opt, harness::Reporter& report);

}  // namespace computeedge::bench::thermal_drift
