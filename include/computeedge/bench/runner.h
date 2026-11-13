#pragma once

// Benchmark runner: warm-up, calibration and repeated measurement with
// per-sample thermal tagging (bench::run, bench::run_with_metrics).

#include <functional>
#include <map>
#include <string>
#include <vector>
#include "computeedge/bench/environment.h"
#include "computeedge/bench/stats.h"

namespace computeedge::bench {

struct Config {
  // Warm-up runs until BOTH at least `warmup` calls and `min_warmup_ns` have
  // elapsed. The time floor matters for small ops: a few µs-long calls do not
  // keep the cores busy long enough to reach steady-state clocks (measured on
  // M4: a 128² SGEMM is ~4-5x slower with a 10-call-only warm-up).
  int warmup = 10;
  uint64_t min_warmup_ns = 200'000'000;
  int repetitions = 30;
  // Calls per sample. 0 = auto: pick the smallest power of two so one sample
  // takes at least min_sample_ns. Each sample is reported per call.
  int inner_iterations = 0;
  uint64_t min_sample_ns = 50'000;
  // Wait for thermal state Nominal before each benchmark (fanless Air).
  bool cooldown = false;
  double cooldown_timeout_s = 300;
};

struct Result {
  std::string name;
  std::map<std::string, std::string> params;
  int inner_iterations = 1;
  std::vector<double> samples_ns;            // per call
  std::vector<ThermalState> sample_thermal;  // state read after each sample
  ThermalState thermal_before = ThermalState::Unknown;
  ThermalState thermal_after = ThermalState::Unknown;
  double flops_per_call = 0;  // 0 if not applicable
  Summary summary;
  // Optional per-sample breakdown (e.g. submit / wait / GPU time), from run_with_metrics.
  std::map<std::string, std::vector<double>> metrics;
  std::map<std::string, Summary> metric_summaries;

  double gflops_median() const;  // 0 if flops_per_call == 0
};

// Runs fn() under the given config. `setup` (optional) runs before each sample,
// outside the timed region.
Result run(const std::string& name, const std::map<std::string, std::string>& params, const Config& config,
           const std::function<void()>& fn, double flops_per_call = 0, const std::function<void()>& setup = {});

// A measured call that also reports named metrics for that call (e.g. phase
// durations in ns). Each sample is exactly one call (inner_iterations is forced
// to 1) so metrics and the sample line up.
using Metrics = std::map<std::string, double>;
using MetricsFn = std::function<void(Metrics&)>;

// `setup` (optional) runs before every call, outside the timed region (e.g. an
// idle gap, or creating a fresh buffer for a first-use measurement).
Result run_with_metrics(const std::string& name, const std::map<std::string, std::string>& params, const Config& config,
                        const MetricsFn& fn, double flops_per_call = 0, const std::function<void()>& setup = {});

}  // namespace computeedge::bench
