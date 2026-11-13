#pragma once

// Benchmark harness: warm-up, repeated measurement, robust statistics, thermal
// tagging and reproducible output (env.json + samples.csv + summary.csv).
//
// Lives in the separate `computeedge_bench` library because the environment
// recorder uses Apple frameworks (Foundation, IOKit).

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace computeedge::bench {

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

struct Summary {
  size_t n = 0;
  double min = 0, max = 0, mean = 0, stddev = 0;
  double median = 0, p25 = 0, p75 = 0, iqr = 0;
  double ci95_lo = 0, ci95_hi = 0;  // bootstrap 95% CI of the median
};

// Linear-interpolated quantile (q in [0,1]) of an unsorted sample.
double quantile(std::vector<double> values, double q);

// Summary statistics; the CI uses `resamples` bootstrap resamples with a fixed seed.
Summary summarize(const std::vector<double>& values, size_t resamples = 2000, uint64_t seed = 0xCE);

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

enum class ThermalState : int8_t { Nominal = 0, Fair = 1, Serious = 2, Critical = 3, Unknown = -1 };
const char* to_string(ThermalState state);

ThermalState thermal_state();

// Polls until the thermal state is at or below `target`, or the timeout expires.
// Returns true if the target was reached.
bool wait_for_thermal(ThermalState target, double timeout_s, double poll_s = 1.0);

struct Environment {
  std::string timestamp_utc;  // ISO 8601
  std::string model;          // hw.model, e.g. Mac16,12
  std::string chip;           // machdep.cpu.brand_string
  int performance_cores = 0;
  int efficiency_cores = 0;
  uint64_t memory_bytes = 0;
  std::string os_version;  // e.g. 26.4.1
  std::string os_build;    // kern.osversion
  std::string power_source;  // "AC Power", "Battery Power", or "unknown"
  bool low_power_mode = false;
  ThermalState thermal_state = ThermalState::Unknown;
  std::string gpu_name;                    // MTLDevice.name
  uint64_t gpu_recommended_working_set = 0;  // MTLDevice.recommendedMaxWorkingSetSize
  std::string compiler;
  std::string compiler_flags;  // CMAKE_CXX_FLAGS + build-type flags
  std::string build_type;
  std::string git_sha;  // captured at CMake configure time
  bool sanitizers = false;
  std::string veclib_maximum_threads;  // VECLIB_MAXIMUM_THREADS, empty if unset
};

Environment record_environment();

// "AC Power", "Battery Power" or "unknown" right now (cheap; for per-cycle checks
// in long runs, which are invalid if the source changes mid-run).
std::string current_power_source();
std::string to_json(const Environment& env);

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

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
  std::vector<double> samples_ns;               // per call
  std::vector<ThermalState> sample_thermal;     // state read after each sample
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
Result run_with_metrics(const std::string& name, const std::map<std::string, std::string>& params,
                        const Config& config, const MetricsFn& fn, double flops_per_call = 0,
                        const std::function<void()>& setup = {});

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

// Writes <root>/<YYYYmmdd-HHMMSS>_<sha>/{env.json, samples.csv, summary.csv} and,
// when any result has metrics, metrics.csv (long format) and metrics_summary.csv.
class ResultWriter {
 public:
  ResultWriter(const std::string& root, const Environment& env);
  void add(const Result& result);
  void flush() const;
  const std::string& directory() const { return dir_; }

 private:
  std::string dir_;
  Environment env_;
  std::vector<Result> results_;
};

// One-line human-readable summary for console output.
std::string format_result(const Result& result);

}  // namespace computeedge::bench
