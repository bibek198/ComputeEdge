#pragma once

// Grouping a long time series of measurements into fixed time windows, e.g.
// per-minute results of a sustained-load (thermal drift) run.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "computeedge/bench/environment.h"
#include "computeedge/bench/runner.h"

namespace computeedge::bench {

// One measured call in a time series.
struct TimedSample {
  double t_s = 0;     // seconds since the start of the run (non-decreasing across samples)
  size_t series = 0;  // index into the series list passed to window_results()
  uint64_t ns = 0;    // duration of the call
  ThermalState thermal = ThermalState::Unknown;
};

struct SeriesInfo {
  std::string name;
  double flops_per_call = 0;  // 0 if not applicable
};

// Returns one Result per (window, series) that has at least one sample, ordered by
// window and then by series index. Window k covers t_s in [k * window_s, (k+1) * window_s).
// Each Result is named `name`, has `base_params` plus {series_key: series name,
// window_key: k}, carries the samples in time order with their thermal states,
// thermal_before / thermal_after from the first / last sample, and its summary.
std::vector<Result> window_results(const std::string& name, const std::vector<TimedSample>& samples,
                                   const std::vector<SeriesInfo>& series, double window_s,
                                   const std::string& series_key, const std::string& window_key,
                                   const std::map<std::string, std::string>& base_params);

}  // namespace computeedge::bench
