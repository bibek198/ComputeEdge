#include "computeedge/bench/windows.h"

#include <algorithm>
#include <utility>

#include "computeedge/bench/stats.h"

namespace computeedge::bench {

std::vector<Result> window_results(const std::string& name, const std::vector<TimedSample>& samples,
                                   const std::vector<SeriesInfo>& series, double window_s,
                                   const std::string& series_key, const std::string& window_key,
                                   const std::map<std::string, std::string>& base_params) {
  std::vector<Result> results;
  if (samples.empty()) return results;
  double last_t = 0;
  for (const TimedSample& s : samples) last_t = std::max(last_t, s.t_s);
  const size_t windows = static_cast<size_t>(last_t / window_s) + 1;
  for (size_t w = 0; w < windows; ++w) {
    for (size_t i = 0; i < series.size(); ++i) {
      Result r;
      r.name = name;
      r.params = base_params;
      r.params[series_key] = series[i].name;
      r.params[window_key] = std::to_string(w);
      r.flops_per_call = series[i].flops_per_call;
      for (const TimedSample& s : samples) {
        if (s.series != i || static_cast<size_t>(s.t_s / window_s) != w) continue;
        r.samples_ns.push_back(double(s.ns));
        r.sample_thermal.push_back(s.thermal);
      }
      if (r.samples_ns.empty()) continue;
      r.thermal_before = r.sample_thermal.front();
      r.thermal_after = r.sample_thermal.back();
      r.summary = summarize(r.samples_ns);
      results.push_back(std::move(r));
    }
  }
  return results;
}

}  // namespace computeedge::bench
