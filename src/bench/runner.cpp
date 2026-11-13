#include "computeedge/bench/runner.h"

#include <map>
#include <string>

#include "computeedge/bench/trace.h"
#include "computeedge/core/error.h"
#include "computeedge/core/instrumentation.h"

namespace computeedge::bench {

double Result::gflops_median() const {
  if (flops_per_call <= 0 || summary.median <= 0) return 0;
  return flops_per_call / summary.median;  // flop/ns == GFLOP/s
}

namespace {

std::string trace_params(const std::map<std::string, std::string>& params) {
  std::string out;
  for (const auto& [k, v] : params) out += (out.empty() ? "" : ",") + k + "=" + v;
  return out;
}

void trace_run(const char* kind, const std::string& name, const std::map<std::string, std::string>& params,
               const Config& c, bool has_setup, double flops_per_call) {
  if (!trace_enabled()) return;
  trace(std::string(kind) + " name=" + name + " params={" + trace_params(params) +
        "} warmup=" + std::to_string(c.warmup) + " min_warmup_ns=" + std::to_string(c.min_warmup_ns) +
        " reps=" + std::to_string(c.repetitions) + " inner=" + std::to_string(c.inner_iterations) +
        " min_sample_ns=" + std::to_string(c.min_sample_ns) + " cooldown=" + (c.cooldown ? "1" : "0") +
        " setup=" + (has_setup ? "1" : "0") + " flops=" + std::to_string(flops_per_call));
}

// `measuring` (optional) is set to true right before the measured samples start,
// so callers can tell warm-up/calibration calls from measured ones.
Result run_impl(const std::string& name, const std::map<std::string, std::string>& params, const Config& config,
                const std::function<void()>& fn, double flops_per_call, const std::function<void()>& setup,
                bool* measuring) {
  if (config.repetitions < 1) throw Error("bench::run: repetitions must be >= 1");

  Result r;
  r.name = name;
  r.params = params;
  r.flops_per_call = flops_per_call;

  if (config.cooldown) wait_for_thermal(ThermalState::Nominal, config.cooldown_timeout_s);
  r.thermal_before = thermal_state();

  const uint64_t warmup_start = now_ns();
  for (int i = 0; i < config.warmup || now_ns() - warmup_start < config.min_warmup_ns; ++i) {
    if (setup) setup();
    fn();
  }

  // Calibrate calls per sample so a sample is long enough to time reliably.
  int inner = config.inner_iterations;
  if (inner <= 0) {
    inner = 1;
    while (inner < (1 << 20)) {
      if (setup) setup();
      const uint64_t t0 = now_ns();
      for (int i = 0; i < inner; ++i) fn();
      if (now_ns() - t0 >= config.min_sample_ns) break;
      inner *= 2;
    }
  }
  r.inner_iterations = inner;

  if (measuring) *measuring = true;
  r.samples_ns.reserve(static_cast<size_t>(config.repetitions));
  r.sample_thermal.reserve(static_cast<size_t>(config.repetitions));
  for (int rep = 0; rep < config.repetitions; ++rep) {
    if (setup) setup();
    const uint64_t t0 = now_ns();
    for (int i = 0; i < inner; ++i) fn();
    const uint64_t t1 = now_ns();
    r.samples_ns.push_back(static_cast<double>(t1 - t0) / inner);
    r.sample_thermal.push_back(thermal_state());
  }

  r.thermal_after = thermal_state();
  r.summary = summarize(r.samples_ns);
  return r;
}

}  // namespace

Result run(const std::string& name, const std::map<std::string, std::string>& params, const Config& config,
           const std::function<void()>& fn, double flops_per_call, const std::function<void()>& setup) {
  trace_run("run", name, params, config, static_cast<bool>(setup), flops_per_call);
  return run_impl(name, params, config, fn, flops_per_call, setup, nullptr);
}

Result run_with_metrics(const std::string& name, const std::map<std::string, std::string>& params, const Config& config,
                        const MetricsFn& fn, double flops_per_call, const std::function<void()>& setup) {
  Config cfg = config;
  cfg.inner_iterations = 1;  // one call per sample, so metrics align with samples
  trace_run("run_with_metrics", name, params, cfg, static_cast<bool>(setup), flops_per_call);
  bool measuring = false;
  std::vector<Metrics> collected;
  collected.reserve(static_cast<size_t>(cfg.repetitions));
  Metrics scratch;

  Result r = run_impl(
      name, params, cfg,
      [&] {
        scratch.clear();
        fn(scratch);
        if (measuring) collected.push_back(scratch);
      },
      flops_per_call, setup, &measuring);

  for (const Metrics& m : collected)
    for (const auto& [key, value] : m) r.metrics[key].push_back(value);
  for (const auto& [key, values] : r.metrics) {
    if (values.size() != r.samples_ns.size()) {
      throw Error("run_with_metrics: metric '" + key + "' was not reported on every call");
    }
    r.metric_summaries[key] = summarize(values);
  }
  return r;
}

}  // namespace computeedge::bench
