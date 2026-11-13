#include "computeedge/benchmark.h"

#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#include "computeedge/error.h"
#include "computeedge/instrumentation.h"

namespace computeedge::bench {

double Result::gflops_median() const {
  if (flops_per_call <= 0 || summary.median <= 0) return 0;
  return flops_per_call / summary.median;  // flop/ns == GFLOP/s
}

namespace {

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
  return run_impl(name, params, config, fn, flops_per_call, setup, nullptr);
}

Result run_with_metrics(const std::string& name, const std::map<std::string, std::string>& params,
                        const Config& config, const MetricsFn& fn, double flops_per_call,
                        const std::function<void()>& setup) {
  Config cfg = config;
  cfg.inner_iterations = 1;  // one call per sample, so metrics align with samples
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

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

namespace {

std::string local_stamp() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  localtime_r(&now, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
  return buf;
}

std::string csv_field(const std::string& s) {
  if (s.find_first_of(",\"\n") == std::string::npos) return s;
  std::string out = "\"";
  for (char c : s) out += (c == '"') ? std::string("\"\"") : std::string(1, c);
  return out + "\"";
}

std::vector<std::string> param_keys(const std::vector<Result>& results) {
  std::set<std::string> keys;
  for (const Result& r : results)
    for (const auto& [k, v] : r.params) keys.insert(k);
  return {keys.begin(), keys.end()};
}

std::string param_value(const Result& r, const std::string& key) {
  auto it = r.params.find(key);
  return it == r.params.end() ? "" : it->second;
}

void write_metrics(const std::filesystem::path& dir, const std::vector<Result>& results,
                   const std::vector<std::string>& keys);

}  // namespace

ResultWriter::ResultWriter(const std::string& root, const Environment& env) : env_(env) {
  std::string sha = env.git_sha.empty() ? "unknown" : env.git_sha;
  std::filesystem::path base = std::filesystem::path(root) / (local_stamp() + "_" + sha);
  std::filesystem::path dir = base;
  for (int i = 1; std::filesystem::exists(dir); ++i) dir = base.string() + "_" + std::to_string(i);
  std::filesystem::create_directories(dir);
  dir_ = dir.string();
}

void ResultWriter::add(const Result& result) { results_.push_back(result); }

void ResultWriter::flush() const {
  const std::filesystem::path dir(dir_);
  {
    std::ofstream f(dir / "env.json");
    if (!f) throw Error("ResultWriter: cannot write env.json in " + dir_);
    f << to_json(env_);
  }

  const std::vector<std::string> keys = param_keys(results_);
  {
    std::ofstream f(dir / "samples.csv");
    f << "benchmark";
    for (const auto& k : keys) f << ',' << csv_field(k);
    f << ",rep,inner_iterations,ns_per_call,thermal\n";
    for (const Result& r : results_) {
      for (size_t i = 0; i < r.samples_ns.size(); ++i) {
        f << csv_field(r.name);
        for (const auto& k : keys) f << ',' << csv_field(param_value(r, k));
        f << ',' << i << ',' << r.inner_iterations << ',' << r.samples_ns[i] << ','
          << to_string(r.sample_thermal[i]) << '\n';
      }
    }
  }
  {
    std::ofstream f(dir / "summary.csv");
    f << "benchmark";
    for (const auto& k : keys) f << ',' << csv_field(k);
    f << ",n_samples,inner_iterations,median_ns,p25_ns,p75_ns,iqr_ns,ci95_lo_ns,ci95_hi_ns,mean_ns,stddev_ns,min_ns,max_ns,"
         "gflops_median,thermal_before,thermal_after\n";
    for (const Result& r : results_) {
      const Summary& s = r.summary;
      f << csv_field(r.name);
      for (const auto& k : keys) f << ',' << csv_field(param_value(r, k));
      f << ',' << s.n << ',' << r.inner_iterations << ',' << s.median << ',' << s.p25 << ',' << s.p75 << ',' << s.iqr
        << ',' << s.ci95_lo << ',' << s.ci95_hi << ',' << s.mean << ',' << s.stddev << ',' << s.min << ',' << s.max
        << ',' << r.gflops_median() << ',' << to_string(r.thermal_before) << ',' << to_string(r.thermal_after)
        << '\n';
    }
  }
  write_metrics(dir, results_, keys);
}

namespace {

void write_metrics(const std::filesystem::path& dir, const std::vector<Result>& results,
                   const std::vector<std::string>& keys) {
  bool any = false;
  for (const Result& r : results) any = any || !r.metrics.empty();
  if (!any) return;

  std::ofstream f(dir / "metrics.csv");
  f << "benchmark";
  for (const auto& k : keys) f << ',' << csv_field(k);
  f << ",rep,metric,value\n";
  std::ofstream g(dir / "metrics_summary.csv");
  g << "benchmark";
  for (const auto& k : keys) g << ',' << csv_field(k);
  g << ",metric,n_samples,median,p25,p75,iqr,ci95_lo,ci95_hi,mean,min,max\n";

  for (const Result& r : results) {
    for (const auto& [metric, values] : r.metrics) {
      for (size_t i = 0; i < values.size(); ++i) {
        f << csv_field(r.name);
        for (const auto& k : keys) f << ',' << csv_field(param_value(r, k));
        f << ',' << i << ',' << csv_field(metric) << ',' << values[i] << '\n';
      }
      const Summary& s = r.metric_summaries.at(metric);
      g << csv_field(r.name);
      for (const auto& k : keys) g << ',' << csv_field(param_value(r, k));
      g << ',' << csv_field(metric) << ',' << s.n << ',' << s.median << ',' << s.p25 << ',' << s.p75 << ',' << s.iqr
        << ',' << s.ci95_lo << ',' << s.ci95_hi << ',' << s.mean << ',' << s.min << ',' << s.max << '\n';
    }
  }
}

}  // namespace

std::string format_result(const Result& r) {
  std::ostringstream os;
  os << r.name;
  for (const auto& [k, v] : r.params) os << ' ' << k << '=' << v;
  char buf[256];
  const Summary& s = r.summary;
  std::snprintf(buf, sizeof(buf), "  median %.3f us  IQR %.3f us  CI95 [%.3f, %.3f] us", s.median / 1e3, s.iqr / 1e3,
                s.ci95_lo / 1e3, s.ci95_hi / 1e3);
  os << buf;
  if (r.flops_per_call > 0) {
    std::snprintf(buf, sizeof(buf), "  %.1f GFLOP/s", r.gflops_median());
    os << buf;
  }
  for (const auto& [metric, ms] : r.metric_summaries) {
    std::snprintf(buf, sizeof(buf), "\n    %-14s median %10.3f us  IQR %8.3f us", metric.c_str(), ms.median / 1e3,
                  ms.iqr / 1e3);
    os << buf;
  }
  if (!r.metric_summaries.empty()) os << "\n   ";
  os << "  [" << to_string(r.thermal_before) << "->" << to_string(r.thermal_after) << ", x" << r.inner_iterations
     << "]";
  return os.str();
}

}  // namespace computeedge::bench
