#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "computeedge/benchmark.h"
#include "computeedge/error.h"
#include "computeedge/instrumentation.h"

#include <unistd.h>
#include <algorithm>

namespace cb = computeedge::bench;

namespace {

std::string read_file(const std::filesystem::path& p) {
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

size_t count_lines(const std::string& s) { return static_cast<size_t>(std::count(s.begin(), s.end(), '\n')); }

}  // namespace

TEST(Stats, Quantiles) {
  const std::vector<double> v{5, 1, 4, 2, 3};
  EXPECT_DOUBLE_EQ(cb::quantile(v, 0.5), 3.0);
  EXPECT_DOUBLE_EQ(cb::quantile(v, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(cb::quantile(v, 1.0), 5.0);
  EXPECT_DOUBLE_EQ(cb::quantile(v, 0.25), 2.0);
  EXPECT_DOUBLE_EQ(cb::quantile({1, 2, 3, 4}, 0.5), 2.5);
  EXPECT_THROW(cb::quantile({}, 0.5), computeedge::Error);
}

TEST(Stats, SummaryFields) {
  std::vector<double> v;
  for (int i = 1; i <= 101; ++i) v.push_back(i);
  const cb::Summary s = cb::summarize(v);
  EXPECT_EQ(s.n, 101u);
  EXPECT_DOUBLE_EQ(s.min, 1);
  EXPECT_DOUBLE_EQ(s.max, 101);
  EXPECT_DOUBLE_EQ(s.mean, 51);
  EXPECT_DOUBLE_EQ(s.median, 51);
  EXPECT_DOUBLE_EQ(s.p25, 26);
  EXPECT_DOUBLE_EQ(s.p75, 76);
  EXPECT_DOUBLE_EQ(s.iqr, 50);
  EXPECT_NEAR(s.stddev, 29.3002, 1e-3);
  EXPECT_LE(s.ci95_lo, s.median);
  EXPECT_GE(s.ci95_hi, s.median);
  EXPECT_LT(s.ci95_hi - s.ci95_lo, s.iqr);
}

TEST(Stats, BootstrapIsDeterministic) {
  const std::vector<double> v{3, 9, 1, 7, 5, 8, 2, 6, 4};
  const cb::Summary a = cb::summarize(v);
  const cb::Summary b = cb::summarize(v);
  EXPECT_EQ(a.ci95_lo, b.ci95_lo);
  EXPECT_EQ(a.ci95_hi, b.ci95_hi);
}

TEST(Stats, ConstantSampleHasZeroSpread) {
  const cb::Summary s = cb::summarize({4, 4, 4, 4});
  EXPECT_EQ(s.iqr, 0);
  EXPECT_EQ(s.stddev, 0);
  EXPECT_EQ(s.ci95_lo, 4);
  EXPECT_EQ(s.ci95_hi, 4);
}

TEST(Environment, RecordsHardwareAndState) {
  const cb::Environment env = cb::record_environment();
  EXPECT_FALSE(env.model.empty());
  EXPECT_FALSE(env.chip.empty());
  EXPECT_GT(env.memory_bytes, 0u);
  EXPECT_GT(env.performance_cores, 0);
  EXPECT_FALSE(env.os_version.empty());
  EXPECT_FALSE(env.os_build.empty());
  EXPECT_FALSE(env.timestamp_utc.empty());
  EXPECT_NE(env.thermal_state, cb::ThermalState::Unknown);

  const std::string json = cb::to_json(env);
  EXPECT_EQ(json.front(), '{');
  EXPECT_NE(json.find("\"chip\": \""), std::string::npos);
  EXPECT_NE(json.find("\"thermal_state\""), std::string::npos);
}

TEST(Runner, CollectsRequestedSamples) {
  cb::Config cfg;
  cfg.warmup = 2;
  cfg.min_warmup_ns = 0;
  cfg.repetitions = 7;
  cfg.inner_iterations = 3;
  int calls = 0, setups = 0;
  const cb::Result r = cb::run("noop", {{"k", "v"}}, cfg, [&] { ++calls; }, 10.0, [&] { ++setups; });
  EXPECT_EQ(r.samples_ns.size(), 7u);
  EXPECT_EQ(r.sample_thermal.size(), 7u);
  EXPECT_EQ(r.inner_iterations, 3);
  EXPECT_EQ(calls, 2 + 7 * 3);
  EXPECT_EQ(setups, 2 + 7);
  EXPECT_EQ(r.summary.n, 7u);
  EXPECT_EQ(r.params.at("k"), "v");
}

TEST(Runner, AutoCalibratesInnerIterations) {
  cb::Config cfg;
  cfg.warmup = 0;
  cfg.min_warmup_ns = 0;
  cfg.repetitions = 3;
  cfg.min_sample_ns = 20'000;
  volatile double sink = 0;
  const cb::Result r = cb::run("tiny", {}, cfg, [&] { sink = sink + 1.0; });
  EXPECT_GT(r.inner_iterations, 1);
  EXPECT_GT(r.summary.median, 0);
}

TEST(Runner, GflopsFromMedian) {
  cb::Result r;
  r.flops_per_call = 2e9;
  r.summary.median = 1e9;  // 1 s
  EXPECT_DOUBLE_EQ(r.gflops_median(), 2.0);
  r.flops_per_call = 0;
  EXPECT_DOUBLE_EQ(r.gflops_median(), 0.0);
}

TEST(Writer, WritesEnvSamplesAndSummary) {
  const auto root = std::filesystem::temp_directory_path() / ("ce_bench_test_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);

  cb::Config cfg;
  cfg.warmup = 0;
  cfg.min_warmup_ns = 0;
  cfg.repetitions = 4;
  cfg.inner_iterations = 1;
  const cb::Result a = cb::run("op", {{"size", "1"}}, cfg, [] {});
  const cb::Result b = cb::run("op", {{"size", "2"}, {"mode", "x,y"}}, cfg, [] {});

  cb::ResultWriter w(root.string(), cb::record_environment());
  w.add(a);
  w.add(b);
  w.flush();

  const std::filesystem::path dir(w.directory());
  ASSERT_TRUE(std::filesystem::exists(dir / "env.json"));
  const std::string samples = read_file(dir / "samples.csv");
  const std::string summary = read_file(dir / "summary.csv");
  EXPECT_EQ(samples.rfind("benchmark,mode,size,rep,inner_iterations,ns_per_call,thermal\n", 0), 0u);
  EXPECT_EQ(count_lines(samples), 1u + 8u);
  EXPECT_EQ(count_lines(summary), 1u + 2u);
  EXPECT_NE(samples.find("\"x,y\""), std::string::npos) << "fields with commas must be quoted";

  // A second writer in the same second must not clobber the first directory.
  cb::ResultWriter w2(root.string(), cb::record_environment());
  EXPECT_NE(w2.directory(), w.directory());

  std::filesystem::remove_all(root);
}

TEST(Runner, WarmupHonoursTimeFloor) {
  cb::Config cfg;
  cfg.warmup = 1;
  cfg.min_warmup_ns = 5'000'000;  // 5 ms
  cfg.repetitions = 1;
  cfg.inner_iterations = 1;
  int calls = 0;
  const uint64_t t0 = computeedge::now_ns();
  cb::run("floor", {}, cfg, [&] { ++calls; });
  EXPECT_GE(computeedge::now_ns() - t0, 5'000'000u);
  EXPECT_GT(calls, 2) << "warm-up must keep running until the time floor";
}

TEST(Writer, ParamNamedNDoesNotCollideWithSummaryColumns) {
  const auto root = std::filesystem::temp_directory_path() / ("ce_bench_cols_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  cb::Config cfg;
  cfg.warmup = 0;
  cfg.min_warmup_ns = 0;
  cfg.repetitions = 2;
  cfg.inner_iterations = 1;
  cb::ResultWriter w(root.string(), cb::record_environment());
  w.add(cb::run("op", {{"n", "128"}}, cfg, [] {}));
  w.flush();
  const std::string summary = read_file(std::filesystem::path(w.directory()) / "summary.csv");
  const std::string header = summary.substr(0, summary.find('\n'));
  EXPECT_EQ(header.rfind("benchmark,n,n_samples,", 0), 0u) << header;
  std::filesystem::remove_all(root);
}
