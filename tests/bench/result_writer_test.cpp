#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "computeedge/bench/environment.h"
#include "computeedge/bench/result_writer.h"
#include "computeedge/bench/runner.h"
#include "computeedge/core/error.h"
#include "computeedge/core/instrumentation.h"

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

TEST(Writer, WritesMetricsFiles) {
  const auto root = std::filesystem::temp_directory_path() / ("ce_bench_metrics_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);
  cb::Config cfg;
  cfg.warmup = 0;
  cfg.min_warmup_ns = 0;
  cfg.repetitions = 3;
  cb::ResultWriter w(root.string(), cb::record_environment());
  w.add(cb::run_with_metrics("op", {{"storage", "nocopy"}}, cfg, [](cb::Metrics& m) {
    m["wait_ns"] = 10;
    m["gpu_ns"] = 5;
  }));
  w.flush();
  const std::filesystem::path dir(w.directory());
  const std::string metrics = read_file(dir / "metrics.csv");
  const std::string msum = read_file(dir / "metrics_summary.csv");
  EXPECT_EQ(metrics.rfind("benchmark,storage,rep,metric,value\n", 0), 0u);
  EXPECT_EQ(count_lines(metrics), 1u + 2u * 3u);
  EXPECT_EQ(count_lines(msum), 1u + 2u);
  EXPECT_NE(read_file(dir / "env.json").find("\"compiler_flags\""), std::string::npos);
  std::filesystem::remove_all(root);
}
