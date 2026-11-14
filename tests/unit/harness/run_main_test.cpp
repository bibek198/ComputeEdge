#include "harness/run_main.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "computeedge/bench/environment.h"
#include "computeedge/bench/runner.h"
#include "computeedge/core/error.h"

namespace cb = computeedge::bench;
namespace harness = computeedge::bench::harness;

namespace {

std::string read_file(const std::filesystem::path& p) {
  std::ifstream f(p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

TEST(RunMain, ReturnsZeroOnSuccess) {
  const int code = harness::run_main([] { return 0; });
  EXPECT_EQ(code, 0);
}

TEST(RunMain, CatchesUsageErrorReturnsTwo) {
  const int code = harness::run_main([]() -> int { throw computeedge::UsageError("bad flag"); });
  EXPECT_EQ(code, 2);
}

TEST(RunMain, CatchesInvalidRunErrorReturnsThree) {
  const int code = harness::run_main([]() -> int { throw computeedge::InvalidRunError("power changed"); });
  EXPECT_EQ(code, 3);
}

TEST(RunMain, CatchesErrorReturnsOne) {
  const int code = harness::run_main([]() -> int { throw computeedge::Error("bad alloc"); });
  EXPECT_EQ(code, 1);
}

TEST(RunMain, CatchesStdExceptionReturnsOne) {
  const int code = harness::run_main([]() -> int { throw std::runtime_error("std error"); });
  EXPECT_EQ(code, 1);
}

TEST(RunMain, FlushesPartialResultsOnFailure) {
  const auto root = std::filesystem::temp_directory_path() / ("ce_run_main_partial_" + std::to_string(::getpid()));
  std::filesystem::remove_all(root);

  harness::CommonOptions opt;
  opt.out = root.string();
  opt.write = true;

  cb::Environment env = cb::record_environment();
  harness::Reporter report;

  cb::Config cfg;
  cfg.warmup = 0;
  cfg.min_warmup_ns = 0;
  cfg.repetitions = 1;
  report.result(cb::run("step1", {}, cfg, [] {}));

  const int code = harness::run_main(&opt, &env, &report,
                                     [&]() -> int { throw computeedge::Error("simulated mid-benchmark failure"); });

  EXPECT_EQ(code, 1);

  // Check that partial output directory was written with PARTIAL marker
  bool found_partial = false;
  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (entry.is_directory() && std::filesystem::exists(entry.path() / "PARTIAL")) {
      found_partial = true;
      EXPECT_TRUE(std::filesystem::exists(entry.path() / "summary.csv"));
      EXPECT_EQ(read_file(entry.path() / "PARTIAL"), "partial\n");
    }
  }
  EXPECT_TRUE(found_partial);

  std::filesystem::remove_all(root);
}
