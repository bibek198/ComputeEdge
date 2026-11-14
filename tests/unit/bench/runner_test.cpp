#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "computeedge/bench/runner.h"
#include "computeedge/bench/trace.h"
#include "computeedge/core/error.h"
#include "computeedge/core/instrumentation.h"

#include <unistd.h>
#include <algorithm>

namespace cb = computeedge::bench;

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

TEST(Runner, MetricsAlignWithMeasuredSamples) {
  cb::Config cfg;
  cfg.warmup = 3;
  cfg.min_warmup_ns = 0;
  cfg.repetitions = 5;
  int call = 0;
  const cb::Result r = cb::run_with_metrics("m", {}, cfg, [&](cb::Metrics& m) {
    ++call;
    m["call"] = call;
    m["const"] = 7;
  });
  EXPECT_EQ(r.inner_iterations, 1);
  ASSERT_EQ(r.samples_ns.size(), 5u);
  ASSERT_EQ(r.metrics.at("call").size(), 5u);
  // Warm-up calls (1..3) are excluded; measured calls are 4..8.
  EXPECT_EQ(r.metrics.at("call").front(), 4);
  EXPECT_EQ(r.metrics.at("call").back(), 8);
  EXPECT_EQ(r.metric_summaries.at("const").median, 7);
  EXPECT_EQ(r.metric_summaries.at("call").median, 6);
}

TEST(Runner, MetricsSetupRunsBeforeEveryCallUntimed) {
  cb::Config cfg;
  cfg.warmup = 3;
  cfg.min_warmup_ns = 0;
  cfg.repetitions = 5;
  int calls = 0, setups = 0;
  const cb::Result r = cb::run_with_metrics(
      "setup", {}, cfg,
      [&](cb::Metrics& m) {
        EXPECT_EQ(setups, calls + 1) << "setup must run right before each call";
        ++calls;
        m["x"] = 1;
      },
      0,
      [&] {
        ++setups;
        const uint64_t until = computeedge::now_ns() + 2'000'000;  // 2 ms, must not be timed
        while (computeedge::now_ns() < until) {
        }
      });
  EXPECT_EQ(calls, 3 + 5);
  EXPECT_EQ(setups, 3 + 5);
  for (double s : r.samples_ns) EXPECT_LT(s, 1'000'000.0);
}

TEST(Runner, MetricsMustBeReportedEveryCall) {
  cb::Config cfg;
  cfg.warmup = 0;
  cfg.min_warmup_ns = 0;
  cfg.repetitions = 4;
  int call = 0;
  EXPECT_THROW(cb::run_with_metrics("m", {}, cfg,
                                    [&](cb::Metrics& m) {
                                      if (++call % 2) m["sometimes"] = 1;
                                    }),
               computeedge::Error);
}

TEST(Trace, InertWithoutEnvironmentVariable) {
  // The verification trace must never activate in normal runs (CE_TRACE unset in tests).
  ASSERT_EQ(std::getenv("CE_TRACE"), nullptr);
  EXPECT_FALSE(cb::trace_enabled());
  cb::trace("this line goes nowhere");  // must be a harmless no-op
}
