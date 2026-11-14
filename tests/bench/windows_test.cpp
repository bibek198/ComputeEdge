// window_results() must reproduce the per-minute windowing that thermal_drift
// did inline before the refactor; the reference below is that code, verbatim
// apart from variable names.

#include "computeedge/bench/windows.h"

#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "computeedge/bench/stats.h"

namespace cb = computeedge::bench;

namespace {

std::vector<cb::Result> reference(const std::vector<cb::TimedSample>& samples,
                                  const std::vector<cb::SeriesInfo>& series, const std::string& load,
                                  bool power_changed) {
  std::vector<cb::Result> results;
  const size_t windows = static_cast<size_t>(samples.back().t_s / 60.0) + 1;
  for (size_t w = 0; w < windows; ++w) {
    for (size_t i = 0; i < series.size(); ++i) {
      cb::Result r;
      r.name = "thermal_drift";
      r.params = {{"workload", series[i].name},
                  {"window_min", std::to_string(w)},
                  {"load", load},
                  {"valid", power_changed ? "no_power_changed" : "yes"}};
      r.flops_per_call = series[i].flops_per_call;
      for (const cb::TimedSample& s : samples) {
        if (s.series != i || static_cast<size_t>(s.t_s / 60.0) != w) continue;
        r.samples_ns.push_back(double(s.ns));
        r.sample_thermal.push_back(s.thermal);
      }
      if (r.samples_ns.empty()) continue;
      r.thermal_before = r.sample_thermal.front();
      r.thermal_after = r.sample_thermal.back();
      r.summary = cb::summarize(r.samples_ns);
      results.push_back(std::move(r));
    }
  }
  return results;
}

void expect_same(const std::vector<cb::Result>& a, const std::vector<cb::Result>& b) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t k = 0; k < a.size(); ++k) {
    EXPECT_EQ(a[k].name, b[k].name);
    EXPECT_EQ(a[k].params, b[k].params);
    EXPECT_EQ(a[k].flops_per_call, b[k].flops_per_call);
    EXPECT_EQ(a[k].samples_ns, b[k].samples_ns);
    EXPECT_EQ(a[k].sample_thermal, b[k].sample_thermal);
    EXPECT_EQ(a[k].thermal_before, b[k].thermal_before);
    EXPECT_EQ(a[k].thermal_after, b[k].thermal_after);
    EXPECT_EQ(a[k].summary.median, b[k].summary.median);
    EXPECT_EQ(a[k].summary.ci95_lo, b[k].summary.ci95_lo);
    EXPECT_EQ(a[k].summary.ci95_hi, b[k].summary.ci95_hi);
  }
}

}  // namespace

TEST(Windows, MatchesThePreRefactorThermalWindowing) {
  const std::vector<cb::SeriesInfo> series{{"cpu_1024", 2e9}, {"cpu_2048", 16e9}, {"gpu_empty", 0}};
  std::mt19937 rng(7);
  for (int trial = 0; trial < 20; ++trial) {
    std::vector<cb::TimedSample> samples;
    double t = 0;
    const int n = 1 + static_cast<int>(rng() % 400);
    for (int k = 0; k < n; ++k) {
      t += (rng() % 2000) / 1000.0;  // 0-2 s steps, so some windows and series are empty
      samples.push_back({t, rng() % series.size(), 1000 + rng() % 100000, static_cast<cb::ThermalState>(rng() % 3)});
    }
    for (bool changed : {false, true}) {
      expect_same(cb::window_results("thermal_drift", samples, series, 60.0, "workload", "window_min",
                                     {{"load", "mixed"}, {"valid", changed ? "no_power_changed" : "yes"}}),
                  reference(samples, series, "mixed", changed));
    }
  }
}

TEST(Windows, EmptyInputGivesNoResults) {
  EXPECT_TRUE(cb::window_results("x", {}, {{"a", 0}}, 60, "s", "w", {}).empty());
}

TEST(Windows, BoundariesAreHalfOpen) {
  const std::vector<cb::TimedSample> samples{{59.999, 0, 1, cb::ThermalState::Nominal},
                                             {60.0, 0, 2, cb::ThermalState::Fair}};
  const auto r = cb::window_results("x", samples, {{"a", 0}}, 60, "s", "w", {});
  ASSERT_EQ(r.size(), 2u);
  EXPECT_EQ(r[0].params.at("w"), "0");
  EXPECT_EQ(r[1].params.at("w"), "1");
  EXPECT_EQ(r[1].thermal_before, cb::ThermalState::Fair);
}

TEST(Windows, SleepDurationGuardsUnderflow) {
  const uint64_t heat_until = 1000;
  const uint64_t now_past = 1500;
  uint64_t sleep_ns = 0;
  if (heat_until > now_past) {
    sleep_ns = heat_until - now_past;
  }
  EXPECT_EQ(sleep_ns, 0u);
}
