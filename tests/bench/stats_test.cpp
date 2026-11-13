#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "computeedge/bench/stats.h"
#include "computeedge/core/error.h"
#include "computeedge/core/instrumentation.h"

#include <unistd.h>
#include <algorithm>

namespace cb = computeedge::bench;

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
