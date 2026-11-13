#pragma once

// Robust statistics for benchmark samples: quantiles, median, IQR and a
// percentile-bootstrap 95% CI of the median.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace computeedge::bench {

struct Summary {
  size_t n = 0;
  double min = 0, max = 0, mean = 0, stddev = 0;
  double median = 0, p25 = 0, p75 = 0, iqr = 0;
  double ci95_lo = 0, ci95_hi = 0;  // bootstrap 95% CI of the median
};

// Linear-interpolated quantile (q in [0,1]) of an unsorted sample.
double quantile(std::vector<double> values, double q);

// Summary statistics; the CI uses `resamples` bootstrap resamples with a fixed seed.
Summary summarize(const std::vector<double>& values, size_t resamples = 2000, uint64_t seed = 0xCE);

}  // namespace computeedge::bench
