#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "computeedge/bench/stats.h"
#include "computeedge/core/error.h"

namespace computeedge::bench {

namespace {

double sorted_quantile(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) throw Error("quantile: empty sample");
  if (q <= 0) return sorted.front();
  if (q >= 1) return sorted.back();
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(pos);
  const size_t hi = std::min(lo + 1, sorted.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return sorted[lo] + frac * (sorted[hi] - sorted[lo]);
}

}  // namespace

double quantile(std::vector<double> values, double q) {
  std::sort(values.begin(), values.end());
  return sorted_quantile(values, q);
}

Summary summarize(const std::vector<double>& values, size_t resamples, uint64_t seed) {
  if (values.empty()) throw Error("summarize: empty sample");
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());

  Summary s;
  s.n = sorted.size();
  s.min = sorted.front();
  s.max = sorted.back();
  s.mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(s.n);
  double sq = 0;
  for (double v : sorted) sq += (v - s.mean) * (v - s.mean);
  s.stddev = s.n > 1 ? std::sqrt(sq / static_cast<double>(s.n - 1)) : 0.0;
  s.median = sorted_quantile(sorted, 0.5);
  s.p25 = sorted_quantile(sorted, 0.25);
  s.p75 = sorted_quantile(sorted, 0.75);
  s.iqr = s.p75 - s.p25;

  // Percentile bootstrap of the median.
  if (s.n == 1 || resamples == 0) {
    s.ci95_lo = s.ci95_hi = s.median;
    return s;
  }
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> pick(0, s.n - 1);
  std::vector<double> medians(resamples);
  std::vector<double> sample(s.n);
  for (size_t r = 0; r < resamples; ++r) {
    for (size_t i = 0; i < s.n; ++i) sample[i] = sorted[pick(rng)];
    std::sort(sample.begin(), sample.end());
    medians[r] = sorted_quantile(sample, 0.5);
  }
  std::sort(medians.begin(), medians.end());
  s.ci95_lo = sorted_quantile(medians, 0.025);
  s.ci95_hi = sorted_quantile(medians, 0.975);
  return s;
}

}  // namespace computeedge::bench
