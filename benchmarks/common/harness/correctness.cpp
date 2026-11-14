#include "harness/correctness.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace computeedge::bench::harness {

Tensor random_square(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n * n);
  for (float& x : v) x = dist(rng);
  return Tensor({n, n}, v);
}

double max_scaled_error(const Tensor& got, const Tensor& ref, size_t k) {
  const std::vector<float> g = got.to_vector(), r = ref.to_vector();
  double worst = 0;
  for (size_t i = 0; i < g.size(); ++i) {
    const double err = std::abs(double(g[i]) - double(r[i])) / (std::abs(double(r[i])) + 1.0);
    worst = std::max(worst, err);
  }
  return worst / std::sqrt(double(k));
}

}  // namespace computeedge::bench::harness
