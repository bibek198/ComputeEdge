#pragma once

// Correctness gates run before timing: every benchmark checks its result against
// a reference before measuring it.

#include <cstddef>
#include <cstdint>

#include "computeedge/core/tensor.h"

namespace computeedge::bench::harness {

// Maximum allowed max_scaled_error() for FP32 GEMM results.
inline constexpr double kCorrectnessTolerance = 1e-5;

// An n x n tensor of U[-1, 1) values from a fixed-seed mt19937.
Tensor random_square(size_t n, uint32_t seed);

// max_i |got_i - ref_i| / (|ref_i| + 1), divided by sqrt(k) to allow for FP32
// accumulation-order differences over an inner dimension of k.
double max_scaled_error(const Tensor& got, const Tensor& ref, size_t k);

}  // namespace computeedge::bench::harness
