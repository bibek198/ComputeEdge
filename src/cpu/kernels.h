#pragma once

// Internal CPU kernels. Callers (src/cpu/matmul.cpp) have already validated shapes
// and guarantee `out` is a contiguous [M,N] tensor.

#include "computeedge/tensor.h"

namespace computeedge::detail {

void matmul_naive_kernel(const Tensor& a, const Tensor& b, Tensor& out);
void matmul_accelerate_kernel(const Tensor& a, const Tensor& b, Tensor& out);

}  // namespace computeedge::detail
