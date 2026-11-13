#pragma once

// Argument validation shared by every matmul backend (CPU and Metal).

#include "computeedge/core/tensor.h"

namespace computeedge {

// Shared argument checks for every matmul backend. Throw Error on misuse.
void validate_matmul(const Tensor& a, const Tensor& b);
void validate_matmul_out(const Tensor& a, const Tensor& b, const Tensor& out);

}  // namespace computeedge
