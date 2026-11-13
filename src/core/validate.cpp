// Argument validation shared by all matmul backends (CPU, Metal).

#include "computeedge/core/validate.h"

namespace computeedge {

void validate_matmul(const Tensor& a, const Tensor& b) {
  if (!a.defined() || !b.defined()) throw Error("matmul: undefined input");
  if (a.ndim() != 2 || b.ndim() != 2) throw Error("matmul: inputs must be 2-D");
  if (b.shape()[0] != a.shape()[1]) {
    throw Error("matmul: inner dimensions differ (" + std::to_string(a.shape()[1]) + " vs " +
                std::to_string(b.shape()[0]) + ")");
  }
}

void validate_matmul_out(const Tensor& a, const Tensor& b, const Tensor& out) {
  validate_matmul(a, b);
  const size_t M = a.shape()[0];
  const size_t N = b.shape()[1];
  if (!out.defined() || out.ndim() != 2 || out.shape()[0] != M || out.shape()[1] != N || !out.is_contiguous()) {
    throw Error("matmul_into: output must be a contiguous [" + std::to_string(M) + ", " + std::to_string(N) +
                "] tensor");
  }
  if (out.storage() == a.storage() || out.storage() == b.storage()) {
    throw Error("matmul_into: output must not alias an input");
  }
}

}  // namespace computeedge
