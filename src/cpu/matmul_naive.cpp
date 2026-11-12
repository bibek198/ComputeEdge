// Naive CPU matrix multiplication. This is the correctness reference, not a
// performance baseline: Accelerate's cblas_sgemm takes that role.

#include <cstring>

#include "kernels.h"

namespace computeedge::detail {

void matmul_naive_kernel(const Tensor& a, const Tensor& b, Tensor& out) {
  const size_t M = a.shape()[0];
  const size_t K = a.shape()[1];
  const size_t N = b.shape()[1];

  const Tensor ac = a.contiguous();
  const Tensor bc = b.contiguous();
  const float* A = ac.numel() ? ac.data() : nullptr;
  const float* B = bc.numel() ? bc.data() : nullptr;
  float* C = out.data();
  if (M * N) std::memset(C, 0, M * N * sizeof(float));

  // i-k-j order: streams rows of B and C, which is cache-friendly for row-major data.
  for (size_t i = 0; i < M; ++i) {
    float* c_row = C + i * N;
    for (size_t k = 0; k < K; ++k) {
      const float a_ik = A[i * K + k];
      const float* b_row = B + k * N;
      for (size_t j = 0; j < N; ++j) c_row[j] += a_ik * b_row[j];
    }
  }
}

}  // namespace computeedge::detail
