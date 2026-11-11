// Naive CPU matrix multiplication. This is the correctness reference, not a
// performance baseline: Accelerate's cblas_sgemm takes that role in M2.

#include "computeedge/instrumentation.h"
#include "computeedge/scoped_timer.h"
#include "computeedge/tensor.h"

namespace computeedge {

Tensor matmul_naive(const Tensor& a, const Tensor& b) {
  if (!a.defined() || !b.defined()) throw Error("matmul: undefined input");
  if (a.ndim() != 2 || b.ndim() != 2) throw Error("matmul: inputs must be 2-D");
  const size_t M = a.shape()[0];
  const size_t K = a.shape()[1];
  const size_t N = b.shape()[1];
  if (b.shape()[0] != K) {
    throw Error("matmul: inner dimensions differ (" + std::to_string(K) + " vs " + std::to_string(b.shape()[0]) +
                ")");
  }

  const uint64_t exec = next_exec_id();
  emit(EventType::Dispatch, exec, M * N * sizeof(float), Backend::CPU, exec, "matmul_naive");
  ScopedTimer timer("matmul_naive", nullptr, exec, Backend::CPU);

  const Tensor ac = a.contiguous();
  const Tensor bc = b.contiguous();
  Tensor c({M, N}, a.storage()->allocator());

  const float* A = ac.data();
  const float* B = bc.data();
  float* C = c.data();

  // i-k-j order: streams rows of B and C, which is cache-friendly for row-major data.
  for (size_t i = 0; i < M; ++i) {
    float* c_row = C + i * N;
    for (size_t k = 0; k < K; ++k) {
      const float a_ik = A[i * K + k];
      const float* b_row = B + k * N;
      for (size_t j = 0; j < N; ++j) c_row[j] += a_ik * b_row[j];
    }
  }
  return c;
}

}  // namespace computeedge
