#pragma once

#include "computeedge/core/tensor.h"

namespace computeedge {

enum class MatmulImpl : uint8_t {
  Naive,       // triple loop; correctness reference only
  Accelerate,  // cblas_sgemm; the optimized CPU baseline
};

const char* to_string(MatmulImpl impl);

// C[M,N] = A[M,K] * B[K,N]. Allocates the output, then calls matmul_into.
// Default implementation is Accelerate. The dispatcher (M8) will extend this
// entry point with CPU / Metal selection.
Tensor matmul(const Tensor& a, const Tensor& b, MatmulImpl impl = MatmulImpl::Accelerate);

// Writes A*B into a pre-allocated, contiguous [M,N] output (its contents are
// overwritten). Use this to time compute without output allocation.
// Emits one Dispatch event and a Begin/End span around the kernel.
void matmul_into(const Tensor& a, const Tensor& b, Tensor& out, MatmulImpl impl = MatmulImpl::Accelerate);

inline Tensor matmul_naive(const Tensor& a, const Tensor& b) { return matmul(a, b, MatmulImpl::Naive); }
inline Tensor matmul_accelerate(const Tensor& a, const Tensor& b) { return matmul(a, b, MatmulImpl::Accelerate); }

}  // namespace computeedge
