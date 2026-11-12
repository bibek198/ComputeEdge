// Accelerate cblas_sgemm: the optimized CPU baseline. On M4, Accelerate routes
// large SGEMMs to the SME/AMX matrix units, so this is the strongest CPU path.
//
// Row-major and transposed 2-D views are passed to BLAS directly (via
// CblasTrans / leading dimension); anything else is materialised first.

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <climits>
#include <cstring>

#include "kernels.h"

namespace computeedge::detail {

namespace {

struct GemmOperand {
  Tensor tensor;  // keeps a materialised copy alive if one was needed
  CBLAS_TRANSPOSE trans = CblasNoTrans;
  size_t ld = 0;
};

// Describes a 2-D tensor as (pointer, trans, ld) for row-major BLAS.
GemmOperand as_gemm_operand(const Tensor& t) {
  const size_t rows = t.shape()[0];
  const size_t cols = t.shape()[1];
  const size_t s0 = t.strides()[0];
  const size_t s1 = t.strides()[1];
  const size_t min_ld_rowmajor = cols > 0 ? cols : 1;
  const size_t min_ld_colmajor = rows > 0 ? rows : 1;

  // Row-major: unit stride along columns.
  if ((cols <= 1 || s1 == 1) && (rows <= 1 || s0 >= min_ld_rowmajor)) {
    return {t, CblasNoTrans, rows <= 1 ? min_ld_rowmajor : s0};
  }
  // Column-major (e.g. a transpose view): unit stride along rows.
  if ((rows <= 1 || s0 == 1) && (cols <= 1 || s1 >= min_ld_colmajor)) {
    return {t, CblasTrans, cols <= 1 ? min_ld_colmajor : s1};
  }
  Tensor c = t.contiguous();
  return {c, CblasNoTrans, min_ld_rowmajor};
}

int checked_int(size_t v, const char* what) {
  if (v > static_cast<size_t>(INT_MAX)) throw Error(std::string("matmul_accelerate: ") + what + " exceeds INT_MAX");
  return static_cast<int>(v);
}

}  // namespace

void matmul_accelerate_kernel(const Tensor& a, const Tensor& b, Tensor& out) {
  const size_t M = a.shape()[0];
  const size_t K = a.shape()[1];
  const size_t N = b.shape()[1];
  if (M == 0 || N == 0) return;
  if (K == 0) {
    std::memset(out.data(), 0, M * N * sizeof(float));
    return;
  }

  const GemmOperand A = as_gemm_operand(a);
  const GemmOperand B = as_gemm_operand(b);
  cblas_sgemm(CblasRowMajor, A.trans, B.trans, checked_int(M, "M"), checked_int(N, "N"), checked_int(K, "K"), 1.0f,
              A.tensor.data(), checked_int(A.ld, "lda"), B.tensor.data(), checked_int(B.ld, "ldb"), 0.0f, out.data(),
              checked_int(N, "ldc"));
}

}  // namespace computeedge::detail
