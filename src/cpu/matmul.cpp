// matmul entry points: validation, instrumentation and implementation selection.

#include "computeedge/instrumentation.h"
#include "computeedge/ops.h"
#include "computeedge/scoped_timer.h"
#include "kernels.h"

namespace computeedge {

const char* to_string(MatmulImpl impl) {
  switch (impl) {
    case MatmulImpl::Naive: return "naive";
    case MatmulImpl::Accelerate: return "accelerate";
  }
  return "unknown";
}

namespace {

void check_inputs(const Tensor& a, const Tensor& b) {
  if (!a.defined() || !b.defined()) throw Error("matmul: undefined input");
  if (a.ndim() != 2 || b.ndim() != 2) throw Error("matmul: inputs must be 2-D");
  if (b.shape()[0] != a.shape()[1]) {
    throw Error("matmul: inner dimensions differ (" + std::to_string(a.shape()[1]) + " vs " +
                std::to_string(b.shape()[0]) + ")");
  }
}

const char* kernel_label(MatmulImpl impl) {
  switch (impl) {
    case MatmulImpl::Naive: return "matmul_naive";
    case MatmulImpl::Accelerate: return "matmul_accelerate";
  }
  return "matmul";
}

}  // namespace

void matmul_into(const Tensor& a, const Tensor& b, Tensor& out, MatmulImpl impl) {
  check_inputs(a, b);
  const size_t M = a.shape()[0];
  const size_t N = b.shape()[1];
  if (!out.defined() || out.ndim() != 2 || out.shape()[0] != M || out.shape()[1] != N || !out.is_contiguous()) {
    throw Error("matmul_into: output must be a contiguous [" + std::to_string(M) + ", " + std::to_string(N) +
                "] tensor");
  }
  if (out.storage() == a.storage() || out.storage() == b.storage()) {
    throw Error("matmul_into: output must not alias an input");
  }

  const uint64_t exec = next_exec_id();
  const char* label = kernel_label(impl);
  emit(EventType::Dispatch, exec, out.nbytes(), Backend::CPU, exec, label);
  ScopedTimer timer(label, nullptr, exec, Backend::CPU);

  switch (impl) {
    case MatmulImpl::Naive: detail::matmul_naive_kernel(a, b, out); break;
    case MatmulImpl::Accelerate: detail::matmul_accelerate_kernel(a, b, out); break;
  }
}

Tensor matmul(const Tensor& a, const Tensor& b, MatmulImpl impl) {
  check_inputs(a, b);
  Tensor out({a.shape()[0], b.shape()[1]}, a.storage()->allocator());
  matmul_into(a, b, out, impl);
  return out;
}

}  // namespace computeedge
