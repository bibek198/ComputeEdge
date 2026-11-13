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

const char* kernel_label(MatmulImpl impl) {
  switch (impl) {
    case MatmulImpl::Naive: return "matmul_naive";
    case MatmulImpl::Accelerate: return "matmul_accelerate";
  }
  return "matmul";
}

}  // namespace

void matmul_into(const Tensor& a, const Tensor& b, Tensor& out, MatmulImpl impl) {
  validate_matmul_out(a, b, out);

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
  validate_matmul(a, b);
  Tensor out({a.shape()[0], b.shape()[1]}, a.storage()->allocator());
  matmul_into(a, b, out, impl);
  return out;
}

}  // namespace computeedge
