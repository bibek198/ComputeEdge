// Links against computeedge_core only. If core ever depends on an Apple
// framework or a backend, this target fails to link.

#include "computeedge/allocator.h"
#include "computeedge/instrumentation.h"
#include "computeedge/ops.h"
#include "computeedge/tensor.h"

int main() {
  using namespace computeedge;
  ScopedEventLogging logging;
  Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor t = a.transpose().contiguous();
  validate_matmul(a, t);
  return (t.at({2, 1}) == 6.0f && EventLog::global().size() > 0) ? 0 : 1;
}
