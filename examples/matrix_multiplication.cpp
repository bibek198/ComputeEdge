// Minimal ComputeEdge example: 2x2 matrix multiplication with instrumentation.
//
//   ./build/examples/matrix_multiplication [events.csv]

#include <cstdio>
#include <iostream>

#include "computeedge/instrumentation.h"
#include "computeedge/scoped_timer.h"
#include "computeedge/tensor.h"

using namespace computeedge;

int main(int argc, char** argv) {
  ScopedEventLogging logging;

  uint64_t elapsed_ns = 0;
  Tensor C;
  {
    Tensor A({2, 2}, {1, 2, 3, 4});
    Tensor B({2, 2}, {5, 6, 7, 8});
    ScopedTimer timer("example_matmul", &elapsed_ns);
    C = matmul(A, B);
  }

  std::cout << "ComputeEdge Tensor Engine\n\n"
            << "Operation: Matrix Multiplication\n\n"
            << "Result:\n";
  C.print();

  const AllocStats s = default_allocator().stats();
  std::cout << "\nBackend: CPU (naive reference)\n"
            << "Execution Time: " << elapsed_ns / 1000.0 << " us\n"
            << "Allocations: " << s.alloc_count << " (live " << s.live_blocks() << ", peak " << s.peak_bytes / 1024
            << " KiB)\n"
            << "Events recorded: " << EventLog::global().size() << "\n";

  if (argc > 1) {
    EventLog::global().write_csv(argv[1]);
    std::cout << "Event log written to " << argv[1] << "\n";
  }
  return 0;
}
