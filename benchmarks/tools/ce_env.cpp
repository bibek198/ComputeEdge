// Prints the benchmark environment record (the same env.json the C++ harness
// writes), so non-C++ benchmarks (e.g. the MLX script) record identical fields.
//
//   ce_env            full environment as JSON
//   ce_env --thermal  current thermal state only (nominal/fair/serious/critical)

#include <cstring>
#include <iostream>

#include "computeedge/benchmark.h"

int main(int argc, char** argv) {
  namespace cb = computeedge::bench;
  if (argc > 1 && std::strcmp(argv[1], "--thermal") == 0) {
    std::cout << cb::to_string(cb::thermal_state()) << '\n';
    return 0;
  }
  std::cout << cb::to_json(cb::record_environment());
  return 0;
}
