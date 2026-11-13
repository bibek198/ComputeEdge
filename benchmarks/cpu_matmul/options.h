#pragma once

// Command-line options of the cpu_matmul benchmark (see main.cpp for usage).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "computeedge/cpu/matmul.h"
#include "harness/cli.h"

namespace computeedge::bench::cpu_matmul {

inline constexpr const char* kSource = "benchmarks/cpu_matmul/main.cpp";

// Binary-specific options on top of the common flags (--warmup --min-warmup-ms --reps --out --cooldown --no-write; no
// --prewarm-ms).
struct Options : harness::CommonOptions {
  std::vector<size_t> sizes{128, 512, 1024, 2048, 4096};
  std::vector<std::string> impls{"accelerate", "naive"};
  std::vector<std::string> modes{"kernel", "e2e"};
  size_t naive_max = 512;
};

// Parses and validates argv; exits with status 2 on a usage error.
Options parse_options(int argc, char** argv);

// The MatmulImpl named `s` (its to_string()); a usage error otherwise.
MatmulImpl impl_from(const std::string& s);

}  // namespace computeedge::bench::cpu_matmul
