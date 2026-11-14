#pragma once

// Command-line options of the followups benchmark (see main.cpp for usage).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "harness/cli.h"

namespace computeedge::bench::followups {

inline constexpr const char* kSource = "benchmarks/followups/main.cpp";

// Binary-specific options on top of the common flags (all common flags).
struct Options : harness::CommonOptions {
  std::vector<std::string> experiments{"wake", "cpu_write", "gpu_write"};
};

// Parses and validates argv; exits with status 2 on a usage error.
Options parse_options(int argc, char** argv);

}  // namespace computeedge::bench::followups
