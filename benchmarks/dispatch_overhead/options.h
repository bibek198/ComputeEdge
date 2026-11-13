#pragma once

// Command-line options of the dispatch_overhead benchmark (see main.cpp for usage).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "harness/cli.h"

namespace computeedge::bench::dispatch_overhead {

inline constexpr const char* kSource = "benchmarks/dispatch_overhead/main.cpp";

// Binary-specific options on top of the common flags (all common flags).
struct Options : harness::CommonOptions {
  std::vector<uint32_t> dispatches{1, 4, 16, 64, 256, 1024};
  std::vector<uint32_t> encoders{1, 4, 16, 64, 256};
  std::vector<uint32_t> cmdbufs{1, 2, 4, 8, 16, 32, 64};
  std::vector<std::string> experiments{"dispatches", "encoders", "cmdbufs"};
};

// Parses and validates argv; exits with status 2 on a usage error.
Options parse_options(int argc, char** argv);

}  // namespace computeedge::bench::dispatch_overhead
