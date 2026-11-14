#pragma once

// Command-line options of the cache_interference benchmark (see main.cpp for usage).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "harness/cli.h"

namespace computeedge::bench::cache_interference {

inline constexpr const char* kSource = "benchmarks/cache_interference/main.cpp";

// Binary-specific options on top of the common flags (all common flags).
struct Options : harness::CommonOptions {
  std::vector<size_t> working_sets = harness::split_bytes("1M,4M,12M,64M");
  std::vector<size_t> gpu_sizes = harness::split_bytes("16M,64M,256M");
  std::vector<std::string> passes{"read", "slc", "sgemm"};
};

// Parses and validates argv; exits with status 2 on a usage error.
Options parse_options(int argc, char** argv);

}  // namespace computeedge::bench::cache_interference
