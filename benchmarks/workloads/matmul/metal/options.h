#pragma once

// Command-line options of the metal_matmul benchmark (see main.cpp for usage).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "computeedge/metal/matmul.h"
#include "harness/cli.h"

namespace computeedge::bench::metal_matmul {

inline constexpr const char* kSource = "benchmarks/metal_matmul/main.cpp";

// Binary-specific options on top of the common flags (all common flags).
struct Options : harness::CommonOptions {
  std::vector<size_t> sizes{128, 512, 1024, 2048, 4096};
  std::vector<std::string> storage{"nocopy", "shared", "private"};
  std::vector<std::string> kernels{"tiled16", "mps"};
};

// Parses and validates argv; exits with status 2 on a usage error.
Options parse_options(int argc, char** argv);

// The metal::StorageMode / metal::Kernel named `s`; a usage error otherwise.
metal::StorageMode storage_from(const std::string& s);
metal::Kernel kernel_from(const std::string& s);

}  // namespace computeedge::bench::metal_matmul
