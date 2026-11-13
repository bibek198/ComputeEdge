#pragma once

// Command-line options of the storage_modes benchmark (see main.cpp for usage).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "computeedge/metal/probes.h"
#include "harness/cli.h"

namespace computeedge::bench::storage_modes {

inline constexpr const char* kSource = "benchmarks/storage_modes/main.cpp";

// Binary-specific options on top of the common flags (all common flags).
struct Options : harness::CommonOptions {
  std::vector<size_t> sizes = harness::split_bytes("16K,256K,4M,64M,256M,1G");
  std::vector<std::string> kinds{"host", "shared", "private", "nocopy", "shared_copy"};
  std::vector<std::string> experiments{"create", "cpu_touch", "gpu_first", "gpu_bw"};
};

// Parses and validates argv; exits with status 2 on a usage error.
Options parse_options(int argc, char** argv);

// The probe::BufferKind named `s` (its to_string()); a usage error otherwise.
metal::probe::BufferKind parse_kind(const std::string& s, const char* src);

}  // namespace computeedge::bench::storage_modes
