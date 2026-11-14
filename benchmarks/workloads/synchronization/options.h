#pragma once

// Command-line options of the sync_methods benchmark (see main.cpp for usage).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "computeedge/metal/probes.h"
#include "harness/cli.h"

namespace computeedge::bench::sync_methods {

inline constexpr const char* kSource = "benchmarks/sync_methods/main.cpp";

// Binary-specific options on top of the common flags (all common flags).
struct Options : harness::CommonOptions {
  std::vector<std::string> methods{"wait_until_completed", "event_wait", "completion_handler", "spin_event",
                                   "spin_status"};
  std::vector<std::string> work{"empty", "4M", "256M"};
  std::vector<double> gaps_ms{0, 0.1, 1, 10, 100};
  std::vector<std::string> experiments{"methods", "idle"};
};

// Parses and validates argv; exits with status 2 on a usage error.
Options parse_options(int argc, char** argv);

// The probe::Sync named `s` (its to_string()); a usage error otherwise.
metal::probe::Sync parse_sync(const std::string& s, const char* src);

}  // namespace computeedge::bench::sync_methods
