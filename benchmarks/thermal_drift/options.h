#pragma once

// Command-line options of the thermal_drift benchmark (see main.cpp for usage).

#include <string>

#include "harness/cli.h"

namespace computeedge::bench::thermal_drift {

inline constexpr const char* kSource = "benchmarks/thermal_drift/main.cpp";

// Binary-specific options on top of all common flags. Note: the common
// --cooldown flag is accepted but unused here; waiting for thermal state
// nominal is on by default and turned off with --no-cooldown.
struct Options : harness::CommonOptions {
  double minutes = 15;
  std::string load = "mixed";  // mixed | cpu | gpu | none
  double interval_ms = 1000;
  bool wait_for_nominal = true;
};

// Parses and validates argv; exits with status 2 on a usage error.
Options parse_options(int argc, char** argv);

}  // namespace computeedge::bench::thermal_drift
