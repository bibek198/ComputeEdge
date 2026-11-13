#pragma once

// Command-line options of the event_log_overhead benchmark (see main.cpp for usage).

#include "harness/cli.h"

namespace computeedge::bench::event_log_overhead {

inline constexpr const char* kSource = "benchmarks/event_log_overhead/main.cpp";

// Only --reps, --out and --no-write are accepted.
struct Options : harness::CommonOptions {};

// Parses argv; exits with status 2 on a usage error.
Options parse_options(int argc, char** argv);

}  // namespace computeedge::bench::event_log_overhead
