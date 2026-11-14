#pragma once

// Measurement logic of the event_log_overhead benchmark.

#include "harness/results.h"
#include "options.h"

namespace computeedge::bench::event_log_overhead {

// Measures emit() and ScopedTimer with the log disabled and enabled, and now_ns().
void run(const Options& opt, harness::Reporter& report);

}  // namespace computeedge::bench::event_log_overhead
