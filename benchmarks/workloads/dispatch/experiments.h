#pragma once

// Measurement logic of the dispatch_overhead benchmark: everything that runs after the banner.
// Reports results and progress through `report`; never prints directly.

#include "harness/results.h"
#include "options.h"

namespace computeedge::bench::dispatch_overhead {

// Runs every configured experiment. Returns 0, or 1 if a correctness gate failed
// (in which case nothing should be written).
int run(const Options& opt, harness::Reporter& report);

}  // namespace computeedge::bench::dispatch_overhead
