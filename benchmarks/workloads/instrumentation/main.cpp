// Instrumentation overhead: cost of one emit() call with the event log disabled
// and enabled, and of a ScopedTimer. Plan §6 requires this to be measured and
// reported alongside any result collected with logging on.
//
// Usage: event_log_overhead [--reps 30] [--out data/scratch] [--no-write]

#include <iostream>

#include "computeedge/bench/environment.h"
#include "experiments.h"
#include "harness/results.h"
#include "harness/run_main.h"
#include "options.h"

namespace harness = computeedge::bench::harness;
using namespace computeedge::bench::event_log_overhead;

int main(int argc, char** argv) {
  Options opt;
  computeedge::bench::Environment env;
  harness::ConsoleReporter report;
  return harness::run_main(&opt, &env, &report, [&] {
    opt = parse_options(argc, argv);
    env = computeedge::bench::record_environment();
    std::cout << "ComputeEdge instrumentation overhead (" << env.chip << ", " << env.build_type << ")\n\n";
    run(opt, report);
    harness::write_results(opt.out, opt.write, env, report.results());
    return 0;
  });
}
