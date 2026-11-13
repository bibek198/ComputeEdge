// Track A3/A4: how the CPU learns the GPU is done, and what idling costs.
//
// Experiments:
//   methods   every Sync method x GPU work {empty kernel, write 4 MiB, write 256 MiB}.
//             wake_ns = CPU resumes - GPUEndTime is the key metric; wait_ns is
//             the whole blocking interval.
//   idle      empty dispatch after the GPU has been idle for G ms (untimed sleep
//             before every call) with waitUntilCompleted and spin_event:
//             does an idle GPU (or sleeping CPU) make the next round trip slower?
//
// Usage:
//   sync_methods [--methods wait_until_completed,event_wait,completion_handler,spin_event,spin_status]
//                [--work empty,4M,256M] [--gaps-ms 0,0.1,1,10,100] [--experiments methods,idle]
//                + common flags (benchmarks/harness/cli.h)

#include "computeedge/bench/environment.h"
#include "experiments.h"
#include "harness/console.h"
#include "harness/gpu.h"
#include "harness/results.h"
#include "options.h"

namespace harness = computeedge::bench::harness;
using namespace computeedge::bench::sync_methods;

int main(int argc, char** argv) {
  const Options opt = parse_options(argc, argv);
  harness::require_metal();
  const computeedge::bench::Environment env =
      harness::banner("ComputeEdge M5 / Track A: synchronization methods and idle gaps", harness::BannerStyle::WithGpu);
  harness::ConsoleReporter report;
  if (const int status = run(opt, report); status != 0) return status;
  harness::write_results(opt.out, opt.write, env, report.results());
  return 0;
}
