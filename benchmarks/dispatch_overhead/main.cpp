// Track A1/A2: fixed cost of GPU work submission, measured with empty kernels
// (one thread, no memory access) so every nanosecond is runtime overhead.
//
// Experiments (all synchronize with waitUntilCompleted):
//   cold          first empty dispatch in the process (includes compiling the probe
//                 library; one sample; the OS shader cache may already hold it)
//   dispatches    1 command buffer, 1 encoder, N dispatches        -> marginal cost per dispatch
//   encoders      1 command buffer, N encoders x 1 dispatch        -> marginal cost per encoder
//   cmdbufs       N command buffers x 1 encoder x 1 dispatch,       -> marginal cost per command buffer
//                 committed back to back, wait on the last
//
// Usage:
//   dispatch_overhead [--dispatches 1,4,16,64,256,1024] [--encoders 1,4,16,64,256]
//                     [--cmdbufs 1,2,4,8,16,32,64] [--experiments dispatches,encoders,cmdbufs]
//                     + common flags (benchmarks/harness/cli.h)

#include "computeedge/bench/environment.h"
#include "experiments.h"
#include "harness/console.h"
#include "harness/gpu.h"
#include "harness/results.h"
#include "options.h"

namespace harness = computeedge::bench::harness;
using namespace computeedge::bench::dispatch_overhead;

int main(int argc, char** argv) {
  const Options opt = parse_options(argc, argv);
  harness::require_metal();
  const computeedge::bench::Environment env =
      harness::banner("ComputeEdge M5 / Track A: dispatch overhead (empty kernels)", harness::BannerStyle::WithGpu);
  harness::ConsoleReporter report;
  if (const int status = run(opt, report); status != 0) return status;
  harness::write_results(opt.out, opt.write, env, report.results());
  return 0;
}
