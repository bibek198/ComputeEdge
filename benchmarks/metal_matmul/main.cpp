// Metal matmul benchmark: tiled FP32 kernel under each storage mode, with a
// per-call phase breakdown (buffers, copies, encode, commit, wait, GPU time,
// queue and wake-up latency).
//
// Before any measurement the GPU is pre-warmed with --prewarm-ms of continuous
// dispatches: without it, whichever configuration runs first is measured
// ~1.7x slower at 128² (the GPU had not reached steady-state clocks).
//
// Usage:
//   metal_matmul [--sizes 128,512,1024,2048,4096] [--storage nocopy,shared,private]
//                [--kernels tiled16,mps]
//                [--warmup 10] [--min-warmup-ms 200] [--prewarm-ms 1000] [--reps 30]
//                [--out data/scratch] [--cooldown] [--no-write]

#include "computeedge/bench/environment.h"
#include "experiments.h"
#include "harness/console.h"
#include "harness/gpu.h"
#include "harness/results.h"
#include "options.h"

namespace harness = computeedge::bench::harness;
using namespace computeedge::bench::metal_matmul;

int main(int argc, char** argv) {
  const Options opt = parse_options(argc, argv);
  harness::require_metal();
  const computeedge::bench::Environment env =
      harness::banner("ComputeEdge Metal matmul benchmark", harness::BannerStyle::WithGpu);
  harness::ConsoleReporter report;
  if (const int status = run(opt, report); status != 0) return status;
  harness::write_results(opt.out, opt.write, env, report.results());
  return 0;
}
