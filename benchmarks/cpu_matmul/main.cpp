// CPU matmul benchmark: Accelerate (baseline) and naive (reference) SGEMM.
//
// Modes:
//   kernel  matmul_into() into a pre-allocated output (compute only)
//   e2e     matmul()  (output allocation + zero-fill + compute + free)
//
// Usage:
//   cpu_matmul [--sizes 128,512,1024,2048,4096] [--impls accelerate,naive]
//              [--modes kernel,e2e] [--warmup 10] [--min-warmup-ms 200] [--reps 30]
//              [--naive-max 512]
//              [--out data/scratch] [--cooldown] [--no-write]

#include "computeedge/bench/environment.h"
#include "experiments.h"
#include "harness/console.h"
#include "harness/results.h"
#include "options.h"

namespace harness = computeedge::bench::harness;
using namespace computeedge::bench::cpu_matmul;

int main(int argc, char** argv) {
  const Options opt = parse_options(argc, argv);
  const computeedge::bench::Environment env =
      harness::banner("ComputeEdge CPU matmul benchmark", harness::BannerStyle::CpuOnly);
  harness::ConsoleReporter report;
  if (const int status = run(opt, report); status != 0) return status;
  harness::write_results(opt.out, opt.write, env, report.results());
  return 0;
}
