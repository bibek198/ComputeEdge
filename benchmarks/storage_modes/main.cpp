// Track B: what each way of getting memory to the GPU costs.
//
// Experiments (sizes default 16K,256K,4M,64M,256M,1G):
//   create      create + release one MTLBuffer: shared, private, nocopy (wrap existing
//               page-aligned host memory), shared_copy (newBufferWithBytes from host)
//   cpu_touch   CPU write to freshly created memory (first touch: page faults), then
//               a second write and a read (warm), for host (posix_memalign), shared,
//               nocopy. Each call gets fresh memory (created in untimed setup).
//   gpu_first   empty kernel with one buffer bound but untouched, on a fresh buffer
//               (created in untimed setup), a fresh NoCopy wrapper over the same
//               resident memory (rewrap), or the same buffer reused: isolates
//               residency / mapping cost, and tests whether it explains the growth
//               of matmul queue time with size seen in M3/M4
//   gpu_bw      GPU read / write / copy streams over shared, private, nocopy buffers
//               (reused; bandwidth = bytes moved / gpu_ns)
//
// Usage:
//   storage_modes [--sizes 16K,256K,4M,64M,256M,1G] [--kinds host,shared,private,nocopy,shared_copy]
//                 [--experiments create,cpu_touch,gpu_first,gpu_bw]
//                 + common flags (benchmarks/harness/cli.h)

#include "computeedge/bench/environment.h"
#include "experiments.h"
#include "harness/console.h"
#include "harness/gpu.h"
#include "harness/results.h"
#include "options.h"

namespace harness = computeedge::bench::harness;
using namespace computeedge::bench::storage_modes;

int main(int argc, char** argv) {
  const Options opt = parse_options(argc, argv);
  harness::require_metal();
  const computeedge::bench::Environment env =
      harness::banner("ComputeEdge M5 / Track B: storage modes", harness::BannerStyle::WithGpu);
  harness::ConsoleReporter report;
  if (const int status = run(opt, report); status != 0) return status;
  harness::write_results(opt.out, opt.write, env, report.results());
  return 0;
}
