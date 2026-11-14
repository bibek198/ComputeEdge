// Track B1/B2/B3/B4: allocation, CPU touch, first GPU access, and sustained
// bandwidth under each Metal storage mode.
//
// Modes:
//   host        posix_memalign page-aligned memory (baseline)
//   shared      MTLResourceStorageModeShared: coherent CPU and GPU access
//   private     MTLResourceStorageModePrivate: GPU-only VRAM; CPU access needs a blit
//   nocopy      newBufferWithBytesNoCopy wrapping host-allocated pages
//   shared_copy newBufferWithBytes allocating shared memory and copying from host
//
// Experiments:
//   create      allocating the buffer (or calling posix_memalign): shows the cost
//               of entering the kernel / allocating pages
//   cpu_touch   CPU sequential write then read: checks whether the Metal mapping
//               imposes a page-fault or write-barrier tax over plain host pages
//   gpu_first   first single-word GPU write into a newly allocated buffer, with
//               either a freshly allocated buffer (new page table entries), an
//               evicted / non-resident memory (rewrap), or the same buffer reused: isolates
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
#include "harness/run_main.h"
#include "options.h"

namespace harness = computeedge::bench::harness;
using namespace computeedge::bench::storage_modes;

int main(int argc, char** argv) {
  Options opt;
  computeedge::bench::Environment env;
  harness::ConsoleReporter report;
  return harness::run_main(&opt, &env, &report, [&] {
    opt = parse_options(argc, argv);
    harness::require_metal();
    env = harness::banner("ComputeEdge M5 / Track B: storage modes", harness::BannerStyle::WithGpu);
    if (const int status = run(opt, report); status != 0) return status;
    harness::write_results(opt.out, opt.write, env, report.results());
    return 0;
  });
}
