// M6 follow-ups: targeted experiments for the three open M5 findings
// (research/findings.md 7, 14, 15). Each experiment tests one hypothesis.
//
//   wake      (finding 7) Wake latency grew with GPU work. Hypothesis: completion
//             includes writing back dirty GPU cache lines. GPU read vs write streams of
//             1M..256M, sync = spin on MTLSharedEvent (lowest floor). Supported if wake
//             grows with bytes written but not with bytes read.
//   cpu_write (finding 14) Warm CPU writes into a Shared buffer were ~1.7x slower than
//             into malloc'd memory. Memory source {posix_memalign, mmap, shared,
//             shared_wc, nocopy over mmap} x write method {memset, vectorised store loop}
//             at 64 MiB, plus a read loop; and writes 1-4 into a newly created buffer.
//             Separates "memset path", "recently created mapping", and "Metal
//             mapping attribute".
//   gpu_write (finding 15) GPU write bandwidth dropped above ~256 MiB. Fine size sweep;
//             one buffer of N bytes vs N/256M buffers of 256 MiB written back to back;
//             second pass over the same buffer. Per-buffer effect (GPU page tables / TLB
//             reach) vs total-footprint effect.
//
// Usage:
//   followups [--experiments wake,cpu_write,gpu_write] + common flags

#include "computeedge/bench/environment.h"
#include "experiments.h"
#include "harness/console.h"
#include "harness/gpu.h"
#include "harness/results.h"
#include "options.h"

namespace harness = computeedge::bench::harness;
using namespace computeedge::bench::followups;

int main(int argc, char** argv) {
  const Options opt = parse_options(argc, argv);
  harness::require_metal();
  const computeedge::bench::Environment env =
      harness::banner("ComputeEdge M6: follow-ups to open M5 findings", harness::BannerStyle::WithGpu);
  harness::ConsoleReporter report;
  if (const int status = run(opt, report); status != 0) return status;
  harness::write_results(opt.out, opt.write, env, report.results());
  return 0;
}
