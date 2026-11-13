// M6 / RQ3: does GPU work slow the CPU's next pass over its own data?
//
// Every sample: warm CPU pass (untimed) -> interference phase (untimed) -> timed
// CPU pass. The CPU passes are
//   read  integer read-sum over W bytes (NoCopy buffer, behaves like host memory)
//   slc   16 MiB touched as one 128 B line per 8 KiB over 1 GiB, alternating direction:
//         the system-level-cache probe of Alpay & Basaran (arXiv 2606.27098)
//   sgemm Accelerate SGEMM 512^2 into a pre-allocated output (3 MiB of operands)
// Interference phases, each paired with a control of equal duration so clock /
// power-state effects (M5 finding 8: idling makes the next call slower) are not
// mistaken for cache effects. The CPU busy-spins through every phase, including
// GPU waits (polling the command buffer's status), so its clocks stay up:
//   idle_spin      control: plain spin for the median duration of gpu_write at this G
//   gpu_tiny       control: same submit + poll path, GPU rewrites a 16 KiB buffer for
//                  the same duration (GPU busy, negligible memory traffic)
//   gpu_write      GPU writes a separate G-byte buffer         -> eviction by GPU writes
//   gpu_read       GPU reads a separate G-byte buffer          -> eviction by GPU reads (no dirty lines)
//   cpu_read       this CPU thread reads a separate G buffer   -> reference: CPU-side eviction
//   idle_short     control for gpu_concurrent: spin 300 us
//   gpu_concurrent GPU keeps writing G while the timed pass runs -> bandwidth contention
//                  (300 us head start so the GPU is running; `overlap` metric checks it)
// Read pass only (handoff case: the CPU reads what the GPU just wrote):
//   idle_same      control: spin for the median duration of gpu_same
//   gpu_tiny_same  control: tiny GPU work of the same duration
//   gpu_same       GPU writes the CPU's own W buffer, then the CPU reads it
//
// Usage:
//   cache_interference [--working-sets 1M,4M,12M,64M] [--gpu-sizes 16M,64M,256M] [--passes read,slc,sgemm]
//                      + common flags (benchmarks/harness/cli.h)

#include "computeedge/bench/environment.h"
#include "experiments.h"
#include "harness/console.h"
#include "harness/gpu.h"
#include "harness/results.h"
#include "options.h"

#include <pthread.h>
#include <sys/qos.h>

namespace harness = computeedge::bench::harness;
using namespace computeedge::bench::cache_interference;

int main(int argc, char** argv) {
  const Options opt = parse_options(argc, argv);
  // Prefer the performance cores for the measuring thread.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  harness::require_metal();
  const computeedge::bench::Environment env =
      harness::banner("ComputeEdge M6 / RQ3: CPU-GPU cache interference", harness::BannerStyle::WithGpu);
  harness::ConsoleReporter report;
  if (const int status = run(opt, report); status != 0) return status;
  harness::write_results(opt.out, opt.write, env, report.results());
  return 0;
}
