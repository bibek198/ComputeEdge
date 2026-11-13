// M6 / RQ4: does performance (and the CPU/GPU crossover) drift as the fanless
// Air heats up under sustained load?
//
// Starts from thermal state `nominal`, then for --minutes runs ~1 s cycles. Each
// cycle measures one synchronous call of every workload (start rotated per cycle
// to avoid order bias):
//   cpu_1024 / cpu_2048   Accelerate SGEMM into a pre-allocated output
//   gpu_1024 / gpu_2048   Metal MPS SGEMM, NoCopy, pre-allocated output (same
//                         allocation policy as the CPU side)
//   gpu_empty             empty-kernel round trip (fixed GPU overhead)
// and spends the rest of the cycle on a heater load:
//   mixed (default)  GPU 64 MiB write streams + CPU SGEMM 1024^2 loop, concurrently
//   cpu | gpu        only one side;  none   sleep (control: does time alone drift?)
//
// Output: one Result per (workload, minute window) through ResultWriter (so
// summary.csv has per-window medians and CIs, samples.csv has every sample with
// its thermal state), plus timeseries.csv (t_s, cycle, workload, ns, thermal, on_ac).
// The power source is checked every cycle; if it changes, the run stops, results
// are tagged valid=no_power_changed and the exit status is 3.
//
// Optional power/frequency trace (needs your password, run in a second terminal):
//   sudo powermetrics --samplers cpu_power,gpu_power,thermal -i 1000 -o <run dir>/powermetrics.txt
//
// Usage:
//   thermal_drift [--minutes 15] [--load mixed|cpu|gpu|none] [--interval-ms 1000] [--no-cooldown]
//                 [--out data/scratch] [--no-write] [--prewarm-ms 1000]

#include <pthread.h>
#include <sys/qos.h>

#include <fstream>
#include <iostream>

#include "computeedge/bench/environment.h"
#include "computeedge/bench/result_writer.h"
#include "experiments.h"
#include "harness/console.h"
#include "harness/gpu.h"
#include "harness/results.h"
#include "options.h"

namespace cb = computeedge::bench;
namespace harness = computeedge::bench::harness;
using namespace computeedge::bench::thermal_drift;

int main(int argc, char** argv) {
  const Options opt = parse_options(argc, argv);
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  harness::require_metal();
  const cb::Environment env =
      harness::banner("ComputeEdge M6 / RQ4: thermal drift under sustained load", harness::BannerStyle::WithGpu);
  harness::ConsoleReporter report;
  const RunOutput run_out = run(opt, report);
  if (run_out.samples.empty()) return run_out.status;

  for (const auto& r : run_out.results)
    if (r.params.at("workload") == "cpu_2048" || r.params.at("workload") == "gpu_2048")
      std::cout << cb::format_result(r) << '\n';

  if (opt.write) {
    cb::ResultWriter writer(opt.out, env);
    for (const auto& r : run_out.results) writer.add(r);
    writer.flush();
    std::ofstream ts(writer.directory() + "/timeseries.csv");
    ts << "t_s,cycle,workload,ns,thermal,on_ac\n";
    for (const Sample& s : run_out.samples)
      ts << s.t_s << ',' << s.cycle << ',' << run_out.workloads[s.workload] << ',' << s.ns << ','
         << cb::to_string(s.thermal) << ',' << (s.on_ac ? 1 : 0) << '\n';
    std::cout << "\nResults written to " << writer.directory() << '\n';
  }
  return run_out.status;
}
