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

#include <algorithm>
#include <chrono>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "computeedge/bench/environment.h"
#include "computeedge/bench/result_writer.h"
#include "computeedge/bench/runner.h"
#include "computeedge/bench/trace.h"
#include "computeedge/core/instrumentation.h"
#include "computeedge/metal/matmul.h"
#include "computeedge/metal/probes.h"
#include "harness/cli.h"
#include "harness/console.h"
#include "harness/gpu.h"
#include "harness/results.h"

namespace cb = computeedge::bench;
namespace mtl = computeedge::metal;
namespace probe = computeedge::metal::probe;
using namespace computeedge::bench::harness;

probe::Sync parse_sync(const std::string& s, const char* src) {
  return parse_enum(s,
                    {probe::Sync::WaitUntilCompleted, probe::Sync::EventWait, probe::Sync::CompletionHandler,
                     probe::Sync::SpinEvent, probe::Sync::SpinStatus},
                    "sync method", src);
}

int main(int argc, char** argv) {
  const char* src = "benchmarks/sync_methods/main.cpp";
  std::vector<std::string> methods{"wait_until_completed", "event_wait", "completion_handler", "spin_event",
                                   "spin_status"};
  std::vector<std::string> work{"empty", "4M", "256M"};
  std::vector<double> gaps_ms{0, 0.1, 1, 10, 100};
  std::vector<std::string> experiments{"methods", "idle"};
  const CommonOptions opt = parse(argc, argv, src, kAllCommonFlags, [&](const std::string& f, const auto& next) {
    if (f == "--methods")
      methods = split(next());
    else if (f == "--work")
      work = split(next());
    else if (f == "--gaps-ms")
      gaps_ms = split_doubles(next());
    else if (f == "--experiments")
      experiments = split(next());
    else
      return false;
    return true;
  });
  for (const auto& m : methods) parse_sync(m, src);  // validate before running anything

  require_metal();
  const cb::Environment env =
      banner("ComputeEdge M5 / Track A: synchronization methods and idle gaps", BannerStyle::WithGpu);
  prewarm(PrewarmWorkload::GpuStreamWritePrivate64MiB, opt.prewarm_ms);
  const cb::Config cfg = config(opt);
  std::vector<cb::Result> results;

  if (wants(experiments, "methods")) {
    for (const std::string& w : work) {
      std::optional<probe::Buffer> buf;
      if (w != "empty") buf = probe::Buffer::create(probe::BufferKind::Private, split_bytes(w).at(0));
      for (const std::string& name : methods) {
        const probe::Sync sync = parse_sync(name, src);
        cb::Result r = cb::run_with_metrics(
            "sync_method", {{"experiment", "methods"}, {"sync", name}, {"work", w}}, cfg, [&](cb::Metrics& m) {
              record_probe_timing(m, buf ? probe::gpu_stream(probe::Stream::Write, *buf, nullptr, sync)
                                         : probe::empty_dispatch({1, 1, 1, sync}));
            });
        std::cout << cb::format_result(r) << std::endl;
        results.push_back(std::move(r));
      }
    }
  }

  if (wants(experiments, "idle")) {
    for (const char* name : {"wait_until_completed", "spin_event"}) {
      const probe::Sync sync = parse_sync(name, src);
      for (double gap : gaps_ms) {
        const auto sleep_for = std::chrono::nanoseconds(static_cast<int64_t>(gap * 1e6));
        char gap_label[32];
        std::snprintf(gap_label, sizeof(gap_label), "%g", gap);
        cb::Result r = cb::run_with_metrics(
            "sync_idle", {{"experiment", "idle"}, {"sync", name}, {"gap_ms", gap_label}}, cfg,
            [&](cb::Metrics& m) { record_probe_timing(m, probe::empty_dispatch({1, 1, 1, sync})); }, 0,
            [&] {
              if (gap > 0) std::this_thread::sleep_for(sleep_for);
            });
        std::cout << cb::format_result(r) << std::endl;
        results.push_back(std::move(r));
      }
    }
  }

  write_results(opt.out, opt.write, env, results);
  return 0;
}
