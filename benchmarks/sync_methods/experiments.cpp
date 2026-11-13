// Measurement logic of the sync_methods benchmark (moved verbatim from main.cpp; only
// printing goes through the Reporter).

#include "experiments.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <sstream>
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

namespace computeedge::bench::sync_methods {

namespace cb = computeedge::bench;
namespace mtl = computeedge::metal;
namespace probe = computeedge::metal::probe;
using namespace computeedge::bench::harness;

int run(const Options& opt, harness::Reporter& report) {
  const char* src = kSource;
  const auto& methods = opt.methods;
  const auto& work = opt.work;
  const auto& gaps_ms = opt.gaps_ms;
  const auto& experiments = opt.experiments;
  prewarm(PrewarmWorkload::GpuStreamWritePrivate64MiB, opt.prewarm_ms, report);
  const cb::Config cfg = config(opt);

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
        report.result(std::move(r));
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
        report.result(std::move(r));
      }
    }
  }
  return 0;
}

}  // namespace computeedge::bench::sync_methods
