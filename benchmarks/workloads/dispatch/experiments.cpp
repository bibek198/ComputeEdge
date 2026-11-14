// Measurement logic of the dispatch_overhead benchmark (moved verbatim from main.cpp; only
// printing goes through the Reporter).

#include "experiments.h"

#include <algorithm>
#include <iostream>
#include <string>
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

namespace computeedge::bench::dispatch_overhead {

namespace cb = computeedge::bench;
namespace mtl = computeedge::metal;
namespace probe = computeedge::metal::probe;
using namespace computeedge::bench::harness;

int run(const Options& opt, harness::Reporter& report) {
  const auto& dispatches = opt.dispatches;
  const auto& encoders = opt.encoders;
  const auto& cmdbufs = opt.cmdbufs;
  const auto& experiments = opt.experiments;

  // Cold: before anything else touches the probe library.
  {
    cb::Config cold;
    cold.warmup = 0;
    cold.min_warmup_ns = 0;
    cold.repetitions = 1;
    cb::Result r = cb::run_with_metrics("dispatch_cold", {{"experiment", "cold"}}, cold, [&](cb::Metrics& m) {
      const uint64_t before = mtl::compile_ns();
      const mtl::Timing t = probe::empty_dispatch({});
      m["compile_ns"] = double(mtl::compile_ns() - before);
      record_probe_timing(m, t);
    });
    report.result(std::move(r));
    report.note("\n");
  }

  prewarm(PrewarmWorkload::GpuStreamWritePrivate64MiB, opt.prewarm_ms, report);
  const cb::Config cfg = config(opt);

  auto measure = [&](const std::string& experiment, uint32_t c, uint32_t e, uint32_t d) {
    const probe::DispatchConfig dc{c, e, d, probe::Sync::WaitUntilCompleted};
    cb::Result r =
        cb::run_with_metrics("dispatch_empty",
                             {{"experiment", experiment},
                              {"command_buffers", std::to_string(c)},
                              {"encoders", std::to_string(e)},
                              {"dispatches", std::to_string(d)},
                              {"sync", probe::to_string(dc.sync)}},
                             cfg, [&](cb::Metrics& m) { record_probe_timing(m, probe::empty_dispatch(dc)); });
    report.result(std::move(r));
  };

  if (wants(experiments, "dispatches"))
    for (uint32_t n : dispatches) measure("dispatches", 1, 1, n);
  if (wants(experiments, "encoders"))
    for (uint32_t n : encoders) measure("encoders", 1, n, 1);
  if (wants(experiments, "cmdbufs"))
    for (uint32_t n : cmdbufs) measure("cmdbufs", n, 1, 1);
  return 0;
}

}  // namespace computeedge::bench::dispatch_overhead
