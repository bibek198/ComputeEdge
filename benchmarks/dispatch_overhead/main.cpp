// Track A1/A2: fixed cost of GPU work submission, measured with empty kernels
// (one thread, no memory access) so every nanosecond is runtime overhead.
//
// Experiments (all synchronize with waitUntilCompleted):
//   cold          first empty dispatch in the process (includes compiling the probe
//                 library; one sample; the OS shader cache may already hold it)
//   dispatches    1 command buffer, 1 encoder, N dispatches        -> marginal cost per dispatch
//   encoders      1 command buffer, N encoders x 1 dispatch        -> marginal cost per encoder
//   cmdbufs       N command buffers x 1 encoder x 1 dispatch,       -> marginal cost per command buffer
//                 committed back to back, wait on the last
//
// Usage:
//   dispatch_overhead [--dispatches 1,4,16,64,256,1024] [--encoders 1,4,16,64,256]
//                     [--cmdbufs 1,2,4,8,16,32,64] [--experiments dispatches,encoders,cmdbufs]
//                     + common flags (benchmarks/harness/bench_common.h)

#include <algorithm>
#include <string>
#include <vector>

#include "harness/bench_common.h"

using namespace ce_bench;

namespace {

std::vector<uint32_t> to_u32(const std::string& s) {
  std::vector<uint32_t> out;
  for (const auto& v : split(s)) out.push_back(static_cast<uint32_t>(std::stoul(v)));
  return out;
}

bool wants(const std::vector<std::string>& list, const std::string& name) {
  return std::find(list.begin(), list.end(), name) != list.end();
}

}  // namespace

int main(int argc, char** argv) {
  const char* src = "benchmarks/dispatch_overhead/main.cpp";
  std::vector<uint32_t> dispatches{1, 4, 16, 64, 256, 1024};
  std::vector<uint32_t> encoders{1, 4, 16, 64, 256};
  std::vector<uint32_t> cmdbufs{1, 2, 4, 8, 16, 32, 64};
  std::vector<std::string> experiments{"dispatches", "encoders", "cmdbufs"};
  const CommonOptions opt = parse(argc, argv, src, [&](const std::string& f, const auto& next) {
    if (f == "--dispatches")
      dispatches = to_u32(next());
    else if (f == "--encoders")
      encoders = to_u32(next());
    else if (f == "--cmdbufs")
      cmdbufs = to_u32(next());
    else if (f == "--experiments")
      experiments = split(next());
    else
      return false;
    return true;
  });

  const cb::Environment env = banner("ComputeEdge M5 / Track A: dispatch overhead (empty kernels)");
  std::vector<cb::Result> results;

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
      record_timing(m, t);
    });
    std::cout << cb::format_result(r) << "\n\n";
    results.push_back(std::move(r));
  }

  prewarm_gpu(opt.prewarm_ms);
  const cb::Config cfg = config(opt);

  auto measure = [&](const std::string& experiment, uint32_t c, uint32_t e, uint32_t d) {
    const probe::DispatchConfig dc{c, e, d, probe::Sync::WaitUntilCompleted};
    cb::Result r = cb::run_with_metrics("dispatch_empty",
                                        {{"experiment", experiment},
                                         {"command_buffers", std::to_string(c)},
                                         {"encoders", std::to_string(e)},
                                         {"dispatches", std::to_string(d)},
                                         {"sync", probe::to_string(dc.sync)}},
                                        cfg, [&](cb::Metrics& m) { record_timing(m, probe::empty_dispatch(dc)); });
    std::cout << cb::format_result(r) << std::endl;
    results.push_back(std::move(r));
  };

  if (wants(experiments, "dispatches"))
    for (uint32_t n : dispatches) measure("dispatches", 1, 1, n);
  if (wants(experiments, "encoders"))
    for (uint32_t n : encoders) measure("encoders", 1, n, 1);
  if (wants(experiments, "cmdbufs"))
    for (uint32_t n : cmdbufs) measure("cmdbufs", n, 1, 1);

  write_results(opt, env, results);
  return 0;
}
