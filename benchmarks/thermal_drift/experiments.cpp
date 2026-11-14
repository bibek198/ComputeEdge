// Measurement logic of the thermal_drift benchmark (moved verbatim from main.cpp;
// only printing goes through the Reporter, and per-minute windowing uses
// bench::window_results, which reproduces the former inline code exactly).

#include "experiments.h"

#include <pthread.h>
#include <sys/qos.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <cstdio>
#include <sstream>
#include "computeedge/bench/environment.h"
#include "computeedge/bench/result_writer.h"
#include "computeedge/bench/runner.h"
#include "computeedge/bench/trace.h"
#include "computeedge/computeedge.h"
#include "computeedge/core/instrumentation.h"
#include "computeedge/metal/matmul.h"
#include "computeedge/metal/probes.h"
#include "harness/cli.h"
#include "harness/console.h"
#include "harness/gpu.h"
#include "harness/results.h"

#include "computeedge/bench/windows.h"

namespace computeedge::bench::thermal_drift {

namespace cb = computeedge::bench;
namespace mtl = computeedge::metal;
namespace probe = computeedge::metal::probe;
using namespace computeedge::bench::harness;
using namespace computeedge;

namespace {

struct Workload {
  std::string name;
  double flops;
  std::function<void()> run;
};

}  // namespace

RunOutput run(const Options& opt, harness::Reporter& report) {
  RunOutput out;
  const double minutes = opt.minutes;
  const std::string& load = opt.load;
  const double interval_ms = opt.interval_ms;
  const bool cooldown = opt.wait_for_nominal;

  if (cooldown) {
    report.note("waiting for thermal state nominal...\n");
    if (!cb::wait_for_thermal(cb::ThermalState::Nominal, 900)) report.note("  WARNING: did not reach nominal\n");
  }
  prewarm(PrewarmWorkload::GpuStreamWritePrivate64MiB, opt.prewarm_ms, report);

  auto square = [](size_t n, float v) { return Tensor::full({n, n}, v); };
  const Tensor a1 = square(1024, 0.5f), b1 = square(1024, 0.25f);
  const Tensor a2 = square(2048, 0.5f), b2 = square(2048, 0.25f);
  Tensor c1c({1024, 1024}), c2c({2048, 2048}), c1g({1024, 1024}), c2g({2048, 2048});
  const metal::MatmulOptions mps{metal::StorageMode::NoCopy, metal::Kernel::MPS};
  const double f1 = 2.0 * 1024.0 * 1024.0 * 1024.0, f2 = 8.0 * f1;
  std::vector<Workload> workloads{
      {"cpu_1024", f1, [&] { matmul_into(a1, b1, c1c); }},
      {"cpu_2048", f2, [&] { matmul_into(a2, b2, c2c); }},
      {"gpu_1024", f1, [&] { metal::matmul_into(a1, b1, c1g, mps); }},
      {"gpu_2048", f2, [&] { metal::matmul_into(a2, b2, c2g, mps); }},
      {"gpu_empty", 0, [&] { probe::empty_dispatch({}); }},
  };
  for (auto& w : workloads) w.run();  // first-use costs (MPS kernel objects, pipelines) out of the way

  probe::Buffer heat_buf = probe::Buffer::create(probe::BufferKind::Private, size_t{64} << 20);
  const uint64_t heat_write_ns = [&] {
    probe::gpu_stream(probe::Stream::Write, heat_buf);
    return probe::gpu_stream(probe::Stream::Write, heat_buf).gpu_ns();
  }();

  {
    std::string names;
    for (const auto& w : workloads) names += (names.empty() ? "" : ",") + w.name;
    cb::trace("thermal workloads=" + names + " load=" + load + " interval_ms=" + std::to_string(interval_ms) +
              " minutes=" + std::to_string(minutes) + " cooldown=" + (cooldown ? "1" : "0") +
              " heater_gpu=gpu_stream_write kind=private bytes=67108864 heater_cpu=sgemm n=1024");
  }
  const uint64_t interval_ns = static_cast<uint64_t>(interval_ms * 1e6);
  const uint64_t start = now_ns();
  const uint64_t end = start + static_cast<uint64_t>(minutes * 60e9);
  std::vector<Sample>& samples = out.samples;
  uint64_t cycle = 0;
  std::vector<uint64_t> last_ns(workloads.size(), 0);
  cb::ThermalState last_state = cb::thermal_state();
  {
    std::ostringstream note;
    note << "running " << minutes << " min, load=" << load << ", interval " << interval_ms << " ms\n";
    report.note(note.str());
  }
  const std::string start_power = cb::current_power_source();
  bool& power_changed = out.power_changed;

  while (now_ns() < end) {
    const uint64_t cycle_start = now_ns();
    // A thermal run is only valid on one power source: unplugging the adapter
    // (or Low Power Mode) changes the power budget mid-run.
    const std::string power = cb::current_power_source();
    const bool on_ac = power == "AC Power";
    if (power != start_power) {
      char line[200];
      std::snprintf(line, sizeof(line), "  t=%6.1f s  POWER SOURCE CHANGED (%s -> %s): stopping; this run is invalid\n",
                    double(now_ns() - start) / 1e9, start_power.c_str(), power.c_str());
      report.note(line);
      power_changed = true;
      break;
    }
    for (size_t k = 0; k < workloads.size(); ++k) {
      const size_t i = (k + cycle) % workloads.size();
      const uint64_t t0 = now_ns();
      workloads[i].run();
      const uint64_t t1 = now_ns();
      samples.push_back({double(t0 - start) / 1e9, cycle, i, t1 - t0, cb::thermal_state(), on_ac});
      last_ns[i] = t1 - t0;
    }

    // Heater until the end of the interval.
    const uint64_t heat_until = cycle_start + interval_ns;
    const uint64_t now = now_ns();
    if (heat_until > now) {
      const uint64_t remaining = heat_until - now;
      std::optional<probe::Submission> gpu_heat;
      if (load == "mixed" || load == "gpu") {
        const uint32_t reps =
            static_cast<uint32_t>(std::max<uint64_t>(1, remaining / std::max<uint64_t>(heat_write_ns, 1)));
        gpu_heat.emplace(probe::gpu_stream_async(probe::Stream::Write, heat_buf, nullptr, reps));
      }
      if (load == "mixed" || load == "cpu") {
        while (now_ns() < heat_until) matmul_into(a1, b1, c1c);
      } else if (load == "none") {
        const uint64_t current = now_ns();
        if (heat_until > current) {
          std::this_thread::sleep_for(std::chrono::nanoseconds(heat_until - current));
        }
      }
      if (gpu_heat) gpu_heat->wait();
    }

    const cb::ThermalState state = cb::thermal_state();
    if (state != last_state || cycle % 60 == 0) {
      char line[200];
      std::snprintf(line, sizeof(line), "  t=%6.1f s  cycle %4llu  thermal %-8s  cpu_2048 %.2f ms  gpu_2048 %.2f ms\n",
                    double(now_ns() - start) / 1e9, static_cast<unsigned long long>(cycle), cb::to_string(state),
                    last_ns[1] / 1e6, last_ns[3] / 1e6);
      report.note(line);
      last_state = state;
    }
    ++cycle;
  }
  report.note("done: " + std::to_string(cycle) + " cycles\n");

  if (samples.empty()) {
    report.note("no samples recorded\n");
    out.status = power_changed ? 3 : 1;
    return out;
  }

  // Per-minute windows as Results.
  std::vector<cb::SeriesInfo> series;
  for (const auto& w : workloads) {
    out.workloads.push_back(w.name);
    series.push_back({w.name, w.flops});
  }
  std::vector<cb::TimedSample> timed;
  timed.reserve(samples.size());
  for (const Sample& s : samples) timed.push_back({s.t_s, s.workload, s.ns, s.thermal});
  out.results = cb::window_results("thermal_drift", timed, series, 60.0, "workload", "window_min",
                                   {{"load", load}, {"valid", power_changed ? "no_power_changed" : "yes"}});
  out.status = power_changed ? 3 : 0;
  return out;
}

}  // namespace computeedge::bench::thermal_drift
