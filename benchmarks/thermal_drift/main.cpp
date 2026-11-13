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
//                 [--out benchmarks/results] [--no-write] [--prewarm-ms 1000]

#include <pthread.h>
#include <sys/qos.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "harness/bench_common.h"
#include "computeedge/computeedge.h"

using namespace ce_bench;
using namespace computeedge;

namespace {

struct Workload {
  std::string name;
  double flops;
  std::function<void()> run;
};

struct Sample {
  double t_s;
  uint64_t cycle;
  size_t workload;
  uint64_t ns;
  cb::ThermalState thermal;
  bool on_ac;
};

}  // namespace

int main(int argc, char** argv) {
  const char* src = "benchmarks/thermal_drift/main.cpp";
  double minutes = 15;
  std::string load = "mixed";
  double interval_ms = 1000;
  bool cooldown = true;
  CommonOptions opt = parse(argc, argv, src, [&](const std::string& f, const auto& next) {
    if (f == "--minutes")
      minutes = std::stod(next());
    else if (f == "--load")
      load = next();
    else if (f == "--interval-ms")
      interval_ms = std::stod(next());
    else if (f == "--no-cooldown")
      cooldown = false;
    else
      return false;
    return true;
  });
  if (load != "mixed" && load != "cpu" && load != "gpu" && load != "none") usage(src, "unknown --load " + load);

  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  const cb::Environment env = banner("ComputeEdge M6 / RQ4: thermal drift under sustained load");
  if (cooldown) {
    std::cout << "waiting for thermal state nominal...\n";
    if (!cb::wait_for_thermal(cb::ThermalState::Nominal, 900)) std::cout << "  WARNING: did not reach nominal\n";
  }
  prewarm_gpu(opt.prewarm_ms);

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
  std::vector<Sample> samples;
  uint64_t cycle = 0;
  std::vector<uint64_t> last_ns(workloads.size(), 0);
  cb::ThermalState last_state = cb::thermal_state();
  std::cout << "running " << minutes << " min, load=" << load << ", interval " << interval_ms << " ms\n";
  const std::string start_power = cb::current_power_source();
  bool power_changed = false;

  while (now_ns() < end) {
    const uint64_t cycle_start = now_ns();
    // A thermal run is only valid on one power source: unplugging the adapter
    // (or Low Power Mode) changes the power budget mid-run.
    const std::string power = cb::current_power_source();
    const bool on_ac = power == "AC Power";
    if (power != start_power) {
      std::printf("  t=%6.1f s  POWER SOURCE CHANGED (%s -> %s): stopping; this run is invalid\n",
                  double(now_ns() - start) / 1e9, start_power.c_str(), power.c_str());
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
        std::this_thread::sleep_for(std::chrono::nanoseconds(heat_until - now_ns()));
      }
      if (gpu_heat) gpu_heat->wait();
    }

    const cb::ThermalState state = cb::thermal_state();
    if (state != last_state || cycle % 60 == 0) {
      std::printf("  t=%6.1f s  cycle %4llu  thermal %-8s  cpu_2048 %.2f ms  gpu_2048 %.2f ms\n",
                  double(now_ns() - start) / 1e9, static_cast<unsigned long long>(cycle), cb::to_string(state),
                  last_ns[1] / 1e6, last_ns[3] / 1e6);
      std::fflush(stdout);
      last_state = state;
    }
    ++cycle;
  }
  std::cout << "done: " << cycle << " cycles\n";

  if (samples.empty()) {
    std::cout << "no samples recorded\n";
    return power_changed ? 3 : 1;
  }

  // Per-minute windows as Results.
  std::vector<cb::Result> results;
  const size_t windows = static_cast<size_t>(samples.back().t_s / 60.0) + 1;
  for (size_t w = 0; w < windows; ++w) {
    for (size_t i = 0; i < workloads.size(); ++i) {
      cb::Result r;
      r.name = "thermal_drift";
      r.params = {{"workload", workloads[i].name},
                  {"window_min", std::to_string(w)},
                  {"load", load},
                  {"valid", power_changed ? "no_power_changed" : "yes"}};
      r.flops_per_call = workloads[i].flops;
      for (const Sample& s : samples) {
        if (s.workload != i || static_cast<size_t>(s.t_s / 60.0) != w) continue;
        r.samples_ns.push_back(double(s.ns));
        r.sample_thermal.push_back(s.thermal);
      }
      if (r.samples_ns.empty()) continue;
      r.thermal_before = r.sample_thermal.front();
      r.thermal_after = r.sample_thermal.back();
      r.summary = cb::summarize(r.samples_ns);
      results.push_back(std::move(r));
    }
  }
  for (const auto& r : results)
    if (r.params.at("workload") == "cpu_2048" || r.params.at("workload") == "gpu_2048")
      std::cout << cb::format_result(r) << '\n';

  if (opt.write) {
    cb::ResultWriter writer(opt.out, env);
    for (const auto& r : results) writer.add(r);
    writer.flush();
    std::ofstream ts(writer.directory() + "/timeseries.csv");
    ts << "t_s,cycle,workload,ns,thermal,on_ac\n";
    for (const Sample& s : samples)
      ts << s.t_s << ',' << s.cycle << ',' << workloads[s.workload].name << ',' << s.ns << ','
         << cb::to_string(s.thermal) << ',' << (s.on_ac ? 1 : 0) << '\n';
    std::cout << "\nResults written to " << writer.directory() << '\n';
  }
  return power_changed ? 3 : 0;
}
