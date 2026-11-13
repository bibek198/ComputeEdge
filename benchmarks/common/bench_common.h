#pragma once

// Shared plumbing for the M5 characterization benchmarks: common command-line
// flags, the environment banner, GPU pre-warm, Timing -> metrics, and output.
//
// Common flags: [--warmup 10] [--min-warmup-ms 200] [--prewarm-ms 1000] [--reps 30]
//               [--out benchmarks/results] [--cooldown] [--no-write]

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "computeedge/bench/environment.h"
#include "computeedge/bench/result_writer.h"
#include "computeedge/bench/runner.h"
#include "computeedge/bench/trace.h"
#include "computeedge/core/instrumentation.h"
#include "computeedge/metal/matmul.h"
#include "computeedge/metal/probes.h"

namespace ce_bench {

namespace cb = computeedge::bench;
namespace mtl = computeedge::metal;
namespace probe = computeedge::metal::probe;

struct CommonOptions {
  int warmup = 10;
  double min_warmup_ms = 200;
  double prewarm_ms = 1000;
  int reps = 30;
  std::string out = "benchmarks/results";
  bool cooldown = false;
  bool write = true;
};

inline std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');)
    if (!item.empty()) out.push_back(item);
  return out;
}

// "16K,4M,1G" -> bytes (binary units).
inline std::vector<size_t> split_bytes(const std::string& s) {
  std::vector<size_t> out;
  for (std::string item : split(s)) {
    size_t mult = 1;
    const char suffix = item.back();
    if (suffix == 'K' || suffix == 'k') mult = size_t{1} << 10;
    if (suffix == 'M' || suffix == 'm') mult = size_t{1} << 20;
    if (suffix == 'G' || suffix == 'g') mult = size_t{1} << 30;
    if (mult != 1) item.pop_back();
    out.push_back(std::stoull(item) * mult);
  }
  return out;
}

inline std::vector<double> split_doubles(const std::string& s) {
  std::vector<double> out;
  for (const auto& item : split(s)) out.push_back(std::stod(item));
  return out;
}

inline std::string bytes_label(size_t bytes) {
  if (bytes >= (size_t{1} << 30) && bytes % (size_t{1} << 30) == 0) return std::to_string(bytes >> 30) + "G";
  if (bytes >= (size_t{1} << 20) && bytes % (size_t{1} << 20) == 0) return std::to_string(bytes >> 20) + "M";
  if (bytes >= (size_t{1} << 10) && bytes % (size_t{1} << 10) == 0) return std::to_string(bytes >> 10) + "K";
  return std::to_string(bytes);
}

[[noreturn]] inline void usage(const char* source, const std::string& msg) {
  std::fprintf(stderr, "error: %s\nsee the header of %s for usage\n", msg.c_str(), source);
  std::exit(2);
}

// `extra(flag, next)` handles benchmark-specific flags; returns false if unknown.
using ExtraFlags = std::function<bool(const std::string& flag, const std::function<std::string()>& next)>;

inline CommonOptions parse(int argc, char** argv, const char* source, const ExtraFlags& extra) {
  CommonOptions o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const std::function<std::string()> next = [&]() -> std::string {
      if (i + 1 >= argc) usage(source, "missing value for " + a);
      return argv[++i];
    };
    if (a == "--warmup") {
      o.warmup = std::stoi(next());
    } else if (a == "--min-warmup-ms") {
      o.min_warmup_ms = std::stod(next());
    } else if (a == "--prewarm-ms") {
      o.prewarm_ms = std::stod(next());
    } else if (a == "--reps") {
      o.reps = std::stoi(next());
    } else if (a == "--out") {
      o.out = next();
    } else if (a == "--cooldown") {
      o.cooldown = true;
    } else if (a == "--no-write") {
      o.write = false;
    } else if (!extra || !extra(a, next)) {
      usage(source, "unknown argument " + a);
    }
  }
  return o;
}

inline cb::Config config(const CommonOptions& o) {
  cb::Config c;
  c.warmup = o.warmup;
  c.min_warmup_ns = static_cast<uint64_t>(o.min_warmup_ms * 1e6);
  c.repetitions = o.reps;
  c.cooldown = o.cooldown;
  return c;
}

// Records the environment, prints it, and warns about unrepresentative setups.
inline cb::Environment banner(const char* title) {
  if (!mtl::available()) {
    std::fprintf(stderr, "Metal is not available on this system\n");
    std::exit(1);
  }
  const cb::Environment env = cb::record_environment();
  std::cout << title << "\n"
            << "  " << env.chip << " / " << env.gpu_name << " (" << env.model << "), macOS " << env.os_version << " ("
            << env.os_build << ")\n"
            << "  power: " << env.power_source << (env.low_power_mode ? ", LOW POWER MODE" : "")
            << ", thermal: " << cb::to_string(env.thermal_state) << ", build: " << env.build_type
            << (env.sanitizers ? " +sanitizers" : "") << ", git " << env.git_sha << "\n";
  if (env.sanitizers || env.build_type != "Release") {
    std::cout << "  WARNING: not a Release build without sanitizers; timings are not representative.\n";
  }
  if (env.power_source != "AC Power") std::cout << "  WARNING: not on AC power.\n";
  std::cout << '\n';
  return env;
}

// Continuous real GPU work (64 MiB write streams) so clocks reach steady state
// before measurement; see the pre-warm note in docs/plan_and_architecture.md §7.2.
inline void prewarm_gpu(double ms) {
  cb::trace("prewarm workload=gpu_stream_write kind=private bytes=67108864 ms=" + std::to_string(ms));
  if (ms <= 0) return;
  probe::Buffer buf = probe::Buffer::create(probe::BufferKind::Private, size_t{64} << 20);
  const uint64_t until = computeedge::now_ns() + static_cast<uint64_t>(ms * 1e6);
  size_t calls = 0;
  while (computeedge::now_ns() < until) {
    probe::gpu_stream(probe::Stream::Write, buf);
    ++calls;
  }
  std::cout << "GPU pre-warm: " << calls << " x 64 MiB writes over " << ms << " ms\n\n";
}

inline void record_timing(cb::Metrics& m, const mtl::Timing& t) {
  m["encode_ns"] = double(t.encode_ns);
  m["commit_ns"] = double(t.commit_ns);
  m["wait_ns"] = double(t.wait_ns);
  m["total_ns"] = double(t.total_ns);
  m["gpu_ns"] = double(t.gpu_ns());
  m["queue_ns"] = double(t.queue_ns());
  m["wake_ns"] = double(t.wake_ns());
}

inline void write_results(const CommonOptions& o, const cb::Environment& env, const std::vector<cb::Result>& results) {
  if (!o.write) return;
  cb::ResultWriter writer(o.out, env);
  for (const auto& r : results) writer.add(r);
  writer.flush();
  std::cout << "\nResults written to " << writer.directory() << '\n';
}

}  // namespace ce_bench
