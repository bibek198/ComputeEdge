// Instrumentation overhead: cost of one emit() call with the event log disabled
// and enabled, and of a ScopedTimer. Plan §6 requires this to be measured and
// reported alongside any result collected with logging on.
//
// Usage: event_log_overhead [--reps 30] [--out benchmarks/results] [--no-write]

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "computeedge/bench/environment.h"
#include "computeedge/bench/result_writer.h"
#include "computeedge/bench/runner.h"
#include "computeedge/core/instrumentation.h"
#include "computeedge/core/scoped_timer.h"

using namespace computeedge;
namespace cb = computeedge::bench;

int main(int argc, char** argv) {
  int reps = 30;
  std::string out = "benchmarks/results";
  bool write = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--reps" && i + 1 < argc) {
      reps = std::stoi(argv[++i]);
    } else if (a == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (a == "--no-write") {
      write = false;
    } else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }

  const cb::Environment env = cb::record_environment();
  std::cout << "ComputeEdge instrumentation overhead (" << env.chip << ", " << env.build_type << ")\n\n";

  cb::Config cfg;
  cfg.repetitions = reps;
  cfg.min_sample_ns = 1'000'000;  // ~1 ms samples: thousands of calls each

  EventLog& log = EventLog::global();
  std::vector<cb::Result> results;
  auto add = [&](cb::Result r) {
    std::cout << cb::format_result(r) << '\n';
    results.push_back(std::move(r));
  };

  uint64_t id = 0;
  log.set_enabled(false);
  add(cb::run("emit", {{"log", "disabled"}}, cfg, [&] {
    const uint64_t i = ++id;
    emit(EventType::Dispatch, i, 64, Backend::CPU, i, "x");
  }));

  log.set_enabled(true);
  add(cb::run(
      "emit", {{"log", "enabled"}}, cfg,
      [&] {
        const uint64_t i = ++id;
        emit(EventType::Dispatch, i, 64, Backend::CPU, i, "x");
      },
      0, [&] { log.clear(); }));

  log.set_enabled(false);
  add(cb::run("scoped_timer", {{"log", "disabled"}}, cfg, [&] { ScopedTimer t("x"); }));

  log.set_enabled(true);
  add(cb::run("scoped_timer", {{"log", "enabled"}}, cfg, [&] { ScopedTimer t("x"); }, 0, [&] { log.clear(); }));
  log.set_enabled(false);
  log.clear();

  add(cb::run("now_ns", {}, cfg, [] {
    volatile uint64_t t = now_ns();
    (void)t;
  }));

  if (write) {
    cb::ResultWriter writer(out, env);
    for (const auto& r : results) writer.add(r);
    writer.flush();
    std::cout << "\nResults written to " << writer.directory() << '\n';
  }
  return 0;
}
