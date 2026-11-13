// Measurement logic of the event_log_overhead benchmark (moved verbatim from
// main.cpp; only printing goes through the Reporter).

#include "experiments.h"

#include <cstdint>
#include <utility>

#include "computeedge/bench/runner.h"
#include "computeedge/core/instrumentation.h"
#include "computeedge/core/scoped_timer.h"

namespace computeedge::bench::event_log_overhead {

using namespace computeedge;
namespace cb = computeedge::bench;

void run(const Options& opt, harness::Reporter& report) {
  const int reps = opt.reps;

  cb::Config cfg;
  cfg.repetitions = reps;
  cfg.min_sample_ns = 1'000'000;  // ~1 ms samples: thousands of calls each

  EventLog& log = EventLog::global();
  auto add = [&](cb::Result r) { report.result(std::move(r)); };

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
}

}  // namespace computeedge::bench::event_log_overhead
