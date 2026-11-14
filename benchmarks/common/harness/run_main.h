#pragma once

#include <iostream>
#include <utility>

#include "computeedge/bench/environment.h"
#include "computeedge/core/error.h"
#include "harness/cli.h"
#include "harness/results.h"

namespace computeedge::bench::harness {

// Standard entry point wrapper for benchmark binaries.
// Catches UsageError (2), InvalidRunError (3), computeedge::Error / std::exception (1),
// and flushes partial results when an error occurs during measurement.
template <typename BodyFn>
int run_main(const CommonOptions* opt, const Environment* env, Reporter* report, BodyFn&& body) {
  try {
    return body();
  } catch (const UsageError& e) {
    std::cerr << "error: " << e.what() << "\n";
    if (opt && env && report && !report->results().empty()) {
      try {
        write_results(opt->out, opt->write, *env, report->results(), /*partial=*/true);
      } catch (...) {
      }
    }
    return 2;
  } catch (const InvalidRunError& e) {
    std::cerr << "invalid run: " << e.what() << "\n";
    if (opt && env && report && !report->results().empty()) {
      try {
        write_results(opt->out, opt->write, *env, report->results(), /*partial=*/true);
      } catch (...) {
      }
    }
    return 3;
  } catch (const computeedge::Error& e) {
    std::cerr << "runtime error: " << e.what() << "\n";
    if (opt && env && report && !report->results().empty()) {
      try {
        write_results(opt->out, opt->write, *env, report->results(), /*partial=*/true);
      } catch (...) {
      }
    }
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "std::exception: " << e.what() << "\n";
    if (opt && env && report && !report->results().empty()) {
      try {
        write_results(opt->out, opt->write, *env, report->results(), /*partial=*/true);
      } catch (...) {
      }
    }
    return 1;
  } catch (...) {
    std::cerr << "unknown error\n";
    if (opt && env && report && !report->results().empty()) {
      try {
        write_results(opt->out, opt->write, *env, report->results(), /*partial=*/true);
      } catch (...) {
      }
    }
    return 1;
  }
}

template <typename BodyFn>
int run_main(BodyFn&& body) {
  return run_main(nullptr, nullptr, nullptr, std::forward<BodyFn>(body));
}

}  // namespace computeedge::bench::harness
