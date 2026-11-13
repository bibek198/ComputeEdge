#pragma once

// Collecting and writing benchmark results.

#include <string>
#include <vector>

#include "computeedge/bench/environment.h"
#include "computeedge/bench/runner.h"

namespace computeedge::bench::harness {

// Prints each result's one-line summary as it arrives and keeps it for writing.
class Collector {
 public:
  void add(Result r);
  const std::vector<Result>& results() const { return results_; }

 private:
  std::vector<Result> results_;
};

// Writes all results to a new run directory under `out` (when `write` is true)
// and prints its path. Returns the directory, or "" if nothing was written.
std::string write_results(const std::string& out, bool write, const Environment& env,
                          const std::vector<Result>& results);

}  // namespace computeedge::bench::harness
