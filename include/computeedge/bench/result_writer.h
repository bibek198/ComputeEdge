#pragma once

// Reproducible output: <root>/<stamp>_<sha>/{env.json, samples.csv, summary.csv,
// metrics.csv, metrics_summary.csv}, and a one-line console summary.

#include <string>
#include <vector>
#include "computeedge/bench/environment.h"
#include "computeedge/bench/runner.h"

namespace computeedge::bench {

// Writes <root>/<YYYYmmdd-HHMMSS>_<sha>/{env.json, samples.csv, summary.csv} and,
// when any result has metrics, metrics.csv (long format) and metrics_summary.csv.
class ResultWriter {
 public:
  ResultWriter(const std::string& root, const Environment& env);
  void add(const Result& result);
  void flush() const;
  const std::string& directory() const { return dir_; }

 private:
  std::string dir_;
  Environment env_;
  std::vector<Result> results_;
};

// One-line human-readable summary for console output.
std::string format_result(const Result& result);

}  // namespace computeedge::bench
