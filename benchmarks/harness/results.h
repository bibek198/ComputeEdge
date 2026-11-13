#pragma once

// Collecting, presenting and writing benchmark results.
//
// Experiments (measurement logic) report through a Reporter and never print
// directly; the binary's main() chooses the presentation (ConsoleReporter for
// normal runs, the silent base Reporter in tests).

#include <string>
#include <vector>

#include "computeedge/bench/environment.h"
#include "computeedge/bench/runner.h"

namespace computeedge::bench::harness {

// Receives results and progress notes from an experiment. The base class keeps
// results and shows nothing.
class Reporter {
 public:
  virtual ~Reporter() = default;

  // A finished measurement: kept for writing, and shown by presenting subclasses.
  void result(Result r);
  // Free-form progress text, shown verbatim (callers include their own newlines).
  void note(const std::string& text) { on_note(text); }

  const std::vector<Result>& results() const { return results_; }

 protected:
  virtual void on_result(const Result& r) { (void)r; }
  virtual void on_note(const std::string& text) { (void)text; }

 private:
  std::vector<Result> results_;
};

// Shows each result's one-line summary and every note on standard output.
class ConsoleReporter : public Reporter {
 protected:
  void on_result(const Result& r) override;
  void on_note(const std::string& text) override;
};

// Writes all results to a new run directory under `out` (when `write` is true)
// and prints its path. Returns the directory, or "" if nothing was written.
std::string write_results(const std::string& out, bool write, const Environment& env,
                          const std::vector<Result>& results);

}  // namespace computeedge::bench::harness
