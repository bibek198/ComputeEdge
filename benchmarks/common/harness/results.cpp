#include "harness/results.h"

#include <iostream>
#include <utility>

#include "computeedge/bench/result_writer.h"

namespace computeedge::bench::harness {

void Reporter::result(Result r) {
  on_result(r);
  results_.push_back(std::move(r));
}

void ConsoleReporter::on_result(const Result& r) { std::cout << format_result(r) << std::endl; }

void ConsoleReporter::on_note(const std::string& text) { std::cout << text << std::flush; }

std::string write_results(const std::string& out, bool write, const Environment& env,
                          const std::vector<Result>& results, bool partial) {
  if (!write || results.empty()) return "";
  ResultWriter writer(out, env);
  if (partial) writer.mark_partial(true);
  for (const auto& r : results) writer.add(r);
  writer.flush();
  if (partial) {
    std::cout << "\nPartial results written to " << writer.directory() << " (partial)\n";
  } else {
    std::cout << "\nResults written to " << writer.directory() << '\n';
  }
  return writer.directory();
}

}  // namespace computeedge::bench::harness
