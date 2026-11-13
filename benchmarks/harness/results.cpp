#include "harness/results.h"

#include <iostream>
#include <utility>

#include "computeedge/bench/result_writer.h"

namespace computeedge::bench::harness {

void Collector::add(Result r) {
  std::cout << format_result(r) << std::endl;
  results_.push_back(std::move(r));
}

std::string write_results(const std::string& out, bool write, const Environment& env,
                          const std::vector<Result>& results) {
  if (!write) return "";
  ResultWriter writer(out, env);
  for (const auto& r : results) writer.add(r);
  writer.flush();
  std::cout << "\nResults written to " << writer.directory() << '\n';
  return writer.directory();
}

}  // namespace computeedge::bench::harness
