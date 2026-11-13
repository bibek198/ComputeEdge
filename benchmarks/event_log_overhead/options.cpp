#include "options.h"

namespace computeedge::bench::event_log_overhead {

using namespace computeedge::bench::harness;

Options parse_options(int argc, char** argv) {
  Options o;
  static_cast<CommonOptions&>(o) = parse(argc, argv, kSource, kReps | kOut | kNoWrite, nullptr);
  return o;
}

}  // namespace computeedge::bench::event_log_overhead
