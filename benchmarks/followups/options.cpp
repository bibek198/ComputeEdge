#include "options.h"

namespace computeedge::bench::followups {

using namespace computeedge::bench::harness;

Options parse_options(int argc, char** argv) {
  Options o;
  const char* src = kSource;
  static_cast<CommonOptions&>(o) = parse(argc, argv, src, kAllCommonFlags, [&](const std::string& f, const auto& next) {
    if (f == "--experiments")
      o.experiments = split(next());
    else
      return false;
    return true;
  });
  return o;
}

}  // namespace computeedge::bench::followups
