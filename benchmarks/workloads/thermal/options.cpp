#include "options.h"

namespace computeedge::bench::thermal_drift {

using namespace computeedge::bench::harness;

Options parse_options(int argc, char** argv) {
  Options o;
  const char* src = kSource;
  static_cast<CommonOptions&>(o) = parse(argc, argv, src, kAllCommonFlags, [&](const std::string& f, const auto& next) {
    if (f == "--minutes")
      o.minutes = std::stod(next());
    else if (f == "--load")
      o.load = next();
    else if (f == "--interval-ms")
      o.interval_ms = std::stod(next());
    else if (f == "--no-cooldown")
      o.wait_for_nominal = false;
    else
      return false;
    return true;
  });
  if (o.load != "mixed" && o.load != "cpu" && o.load != "gpu" && o.load != "none")
    usage(src, "unknown --load " + o.load);
  return o;
}

}  // namespace computeedge::bench::thermal_drift
