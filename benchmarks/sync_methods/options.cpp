#include "options.h"

namespace computeedge::bench::sync_methods {

using namespace computeedge::bench::harness;

metal::probe::Sync parse_sync(const std::string& s, const char* src) {
  namespace probe = computeedge::metal::probe;
  return parse_enum(s,
                    {probe::Sync::WaitUntilCompleted, probe::Sync::EventWait, probe::Sync::CompletionHandler,
                     probe::Sync::SpinEvent, probe::Sync::SpinStatus},
                    "sync method", src);
}

Options parse_options(int argc, char** argv) {
  Options o;
  const char* src = kSource;
  static_cast<CommonOptions&>(o) = parse(argc, argv, src, kAllCommonFlags, [&](const std::string& f, const auto& next) {
    if (f == "--methods")
      o.methods = split(next());
    else if (f == "--work")
      o.work = split(next());
    else if (f == "--gaps-ms")
      o.gaps_ms = split_doubles(next());
    else if (f == "--experiments")
      o.experiments = split(next());
    else
      return false;
    return true;
  });
  for (const auto& m : o.methods) parse_sync(m, src);  // validate before running anything
  return o;
}

}  // namespace computeedge::bench::sync_methods
