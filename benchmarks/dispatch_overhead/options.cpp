#include "options.h"

namespace computeedge::bench::dispatch_overhead {

using namespace computeedge::bench::harness;

namespace {

std::vector<uint32_t> to_u32(const std::string& s) {
  std::vector<uint32_t> out;
  for (const auto& v : split(s)) out.push_back(static_cast<uint32_t>(std::stoul(v)));
  return out;
}

}  // namespace

Options parse_options(int argc, char** argv) {
  Options o;
  const char* src = kSource;
  static_cast<CommonOptions&>(o) = parse(argc, argv, src, kAllCommonFlags, [&](const std::string& f, const auto& next) {
    if (f == "--dispatches")
      o.dispatches = to_u32(next());
    else if (f == "--encoders")
      o.encoders = to_u32(next());
    else if (f == "--cmdbufs")
      o.cmdbufs = to_u32(next());
    else if (f == "--experiments")
      o.experiments = split(next());
    else
      return false;
    return true;
  });
  return o;
}

}  // namespace computeedge::bench::dispatch_overhead
