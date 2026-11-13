#include "options.h"

namespace computeedge::bench::cache_interference {

using namespace computeedge::bench::harness;

Options parse_options(int argc, char** argv) {
  Options o;
  const char* src = kSource;
  static_cast<CommonOptions&>(o) = parse(argc, argv, src, kAllCommonFlags, [&](const std::string& f, const auto& next) {
    if (f == "--working-sets")
      o.working_sets = split_bytes(next());
    else if (f == "--gpu-sizes")
      o.gpu_sizes = split_bytes(next());
    else if (f == "--passes")
      o.passes = split(next());
    else
      return false;
    return true;
  });
  return o;
}

}  // namespace computeedge::bench::cache_interference
