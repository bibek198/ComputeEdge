#include "options.h"

namespace computeedge::bench::metal_matmul {

using namespace computeedge::bench::harness;

metal::StorageMode storage_from(const std::string& s) {
  return parse_enum(s, {metal::StorageMode::NoCopy, metal::StorageMode::Shared, metal::StorageMode::Private},
                    "storage mode", kSource);
}

metal::Kernel kernel_from(const std::string& s) {
  return parse_enum(s, {metal::Kernel::Tiled16, metal::Kernel::MPS}, "kernel", kSource);
}

Options parse_options(int argc, char** argv) {
  Options o;
  const char* src = kSource;
  static_cast<CommonOptions&>(o) = parse(argc, argv, src, kAllCommonFlags, [&](const std::string& f, const auto& next) {
    if (f == "--sizes") {
      o.sizes.clear();
      for (const auto& s : split(next())) o.sizes.push_back(std::stoul(s));
    } else if (f == "--storage") {
      o.storage = split(next());
    } else if (f == "--kernels") {
      o.kernels = split(next());
    } else {
      return false;
    }
    return true;
  });
  return o;
}

}  // namespace computeedge::bench::metal_matmul
