#include "options.h"

namespace computeedge::bench::cpu_matmul {

using namespace computeedge::bench::harness;

MatmulImpl impl_from(const std::string& s) {
  return parse_enum(s, {MatmulImpl::Accelerate, MatmulImpl::Naive}, "impl", kSource);
}

Options parse_options(int argc, char** argv) {
  Options o;
  const char* src = kSource;
  static_cast<CommonOptions&>(o) = parse(argc, argv, src, kWarmup | kMinWarmupMs | kReps | kOut | kCooldown | kNoWrite,
                                         [&](const std::string& f, const auto& next) {
                                           if (f == "--sizes") {
                                             o.sizes.clear();
                                             for (const auto& s : split(next())) o.sizes.push_back(std::stoul(s));
                                           } else if (f == "--impls") {
                                             o.impls = split(next());
                                           } else if (f == "--modes") {
                                             o.modes = split(next());
                                           } else if (f == "--naive-max") {
                                             o.naive_max = std::stoul(next());
                                           } else {
                                             return false;
                                           }
                                           return true;
                                         });
  return o;
}

}  // namespace computeedge::bench::cpu_matmul
