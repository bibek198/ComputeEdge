// CPU matmul benchmark: Accelerate (baseline) and naive (reference) SGEMM.
//
// Modes:
//   kernel  matmul_into() into a pre-allocated output (compute only)
//   e2e     matmul()  (output allocation + zero-fill + compute + free)
//
// Usage:
//   cpu_matmul [--sizes 128,512,1024,2048,4096] [--impls accelerate,naive]
//              [--modes kernel,e2e] [--warmup 10] [--min-warmup-ms 200] [--reps 30]
//              [--naive-max 512]
//              [--out data/scratch] [--cooldown] [--no-write]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "computeedge/bench/environment.h"
#include "computeedge/bench/result_writer.h"
#include "computeedge/bench/runner.h"
#include "computeedge/bench/trace.h"
#include "computeedge/computeedge.h"
#include "harness/cli.h"
#include "harness/console.h"
#include "harness/correctness.h"
#include "harness/results.h"

using namespace computeedge;
namespace cb = computeedge::bench;
using namespace computeedge::bench::harness;

namespace {

constexpr const char* kSource = "benchmarks/cpu_matmul/main.cpp";

struct Options {
  std::vector<size_t> sizes{128, 512, 1024, 2048, 4096};
  std::vector<std::string> impls{"accelerate", "naive"};
  std::vector<std::string> modes{"kernel", "e2e"};
  int warmup = 10;
  double min_warmup_ms = 200;
  int reps = 30;
  size_t naive_max = 512;
  std::string out = "data/scratch";
  bool cooldown = false;
  bool write = true;
};

// Accepted common flags: --warmup --min-warmup-ms --reps --out --cooldown --no-write (no --prewarm-ms).
Options parse_options(int argc, char** argv) {
  Options o;
  const CommonOptions c = parse(argc, argv, kSource, kWarmup | kMinWarmupMs | kReps | kOut | kCooldown | kNoWrite,
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
  o.warmup = c.warmup;
  o.min_warmup_ms = c.min_warmup_ms;
  o.reps = c.reps;
  o.out = c.out;
  o.cooldown = c.cooldown;
  o.write = c.write;
  return o;
}

MatmulImpl impl_from(const std::string& s) {
  return parse_enum(s, {MatmulImpl::Accelerate, MatmulImpl::Naive}, "impl", kSource);
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_options(argc, argv);
  const cb::Environment env = banner("ComputeEdge CPU matmul benchmark", BannerStyle::CpuOnly);

  cb::Config cfg;
  cfg.warmup = opt.warmup;
  cfg.min_warmup_ns = static_cast<uint64_t>(opt.min_warmup_ms * 1e6);
  cfg.repetitions = opt.reps;
  cfg.cooldown = opt.cooldown;

  std::vector<cb::Result> results;
  for (size_t n : opt.sizes) {
    const Tensor A = random_square(n, 1);
    const Tensor B = random_square(n, 2);
    const double flops = 2.0 * double(n) * double(n) * double(n);

    // Correctness gate: Accelerate vs naive reference where affordable.
    if (n <= opt.naive_max) {
      cb::trace("gate n=" + std::to_string(n) + " impl=accelerate ref=naive tol=1e-5");
      const double err = max_scaled_error(matmul(A, B, MatmulImpl::Accelerate), matmul(A, B, MatmulImpl::Naive), n);
      if (err > kCorrectnessTolerance) {
        std::fprintf(stderr, "correctness check FAILED at n=%zu (scaled error %.3g)\n", n, err);
        return 1;
      }
    }

    for (const std::string& impl_name : opt.impls) {
      const MatmulImpl impl = impl_from(impl_name);
      if (impl == MatmulImpl::Naive && n > opt.naive_max) continue;
      for (const std::string& mode : opt.modes) {
        const std::map<std::string, std::string> params{{"impl", impl_name},      {"mode", mode},
                                                        {"m", std::to_string(n)}, {"n", std::to_string(n)},
                                                        {"k", std::to_string(n)}, {"dtype", "f32"}};
        cb::Result r;
        if (mode == "kernel") {
          Tensor C({n, n});
          r = cb::run("matmul", params, cfg, [&] { matmul_into(A, B, C, impl); }, flops);
        } else if (mode == "e2e") {
          r = cb::run("matmul", params, cfg, [&] { Tensor C = matmul(A, B, impl); }, flops);
        } else {
          usage(kSource, "unknown mode " + mode);
        }
        std::cout << cb::format_result(r) << std::endl;
        results.push_back(std::move(r));
      }
    }
  }

  write_results(opt.out, opt.write, env, results);
  return 0;
}
