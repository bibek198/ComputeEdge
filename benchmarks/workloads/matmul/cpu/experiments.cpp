// Measurement logic of the cpu_matmul benchmark (moved verbatim from main.cpp; only
// printing goes through the Reporter).

#include "experiments.h"

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
#include <sstream>

namespace computeedge::bench::cpu_matmul {

namespace cb = computeedge::bench;
using namespace computeedge::bench::harness;

int run(const Options& opt, harness::Reporter& report) {
  cb::Config cfg;
  cfg.warmup = opt.warmup;
  cfg.min_warmup_ns = static_cast<uint64_t>(opt.min_warmup_ms * 1e6);
  cfg.repetitions = opt.reps;
  cfg.cooldown = opt.cooldown;

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
        report.result(std::move(r));
      }
    }
  }
  return 0;
}

}  // namespace computeedge::bench::cpu_matmul
