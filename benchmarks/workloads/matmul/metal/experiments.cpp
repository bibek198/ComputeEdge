// Measurement logic of the metal_matmul benchmark (moved verbatim from main.cpp; only
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
#include "computeedge/metal/matmul.h"
#include "harness/cli.h"
#include "harness/console.h"
#include "harness/correctness.h"
#include "harness/gpu.h"
#include "harness/results.h"

using namespace computeedge;
#include <sstream>

namespace computeedge::bench::metal_matmul {

namespace cb = computeedge::bench;
namespace mtl = computeedge::metal;
using namespace computeedge::bench::harness;

int run(const Options& opt, harness::Reporter& report) {
  // Cold start: shader compilation + first dispatch, measured once per process.
  {
    const Tensor A = random_square(64, 1), B = random_square(64, 2);
    Tensor C({64, 64});
    cb::Config cold;
    cold.warmup = 0;
    cold.min_warmup_ns = 0;
    cold.repetitions = 1;
    cb::Result r = cb::run_with_metrics(
        "matmul_metal_cold", {{"m", "64"}, {"n", "64"}, {"k", "64"}, {"storage", "nocopy"}}, cold, [&](cb::Metrics& m) {
          const uint64_t before = mtl::compile_ns();
          mtl::Timing t;
          mtl::matmul_into(A, B, C, {mtl::StorageMode::NoCopy}, &t);
          m["compile_ns"] = double(mtl::compile_ns() - before);
          record_matmul_timing(m, t);
        });
    report.result(std::move(r));
    report.note("\n");
  }

  // Process-level GPU pre-warm (excluded from all results).
  prewarm(PrewarmWorkload::MetalMatmulShared256, opt.prewarm_ms, report);

  cb::Config cfg;
  cfg.warmup = opt.warmup;
  cfg.min_warmup_ns = static_cast<uint64_t>(opt.min_warmup_ms * 1e6);
  cfg.repetitions = opt.reps;
  cfg.cooldown = opt.cooldown;

  for (size_t n : opt.sizes) {
    const Tensor A = random_square(n, 1);
    const Tensor B = random_square(n, 2);
    const double flops = 2.0 * double(n) * double(n) * double(n);
    const Tensor ref = matmul(A, B, MatmulImpl::Accelerate);

    for (const std::string& kernel_name : opt.kernels) {
      for (const std::string& storage_name : opt.storage) {
        const mtl::MatmulOptions mo{storage_from(storage_name), kernel_from(kernel_name)};
        Tensor C({n, n});

        cb::trace("gate n=" + std::to_string(n) + " kernel=" + kernel_name + " storage=" + storage_name +
                  " ref=accelerate tol=1e-5");
        mtl::matmul_into(A, B, C, mo);  // correctness gate against Accelerate
        const double err = max_scaled_error(C, ref, n);
        if (err > kCorrectnessTolerance) {
          std::fprintf(stderr, "correctness check FAILED at n=%zu kernel=%s storage=%s (scaled error %.3g)\n", n,
                       kernel_name.c_str(), storage_name.c_str(), err);
          return 1;
        }

        const std::map<std::string, std::string> params{
            {"backend", "metal"},     {"kernel", kernel_name},  {"storage", storage_name}, {"m", std::to_string(n)},
            {"n", std::to_string(n)}, {"k", std::to_string(n)}, {"dtype", "f32"}};
        cb::Result r = cb::run_with_metrics(
            "matmul_metal", params, cfg,
            [&](cb::Metrics& m) {
              mtl::Timing t;
              mtl::matmul_into(A, B, C, mo, &t);
              record_matmul_timing(m, t);
            },
            flops);
        report.result(std::move(r));
      }
    }
  }
  return 0;
}

}  // namespace computeedge::bench::metal_matmul
