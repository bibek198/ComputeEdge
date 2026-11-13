// Metal matmul benchmark: tiled FP32 kernel under each storage mode, with a
// per-call phase breakdown (buffers, copies, encode, commit, wait, GPU time,
// queue and wake-up latency).
//
// Before any measurement the GPU is pre-warmed with --prewarm-ms of continuous
// dispatches: without it, whichever configuration runs first is measured
// ~1.7x slower at 128² (the GPU had not reached steady-state clocks).
//
// Usage:
//   metal_matmul [--sizes 128,512,1024,2048,4096] [--storage nocopy,shared,private]
//                [--kernels tiled16,mps]
//                [--warmup 10] [--min-warmup-ms 200] [--prewarm-ms 1000] [--reps 30]
//                [--out data/scratch] [--cooldown] [--no-write]

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
namespace cb = computeedge::bench;
namespace mtl = computeedge::metal;
using namespace computeedge::bench::harness;

namespace {

constexpr const char* kSource = "benchmarks/metal_matmul/main.cpp";

struct Options {
  std::vector<size_t> sizes{128, 512, 1024, 2048, 4096};
  std::vector<std::string> storage{"nocopy", "shared", "private"};
  std::vector<std::string> kernels{"tiled16", "mps"};
  int warmup = 10;
  double min_warmup_ms = 200;
  double prewarm_ms = 1000;
  int reps = 30;
  std::string out = "data/scratch";
  bool cooldown = false;
  bool write = true;
};

// Accepts every common flag.
Options parse_options(int argc, char** argv) {
  Options o;
  const CommonOptions c = parse(argc, argv, kSource, kAllCommonFlags, [&](const std::string& f, const auto& next) {
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
  o.warmup = c.warmup;
  o.min_warmup_ms = c.min_warmup_ms;
  o.prewarm_ms = c.prewarm_ms;
  o.reps = c.reps;
  o.out = c.out;
  o.cooldown = c.cooldown;
  o.write = c.write;
  return o;
}

mtl::StorageMode storage_from(const std::string& s) {
  return parse_enum(s, {mtl::StorageMode::NoCopy, mtl::StorageMode::Shared, mtl::StorageMode::Private}, "storage mode",
                    kSource);
}

mtl::Kernel kernel_from(const std::string& s) {
  return parse_enum(s, {mtl::Kernel::Tiled16, mtl::Kernel::MPS}, "kernel", kSource);
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse_options(argc, argv);
  require_metal();
  const cb::Environment env = banner("ComputeEdge Metal matmul benchmark", BannerStyle::WithGpu);

  std::vector<cb::Result> results;

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
    std::cout << cb::format_result(r) << "\n\n";
    results.push_back(std::move(r));
  }

  // Process-level GPU pre-warm (excluded from all results).
  prewarm(PrewarmWorkload::MetalMatmulShared256, opt.prewarm_ms);

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
        std::cout << cb::format_result(r) << std::endl;
        results.push_back(std::move(r));
      }
    }
  }

  write_results(opt.out, opt.write, env, results);
  return 0;
}
