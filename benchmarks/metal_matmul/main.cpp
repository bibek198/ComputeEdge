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
//                [--out benchmarks/results] [--cooldown] [--no-write]

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

using namespace computeedge;
namespace cb = computeedge::bench;
namespace mtl = computeedge::metal;

namespace {

struct Options {
  std::vector<size_t> sizes{128, 512, 1024, 2048, 4096};
  std::vector<std::string> storage{"nocopy", "shared", "private"};
  std::vector<std::string> kernels{"tiled16", "mps"};
  int warmup = 10;
  double min_warmup_ms = 200;
  double prewarm_ms = 1000;
  int reps = 30;
  std::string out = "benchmarks/results";
  bool cooldown = false;
  bool write = true;
};

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');)
    if (!item.empty()) out.push_back(item);
  return out;
}

[[noreturn]] void usage(const char* msg) {
  std::fprintf(stderr, "error: %s\nsee the header of benchmarks/metal_matmul/main.cpp for usage\n", msg);
  std::exit(2);
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage(("missing value for " + a).c_str());
      return argv[++i];
    };
    if (a == "--sizes") {
      o.sizes.clear();
      for (const auto& s : split(next())) o.sizes.push_back(std::stoul(s));
    } else if (a == "--storage") {
      o.storage = split(next());
    } else if (a == "--kernels") {
      o.kernels = split(next());
    } else if (a == "--warmup") {
      o.warmup = std::stoi(next());
    } else if (a == "--min-warmup-ms") {
      o.min_warmup_ms = std::stod(next());
    } else if (a == "--prewarm-ms") {
      o.prewarm_ms = std::stod(next());
    } else if (a == "--reps") {
      o.reps = std::stoi(next());
    } else if (a == "--out") {
      o.out = next();
    } else if (a == "--cooldown") {
      o.cooldown = true;
    } else if (a == "--no-write") {
      o.write = false;
    } else {
      usage(("unknown argument " + a).c_str());
    }
  }
  return o;
}

mtl::StorageMode storage_from(const std::string& s) {
  if (s == "nocopy") return mtl::StorageMode::NoCopy;
  if (s == "shared") return mtl::StorageMode::Shared;
  if (s == "private") return mtl::StorageMode::Private;
  usage(("unknown storage mode " + s).c_str());
}

mtl::Kernel kernel_from(const std::string& s) {
  if (s == "tiled16") return mtl::Kernel::Tiled16;
  if (s == "mps") return mtl::Kernel::MPS;
  usage(("unknown kernel " + s).c_str());
}

Tensor random_square(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n * n);
  for (float& x : v) x = dist(rng);
  return Tensor({n, n}, v);
}

double max_scaled_error(const Tensor& got, const Tensor& ref, size_t k) {
  const std::vector<float> g = got.to_vector(), r = ref.to_vector();
  double worst = 0;
  for (size_t i = 0; i < g.size(); ++i)
    worst = std::max(worst, std::abs(double(g[i]) - double(r[i])) / (std::abs(double(r[i])) + 1.0));
  return worst / std::sqrt(double(k));
}

void record_timing(cb::Metrics& m, const mtl::Timing& t) {
  m["buffer_ns"] = double(t.buffer_ns);
  m["copy_in_ns"] = double(t.copy_in_ns);
  m["encode_ns"] = double(t.encode_ns);
  m["commit_ns"] = double(t.commit_ns);
  m["wait_ns"] = double(t.wait_ns);
  m["copy_out_ns"] = double(t.copy_out_ns);
  m["sched_ns"] = double(t.sched_ns());
  m["gpu_ns"] = double(t.gpu_ns());
  m["queue_ns"] = double(t.queue_ns());
  m["wake_ns"] = double(t.wake_ns());
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse(argc, argv);
  if (!mtl::available()) {
    std::fprintf(stderr, "Metal is not available on this system\n");
    return 1;
  }
  const cb::Environment env = cb::record_environment();
  std::cout << "ComputeEdge Metal matmul benchmark\n"
            << "  " << env.chip << " / " << env.gpu_name << " (" << env.model << "), macOS " << env.os_version << " ("
            << env.os_build << ")\n"
            << "  power: " << env.power_source << (env.low_power_mode ? ", LOW POWER MODE" : "")
            << ", thermal: " << cb::to_string(env.thermal_state) << ", build: " << env.build_type
            << (env.sanitizers ? " +sanitizers" : "") << ", git " << env.git_sha << "\n";
  if (env.sanitizers || env.build_type != "Release") {
    std::cout << "  WARNING: not a Release build without sanitizers; timings are not representative.\n";
  }
  if (env.power_source != "AC Power") std::cout << "  WARNING: not on AC power.\n";
  std::cout << '\n';

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
          record_timing(m, t);
        });
    std::cout << cb::format_result(r) << "\n\n";
    results.push_back(std::move(r));
  }

  // Process-level GPU pre-warm (excluded from all results).
  cb::trace("prewarm workload=metal_matmul n=256 storage=shared kernel=tiled16 ms=" + std::to_string(opt.prewarm_ms));
  {
    const Tensor A = random_square(256, 3), B = random_square(256, 4);
    Tensor C({256, 256});
    const uint64_t until = now_ns() + static_cast<uint64_t>(opt.prewarm_ms * 1e6);
    size_t calls = 0;
    while (now_ns() < until) {
      mtl::matmul_into(A, B, C, {mtl::StorageMode::Shared});
      ++calls;
    }
    std::cout << "GPU pre-warm: " << calls << " dispatches over " << opt.prewarm_ms << " ms\n\n";
  }

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
        if (err > 1e-5) {
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
              record_timing(m, t);
            },
            flops);
        std::cout << cb::format_result(r) << std::endl;
        results.push_back(std::move(r));
      }
    }
  }

  if (opt.write) {
    cb::ResultWriter writer(opt.out, env);
    for (const auto& r : results) writer.add(r);
    writer.flush();
    std::cout << "\nResults written to " << writer.directory() << '\n';
  }
  return 0;
}
