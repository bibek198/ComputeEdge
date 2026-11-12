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
//              [--out benchmarks/results] [--cooldown] [--no-write]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "computeedge/benchmark.h"
#include "computeedge/computeedge.h"

using namespace computeedge;
namespace cb = computeedge::bench;

namespace {

struct Options {
  std::vector<size_t> sizes{128, 512, 1024, 2048, 4096};
  std::vector<std::string> impls{"accelerate", "naive"};
  std::vector<std::string> modes{"kernel", "e2e"};
  int warmup = 10;
  double min_warmup_ms = 200;
  int reps = 30;
  size_t naive_max = 512;
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
  std::fprintf(stderr, "error: %s\nsee the header of benchmarks/cpu_matmul.cpp for usage\n", msg);
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
    } else if (a == "--impls") {
      o.impls = split(next());
    } else if (a == "--modes") {
      o.modes = split(next());
    } else if (a == "--warmup") {
      o.warmup = std::stoi(next());
    } else if (a == "--min-warmup-ms") {
      o.min_warmup_ms = std::stod(next());
    } else if (a == "--reps") {
      o.reps = std::stoi(next());
    } else if (a == "--naive-max") {
      o.naive_max = std::stoul(next());
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

MatmulImpl impl_from(const std::string& s) {
  if (s == "accelerate") return MatmulImpl::Accelerate;
  if (s == "naive") return MatmulImpl::Naive;
  usage(("unknown impl " + s).c_str());
}

Tensor random_square(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n * n);
  for (float& x : v) x = dist(rng);
  return Tensor({n, n}, v);
}

// Max relative error of `got` vs `ref`, scaled by sqrt(K) (FP32 accumulation-order differences).
double max_scaled_error(const Tensor& got, const Tensor& ref, size_t k) {
  const std::vector<float> g = got.to_vector(), r = ref.to_vector();
  double worst = 0;
  for (size_t i = 0; i < g.size(); ++i) {
    const double err = std::abs(double(g[i]) - double(r[i])) / (std::abs(double(r[i])) + 1.0);
    worst = std::max(worst, err);
  }
  return worst / std::sqrt(double(k));
}

}  // namespace

int main(int argc, char** argv) {
  const Options opt = parse(argc, argv);
  const cb::Environment env = cb::record_environment();

  std::cout << "ComputeEdge CPU matmul benchmark\n"
            << "  " << env.chip << " (" << env.model << "), macOS " << env.os_version << " (" << env.os_build << ")\n"
            << "  power: " << env.power_source << (env.low_power_mode ? ", LOW POWER MODE" : "")
            << ", thermal: " << cb::to_string(env.thermal_state) << ", build: " << env.build_type
            << (env.sanitizers ? " +sanitizers" : "") << ", git " << env.git_sha << "\n";
  if (env.sanitizers || env.build_type != "Release") {
    std::cout << "  WARNING: not a Release build without sanitizers; timings are not representative.\n";
  }
  if (env.power_source != "AC Power") std::cout << "  WARNING: not on AC power.\n";
  std::cout << '\n';

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
      const double err = max_scaled_error(matmul(A, B, MatmulImpl::Accelerate), matmul(A, B, MatmulImpl::Naive), n);
      if (err > 1e-5) {
        std::fprintf(stderr, "correctness check FAILED at n=%zu (scaled error %.3g)\n", n, err);
        return 1;
      }
    }

    for (const std::string& impl_name : opt.impls) {
      const MatmulImpl impl = impl_from(impl_name);
      if (impl == MatmulImpl::Naive && n > opt.naive_max) continue;
      for (const std::string& mode : opt.modes) {
        const std::map<std::string, std::string> params{
            {"impl", impl_name}, {"mode", mode}, {"m", std::to_string(n)}, {"n", std::to_string(n)},
            {"k", std::to_string(n)}, {"dtype", "f32"}};
        cb::Result r;
        if (mode == "kernel") {
          Tensor C({n, n});
          r = cb::run("matmul", params, cfg, [&] { matmul_into(A, B, C, impl); }, flops);
        } else if (mode == "e2e") {
          r = cb::run("matmul", params, cfg, [&] { Tensor C = matmul(A, B, impl); }, flops);
        } else {
          usage(("unknown mode " + mode).c_str());
        }
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
