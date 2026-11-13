// The command line of every benchmark binary is a contract: its defaults define
// what a session measures. These tests compile each binary's real options.cpp and
// compare its defaults with the values in the pre-refactor source (git tag
// pre-refactor), and check which common flags each binary accepts.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "cache_interference/options.h"
#include "cpu_matmul/options.h"
#include "dispatch_overhead/options.h"
#include "event_log_overhead/options.h"
#include "followups/options.h"
#include "metal_matmul/options.h"
#include "storage_modes/options.h"
#include "sync_methods/options.h"
#include "thermal_drift/options.h"

namespace b = computeedge::bench;
using Strings = std::vector<std::string>;

namespace {

struct Argv {
  explicit Argv(std::vector<std::string> args) : storage(std::move(args)) {
    storage.insert(storage.begin(), "bench");
    for (auto& s : storage) ptrs.push_back(s.data());
  }
  int argc() { return static_cast<int>(ptrs.size()); }
  char** argv() { return ptrs.data(); }
  std::vector<std::string> storage;
  std::vector<char*> ptrs;
};

template <typename Options>
void expect_common_defaults(const Options& o) {
  EXPECT_EQ(o.warmup, 10);
  EXPECT_EQ(o.min_warmup_ms, 200);
  EXPECT_EQ(o.prewarm_ms, 1000);
  EXPECT_EQ(o.reps, 30);
  EXPECT_EQ(o.out, "data/scratch");  // was benchmarks/results before the data move (Phase 2c)
  EXPECT_FALSE(o.cooldown);
  EXPECT_TRUE(o.write);
}

const std::vector<size_t> kMatmulSizes{128, 512, 1024, 2048, 4096};
constexpr size_t K = size_t{1} << 10, M = size_t{1} << 20, G = size_t{1} << 30;

}  // namespace

TEST(BenchmarkOptions, CpuMatmulDefaults) {
  Argv a({});
  const auto o = b::cpu_matmul::parse_options(a.argc(), a.argv());
  expect_common_defaults(o);
  EXPECT_EQ(o.sizes, kMatmulSizes);
  EXPECT_EQ(o.impls, (Strings{"accelerate", "naive"}));
  EXPECT_EQ(o.modes, (Strings{"kernel", "e2e"}));
  EXPECT_EQ(o.naive_max, 512u);
}

TEST(BenchmarkOptions, MetalMatmulDefaults) {
  Argv a({});
  const auto o = b::metal_matmul::parse_options(a.argc(), a.argv());
  expect_common_defaults(o);
  EXPECT_EQ(o.sizes, kMatmulSizes);
  EXPECT_EQ(o.storage, (Strings{"nocopy", "shared", "private"}));
  EXPECT_EQ(o.kernels, (Strings{"tiled16", "mps"}));
}

TEST(BenchmarkOptions, EventLogOverheadDefaults) {
  Argv a({});
  expect_common_defaults(b::event_log_overhead::parse_options(a.argc(), a.argv()));
}

TEST(BenchmarkOptions, DispatchOverheadDefaults) {
  Argv a({});
  const auto o = b::dispatch_overhead::parse_options(a.argc(), a.argv());
  expect_common_defaults(o);
  EXPECT_EQ(o.dispatches, (std::vector<uint32_t>{1, 4, 16, 64, 256, 1024}));
  EXPECT_EQ(o.encoders, (std::vector<uint32_t>{1, 4, 16, 64, 256}));
  EXPECT_EQ(o.cmdbufs, (std::vector<uint32_t>{1, 2, 4, 8, 16, 32, 64}));
  EXPECT_EQ(o.experiments, (Strings{"dispatches", "encoders", "cmdbufs"}));
}

TEST(BenchmarkOptions, SyncMethodsDefaults) {
  Argv a({});
  const auto o = b::sync_methods::parse_options(a.argc(), a.argv());
  expect_common_defaults(o);
  EXPECT_EQ(o.methods,
            (Strings{"wait_until_completed", "event_wait", "completion_handler", "spin_event", "spin_status"}));
  EXPECT_EQ(o.work, (Strings{"empty", "4M", "256M"}));
  EXPECT_EQ(o.gaps_ms, (std::vector<double>{0, 0.1, 1, 10, 100}));
  EXPECT_EQ(o.experiments, (Strings{"methods", "idle"}));
}

TEST(BenchmarkOptions, StorageModesDefaults) {
  Argv a({});
  const auto o = b::storage_modes::parse_options(a.argc(), a.argv());
  expect_common_defaults(o);
  EXPECT_EQ(o.sizes, (std::vector<size_t>{16 * K, 256 * K, 4 * M, 64 * M, 256 * M, 1 * G}));
  EXPECT_EQ(o.kinds, (Strings{"host", "shared", "private", "nocopy", "shared_copy"}));
  EXPECT_EQ(o.experiments, (Strings{"create", "cpu_touch", "gpu_first", "gpu_bw"}));
}

TEST(BenchmarkOptions, CacheInterferenceDefaults) {
  Argv a({});
  const auto o = b::cache_interference::parse_options(a.argc(), a.argv());
  expect_common_defaults(o);
  EXPECT_EQ(o.working_sets, (std::vector<size_t>{1 * M, 4 * M, 12 * M, 64 * M}));
  EXPECT_EQ(o.gpu_sizes, (std::vector<size_t>{16 * M, 64 * M, 256 * M}));
  EXPECT_EQ(o.passes, (Strings{"read", "slc", "sgemm"}));
}

TEST(BenchmarkOptions, FollowupsDefaults) {
  Argv a({});
  const auto o = b::followups::parse_options(a.argc(), a.argv());
  expect_common_defaults(o);
  EXPECT_EQ(o.experiments, (Strings{"wake", "cpu_write", "gpu_write"}));
}

TEST(BenchmarkOptions, ThermalDriftDefaults) {
  Argv a({});
  const auto o = b::thermal_drift::parse_options(a.argc(), a.argv());
  expect_common_defaults(o);
  EXPECT_EQ(o.minutes, 15);
  EXPECT_EQ(o.load, "mixed");
  EXPECT_EQ(o.interval_ms, 1000);
  EXPECT_TRUE(o.wait_for_nominal) << "waiting for nominal is on unless --no-cooldown";
  Argv off({"--no-cooldown"});
  EXPECT_FALSE(b::thermal_drift::parse_options(off.argc(), off.argv()).wait_for_nominal);
}

TEST(BenchmarkOptions, BinarySpecificFlagsParse) {
  Argv a({"--sizes", "64,128", "--impls", "naive", "--modes", "e2e", "--naive-max", "64"});
  const auto o = b::cpu_matmul::parse_options(a.argc(), a.argv());
  EXPECT_EQ(o.sizes, (std::vector<size_t>{64, 128}));
  EXPECT_EQ(o.impls, (Strings{"naive"}));
  EXPECT_EQ(o.naive_max, 64u);
  Argv s({"--gaps-ms", "0,0.5", "--methods", "spin_event"});
  const auto so = b::sync_methods::parse_options(s.argc(), s.argv());
  EXPECT_EQ(so.gaps_ms, (std::vector<double>{0, 0.5}));
  EXPECT_EQ(so.methods, (Strings{"spin_event"}));
}

using BenchmarkOptionsDeathTest = ::testing::Test;

TEST(BenchmarkOptionsDeathTest, CommonFlagSetsPerBinaryAreUnchanged) {
  Argv prewarm({"--prewarm-ms", "1"});
  EXPECT_EXIT(b::cpu_matmul::parse_options(prewarm.argc(), prewarm.argv()), ::testing::ExitedWithCode(2),
              "unknown argument --prewarm-ms");
  Argv warmup({"--warmup", "1"});
  EXPECT_EXIT(b::event_log_overhead::parse_options(warmup.argc(), warmup.argv()), ::testing::ExitedWithCode(2),
              "unknown argument --warmup");
  Argv cooldown({"--cooldown"});
  EXPECT_EXIT(b::event_log_overhead::parse_options(cooldown.argc(), cooldown.argv()), ::testing::ExitedWithCode(2),
              "unknown argument --cooldown");
}

TEST(BenchmarkOptionsDeathTest, ValidationStillRunsBeforeMeasurement) {
  Argv bad_method({"--methods", "nope"});
  EXPECT_EXIT(b::sync_methods::parse_options(bad_method.argc(), bad_method.argv()), ::testing::ExitedWithCode(2),
              "unknown sync method nope");
  Argv bad_size({"--sizes", "1000"});
  EXPECT_EXIT(b::storage_modes::parse_options(bad_size.argc(), bad_size.argv()), ::testing::ExitedWithCode(2),
              "multiples of the page size");
  Argv bad_load({"--load", "both"});
  EXPECT_EXIT(b::thermal_drift::parse_options(bad_load.argc(), bad_load.argv()), ::testing::ExitedWithCode(2),
              "unknown --load both");
}
