// Tests for benchmarks/harness/cli: helpers and the common-flag parser. The
// defaults must equal the pre-refactor values every benchmark used.

#include "harness/cli.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace h = computeedge::bench::harness;

namespace {

// Builds a mutable argv for parse().
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

}  // namespace

TEST(HarnessCli, DefaultsMatchPreRefactorValues) {
  Argv a({});
  const h::CommonOptions o = h::parse(a.argc(), a.argv(), "src", h::kAllCommonFlags, nullptr);
  EXPECT_EQ(o.warmup, 10);
  EXPECT_EQ(o.min_warmup_ms, 200);
  EXPECT_EQ(o.prewarm_ms, 1000);
  EXPECT_EQ(o.reps, 30);
  EXPECT_EQ(o.out, "data/scratch");
  EXPECT_FALSE(o.cooldown);
  EXPECT_TRUE(o.write);
}

TEST(HarnessCli, ParsesEveryCommonFlag) {
  Argv a({"--warmup", "3", "--min-warmup-ms", "50", "--prewarm-ms", "0", "--reps", "7", "--out", "x", "--cooldown",
          "--no-write"});
  const h::CommonOptions o = h::parse(a.argc(), a.argv(), "src", h::kAllCommonFlags, nullptr);
  EXPECT_EQ(o.warmup, 3);
  EXPECT_EQ(o.min_warmup_ms, 50);
  EXPECT_EQ(o.prewarm_ms, 0);
  EXPECT_EQ(o.reps, 7);
  EXPECT_EQ(o.out, "x");
  EXPECT_TRUE(o.cooldown);
  EXPECT_FALSE(o.write);
}

TEST(HarnessCli, ConfigMapsCommonOptions) {
  h::CommonOptions o;
  o.warmup = 4;
  o.min_warmup_ms = 12.5;
  o.reps = 9;
  o.cooldown = true;
  const auto c = h::config(o);
  EXPECT_EQ(c.warmup, 4);
  EXPECT_EQ(c.min_warmup_ns, 12'500'000u);
  EXPECT_EQ(c.repetitions, 9);
  EXPECT_TRUE(c.cooldown);
  EXPECT_EQ(c.inner_iterations, 0) << "auto-calibration stays the default";
}

TEST(HarnessCli, ExtraFlagsAreOfferedToTheCallback) {
  std::vector<std::string> seen;
  Argv a({"--sizes", "1,2", "--reps", "5"});
  const h::CommonOptions o = h::parse(a.argc(), a.argv(), "src", h::kReps, [&](const std::string& f, const auto& next) {
    if (f != "--sizes") return false;
    seen.push_back(next());
    return true;
  });
  EXPECT_EQ(seen, (std::vector<std::string>{"1,2"}));
  EXPECT_EQ(o.reps, 5);
}

using HarnessCliDeathTest = ::testing::Test;

TEST(HarnessCliDeathTest, DisabledCommonFlagIsRejected) {
  // cpu_matmul does not accept --prewarm-ms; the harness must keep rejecting it.
  Argv a({"--prewarm-ms", "5"});
  EXPECT_EXIT(h::parse(a.argc(), a.argv(), "src.cpp", h::kAllCommonFlags & ~h::kPrewarmMs, nullptr),
              ::testing::ExitedWithCode(2), "unknown argument --prewarm-ms");
}

TEST(HarnessCliDeathTest, MissingValueIsAUsageError) {
  Argv a({"--reps"});
  EXPECT_EXIT(h::parse(a.argc(), a.argv(), "src.cpp", h::kReps, nullptr), ::testing::ExitedWithCode(2),
              "missing value for --reps");
}

TEST(HarnessCli, SplitHelpers) {
  EXPECT_EQ(h::split("a,,b,c,"), (std::vector<std::string>{"a", "b", "c"}));
  EXPECT_EQ(h::split_bytes("16K,4M,1G,100"), (std::vector<size_t>{16 << 10, 4 << 20, size_t{1} << 30, 100}));
  EXPECT_EQ(h::split_bytes("2k,3m,1g"), (std::vector<size_t>{2 << 10, 3 << 20, size_t{1} << 30}));
  EXPECT_EQ(h::split_doubles("0,0.1,100"), (std::vector<double>{0, 0.1, 100}));
  EXPECT_EQ(h::bytes_label(16 << 10), "16K");
  EXPECT_EQ(h::bytes_label(64 << 20), "64M");
  EXPECT_EQ(h::bytes_label(size_t{1} << 30), "1G");
  EXPECT_EQ(h::bytes_label(1000), "1000");
  EXPECT_EQ(h::bytes_label((16 << 10) + 1), std::to_string((16 << 10) + 1));
  EXPECT_TRUE(h::wants({"a", "b"}, "b"));
  EXPECT_FALSE(h::wants({"a", "b"}, "c"));
}
