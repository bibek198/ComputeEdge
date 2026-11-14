// The GPU pre-warm workloads (audit finding D14) and the two timing key sets are
// part of the measurement contract. These tests pin them to their pre-refactor
// definitions: unifying the workloads or changing a key set must fail here.

#include "harness/gpu.h"

#include <gtest/gtest.h>

#include <set>
#include <string>

namespace h = computeedge::bench::harness;
namespace mtl = computeedge::metal;

namespace {

std::set<std::string> keys(const computeedge::bench::Metrics& m) {
  std::set<std::string> out;
  for (const auto& [k, v] : m) out.insert(k);
  return out;
}

}  // namespace

TEST(HarnessGpu, PrewarmWorkloadsKeepTheirPreRefactorIdentity) {
  // Exactly the strings in docs/verification/trace_pre_refactor/*.trace.
  EXPECT_EQ(h::describe(h::PrewarmWorkload::GpuStreamWritePrivate64MiB),
            "gpu_stream_write kind=private bytes=67108864");
  EXPECT_EQ(h::describe(h::PrewarmWorkload::MetalMatmulShared256), "metal_matmul n=256 storage=shared kernel=tiled16");
}

TEST(HarnessGpu, PrewarmWorkloadsStayDistinct) {
  EXPECT_NE(h::describe(h::PrewarmWorkload::GpuStreamWritePrivate64MiB),
            h::describe(h::PrewarmWorkload::MetalMatmulShared256));
}

TEST(HarnessGpu, ProbeTimingKeys) {
  computeedge::bench::Metrics m;
  h::record_probe_timing(m, mtl::Timing{});
  EXPECT_EQ(keys(m),
            (std::set<std::string>{"encode_ns", "commit_ns", "wait_ns", "total_ns", "gpu_ns", "queue_ns", "wake_ns"}));
}

TEST(HarnessGpu, MatmulTimingKeys) {
  computeedge::bench::Metrics m;
  h::record_matmul_timing(m, mtl::Timing{});
  EXPECT_EQ(keys(m), (std::set<std::string>{"buffer_ns", "copy_in_ns", "encode_ns", "commit_ns", "wait_ns",
                                            "copy_out_ns", "sched_ns", "gpu_ns", "queue_ns", "wake_ns"}));
}

TEST(HarnessGpu, PrewarmWithZeroDurationRunsNothing) {
  if (!mtl::available()) GTEST_SKIP() << "Metal not available";
  h::Reporter silent;
  h::prewarm(h::PrewarmWorkload::GpuStreamWritePrivate64MiB, 0, silent);
  h::prewarm(h::PrewarmWorkload::MetalMatmulShared256, 0, silent);
  EXPECT_TRUE(silent.results().empty());
}
