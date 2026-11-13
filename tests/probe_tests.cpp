// Tests for the Metal characterization probes (M5): every synchronization
// method completes with ordered timestamps, buffers of every kind behave as
// documented, and the streaming kernels move the data they claim to.

#include <gtest/gtest.h>

#include <mach/mach.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include "computeedge/error.h"
#include "computeedge/instrumentation.h"
#include "computeedge/metal_probes.h"

using namespace computeedge;
namespace mtl = computeedge::metal;
namespace probe = computeedge::metal::probe;

namespace {

class ProbeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!mtl::available()) GTEST_SKIP() << "Metal not available";
  }
};

class ProbeSync : public ::testing::TestWithParam<probe::Sync> {
 protected:
  void SetUp() override {
    if (!mtl::available()) GTEST_SKIP() << "Metal not available";
  }
};

void expect_ordered(const mtl::Timing& t, uint64_t before, uint64_t after) {
  EXPECT_GT(t.total_ns, 0u);
  EXPECT_GT(t.gpu_start_ns, before);
  EXPECT_LE(t.gpu_start_ns, t.gpu_end_ns);
  EXPECT_GE(t.commit_end_ns, before);
  EXPECT_GE(t.wait_end_ns, t.commit_end_ns);
  EXPECT_LE(t.wait_end_ns, after);
  EXPECT_LE(t.encode_ns + t.commit_ns + t.wait_ns, t.total_ns);
}

uint64_t phys_footprint() {
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) return 0;
  return info.phys_footprint;
}

struct HostPages {
  explicit HostPages(size_t bytes) : bytes(bytes) {
    if (posix_memalign(&ptr, static_cast<size_t>(getpagesize()), bytes) != 0) ptr = nullptr;
  }
  ~HostPages() { std::free(ptr); }
  void* ptr = nullptr;
  size_t bytes;
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(All, ProbeSync,
                         ::testing::Values(probe::Sync::WaitUntilCompleted, probe::Sync::EventWait,
                                           probe::Sync::CompletionHandler, probe::Sync::SpinEvent,
                                           probe::Sync::SpinStatus),
                         [](const auto& info) { return std::string(probe::to_string(info.param)); });

TEST_P(ProbeSync, EmptyDispatchCompletesWithOrderedTimestamps) {
  for (int i = 0; i < 3; ++i) {  // the first call may compile the probe library
    const uint64_t before = now_ns();
    const mtl::Timing t = probe::empty_dispatch({1, 1, 1, GetParam()});
    const uint64_t after = now_ns();
    expect_ordered(t, before, after);
  }
}

TEST_P(ProbeSync, BatchedSubmission) {
  const uint64_t before = now_ns();
  const mtl::Timing t = probe::empty_dispatch({4, 2, 8, GetParam()});
  expect_ordered(t, before, now_ns());
}

TEST_P(ProbeSync, BindDispatch) {
  probe::Buffer buf = probe::Buffer::create(probe::BufferKind::Shared, 1 << 16);
  const uint64_t before = now_ns();
  expect_ordered(probe::bind_dispatch(buf, GetParam()), before, now_ns());
}

TEST_F(ProbeTest, EventValuesAdvanceAcrossCalls) {
  // Two event-based calls in a row must each wait for their own signal.
  probe::empty_dispatch({1, 1, 1, probe::Sync::EventWait});
  const mtl::Timing t = probe::empty_dispatch({1, 1, 1, probe::Sync::SpinEvent});
  EXPECT_GT(t.gpu_end_ns, 0u);
}

TEST_F(ProbeTest, RejectsBadDispatchConfig) {
  EXPECT_THROW(probe::empty_dispatch({0, 1, 1}), Error);
  EXPECT_THROW(probe::empty_dispatch({1, 0, 1}), Error);
  EXPECT_THROW(probe::empty_dispatch({1, 1, 0}), Error);
  EXPECT_THROW(probe::empty_dispatch({65, 1, 1}), Error);
  EXPECT_NO_THROW(probe::empty_dispatch({64, 1, 1}));
}

TEST_F(ProbeTest, BufferKinds) {
  const size_t page = static_cast<size_t>(getpagesize());
  probe::Buffer shared = probe::Buffer::create(probe::BufferKind::Shared, 4 * page);
  EXPECT_TRUE(shared);
  EXPECT_EQ(shared.bytes(), 4 * page);
  EXPECT_NE(shared.contents(), nullptr);

  probe::Buffer wc = probe::Buffer::create(probe::BufferKind::SharedWriteCombined, 4 * page);
  EXPECT_NE(wc.contents(), nullptr);
  std::memset(wc.contents(), 7, 4 * page);
  EXPECT_EQ(static_cast<unsigned char*>(wc.contents())[4 * page - 1], 7);

  probe::Buffer priv = probe::Buffer::create(probe::BufferKind::Private, 4 * page);
  EXPECT_EQ(priv.contents(), nullptr) << "Private buffers have no CPU mapping";

  HostPages host(4 * page);
  ASSERT_NE(host.ptr, nullptr);
  probe::Buffer wrapped = probe::Buffer::create(probe::BufferKind::NoCopy, host.bytes, host.ptr);
  EXPECT_EQ(wrapped.contents(), host.ptr) << "NoCopy must alias the host memory";

  probe::Buffer owned = probe::Buffer::create(probe::BufferKind::NoCopy, 1000);  // rounded up internally
  EXPECT_NE(owned.contents(), nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(owned.contents()) % page, 0u);

  std::memset(host.ptr, 0x5A, host.bytes);
  probe::Buffer copied = probe::Buffer::create(probe::BufferKind::SharedCopy, host.bytes, host.ptr);
  EXPECT_NE(copied.contents(), host.ptr);
  EXPECT_EQ(std::memcmp(copied.contents(), host.ptr, host.bytes), 0);
}

TEST_F(ProbeTest, BufferErrors) {
  const size_t page = static_cast<size_t>(getpagesize());
  EXPECT_THROW(probe::Buffer::create(probe::BufferKind::Shared, 0), Error);
  EXPECT_THROW(probe::Buffer::create(probe::BufferKind::SharedCopy, page), Error);
  HostPages host(2 * page);
  EXPECT_THROW(probe::Buffer::create(probe::BufferKind::NoCopy, page, static_cast<char*>(host.ptr) + 16), Error);
  EXPECT_THROW(probe::Buffer::create(probe::BufferKind::NoCopy, page + 16, host.ptr), Error);
}

TEST_F(ProbeTest, BufferMoveAndReset) {
  probe::Buffer a = probe::Buffer::create(probe::BufferKind::Shared, 1 << 14);
  void* p = a.contents();
  probe::Buffer b = std::move(a);
  EXPECT_FALSE(a);
  EXPECT_EQ(b.contents(), p);
  b.reset();
  EXPECT_FALSE(b);
  EXPECT_EQ(b.bytes(), 0u);
  b.reset();  // idempotent
}

TEST_F(ProbeTest, ReleasedBuffersFreeTheirMemory) {
  // Regression: contents() once ran outside an autorelease pool, so every
  // touched buffer stayed alive after reset() (C++ callers have no pool).
  constexpr size_t kBytes = size_t{64} << 20;
  for (probe::BufferKind kind : {probe::BufferKind::Shared, probe::BufferKind::NoCopy}) {
    const uint64_t before = phys_footprint();
    for (int i = 0; i < 16; ++i) {  // 1 GiB in total if nothing were freed
      probe::Buffer buf = probe::Buffer::create(kind, kBytes);
      std::memset(buf.contents(), 1, kBytes);
    }
    EXPECT_LT(phys_footprint(), before + 4 * kBytes) << probe::to_string(kind);
  }
}

TEST_F(ProbeTest, WriteStreamFillsCpuVisibleBuffers) {
  for (probe::BufferKind kind : {probe::BufferKind::Shared, probe::BufferKind::NoCopy}) {
    probe::Buffer buf = probe::Buffer::create(kind, 1 << 20);
    std::memset(buf.contents(), 0, buf.bytes());
    const mtl::Timing t = probe::gpu_stream(probe::Stream::Write, buf);
    EXPECT_GT(t.gpu_ns(), 0u);
    const float* f = static_cast<const float*>(buf.contents());
    for (size_t i = 0; i < buf.bytes() / sizeof(float); ++i)
      ASSERT_EQ(f[i], 1.0f) << probe::to_string(kind) << " at " << i;
  }
}

TEST_F(ProbeTest, CopyRoundTripsThroughPrivate) {
  const size_t bytes = (1 << 20) + 48;  // not a multiple of the grid: exercises the stride tail
  probe::Buffer src = probe::Buffer::create(probe::BufferKind::Shared, bytes);
  probe::Buffer mid = probe::Buffer::create(probe::BufferKind::Private, bytes);
  probe::Buffer dst = probe::Buffer::create(probe::BufferKind::Shared, bytes);
  float* s = static_cast<float*>(src.contents());
  for (size_t i = 0; i < bytes / sizeof(float); ++i) s[i] = float(i % 1000) * 0.5f;
  std::memset(dst.contents(), 0, bytes);
  probe::gpu_stream(probe::Stream::Copy, src, &mid);
  probe::gpu_stream(probe::Stream::Copy, mid, &dst);
  EXPECT_EQ(std::memcmp(src.contents(), dst.contents(), bytes), 0);
  EXPECT_GT(probe::gpu_stream(probe::Stream::Read, mid).gpu_ns(), 0u);
}

TEST_F(ProbeTest, StreamErrors) {
  probe::Buffer a = probe::Buffer::create(probe::BufferKind::Shared, 1 << 12);
  probe::Buffer odd = probe::Buffer::create(probe::BufferKind::Shared, 100);
  probe::Buffer small = probe::Buffer::create(probe::BufferKind::Shared, 1 << 11);
  EXPECT_THROW(probe::gpu_stream(probe::Stream::Write, odd), Error);
  EXPECT_THROW(probe::gpu_stream(probe::Stream::Copy, a), Error);
  EXPECT_THROW(probe::gpu_stream(probe::Stream::Copy, a, &small), Error);
  probe::Buffer empty;
  EXPECT_THROW(probe::gpu_stream(probe::Stream::Read, empty), Error);
  EXPECT_THROW(probe::bind_dispatch(empty), Error);
}

TEST_F(ProbeTest, AsyncStreamCompletesAndReportsTiming) {
  probe::Buffer buf = probe::Buffer::create(probe::BufferKind::Shared, 1 << 22);
  std::memset(buf.contents(), 0, buf.bytes());
  const uint64_t before = now_ns();
  probe::Submission sub = probe::gpu_stream_async(probe::Stream::Write, buf, nullptr, 3);
  const mtl::Timing t = sub.wait();
  EXPECT_TRUE(sub.done());
  EXPECT_GT(t.gpu_start_ns, before);
  EXPECT_LE(t.gpu_start_ns, t.gpu_end_ns);
  EXPECT_LE(t.gpu_end_ns, t.wait_end_ns);
  EXPECT_LE(t.encode_ns + t.commit_ns + t.wait_ns, t.total_ns);
  const float* f = static_cast<const float*>(buf.contents());
  for (size_t i = 0; i < buf.bytes() / sizeof(float); i += 997) ASSERT_EQ(f[i], 1.0f);
  EXPECT_EQ(sub.wait().wait_end_ns, t.wait_end_ns) << "second wait returns the same timing";
}

TEST_F(ProbeTest, AsyncStreamOverlapsWithCpu) {
  probe::Buffer buf = probe::Buffer::create(probe::BufferKind::Private, size_t{64} << 20);
  probe::Submission sub = probe::gpu_stream_async(probe::Stream::Write, buf, nullptr, 8);
  const uint64_t cpu_mark = now_ns();  // CPU is free while the GPU works
  const mtl::Timing t = sub.wait();
  EXPECT_LT(cpu_mark, t.gpu_end_ns) << "commit must return before the GPU finishes";
}

TEST_F(ProbeTest, AsyncSubmissionWaitsOnDestructionAndMove) {
  probe::Buffer buf = probe::Buffer::create(probe::BufferKind::Shared, 1 << 20);
  std::memset(buf.contents(), 0, buf.bytes());
  {
    probe::Submission a = probe::gpu_stream_async(probe::Stream::Write, buf);
    probe::Submission b = std::move(a);
    EXPECT_TRUE(a.done());
  }  // b's destructor waits
  EXPECT_EQ(static_cast<const float*>(buf.contents())[buf.bytes() / sizeof(float) - 1], 1.0f);
  EXPECT_THROW(probe::gpu_stream_async(probe::Stream::Write, buf, nullptr, 0), Error);
}
