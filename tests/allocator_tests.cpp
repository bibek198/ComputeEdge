#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "computeedge/allocator.h"
#include "computeedge/tensor.h"

using namespace computeedge;

TEST(Allocator, AlignmentIsAtLeast16KB) {
  HostAllocator alloc;
  EXPECT_GE(alloc.alignment(), 16384u);
  EXPECT_EQ(alloc.alignment() % 16384u, 0u);
}

TEST(Allocator, BlocksArePageAlignedAndRounded) {
  HostAllocator alloc;
  for (size_t req : {size_t{1}, size_t{100}, size_t{16384}, size_t{16385}, size_t{1} << 20}) {
    Block b = alloc.allocate(req);
    ASSERT_NE(b.ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(b.ptr) % alloc.alignment(), 0u) << "req=" << req;
    EXPECT_EQ(b.bytes % alloc.alignment(), 0u);
    EXPECT_GE(b.bytes, req);
    EXPECT_LT(b.bytes - req, alloc.alignment());
    EXPECT_EQ(b.requested, req);
    std::memset(b.ptr, 0xAB, b.bytes);  // whole rounded block must be writable
    alloc.deallocate(b);
  }
}

TEST(Allocator, ZeroByteRequestGetsOnePage) {
  HostAllocator alloc;
  Block b = alloc.allocate(0);
  EXPECT_NE(b.ptr, nullptr);
  EXPECT_EQ(b.bytes, alloc.alignment());
  alloc.deallocate(b);
}

TEST(Allocator, BufferIdsAreUnique) {
  HostAllocator alloc;
  Block a = alloc.allocate(10);
  Block b = alloc.allocate(10);
  EXPECT_NE(a.id, 0u);
  EXPECT_NE(a.id, b.id);
  alloc.deallocate(a);
  alloc.deallocate(b);
}

TEST(Allocator, StatsTrackCurrentAndPeak) {
  HostAllocator alloc;
  const size_t page = alloc.alignment();
  Block a = alloc.allocate(page);
  Block b = alloc.allocate(3 * page);
  AllocStats s = alloc.stats();
  EXPECT_EQ(s.alloc_count, 2u);
  EXPECT_EQ(s.current_bytes, 4 * page);
  EXPECT_EQ(s.peak_bytes, 4 * page);
  EXPECT_EQ(s.live_blocks(), 2u);

  alloc.deallocate(b);
  s = alloc.stats();
  EXPECT_EQ(s.free_count, 1u);
  EXPECT_EQ(s.current_bytes, page);
  EXPECT_EQ(s.peak_bytes, 4 * page);

  alloc.reset_peak();
  EXPECT_EQ(alloc.stats().peak_bytes, page);
  alloc.deallocate(a);
  EXPECT_EQ(alloc.stats().current_bytes, 0u);
  EXPECT_EQ(alloc.stats().total_allocated_bytes, 4 * page);
}

TEST(Allocator, SizeHistogram) {
  EXPECT_EQ(size_bucket(1), 0u);
  EXPECT_EQ(size_bucket(16384), 14u);
  EXPECT_EQ(size_bucket(16385), 14u);
  EXPECT_EQ(size_bucket(32768), 15u);

  HostAllocator alloc;
  Block b = alloc.allocate(1);  // rounds to one page
  EXPECT_EQ(alloc.stats().size_histogram[size_bucket(alloc.alignment())], 1u);
  alloc.deallocate(b);
}

TEST(Allocator, TensorLifetimeFollowsLastView) {
  HostAllocator alloc;
  {
    Tensor t({4, 4}, alloc);
    EXPECT_EQ(alloc.stats().live_blocks(), 1u);
    Tensor view = t.transpose();
    Tensor reshaped = t.reshape({16});
    EXPECT_EQ(alloc.stats().alloc_count, 1u) << "views must not allocate";
    t = Tensor();
    EXPECT_EQ(alloc.stats().live_blocks(), 1u) << "storage alive while views exist";
  }
  EXPECT_EQ(alloc.stats().live_blocks(), 0u);
  EXPECT_EQ(alloc.stats().current_bytes, 0u);
}

TEST(Allocator, TensorStorageIsPageAligned) {
  Tensor t({3, 5});
  EXPECT_EQ(reinterpret_cast<uintptr_t>(t.storage()->data()) % 16384u, 0u);
  EXPECT_EQ(t.storage()->bytes() % 16384u, 0u);
  EXPECT_EQ(t.storage()->requested_bytes(), 15 * sizeof(float));
}
