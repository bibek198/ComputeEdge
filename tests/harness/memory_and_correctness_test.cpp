// Tests for the harness host-memory, correctness and hot-path helpers.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <utility>

#include "harness/correctness.h"
#include "harness/host_memory.h"
#include "harness/hot_path.h"

namespace h = computeedge::bench::harness;

TEST(HarnessHostMemory, PageAlignedForBothSources) {
  for (auto source : {h::HostMemory::Source::Memalign, h::HostMemory::Source::Mmap}) {
    h::HostMemory m(4 * h::page_size(), source);
    ASSERT_NE(m.data(), nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(m.data()) % h::page_size(), 0u);
    std::memset(m.data(), 7, m.bytes());
    EXPECT_EQ(static_cast<unsigned char*>(m.data())[m.bytes() - 1], 7);
  }
}

TEST(HarnessHostMemory, MoveTransfersOwnership) {
  h::HostMemory a(h::page_size());
  void* p = a.data();
  h::HostMemory b = std::move(a);
  EXPECT_EQ(b.data(), p);
  EXPECT_EQ(a.data(), nullptr);
  EXPECT_EQ(a.bytes(), 0u);
}

TEST(HarnessCorrectness, RandomSquareIsDeterministicAndBounded) {
  const auto a = h::random_square(16, 1).to_vector();
  const auto b = h::random_square(16, 1).to_vector();
  EXPECT_EQ(a, b);
  for (float x : a) {
    EXPECT_GE(x, -1.0f);
    EXPECT_LT(x, 1.0f);
  }
  EXPECT_NE(h::random_square(16, 2).to_vector(), a);
}

TEST(HarnessCorrectness, MaxScaledError) {
  const computeedge::Tensor ref({1, 2}, {1.0f, -3.0f});
  EXPECT_EQ(h::max_scaled_error(ref, ref, 4), 0.0);
  const computeedge::Tensor got({1, 2}, {1.5f, -3.0f});
  // |1.5 - 1| / (1 + 1) = 0.25, scaled by 1/sqrt(4).
  EXPECT_DOUBLE_EQ(h::max_scaled_error(got, ref, 4), 0.125);
  EXPECT_DOUBLE_EQ(h::kCorrectnessTolerance, 1e-5);
}

TEST(HarnessHotPath, ReadAllSumsWords) {
  const uint64_t words[4] = {1, 2, 3, 4};
  EXPECT_EQ(h::read_all(words, sizeof(words)), 10u);
  EXPECT_EQ(h::read_all(words, 0), 0u);
}

TEST(HarnessHotPath, TimersAreMonotonic) {
  int calls = 0;
  EXPECT_GE(h::median_ns([&] { ++calls; }, 5), 0u);
  EXPECT_EQ(calls, 5);
  const uint64_t spun = h::timed_ns([] { h::spin_for(200'000); });
  EXPECT_GE(spun, 200'000u);
}
