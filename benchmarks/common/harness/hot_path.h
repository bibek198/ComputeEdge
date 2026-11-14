#pragma once

// Helpers that run inside timed regions. Header-only and inline on purpose: they
// were defined in each benchmark's own translation unit before the harness
// existed, and keeping them inlinable keeps the generated code of the measured
// loops unchanged.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "computeedge/core/instrumentation.h"

namespace computeedge::bench::harness {

// Sink for values a measured loop computes, so the compiler cannot remove the loop.
inline volatile uint64_t g_sink = 0;

// Integer sum over `bytes` of memory: vectorises and is bandwidth-bound, not latency-bound.
inline uint64_t read_all(const void* p, size_t bytes) {
  const uint64_t* w = static_cast<const uint64_t*>(p);
  uint64_t acc = 0;
  for (size_t i = 0; i < bytes / sizeof(uint64_t); ++i) acc += w[i];
  return acc;
}

// Busy-waits for `ns` nanoseconds (keeps the CPU clocked up, unlike sleeping).
inline void spin_for(uint64_t ns) {
  const uint64_t until = now_ns() + ns;
  while (now_ns() < until) {
  }
}

// Wall time of one call of fn(), in ns.
inline uint64_t timed_ns(const auto& fn) {
  const uint64_t t0 = now_ns();
  fn();
  return now_ns() - t0;
}

// Median wall time of `runs` calls of fn(), in ns.
template <typename Fn>
uint64_t median_ns(Fn&& fn, int runs = 7) {
  std::vector<uint64_t> v;
  for (int i = 0; i < runs; ++i) {
    const uint64_t t0 = now_ns();
    fn();
    v.push_back(now_ns() - t0);
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace computeedge::bench::harness
