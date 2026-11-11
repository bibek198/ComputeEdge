#pragma once

// Instrumented host memory allocation.
//
// HostAllocator returns blocks aligned to 16 KB (the Apple Silicon VM page size)
// with sizes rounded up to whole pages. This makes every block a valid target for
// MTLDevice newBufferWithBytesNoCopy, so zero-copy GPU access can later be enabled
// as an experimental variable without changing the memory layer.

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace computeedge {

struct Block {
  void* ptr = nullptr;
  size_t bytes = 0;      // actual (page-rounded) size
  size_t requested = 0;  // size the caller asked for
  uint64_t id = 0;       // process-unique buffer id (never 0 for a live block)
};

struct AllocStats {
  static constexpr size_t kBuckets = 48;  // bucket i counts blocks with 2^i <= bytes < 2^(i+1)

  uint64_t alloc_count = 0;
  uint64_t free_count = 0;
  uint64_t reuse_count = 0;  // reserved for the pooling allocator (M7)
  uint64_t current_bytes = 0;
  uint64_t peak_bytes = 0;
  uint64_t total_allocated_bytes = 0;
  std::array<uint64_t, kBuckets> size_histogram{};

  uint64_t live_blocks() const { return alloc_count - free_count; }
};

class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual Block allocate(size_t bytes) = 0;
  virtual void deallocate(const Block& block) = 0;
  virtual AllocStats stats() const = 0;
  virtual void reset_peak() = 0;
  virtual size_t alignment() const = 0;
};

class HostAllocator final : public Allocator {
 public:
  static constexpr size_t kMinAlignment = 16384;

  HostAllocator();

  Block allocate(size_t bytes) override;
  void deallocate(const Block& block) override;
  AllocStats stats() const override;
  void reset_peak() override;
  size_t alignment() const override { return alignment_; }

  // Rounds bytes up to a multiple of alignment() (minimum one page).
  size_t round_up(size_t bytes) const;

 private:
  size_t alignment_;
  mutable std::mutex mutex_;
  AllocStats stats_;
};

// Process-wide allocator used by Tensor unless another one is supplied.
Allocator& default_allocator();

// Histogram bucket for a size in bytes (floor(log2(bytes)), clamped).
size_t size_bucket(size_t bytes);

}  // namespace computeedge
