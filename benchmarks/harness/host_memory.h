#pragma once

// Page-aligned host memory that is not a Metal buffer (the allocator's layout),
// for storage-mode and CPU-bandwidth experiments.

#include <cstddef>

namespace computeedge::bench::harness {

// The VM page size (16 KiB on Apple silicon).
size_t page_size();

// Move-only owner of `bytes` of page-aligned memory from posix_memalign or mmap.
// On allocation failure prints "host allocation of N bytes failed" and exits
// with status 1 (benchmark processes cannot continue without the memory).
class HostMemory {
 public:
  enum class Source { Memalign, Mmap };

  explicit HostMemory(size_t bytes, Source source = Source::Memalign);
  HostMemory(HostMemory&& other) noexcept;
  HostMemory& operator=(HostMemory&& other) noexcept;
  HostMemory(const HostMemory&) = delete;
  HostMemory& operator=(const HostMemory&) = delete;
  ~HostMemory();

  void* data() const { return ptr_; }
  size_t bytes() const { return bytes_; }

 private:
  void release();
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
  Source source_ = Source::Memalign;
};

}  // namespace computeedge::bench::harness
