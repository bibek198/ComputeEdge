#include "harness/host_memory.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

namespace computeedge::bench::harness {

size_t page_size() { return static_cast<size_t>(getpagesize()); }

HostMemory::HostMemory(size_t bytes, Source source) : bytes_(bytes), source_(source) {
  if (source == Source::Memalign) {
    if (posix_memalign(&ptr_, page_size(), bytes) != 0) ptr_ = nullptr;
  } else {
    ptr_ = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (ptr_ == MAP_FAILED) ptr_ = nullptr;
  }
  if (!ptr_) {
    std::fprintf(stderr, "host allocation of %zu bytes failed\n", bytes);
    std::exit(1);
  }
}

HostMemory::HostMemory(HostMemory&& other) noexcept : ptr_(other.ptr_), bytes_(other.bytes_), source_(other.source_) {
  other.ptr_ = nullptr;
  other.bytes_ = 0;
}

HostMemory& HostMemory::operator=(HostMemory&& other) noexcept {
  if (this != &other) {
    release();
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    source_ = other.source_;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

HostMemory::~HostMemory() { release(); }

void HostMemory::release() {
  if (!ptr_) return;
  if (source_ == Source::Memalign) {
    std::free(ptr_);
  } else {
    munmap(ptr_, bytes_);
  }
  ptr_ = nullptr;
}

}  // namespace computeedge::bench::harness
