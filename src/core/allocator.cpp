#include "computeedge/core/allocator.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <new>

#include "computeedge/core/instrumentation.h"

namespace computeedge {

namespace {

uint64_t next_buffer_id() {
  static std::atomic<uint64_t> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

size_t system_page_size() {
  const long page = sysconf(_SC_PAGESIZE);
  return page > 0 ? static_cast<size_t>(page) : HostAllocator::kMinAlignment;
}

}  // namespace

size_t size_bucket(size_t bytes) {
  if (bytes == 0) return 0;
  size_t bucket = 0;
  while (bytes >>= 1) ++bucket;
  return std::min(bucket, AllocStats::kBuckets - 1);
}

HostAllocator::HostAllocator() : alignment_(std::max(kMinAlignment, system_page_size())) {}

size_t HostAllocator::round_up(size_t bytes) const {
  if (bytes == 0) return alignment_;
  if (bytes > SIZE_MAX - (alignment_ - 1)) throw std::bad_alloc();
  return (bytes + alignment_ - 1) / alignment_ * alignment_;
}

Block HostAllocator::allocate(size_t bytes) {
  const size_t rounded = round_up(bytes);
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment_, rounded) != 0 || ptr == nullptr) throw std::bad_alloc();

  Block block{ptr, rounded, bytes, next_buffer_id()};
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.alloc_count++;
    stats_.current_bytes += rounded;
    stats_.total_allocated_bytes += rounded;
    stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.current_bytes);
    stats_.size_histogram[size_bucket(rounded)]++;
  }
  emit(EventType::Alloc, block.id, rounded, Backend::CPU, 0, "host");
  return block;
}

void HostAllocator::deallocate(const Block& block) {
  if (block.ptr == nullptr) return;
  emit(EventType::Free, block.id, block.bytes, Backend::CPU, 0, "host");
  std::free(block.ptr);
  std::lock_guard<std::mutex> lock(mutex_);
  stats_.free_count++;
  stats_.current_bytes -= block.bytes;
}

AllocStats HostAllocator::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void HostAllocator::reset_peak() {
  std::lock_guard<std::mutex> lock(mutex_);
  stats_.peak_bytes = stats_.current_bytes;
}

Allocator& default_allocator() {
  static HostAllocator allocator;
  return allocator;
}

}  // namespace computeedge
