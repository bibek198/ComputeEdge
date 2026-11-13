#pragma once

// Characterization probes for the Metal runtime (M5, Tracks A and B).
//
// Each probe isolates one cost: dispatch and command-buffer overhead, the
// synchronization mechanism, buffer creation per storage mode, buffer residency
// on first GPU use, and streaming bandwidth. Results use metal::Timing, so the
// CPU-side phases and GPU timestamps mean exactly what they mean for matmul.
// Pure C++ interface: no Objective-C types are exposed.

#include <cstddef>
#include <cstdint>

#include "computeedge/metal.h"

namespace computeedge::metal::probe {

// How the CPU learns that the GPU finished (Track A).
enum class Sync : uint8_t {
  WaitUntilCompleted,  // [MTLCommandBuffer waitUntilCompleted] (what matmul uses)
  EventWait,           // GPU signals an MTLSharedEvent; CPU blocks in waitUntilSignaledValue:timeoutMS:
  CompletionHandler,   // addCompletedHandler signals a dispatch semaphore the CPU waits on
  SpinEvent,           // CPU busy-polls MTLSharedEvent.signaledValue (burns a core; lower bound)
  SpinStatus,          // CPU busy-polls MTLCommandBuffer.status
};

const char* to_string(Sync sync);

// Shape of one empty-kernel submission (Track A). All command buffers are encoded
// first, then committed back to back; the CPU waits only for the last one.
struct DispatchConfig {
  uint32_t command_buffers = 1;
  uint32_t encoders_per_buffer = 1;
  uint32_t dispatches_per_encoder = 1;
  Sync sync = Sync::WaitUntilCompleted;
};

// Runs empty kernels (one thread each) as configured. In the returned Timing:
//   encode_ns  creating + encoding every command buffer
//   commit_ns  sum of the commit() calls; commit_end_ns = first commit returned
//   wait_ns    the synchronization method alone (after the last commit)
//   gpu_start_ns / gpu_end_ns = first buffer's GPUStartTime / last buffer's GPUEndTime
// so queue_ns() is the first buffer's queueing delay and wake_ns() the latency
// from the end of GPU work to the CPU resuming. With the event methods the event
// is signalled from inside the last buffer, so wake_ns() can be slightly negative.
Timing empty_dispatch(const DispatchConfig& config);

// How a buffer's memory is obtained (Track B).
enum class BufferKind : uint8_t {
  Shared,      // newBufferWithLength, MTLResourceStorageModeShared
  Private,     // newBufferWithLength, MTLResourceStorageModePrivate (no CPU access)
  NoCopy,      // newBufferWithBytesNoCopy over page-aligned host memory
  SharedCopy,  // newBufferWithBytes: Shared buffer initialised by copying host memory
};

const char* to_string(BufferKind kind);

// Owning handle to one MTLBuffer.
class Buffer {
 public:
  // `bytes` must be > 0. NoCopy wraps `host` if given (page-aligned, size a
  // multiple of the page size, not owned); with host == nullptr it allocates
  // page-aligned memory that is freed with the buffer. SharedCopy copies from
  // `host`, which is required.
  static Buffer create(BufferKind kind, size_t bytes, void* host = nullptr);

  Buffer() = default;
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer();

  void reset();  // releases the MTLBuffer now
  explicit operator bool() const { return handle_ != nullptr; }
  BufferKind kind() const { return kind_; }
  size_t bytes() const { return bytes_; }
  void* contents() const;  // nullptr for Private
  // The underlying MTLBuffer object as an opaque, non-owning pointer (for the backend).
  void* native_handle() const { return handle_; }

 private:
  void* handle_ = nullptr;  // retained MTLBuffer object
  BufferKind kind_ = BufferKind::Shared;
  size_t bytes_ = 0;
};

// Dispatches probe_bind (one thread) with `buffer` bound but untouched: any cost
// that grows with the buffer's size is residency / mapping, not traffic.
Timing bind_dispatch(const Buffer& buffer, Sync sync = Sync::WaitUntilCompleted);

enum class Stream : uint8_t { Read, Write, Copy };
const char* to_string(Stream stream);

// Streams all of `a` on the GPU (Read or Write), or copies `a` into `b` (Copy;
// same size). Sizes must be multiples of 16 bytes. Bandwidth = bytes moved / gpu_ns().
Timing gpu_stream(Stream stream, const Buffer& a, const Buffer* b = nullptr, Sync sync = Sync::WaitUntilCompleted);

}  // namespace computeedge::metal::probe
