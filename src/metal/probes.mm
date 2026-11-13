// Metal characterization probes (Objective-C++, ARC). See metal_probes.h.
//
// As in metal_backend.mm, errors raised inside an @autoreleasepool are captured
// as strings and thrown after the pool has been popped.

#include "context.h"

#include <dispatch/dispatch.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include "computeedge/error.h"
#include "computeedge/instrumentation.h"
#include "computeedge/metal_probes.h"

namespace computeedge::metal::probe {

namespace {

using detail::Context;
using detail::context;
using detail::ns_error;
using detail::to_ns;

// newCommandQueue allows 64 command buffers in flight; asking for more before
// committing any would block forever in -commandBuffer.
constexpr uint32_t kMaxCommandBuffers = 64;
constexpr uint64_t kSpinTimeoutNs = 10'000'000'000;  // a failed command buffer never signals

void require_available() {
  if (!available()) throw Error("Metal is not available on this system");
}

id<MTLBuffer> mtl(const Buffer& b) { return (__bridge id<MTLBuffer>)b.native_handle(); }

size_t page_size() { return static_cast<size_t>(getpagesize()); }

// Per-submission synchronization state, prepared on the last command buffer
// before it is committed.
struct SyncState {
  Sync sync = Sync::WaitUntilCompleted;
  uint64_t value = 0;               // event methods
  dispatch_semaphore_t sem = nil;   // completion handler
};

SyncState prepare_sync(id<MTLCommandBuffer> last, Sync sync) {
  SyncState s;
  s.sync = sync;
  Context& ctx = context();
  switch (sync) {
    case Sync::EventWait:
    case Sync::SpinEvent: {
      {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        s.value = ++ctx.event_value;
      }
      [last encodeSignalEvent:ctx.event value:s.value];
      break;
    }
    case Sync::CompletionHandler: {
      dispatch_semaphore_t sem = dispatch_semaphore_create(0);
      s.sem = sem;
      [last addCompletedHandler:^(id<MTLCommandBuffer>) {
        dispatch_semaphore_signal(sem);
      }];
      break;
    }
    case Sync::WaitUntilCompleted:
    case Sync::SpinStatus:
      break;
  }
  return s;
}

// Commits `cbs` in order, synchronizes on the last one, then (untimed) waits for
// every buffer so the GPU timestamps are final. Returns an error message or "".
std::string submit_and_sync(const std::vector<id<MTLCommandBuffer>>& cbs, const SyncState& s, Timing& t) {
  for (size_t i = 0; i < cbs.size(); ++i) {
    const uint64_t c0 = now_ns();
    [cbs[i] commit];
    const uint64_t c1 = now_ns();
    t.commit_ns += c1 - c0;
    if (i == 0) t.commit_end_ns = c1;
  }
  id<MTLCommandBuffer> last = cbs.back();
  Context& ctx = context();
  std::string failure;

  const uint64_t w0 = now_ns();
  switch (s.sync) {
    case Sync::WaitUntilCompleted:
      [last waitUntilCompleted];
      break;
    case Sync::EventWait:
      if (![ctx.event waitUntilSignaledValue:s.value timeoutMS:10000]) failure = "probe: event wait timed out";
      break;
    case Sync::CompletionHandler:
      dispatch_semaphore_wait(s.sem, DISPATCH_TIME_FOREVER);
      break;
    case Sync::SpinEvent:
      while (ctx.event.signaledValue < s.value) {
        if (now_ns() - w0 > kSpinTimeoutNs) {
          failure = "probe: spin on event timed out";
          break;
        }
      }
      break;
    case Sync::SpinStatus:
      while ([last status] < MTLCommandBufferStatusCompleted) {
      }
      break;
  }
  t.wait_end_ns = now_ns();
  t.wait_ns = t.wait_end_ns - w0;

  for (id<MTLCommandBuffer> cb : cbs) {
    [cb waitUntilCompleted];  // no-op once complete; makes timestamps final
    if ([cb status] != MTLCommandBufferStatusCompleted && failure.empty()) {
      failure = "probe: command buffer failed: " + ns_error([cb error]);
    }
  }
  if (failure.empty()) {
    t.sched_start_ns = to_ns([cbs.front() kernelStartTime]);
    t.sched_end_ns = to_ns([last kernelEndTime]);
    t.gpu_start_ns = to_ns([cbs.front() GPUStartTime]);
    t.gpu_end_ns = to_ns([last GPUEndTime]);
  }
  return failure;
}

// Runs a single command buffer with one compute encoder built by `encode`.
template <typename EncodeFn>
Timing run_single(Sync sync, EncodeFn&& encode) {
  Timing t;
  std::string failure;
  @autoreleasepool {
    Context& ctx = context();
    const uint64_t start = now_ns();
    id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    encode(enc);
    [enc endEncoding];
    const SyncState s = prepare_sync(cb, sync);
    t.encode_ns = now_ns() - start;
    failure = submit_and_sync({cb}, s, t);
    t.total_ns = t.wait_end_ns - start;
  }
  if (!failure.empty()) throw Error(failure);
  return t;
}

}  // namespace

const char* to_string(Sync sync) {
  switch (sync) {
    case Sync::WaitUntilCompleted: return "wait_until_completed";
    case Sync::EventWait: return "event_wait";
    case Sync::CompletionHandler: return "completion_handler";
    case Sync::SpinEvent: return "spin_event";
    case Sync::SpinStatus: return "spin_status";
  }
  return "unknown";
}

const char* to_string(BufferKind kind) {
  switch (kind) {
    case BufferKind::Shared: return "shared";
    case BufferKind::Private: return "private";
    case BufferKind::NoCopy: return "nocopy";
    case BufferKind::SharedCopy: return "shared_copy";
  }
  return "unknown";
}

const char* to_string(Stream stream) {
  switch (stream) {
    case Stream::Read: return "read";
    case Stream::Write: return "write";
    case Stream::Copy: return "copy";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Track A: dispatch and synchronization
// ---------------------------------------------------------------------------

Timing empty_dispatch(const DispatchConfig& config) {
  require_available();
  if (config.command_buffers == 0 || config.encoders_per_buffer == 0 || config.dispatches_per_encoder == 0) {
    throw Error("probe::empty_dispatch: counts must be >= 1");
  }
  if (config.command_buffers > kMaxCommandBuffers) {
    throw Error("probe::empty_dispatch: at most 64 command buffers (command queue limit)");
  }
  id<MTLComputePipelineState> pso = detail::pipeline("probes", "probe_empty");

  Timing t;
  std::string failure;
  @autoreleasepool {
    Context& ctx = context();
    const MTLSize one = MTLSizeMake(1, 1, 1);
    const uint64_t start = now_ns();
    std::vector<id<MTLCommandBuffer>> cbs;
    cbs.reserve(config.command_buffers);
    for (uint32_t c = 0; c < config.command_buffers; ++c) {
      id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
      for (uint32_t e = 0; e < config.encoders_per_buffer; ++e) {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        for (uint32_t d = 0; d < config.dispatches_per_encoder; ++d) {
          [enc dispatchThreads:one threadsPerThreadgroup:one];
        }
        [enc endEncoding];
      }
      cbs.push_back(cb);
    }
    const SyncState s = prepare_sync(cbs.back(), config.sync);
    t.encode_ns = now_ns() - start;
    failure = submit_and_sync(cbs, s, t);
    t.total_ns = t.wait_end_ns - start;
  }
  if (!failure.empty()) throw Error(failure);
  return t;
}

// ---------------------------------------------------------------------------
// Track B: buffers
// ---------------------------------------------------------------------------

Buffer Buffer::create(BufferKind kind, size_t bytes, void* host) {
  require_available();
  if (bytes == 0) throw Error("probe::Buffer: size must be > 0");
  const size_t page = page_size();
  std::string failure;
  Buffer out;
  @autoreleasepool {
    id<MTLDevice> device = context().device;
    id<MTLBuffer> buf = nil;
    switch (kind) {
      case BufferKind::Shared:
        buf = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        break;
      case BufferKind::Private:
        buf = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        break;
      case BufferKind::NoCopy: {
        if (host) {
          if (reinterpret_cast<uintptr_t>(host) % page != 0 || bytes % page != 0) {
            failure = "probe::Buffer: NoCopy host memory must be page-aligned with a page-multiple size";
            break;
          }
          buf = [device newBufferWithBytesNoCopy:host length:bytes options:MTLResourceStorageModeShared
                                     deallocator:nil];
        } else {
          const size_t rounded = (bytes + page - 1) / page * page;
          void* mem = nullptr;
          if (posix_memalign(&mem, page, rounded) != 0) {
            failure = "probe::Buffer: host allocation failed";
            break;
          }
          buf = [device newBufferWithBytesNoCopy:mem length:rounded options:MTLResourceStorageModeShared
                                     deallocator:^(void* p, NSUInteger) {
                                       std::free(p);
                                     }];
          if (!buf) std::free(mem);
        }
        break;
      }
      case BufferKind::SharedCopy:
        if (!host) {
          failure = "probe::Buffer: SharedCopy needs host memory to copy from";
          break;
        }
        buf = [device newBufferWithBytes:host length:bytes options:MTLResourceStorageModeShared];
        break;
    }
    if (failure.empty() && !buf) failure = std::string("probe::Buffer: ") + to_string(kind) + " buffer creation failed";
    if (failure.empty()) {
      out.handle_ = (__bridge_retained void*)buf;
      out.kind_ = kind;
      out.bytes_ = bytes;
    }
  }
  if (!failure.empty()) throw Error(failure);
  return out;
}

Buffer::Buffer(Buffer&& other) noexcept : handle_(other.handle_), kind_(other.kind_), bytes_(other.bytes_) {
  other.handle_ = nullptr;
  other.bytes_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    reset();
    handle_ = other.handle_;
    kind_ = other.kind_;
    bytes_ = other.bytes_;
    other.handle_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

Buffer::~Buffer() { reset(); }

void Buffer::reset() {
  if (!handle_) return;
  @autoreleasepool {
    // Balances the __bridge_retained in create(): the MTLBuffer (and a NoCopy
    // allocation's memory) is freed here.
    CFRelease(handle_);
    handle_ = nullptr;
  }
  bytes_ = 0;
}

void* Buffer::contents() const {
  if (!handle_ || kind_ == BufferKind::Private) return nullptr;
  return [mtl(*this) contents];
}

Timing bind_dispatch(const Buffer& buffer, Sync sync) {
  require_available();
  if (!buffer) throw Error("probe::bind_dispatch: empty buffer");
  id<MTLComputePipelineState> pso = detail::pipeline("probes", "probe_bind");
  id<MTLBuffer> buf = mtl(buffer);
  return run_single(sync, [&](id<MTLComputeCommandEncoder> enc) {
    const uint32_t flag = 0;
    [enc setComputePipelineState:pso];
    [enc setBuffer:buf offset:0 atIndex:0];
    [enc setBytes:&flag length:sizeof(flag) atIndex:1];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  });
}

Timing gpu_stream(Stream stream, const Buffer& a, const Buffer* b, Sync sync) {
  require_available();
  if (!a) throw Error("probe::gpu_stream: empty buffer");
  if (a.bytes() % 16 != 0) throw Error("probe::gpu_stream: size must be a multiple of 16 bytes");
  if (stream == Stream::Copy && (!b || !*b || b->bytes() != a.bytes())) {
    throw Error("probe::gpu_stream: Copy needs a destination buffer of the same size");
  }
  const size_t n4_full = a.bytes() / 16;
  if (n4_full > UINT32_MAX) throw Error("probe::gpu_stream: buffer too large");
  const uint32_t n4 = static_cast<uint32_t>(n4_full);

  const char* fn = stream == Stream::Read ? "probe_read" : stream == Stream::Write ? "probe_write" : "probe_copy";
  id<MTLComputePipelineState> pso = detail::pipeline("probes", fn);
  // Grid-stride: enough threads to fill the GPU, each looping over the buffer.
  const NSUInteger threads = std::min<NSUInteger>(n4, NSUInteger{1} << 18);
  const NSUInteger group = std::min<NSUInteger>(256, [pso maxTotalThreadsPerThreadgroup]);
  static id<MTLBuffer> sink = [context().device newBufferWithLength:16 options:MTLResourceStorageModeShared];

  id<MTLBuffer> src = mtl(a);
  id<MTLBuffer> dst = b ? mtl(*b) : nil;
  return run_single(sync, [&](id<MTLComputeCommandEncoder> enc) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:src offset:0 atIndex:0];
    if (stream == Stream::Read) [enc setBuffer:sink offset:0 atIndex:1];
    if (stream == Stream::Copy) [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&n4 length:sizeof(n4) atIndex:2];
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
  });
}

}  // namespace computeedge::metal::probe
