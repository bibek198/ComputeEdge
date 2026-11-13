// Metal backend (Objective-C++, ARC).
//
// Objective-C exceptions and C++ exceptions do not mix well across
// @autoreleasepool boundaries, so errors raised inside a pool are captured as
// strings and thrown after the pool has been popped.

#include "context.h"

#include <unistd.h>

#include <climits>
#include <cstring>
#include <string>

#include "computeedge/core/error.h"
#include "computeedge/core/instrumentation.h"
#include "computeedge/core/validate.h"
#include "computeedge/cpu/matmul.h"
#include "computeedge/kernel_sources.h"  // generated
#include "computeedge/metal/matmul.h"

namespace computeedge::metal {

namespace detail {

Context& context() {
  static Context* ctx = [] {
    auto* c = new Context;  // intentionally leaked: lives for the process
    @autoreleasepool {
      c->device = MTLCreateSystemDefaultDevice();
      if (c->device) {
        c->queue = [c->device newCommandQueue];
        c->event = [c->device newSharedEvent];
      }
    }
    return c;
  }();
  return *ctx;
}

std::string ns_error(NSError* error) {
  return error ? std::string([[error localizedDescription] UTF8String]) : std::string("unknown error");
}

namespace {

const char* library_source(const std::string& name) {
  if (name == "matmul") return kMatmulSource;
  if (name == "probes") return kProbesSource;
  throw Error("unknown Metal library: " + name);
}

// Compiles the named library on first use. Caller holds ctx.mutex.
id<MTLLibrary> library_locked(Context& ctx, const std::string& name) {
  if (auto it = ctx.libraries.find(name); it != ctx.libraries.end()) return it->second;
  const char* src = library_source(name);
  std::string failure;
  id<MTLLibrary> lib = nil;
  const uint64_t t0 = now_ns();
  @autoreleasepool {
    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:src];
    lib = [ctx.device newLibraryWithSource:source options:[MTLCompileOptions new] error:&error];
    if (!lib) failure = "Metal shader compilation failed (" + name + "): " + ns_error(error);
  }
  ctx.compile_ns += now_ns() - t0;
  if (!failure.empty()) throw Error(failure);
  ctx.libraries.emplace(name, lib);
  return lib;
}

}  // namespace

id<MTLComputePipelineState> pipeline(const std::string& library, const std::string& function) {
  Context& ctx = context();
  std::lock_guard<std::mutex> lock(ctx.mutex);
  const std::string key = library + "/" + function;
  if (auto it = ctx.pipelines.find(key); it != ctx.pipelines.end()) return it->second;
  id<MTLLibrary> lib = library_locked(ctx, library);

  std::string failure;
  id<MTLComputePipelineState> pso = nil;
  const uint64_t t0 = now_ns();
  @autoreleasepool {
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:function.c_str()]];
    if (!fn) {
      failure = "Metal function not found: " + key;
    } else {
      NSError* error = nil;
      pso = [ctx.device newComputePipelineStateWithFunction:fn error:&error];
      if (!pso) failure = "Metal pipeline creation failed for " + key + ": " + ns_error(error);
    }
  }
  ctx.compile_ns += now_ns() - t0;
  if (!failure.empty()) throw Error(failure);
  ctx.pipelines.emplace(key, pso);
  return pso;
}

}  // namespace detail

namespace {

using detail::Context;
using detail::context;
using detail::ns_error;
using detail::to_ns;

constexpr NSUInteger kTile = 16;

struct MatmulParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
};

uint32_t checked_u32(size_t v, const char* what) {
  if (v > UINT32_MAX) throw Error(std::string("metal::matmul: ") + what + " exceeds UINT32_MAX");
  return static_cast<uint32_t>(v);
}

bool page_aligned(const Storage& s) {
  const size_t page = static_cast<size_t>(getpagesize());
  return reinterpret_cast<uintptr_t>(s.data()) % page == 0 && s.bytes() % page == 0;
}

// Emits an event at an explicit timestamp (used for GPU-side timestamps).
void emit_at(uint64_t t_ns, EventType type, uint64_t exec, const char* label) {
  EventLog& log = EventLog::global();
  if (!log.enabled() || t_ns == 0) return;
  log.record(Event{t_ns, type, exec, exec, 0, Backend::Metal, label});
}

MPSMatrixMultiplication* mps_kernel(size_t M, size_t N, size_t K) {
  Context& ctx = context();
  std::lock_guard<std::mutex> lock(ctx.mutex);
  const auto key = std::make_tuple(M, N, K);
  if (auto it = ctx.mps_kernels.find(key); it != ctx.mps_kernels.end()) return it->second;
  MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:ctx.device
                                                                     resultRows:M
                                                                  resultColumns:N
                                                                interiorColumns:K];
  ctx.mps_kernels.emplace(key, mm);
  return mm;
}

MPSMatrix* mps_matrix(id<MTLBuffer> buffer, NSUInteger offset, size_t rows, size_t cols) {
  MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                    columns:cols
                                                                   rowBytes:cols * sizeof(float)
                                                                   dataType:MPSDataTypeFloat32];
  return [[MPSMatrix alloc] initWithBuffer:buffer offset:offset descriptor:desc];
}

}  // namespace

const char* to_string(Kernel kernel) {
  switch (kernel) {
    case Kernel::Tiled16: return "tiled16";
    case Kernel::MPS: return "mps";
  }
  return "unknown";
}

const char* to_string(StorageMode mode) {
  switch (mode) {
    case StorageMode::NoCopy: return "nocopy";
    case StorageMode::Shared: return "shared";
    case StorageMode::Private: return "private";
  }
  return "unknown";
}

bool available() { return context().device != nil && context().queue != nil; }

DeviceInfo device_info() {
  DeviceInfo info;
  Context& ctx = context();
  if (!ctx.device) return info;
  @autoreleasepool {
    info.name = [[ctx.device name] UTF8String];
    info.has_unified_memory = [ctx.device hasUnifiedMemory];
    info.recommended_max_working_set_bytes = [ctx.device recommendedMaxWorkingSetSize];
    info.max_buffer_length = [ctx.device maxBufferLength];
  }
  return info;
}

uint64_t warmup() {
  if (!available()) throw Error("Metal is not available on this system");
  const uint64_t before = compile_ns();
  detail::pipeline("matmul", "matmul_tiled");
  return compile_ns() - before;
}

uint64_t compile_ns() {
  Context& ctx = context();
  std::lock_guard<std::mutex> lock(ctx.mutex);
  return ctx.compile_ns;
}

void matmul_into(const Tensor& a, const Tensor& b, Tensor& out, const MatmulOptions& options, Timing* timing) {
  validate_matmul_out(a, b, out);
  if (!available()) throw Error("Metal is not available on this system");

  Timing t;
  const uint64_t start = now_ns();
  const size_t M = a.shape()[0];
  const size_t K = a.shape()[1];
  const size_t N = b.shape()[1];
  const uint64_t exec = next_exec_id();
  emit(EventType::Dispatch, exec, out.nbytes(), Backend::Metal, exec,
       options.kernel == Kernel::MPS ? "matmul_metal_mps" : "matmul_metal");

  if (M == 0 || N == 0) {
    t.total_ns = now_ns() - start;
    if (timing) *timing = t;
    return;
  }
  const MatmulParams params{checked_u32(M, "M"), checked_u32(N, "N"), checked_u32(K, "K")};
  if (options.storage == StorageMode::NoCopy && !page_aligned(*out.storage())) {
    throw Error("metal::matmul: NoCopy needs page-aligned output storage; use StorageMode::Shared");
  }

  // K == 0 is an all-zeros result; MPS rejects zero-sized matrices, so the
  // tiled kernel handles that case for both choices.
  const bool use_mps = options.kernel == Kernel::MPS && K > 0;
  id<MTLComputePipelineState> pso = detail::pipeline("matmul", "matmul_tiled");  // may compile (cold start)
  MPSMatrixMultiplication* mps = use_mps ? mps_kernel(M, N, K) : nil;
  Context& ctx = context();

  // Inputs must be contiguous row-major; copies (if any) count as copy_in.
  uint64_t t0 = now_ns();
  const Tensor ac = a.contiguous();
  const Tensor bc = b.contiguous();
  t.copy_in_ns += now_ns() - t0;
  if (options.storage == StorageMode::NoCopy && (!page_aligned(*ac.storage()) || !page_aligned(*bc.storage()))) {
    throw Error("metal::matmul: NoCopy needs page-aligned input storage; use StorageMode::Shared");
  }

  const size_t bytes_a = M * K * sizeof(float);
  const size_t bytes_b = K * N * sizeof(float);
  const size_t bytes_c = M * N * sizeof(float);
  std::string failure;

  @autoreleasepool {
    id<MTLDevice> device = ctx.device;
    // Buffers bound to the kernel (gpu_*) and, for copy modes, host-visible staging (host_*).
    id<MTLBuffer> gpu_a = nil, gpu_b = nil, gpu_c = nil;
    id<MTLBuffer> host_a = nil, host_b = nil, host_c = nil;
    NSUInteger off_a = 0, off_b = 0, off_c = 0;

    // ---- buffers -----------------------------------------------------------
    t0 = now_ns();
    switch (options.storage) {
      case StorageMode::NoCopy: {
        auto wrap = [&](const Storage& s) {
          return [device newBufferWithBytesNoCopy:s.data()
                                           length:s.bytes()
                                          options:MTLResourceStorageModeShared
                                      deallocator:nil];
        };
        gpu_a = wrap(*ac.storage());
        gpu_b = wrap(*bc.storage());
        gpu_c = wrap(*out.storage());
        off_a = ac.offset() * sizeof(float);
        off_b = bc.offset() * sizeof(float);
        off_c = out.offset() * sizeof(float);
        break;
      }
      case StorageMode::Shared:
      case StorageMode::Private: {
        host_a = [device newBufferWithLength:bytes_a options:MTLResourceStorageModeShared];
        host_b = [device newBufferWithLength:bytes_b options:MTLResourceStorageModeShared];
        host_c = [device newBufferWithLength:bytes_c options:MTLResourceStorageModeShared];
        if (options.storage == StorageMode::Shared) {
          gpu_a = host_a;
          gpu_b = host_b;
          gpu_c = host_c;
        } else {
          gpu_a = [device newBufferWithLength:bytes_a options:MTLResourceStorageModePrivate];
          gpu_b = [device newBufferWithLength:bytes_b options:MTLResourceStorageModePrivate];
          gpu_c = [device newBufferWithLength:bytes_c options:MTLResourceStorageModePrivate];
        }
        break;
      }
    }
    t.buffer_ns = now_ns() - t0;
    // K == 0 gives zero-length A/B: bind any valid buffer, the kernel never reads it.
    if (K == 0) {
      if (!gpu_a) gpu_a = gpu_c;
      if (!gpu_b) gpu_b = gpu_c;
      if (!host_a) host_a = host_c;
      if (!host_b) host_b = host_c;
    }

    if (!gpu_a || !gpu_b || !gpu_c) {
      failure = "metal::matmul: buffer creation failed";
    } else {
      // ---- copy in -----------------------------------------------------------
      if (options.storage != StorageMode::NoCopy) {
        t0 = now_ns();
        if (bytes_a) std::memcpy([host_a contents], ac.data(), bytes_a);
        if (bytes_b) std::memcpy([host_b contents], bc.data(), bytes_b);
        t.copy_in_ns += now_ns() - t0;
      }

      // ---- encode ------------------------------------------------------------
      t0 = now_ns();
      id<MTLCommandBuffer> cb = [ctx.queue commandBuffer];
      if (options.storage == StorageMode::Private && K > 0) {
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:host_a sourceOffset:0 toBuffer:gpu_a destinationOffset:0 size:bytes_a];
        [blit copyFromBuffer:host_b sourceOffset:0 toBuffer:gpu_b destinationOffset:0 size:bytes_b];
        [blit endEncoding];
      }
      if (use_mps) {
        [mps encodeToCommandBuffer:cb
                        leftMatrix:mps_matrix(gpu_a, off_a, M, K)
                       rightMatrix:mps_matrix(gpu_b, off_b, K, N)
                      resultMatrix:mps_matrix(gpu_c, off_c, M, N)];
      } else {
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:gpu_a offset:off_a atIndex:0];
        [enc setBuffer:gpu_b offset:off_b atIndex:1];
        [enc setBuffer:gpu_c offset:off_c atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((N + kTile - 1) / kTile, (M + kTile - 1) / kTile, 1)
            threadsPerThreadgroup:MTLSizeMake(kTile, kTile, 1)];
        [enc endEncoding];
      }
      if (options.storage == StorageMode::Private) {
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromBuffer:gpu_c sourceOffset:0 toBuffer:host_c destinationOffset:0 size:bytes_c];
        [blit endEncoding];
      }
      t.encode_ns = now_ns() - t0;

      // ---- commit + wait -----------------------------------------------------
      emit(EventType::Submit, exec, 0, Backend::Metal, exec, "matmul_metal");
      t0 = now_ns();
      [cb commit];
      t.commit_end_ns = now_ns();
      t.commit_ns = t.commit_end_ns - t0;

      emit(EventType::Begin, exec, 0, Backend::Metal, exec, "metal_wait");
      [cb waitUntilCompleted];
      t.wait_end_ns = now_ns();
      t.wait_ns = t.wait_end_ns - t.commit_end_ns;
      emit(EventType::End, exec, 0, Backend::Metal, exec, "metal_wait");
      emit(EventType::Sync, exec, 0, Backend::Metal, exec, "matmul_metal");

      if ([cb status] != MTLCommandBufferStatusCompleted) {
        failure = "metal::matmul: command buffer failed: " + ns_error([cb error]);
      } else {
        t.sched_start_ns = to_ns([cb kernelStartTime]);
        t.sched_end_ns = to_ns([cb kernelEndTime]);
        t.gpu_start_ns = to_ns([cb GPUStartTime]);
        t.gpu_end_ns = to_ns([cb GPUEndTime]);
        emit_at(t.gpu_start_ns, EventType::Begin, exec, "gpu_exec");
        emit_at(t.gpu_end_ns, EventType::End, exec, "gpu_exec");

        // ---- copy out ----------------------------------------------------------
        if (options.storage != StorageMode::NoCopy) {
          t0 = now_ns();
          std::memcpy(out.data(), [host_c contents], bytes_c);
          t.copy_out_ns = now_ns() - t0;
        }
      }
    }
  }
  if (!failure.empty()) throw Error(failure);

  t.total_ns = now_ns() - start;
  if (timing) *timing = t;
}

Tensor matmul(const Tensor& a, const Tensor& b, const MatmulOptions& options, Timing* timing) {
  validate_matmul(a, b);
  Tensor out({a.shape()[0], b.shape()[1]}, a.storage()->allocator());
  matmul_into(a, b, out, options, timing);
  return out;
}

}  // namespace computeedge::metal
