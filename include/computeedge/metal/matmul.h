#pragma once

// Metal GPU backend. Pure C++ interface: no Objective-C types are exposed.
//
// Every call is synchronous (commit + waitUntilCompleted) and fills a Timing
// record that splits the cost into CPU-side phases and GPU-side execution.
// All timestamps share the now_ns() timebase (CLOCK_UPTIME_RAW), so CPU and GPU
// events can be placed on one timeline.

#include <cstdint>
#include <string>

#include "computeedge/core/tensor.h"

namespace computeedge::metal {

// How tensor memory reaches the GPU. This is an experimental variable (Track B).
enum class StorageMode : uint8_t {
  NoCopy,   // wrap the tensor's 16 KB-aligned host storage (newBufferWithBytesNoCopy); zero-copy
  Shared,   // copy into a fresh MTLStorageModeShared buffer, copy the result back
  Private,  // Shared staging + blit to/from MTLStorageModePrivate buffers (blits run on the GPU)
};

const char* to_string(StorageMode mode);

// Which GPU kernel computes the product.
enum class Kernel : uint8_t {
  Tiled16,  // ComputeEdge's own 16x16 threadgroup-tiled kernel (src/metal/kernels/matmul.metal)
  MPS,      // Apple's MPSMatrixMultiplication: the tuned-kernel reference
};

const char* to_string(Kernel kernel);

struct DeviceInfo {
  std::string name;
  bool has_unified_memory = false;
  uint64_t recommended_max_working_set_bytes = 0;
  uint64_t max_buffer_length = 0;
};

bool available();
DeviceInfo device_info();

// Compiles the shader library and pipelines if not done yet. Returns the
// one-time compile cost in ns (0 if already compiled). Call before steady-state
// measurement so cold start is excluded.
uint64_t warmup();

// Time spent compiling the library + pipelines (0 until compiled).
uint64_t compile_ns();

struct Timing {
  // CPU-side phase durations (ns).
  uint64_t buffer_ns = 0;    // creating / wrapping MTLBuffers
  uint64_t copy_in_ns = 0;   // host copies into GPU-visible memory (incl. making inputs contiguous)
  uint64_t encode_ns = 0;    // command buffer + encoder creation and encoding
  uint64_t commit_ns = 0;    // the commit() call
  uint64_t wait_ns = 0;      // waitUntilCompleted
  uint64_t copy_out_ns = 0;  // host copy of the result (Shared / Private only)
  uint64_t total_ns = 0;     // whole call, after validation

  // Absolute timestamps in the now_ns() timebase.
  uint64_t commit_end_ns = 0;   // CPU: commit() returned
  uint64_t sched_start_ns = 0;  // MTLCommandBuffer.kernelStartTime: OS kernel/driver began scheduling (CPU side)
  uint64_t sched_end_ns = 0;    // MTLCommandBuffer.kernelEndTime
  uint64_t gpu_start_ns = 0;    // MTLCommandBuffer.GPUStartTime: GPU began executing
  uint64_t gpu_end_ns = 0;      // MTLCommandBuffer.GPUEndTime
  uint64_t wait_end_ns = 0;     // CPU: waitUntilCompleted returned

  uint64_t gpu_ns() const { return gpu_end_ns > gpu_start_ns ? gpu_end_ns - gpu_start_ns : 0; }
  uint64_t sched_ns() const { return sched_end_ns > sched_start_ns ? sched_end_ns - sched_start_ns : 0; }
  // Signed: the GPU may start before commit() returns.
  int64_t queue_ns() const { return static_cast<int64_t>(gpu_start_ns) - static_cast<int64_t>(commit_end_ns); }
  int64_t wake_ns() const { return static_cast<int64_t>(wait_end_ns) - static_cast<int64_t>(gpu_end_ns); }
};

struct MatmulOptions {
  StorageMode storage = StorageMode::NoCopy;
  Kernel kernel = Kernel::Tiled16;
};

// C[M,N] = A[M,K] * B[K,N] on the GPU (FP32). `out` must be a
// contiguous [M,N] tensor; for NoCopy its storage must be page-aligned (the
// default allocator guarantees this).
void matmul_into(const Tensor& a, const Tensor& b, Tensor& out, const MatmulOptions& options = {},
                 Timing* timing = nullptr);

Tensor matmul(const Tensor& a, const Tensor& b, const MatmulOptions& options = {}, Timing* timing = nullptr);

}  // namespace computeedge::metal
