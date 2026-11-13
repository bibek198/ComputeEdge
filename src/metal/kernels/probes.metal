// ComputeEdge characterization probes (M5, Tracks A and B).
//
// Compiled at runtime (newLibraryWithSource:), embedded into the binary by CMake.
// These kernels exist to measure the runtime around them, not to compute anything.

#include <metal_stdlib>
using namespace metal;

// Does nothing: the cost of a dispatch is pure encode / submit / schedule / sync.
kernel void probe_empty(uint tid [[thread_position_in_grid]]) {}

// Binds one buffer and (in practice) never touches it. The never-true branch
// keeps the argument live, so the driver must make the buffer resident; any cost
// that grows with the buffer's size is residency / mapping, not memory traffic.
kernel void probe_bind(device float* buf [[buffer(0)]],
                       constant uint& flag [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {
  if (flag == 0xFFFFFFFFu && tid == 0) buf[0] = 0.0f;
}

// Streaming kernels, float4 per access, grid-stride loop (coalesced per SIMD group).
kernel void probe_read(device const float4* src [[buffer(0)]],
                       device float* sink [[buffer(1)]],
                       constant uint& n4 [[buffer(2)]],
                       uint tid [[thread_position_in_grid]],
                       uint threads [[threads_per_grid]]) {
  float4 acc = 0.0f;
  for (uint i = tid; i < n4; i += threads) acc += src[i];
  const float s = acc.x + acc.y + acc.z + acc.w;
  if (s == 1234.5678f) sink[0] = s;  // defeats dead-code elimination; never true for our inputs
}

kernel void probe_write(device float4* dst [[buffer(0)]],
                        constant uint& n4 [[buffer(2)]],
                        uint tid [[thread_position_in_grid]],
                        uint threads [[threads_per_grid]]) {
  for (uint i = tid; i < n4; i += threads) dst[i] = float4(1.0f);
}

kernel void probe_copy(device const float4* src [[buffer(0)]],
                       device float4* dst [[buffer(1)]],
                       constant uint& n4 [[buffer(2)]],
                       uint tid [[thread_position_in_grid]],
                       uint threads [[threads_per_grid]]) {
  for (uint i = tid; i < n4; i += threads) dst[i] = src[i];
}
