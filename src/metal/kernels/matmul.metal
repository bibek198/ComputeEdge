// ComputeEdge Metal kernels: FP32 matrix multiplication.
//
// Compiled at runtime (newLibraryWithSource:), embedded into the binary by CMake.
// Row-major, contiguous A[M,K], B[K,N], C[M,N].

#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
  uint M;
  uint N;
  uint K;
};

constant constexpr uint TILE = 16;

// Classic shared-memory tiling: each 16x16 threadgroup computes a 16x16 block of
// C, staging 16x16 tiles of A and B through threadgroup memory. Dispatched with
// dispatchThreadgroups, so every thread reaches the barriers; out-of-range
// threads load zeros and skip the store.
kernel void matmul_tiled(device const float* A [[buffer(0)]],
                         device const float* B [[buffer(1)]],
                         device float* C [[buffer(2)]],
                         constant MatmulParams& p [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]],
                         uint2 tid [[thread_position_in_threadgroup]]) {
  threadgroup float As[TILE][TILE];
  threadgroup float Bs[TILE][TILE];

  const uint row = gid.y;
  const uint col = gid.x;
  float acc = 0.0f;

  const uint tiles = (p.K + TILE - 1) / TILE;
  for (uint t = 0; t < tiles; ++t) {
    const uint a_col = t * TILE + tid.x;
    const uint b_row = t * TILE + tid.y;
    As[tid.y][tid.x] = (row < p.M && a_col < p.K) ? A[row * p.K + a_col] : 0.0f;
    Bs[tid.y][tid.x] = (b_row < p.K && col < p.N) ? B[b_row * p.N + col] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < TILE; ++k) acc = fma(As[tid.y][k], Bs[k][tid.x], acc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < p.M && col < p.N) C[row * p.N + col] = acc;
}
