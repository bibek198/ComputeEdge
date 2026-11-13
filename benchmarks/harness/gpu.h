#pragma once

// GPU-side plumbing shared by the Metal benchmarks.

#include <string>

#include "computeedge/bench/runner.h"
#include "computeedge/metal/matmul.h"

namespace computeedge::bench::harness {

// Exits with status 1 and "Metal is not available on this system" when there is no usable GPU.
void require_metal();

// Process-level GPU pre-warm workloads. They differ on purpose (audit finding
// D14): each benchmark keeps the pre-warm it was characterized with.
enum class PrewarmWorkload {
  GpuStreamWritePrivate64MiB,  // 64 MiB Private-buffer write streams (probe benchmarks)
  MetalMatmulShared256,        // 256^2 tiled16 matmul, Shared storage (metal_matmul)
};

// The workload's identity, as written to the execution trace.
std::string describe(PrewarmWorkload workload);

// Runs `workload` continuously for `ms` milliseconds (results excluded) and
// prints how many iterations ran.
void prewarm(PrewarmWorkload workload, double ms);

// Phase metrics of a probe call (empty dispatch, sync, storage, cache, followups).
void record_probe_timing(Metrics& m, const metal::Timing& t);
// Phase metrics of a metal::matmul call (adds buffer, copy and scheduling phases).
void record_matmul_timing(Metrics& m, const metal::Timing& t);

}  // namespace computeedge::bench::harness
