#pragma once

// Internal (Objective-C++ only): the process-wide Metal context shared by the
// matmul backend (metal_backend.mm) and the characterization probes
// (probes.mm). Never included from public headers.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

namespace computeedge::metal::detail {

struct Context {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  // One library per embedded .metal source, compiled on first use.
  std::unordered_map<std::string, id<MTLLibrary>> libraries;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
  // MPS kernels are shape-specific; cache them so steady-state calls do not pay
  // for object creation (as a framework would).
  std::map<std::tuple<size_t, size_t, size_t>, MPSMatrixMultiplication*> mps_kernels;
  // Monotonic event used by the event-based synchronization probes.
  id<MTLSharedEvent> event = nil;
  uint64_t event_value = 0;
  uint64_t compile_ns = 0;
  std::mutex mutex;
};

Context& context();

// Pipeline for `function` in the library built from the named embedded source
// ("matmul" or "probes"). Compiles the library and pipeline on first use; the
// time is added to Context::compile_ns.
id<MTLComputePipelineState> pipeline(const std::string& library, const std::string& function);

std::string ns_error(NSError* error);

// MTLCommandBuffer timestamps (seconds, CPU timebase) to ns; 0 if unset.
inline uint64_t to_ns(CFTimeInterval seconds) { return seconds > 0 ? static_cast<uint64_t>(seconds * 1e9) : 0; }

}  // namespace computeedge::metal::detail
