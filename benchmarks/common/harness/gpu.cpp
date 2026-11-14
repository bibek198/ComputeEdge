#include "harness/gpu.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>

#include "computeedge/bench/trace.h"
#include "computeedge/core/instrumentation.h"
#include "computeedge/metal/probes.h"
#include "harness/correctness.h"

namespace computeedge::bench::harness {

namespace probe = computeedge::metal::probe;

void require_metal() {
  if (!metal::available()) {
    std::fprintf(stderr, "Metal is not available on this system\n");
    std::exit(1);
  }
}

std::string describe(PrewarmWorkload workload) {
  switch (workload) {
    case PrewarmWorkload::GpuStreamWritePrivate64MiB: return "gpu_stream_write kind=private bytes=67108864";
    case PrewarmWorkload::MetalMatmulShared256: return "metal_matmul n=256 storage=shared kernel=tiled16";
  }
  return "unknown";
}

void prewarm(PrewarmWorkload workload, double ms, Reporter& out) {
  trace("prewarm workload=" + describe(workload) + " ms=" + std::to_string(ms));
  switch (workload) {
    case PrewarmWorkload::GpuStreamWritePrivate64MiB: {
      if (ms <= 0) return;
      probe::Buffer buf = probe::Buffer::create(probe::BufferKind::Private, size_t{64} << 20);
      const uint64_t until = now_ns() + static_cast<uint64_t>(ms * 1e6);
      size_t calls = 0;
      while (now_ns() < until) {
        probe::gpu_stream(probe::Stream::Write, buf);
        ++calls;
      }
      std::ostringstream note;
      note << "GPU pre-warm: " << calls << " x 64 MiB writes over " << ms << " ms\n\n";
      out.note(note.str());
      return;
    }
    case PrewarmWorkload::MetalMatmulShared256: {
      const Tensor A = random_square(256, 3), B = random_square(256, 4);
      Tensor C({256, 256});
      const uint64_t until = now_ns() + static_cast<uint64_t>(ms * 1e6);
      size_t calls = 0;
      while (now_ns() < until) {
        metal::matmul_into(A, B, C, {metal::StorageMode::Shared});
        ++calls;
      }
      std::ostringstream note;
      note << "GPU pre-warm: " << calls << " dispatches over " << ms << " ms\n\n";
      out.note(note.str());
      return;
    }
  }
}

void record_probe_timing(Metrics& m, const metal::Timing& t) {
  m["encode_ns"] = double(t.encode_ns);
  m["commit_ns"] = double(t.commit_ns);
  m["wait_ns"] = double(t.wait_ns);
  m["total_ns"] = double(t.total_ns);
  m["gpu_ns"] = double(t.gpu_ns());
  m["queue_ns"] = double(t.queue_ns());
  m["wake_ns"] = double(t.wake_ns());
}

void record_matmul_timing(Metrics& m, const metal::Timing& t) {
  m["buffer_ns"] = double(t.buffer_ns);
  m["copy_in_ns"] = double(t.copy_in_ns);
  m["encode_ns"] = double(t.encode_ns);
  m["commit_ns"] = double(t.commit_ns);
  m["wait_ns"] = double(t.wait_ns);
  m["copy_out_ns"] = double(t.copy_out_ns);
  m["sched_ns"] = double(t.sched_ns());
  m["gpu_ns"] = double(t.gpu_ns());
  m["queue_ns"] = double(t.queue_ns());
  m["wake_ns"] = double(t.wake_ns());
}

}  // namespace computeedge::bench::harness
