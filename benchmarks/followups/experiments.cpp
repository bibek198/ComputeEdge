// Measurement logic of the followups benchmark (moved verbatim from main.cpp; only
// printing goes through the Reporter).

#include "experiments.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <sstream>
#include "computeedge/bench/environment.h"
#include "computeedge/bench/result_writer.h"
#include "computeedge/bench/runner.h"
#include "computeedge/bench/trace.h"
#include "computeedge/core/instrumentation.h"
#include "computeedge/metal/matmul.h"
#include "computeedge/metal/probes.h"
#include "harness/cli.h"
#include "harness/console.h"
#include "harness/gpu.h"
#include "harness/host_memory.h"
#include "harness/hot_path.h"
#include "harness/results.h"

namespace computeedge::bench::followups {

namespace cb = computeedge::bench;
namespace mtl = computeedge::metal;
namespace probe = computeedge::metal::probe;
using computeedge::now_ns;

namespace {

using namespace computeedge::bench::harness;

// Distinct values per word so the compiler cannot turn this into memset; it
// vectorises to plain 128-bit stores.
__attribute__((noinline)) void store_loop(void* p, size_t bytes, uint64_t seed) {
  uint64_t* w = static_cast<uint64_t*>(p);
  for (size_t i = 0; i < bytes / sizeof(uint64_t); ++i) w[i] = i ^ seed;
}

}  // namespace
using namespace computeedge::bench::harness;

int run(const Options& opt, harness::Reporter& report) {
  const auto& experiments = opt.experiments;
  prewarm(PrewarmWorkload::GpuStreamWritePrivate64MiB, opt.prewarm_ms, report);
  const cb::Config cfg = config(opt);
  auto keep = [&](cb::Result r) { report.result(std::move(r)); };

  // ---- finding 7: wake latency vs bytes read / written --------------------------
  if (wants(experiments, "wake")) {
    for (size_t bytes : split_bytes("1M,4M,16M,64M,256M")) {
      probe::Buffer buf = probe::Buffer::create(probe::BufferKind::Private, bytes);
      probe::gpu_stream(probe::Stream::Write, buf);
      for (probe::Stream s : {probe::Stream::Read, probe::Stream::Write}) {
        keep(cb::run_with_metrics("followup_wake",
                                  {{"experiment", "wake"},
                                   {"stream", probe::to_string(s)},
                                   {"size", bytes_label(bytes)},
                                   {"bytes", std::to_string(bytes)},
                                   {"sync", "spin_event"}},
                                  cfg, [&](cb::Metrics& m) {
                                    record_probe_timing(m, probe::gpu_stream(s, buf, nullptr, probe::Sync::SpinEvent));
                                  }));
      }
    }
  }

  // ---- finding 14: CPU write rate by memory source and write method --------------
  if (wants(experiments, "cpu_write")) {
    const size_t bytes = size_t{64} << 20;
    struct Source {
      std::string name;
      void* ptr;
    };
    HostMemory memalign(bytes, HostMemory::Source::Memalign), mapped(bytes, HostMemory::Source::Mmap),
        mapped_for_nocopy(bytes, HostMemory::Source::Mmap);
    probe::Buffer shared = probe::Buffer::create(probe::BufferKind::Shared, bytes);
    probe::Buffer shared_wc = probe::Buffer::create(probe::BufferKind::SharedWriteCombined, bytes);
    probe::Buffer nocopy = probe::Buffer::create(probe::BufferKind::NoCopy, bytes, mapped_for_nocopy.data());
    const std::vector<Source> sources{{"memalign", memalign.data()},
                                      {"mmap", mapped.data()},
                                      {"shared", shared.contents()},
                                      {"shared_wc", shared_wc.contents()},
                                      {"nocopy_mmap", nocopy.contents()}};
    for (const Source& s : sources) {
      std::memset(s.ptr, 1, bytes);  // fault everything in: all measurements are warm
      store_loop(s.ptr, bytes, 3);
      uint64_t seed = 0;
      keep(cb::run_with_metrics("followup_cpu_write",
                                {{"experiment", "cpu_write"},
                                 {"source", s.name},
                                 {"size", bytes_label(bytes)},
                                 {"bytes", std::to_string(bytes)}},
                                cfg, [&](cb::Metrics& m) {
                                  ++seed;
                                  m["memset_ns"] =
                                      double(timed_ns([&] { std::memset(s.ptr, int(seed & 0xFF), bytes); }));
                                  m["store_loop_ns"] = double(timed_ns([&] { store_loop(s.ptr, bytes, seed); }));
                                  m["read_ns"] = double(timed_ns([&] { g_sink = read_all(s.ptr, bytes); }));
                                }));
    }
  }

  // ---- finding 14b: successive writes into a newly created buffer ---------------
  // M5 timed the 2nd write right after creating the buffer; is the slowdown a
  // property of Shared memory or of a recently created mapping?
  if (wants(experiments, "cpu_write")) {
    const size_t bytes = size_t{64} << 20;
    for (const char* kind : {"shared", "nocopy", "memalign"}) {
      std::optional<probe::Buffer> buf;
      std::optional<HostMemory> host;
      void* ptr = nullptr;
      keep(cb::run_with_metrics(
          "followup_cpu_write_fresh",
          {{"experiment", "cpu_write_fresh"},
           {"source", kind},
           {"size", bytes_label(bytes)},
           {"bytes", std::to_string(bytes)}},
          cfg,
          [&](cb::Metrics& m) {
            for (int i = 1; i <= 4; ++i)
              m["write" + std::to_string(i) + "_ns"] = double(timed_ns([&] { std::memset(ptr, i, bytes); }));
          },
          0,
          [&] {
            buf.reset();
            host.reset();
            if (std::string(kind) == "memalign") {
              host.emplace(bytes, HostMemory::Source::Memalign);
              ptr = host->data();
            } else {
              buf = probe::Buffer::create(
                  std::string(kind) == "shared" ? probe::BufferKind::Shared : probe::BufferKind::NoCopy, bytes);
              ptr = buf->contents();
            }
          }));
    }
  }

  // ---- finding 15: GPU write bandwidth vs buffer size ----------------------------
  if (wants(experiments, "gpu_write")) {
    const size_t chunk = size_t{256} << 20;
    for (size_t bytes : split_bytes("128M,192M,256M,320M,384M,512M,768M,1G")) {
      // (a) one buffer, two passes in one command buffer (first vs second pass).
      {
        probe::Buffer buf = probe::Buffer::create(probe::BufferKind::Private, bytes);
        probe::gpu_stream(probe::Stream::Write, buf);
        keep(cb::run_with_metrics("followup_gpu_write",
                                  {{"experiment", "gpu_write"},
                                   {"layout", "one_buffer"},
                                   {"size", bytes_label(bytes)},
                                   {"bytes", std::to_string(bytes)}},
                                  cfg, [&](cb::Metrics& m) {
                                    record_probe_timing(m, probe::gpu_stream(probe::Stream::Write, buf));
                                    m["two_pass_gpu_ns"] = double(
                                        probe::gpu_stream_async(probe::Stream::Write, buf, nullptr, 2).wait().gpu_ns());
                                  }));
      }
      // (b) the same bytes as 256 MiB buffers, committed back to back.
      if (bytes > chunk && bytes % chunk == 0) {
        std::vector<probe::Buffer> parts;
        for (size_t i = 0; i < bytes / chunk; ++i) {
          parts.push_back(probe::Buffer::create(probe::BufferKind::Private, chunk));
          probe::gpu_stream(probe::Stream::Write, parts.back());
        }
        keep(cb::run_with_metrics("followup_gpu_write",
                                  {{"experiment", "gpu_write"},
                                   {"layout", "256M_chunks"},
                                   {"size", bytes_label(bytes)},
                                   {"bytes", std::to_string(bytes)}},
                                  cfg, [&](cb::Metrics& m) {
                                    std::vector<probe::Submission> subs;
                                    for (const auto& p : parts)
                                      subs.push_back(probe::gpu_stream_async(probe::Stream::Write, p));
                                    uint64_t first_start = UINT64_MAX, last_end = 0;
                                    for (auto& sub : subs) {
                                      const mtl::Timing t = sub.wait();
                                      first_start = std::min(first_start, t.gpu_start_ns);
                                      last_end = std::max(last_end, t.gpu_end_ns);
                                    }
                                    m["gpu_ns"] = double(last_end - first_start);
                                  }));
      }
    }
  }
  return 0;
}

}  // namespace computeedge::bench::followups
