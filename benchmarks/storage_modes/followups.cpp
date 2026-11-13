// M6 follow-ups: targeted experiments for the three open M5 findings
// (research/findings.md 7, 14, 15). Each experiment tests one hypothesis.
//
//   wake      (finding 7) Wake latency grew with GPU work. Hypothesis: completion
//             includes writing back dirty GPU cache lines. GPU read vs write streams of
//             1M..256M, sync = spin on MTLSharedEvent (lowest floor). Supported if wake
//             grows with bytes written but not with bytes read.
//   cpu_write (finding 14) Warm CPU writes into a Shared buffer were ~1.7x slower than
//             into malloc'd memory. Memory source {posix_memalign, mmap, shared,
//             shared_wc, nocopy over mmap} x write method {memset, vectorised store loop}
//             at 64 MiB, plus a read loop; and writes 1-4 into a newly created buffer.
//             Separates "memset path", "recently created mapping", and "Metal
//             mapping attribute".
//   gpu_write (finding 15) GPU write bandwidth dropped above ~256 MiB. Fine size sweep;
//             one buffer of N bytes vs N/256M buffers of 256 MiB written back to back;
//             second pass over the same buffer. Per-buffer effect (GPU page tables / TLB
//             reach) vs total-footprint effect.
//
// Usage:
//   followups [--experiments wake,cpu_write,gpu_write] + common flags

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../common/bench_common.h"

using namespace ce_bench;
using computeedge::now_ns;

namespace {

bool wants(const std::vector<std::string>& list, const std::string& name) {
  return std::find(list.begin(), list.end(), name) != list.end();
}

volatile uint64_t g_sink;

uint64_t read_all(const void* p, size_t bytes) {
  const uint64_t* w = static_cast<const uint64_t*>(p);
  uint64_t acc = 0;
  for (size_t i = 0; i < bytes / sizeof(uint64_t); ++i) acc += w[i];
  return acc;
}

// Distinct values per word so the compiler cannot turn this into memset; it
// vectorises to plain 128-bit stores.
__attribute__((noinline)) void store_loop(void* p, size_t bytes, uint64_t seed) {
  uint64_t* w = static_cast<uint64_t*>(p);
  for (size_t i = 0; i < bytes / sizeof(uint64_t); ++i) w[i] = i ^ seed;
}

uint64_t timed(const auto& fn) {
  const uint64_t t0 = now_ns();
  fn();
  return now_ns() - t0;
}

// Host memory from a chosen source (not a Metal buffer).
struct HostMem {
  enum class Source { Memalign, Mmap };
  HostMem(Source source, size_t bytes) : source(source), bytes(bytes) {
    if (source == Source::Memalign) {
      if (posix_memalign(&ptr, static_cast<size_t>(getpagesize()), bytes) != 0) ptr = nullptr;
    } else {
      ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
      if (ptr == MAP_FAILED) ptr = nullptr;
    }
    if (!ptr) {
      std::fprintf(stderr, "host allocation failed\n");
      std::exit(1);
    }
  }
  HostMem(const HostMem&) = delete;
  HostMem& operator=(const HostMem&) = delete;
  ~HostMem() {
    if (source == Source::Memalign)
      std::free(ptr);
    else
      munmap(ptr, bytes);
  }
  Source source;
  size_t bytes;
  void* ptr = nullptr;
};

}  // namespace

int main(int argc, char** argv) {
  const char* src = "benchmarks/storage_modes/followups.cpp";
  std::vector<std::string> experiments{"wake", "cpu_write", "gpu_write"};
  const CommonOptions opt = parse(argc, argv, src, [&](const std::string& f, const auto& next) {
    if (f == "--experiments")
      experiments = split(next());
    else
      return false;
    return true;
  });

  const cb::Environment env = banner("ComputeEdge M6: follow-ups to open M5 findings");
  prewarm_gpu(opt.prewarm_ms);
  const cb::Config cfg = config(opt);
  std::vector<cb::Result> results;
  auto keep = [&](cb::Result r) {
    std::cout << cb::format_result(r) << std::endl;
    results.push_back(std::move(r));
  };

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
                                    record_timing(m, probe::gpu_stream(s, buf, nullptr, probe::Sync::SpinEvent));
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
    HostMem memalign(HostMem::Source::Memalign, bytes), mapped(HostMem::Source::Mmap, bytes),
        mapped_for_nocopy(HostMem::Source::Mmap, bytes);
    probe::Buffer shared = probe::Buffer::create(probe::BufferKind::Shared, bytes);
    probe::Buffer shared_wc = probe::Buffer::create(probe::BufferKind::SharedWriteCombined, bytes);
    probe::Buffer nocopy = probe::Buffer::create(probe::BufferKind::NoCopy, bytes, mapped_for_nocopy.ptr);
    const std::vector<Source> sources{{"memalign", memalign.ptr},
                                      {"mmap", mapped.ptr},
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
                                  m["memset_ns"] = double(timed([&] { std::memset(s.ptr, int(seed & 0xFF), bytes); }));
                                  m["store_loop_ns"] = double(timed([&] { store_loop(s.ptr, bytes, seed); }));
                                  m["read_ns"] = double(timed([&] { g_sink = read_all(s.ptr, bytes); }));
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
      std::optional<HostMem> host;
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
              m["write" + std::to_string(i) + "_ns"] = double(timed([&] { std::memset(ptr, i, bytes); }));
          },
          0,
          [&] {
            buf.reset();
            host.reset();
            if (std::string(kind) == "memalign") {
              host.emplace(HostMem::Source::Memalign, bytes);
              ptr = host->ptr;
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
                                    record_timing(m, probe::gpu_stream(probe::Stream::Write, buf));
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

  write_results(opt, env, results);
  return 0;
}
