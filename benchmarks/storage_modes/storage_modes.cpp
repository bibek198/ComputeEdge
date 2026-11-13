// Track B: what each way of getting memory to the GPU costs.
//
// Experiments (sizes default 16K,256K,4M,64M,256M,1G):
//   create      create + release one MTLBuffer: shared, private, nocopy (wrap existing
//               page-aligned host memory), shared_copy (newBufferWithBytes from host)
//   cpu_touch   CPU write to freshly created memory (first touch: page faults), then
//               a second write and a read (warm), for host (posix_memalign), shared,
//               nocopy. Each call gets fresh memory (created in untimed setup).
//   gpu_first   empty kernel with one buffer bound but untouched, on a fresh buffer
//               (created in untimed setup), a fresh NoCopy wrapper over the same
//               resident memory (rewrap), or the same buffer reused: isolates
//               residency / mapping cost, and tests whether it explains the growth
//               of matmul queue time with size seen in M3/M4
//   gpu_bw      GPU read / write / copy streams over shared, private, nocopy buffers
//               (reused; bandwidth = bytes moved / gpu_ns)
//
// Usage:
//   storage_modes [--sizes 16K,256K,4M,64M,256M,1G] [--kinds host,shared,private,nocopy,shared_copy]
//                 [--experiments create,cpu_touch,gpu_first,gpu_bw]
//                 + common flags (benchmarks/common/bench_common.h)

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

probe::BufferKind kind_from(const std::string& s, const char* src) {
  for (probe::BufferKind k : {probe::BufferKind::Shared, probe::BufferKind::Private, probe::BufferKind::NoCopy,
                              probe::BufferKind::SharedCopy})
    if (s == probe::to_string(k)) return k;
  usage(src, "unknown buffer kind " + s);
}

size_t page_size() { return static_cast<size_t>(getpagesize()); }

// Page-aligned host memory (the allocator's layout) that is not a Metal buffer.
struct HostPages {
  explicit HostPages(size_t bytes) : bytes(bytes) {
    if (posix_memalign(&ptr, page_size(), bytes) != 0) {
      std::fprintf(stderr, "host allocation of %zu bytes failed\n", bytes);
      std::exit(1);
    }
  }
  HostPages(const HostPages&) = delete;
  HostPages& operator=(const HostPages&) = delete;
  ~HostPages() { std::free(ptr); }
  void* ptr = nullptr;
  size_t bytes;
};

// Integer sum so the loop vectorises and is bandwidth-bound, not latency-bound.
volatile uint64_t g_sink;
uint64_t read_all(const void* p, size_t bytes) {
  const uint64_t* w = static_cast<const uint64_t*>(p);
  uint64_t acc = 0;
  for (size_t i = 0; i < bytes / sizeof(uint64_t); ++i) acc += w[i];
  return acc;
}

uint64_t timed_ns(const auto& fn) {
  const uint64_t t0 = now_ns();
  fn();
  return now_ns() - t0;
}

}  // namespace

int main(int argc, char** argv) {
  const char* src = "benchmarks/storage_modes/storage_modes.cpp";
  std::vector<size_t> sizes = split_bytes("16K,256K,4M,64M,256M,1G");
  std::vector<std::string> kinds{"host", "shared", "private", "nocopy", "shared_copy"};
  std::vector<std::string> experiments{"create", "cpu_touch", "gpu_first", "gpu_bw"};
  const CommonOptions opt = parse(argc, argv, src, [&](const std::string& f, const auto& next) {
    if (f == "--sizes")
      sizes = split_bytes(next());
    else if (f == "--kinds")
      kinds = split(next());
    else if (f == "--experiments")
      experiments = split(next());
    else
      return false;
    return true;
  });
  for (const auto& k : kinds)
    if (k != "host") kind_from(k, src);
  // Each experiment runs the requested kinds it supports, in this fixed order.
  auto kinds_for = [&](std::initializer_list<const char*> supported) {
    std::vector<std::string> out;
    for (const char* k : supported)
      if (wants(kinds, k)) out.push_back(k);
    return out;
  };
  for (size_t s : sizes)
    if (s == 0 || s % page_size() != 0) usage(src, "sizes must be non-zero multiples of the page size (16K)");

  const cb::Environment env = banner("ComputeEdge M5 / Track B: storage modes");
  prewarm_gpu(opt.prewarm_ms);
  const cb::Config cfg = config(opt);
  std::vector<cb::Result> results;
  auto keep = [&](cb::Result r) {
    std::cout << cb::format_result(r) << std::endl;
    results.push_back(std::move(r));
  };
  auto base = [](const std::string& experiment, const std::string& kind, size_t bytes) {
    return std::map<std::string, std::string>{
        {"experiment", experiment}, {"kind", kind}, {"bytes", std::to_string(bytes)}, {"size", bytes_label(bytes)}};
  };

  // ---- create ----------------------------------------------------------------
  if (wants(experiments, "create")) {
    for (size_t bytes : sizes) {
      HostPages host(bytes);
      std::memset(host.ptr, 1, bytes);  // resident, so wrap/copy costs exclude host page faults
      for (const std::string& k : kinds_for({"shared", "private", "nocopy", "shared_copy"})) {
        const probe::BufferKind kind = kind_from(k, src);
        keep(cb::run_with_metrics("storage_create", base("create", k, bytes), cfg, [&](cb::Metrics& m) {
          const uint64_t t0 = now_ns();
          probe::Buffer buf = probe::Buffer::create(kind, bytes, host.ptr);
          const uint64_t t1 = now_ns();
          buf.reset();
          const uint64_t t2 = now_ns();
          m["create_ns"] = double(t1 - t0);
          m["release_ns"] = double(t2 - t1);
        }));
      }
    }
  }

  // ---- cpu_touch -------------------------------------------------------------
  if (wants(experiments, "cpu_touch")) {
    for (size_t bytes : sizes) {
      for (const std::string& k : kinds_for({"host", "shared", "nocopy"})) {
        std::optional<HostPages> host;
        std::optional<probe::Buffer> buf;
        void* mem = nullptr;
        auto fresh = [&] {  // untimed: new memory for every call
          buf.reset();
          host.reset();
          if (k == "host") {
            host.emplace(bytes);
            mem = host->ptr;
          } else {
            buf = probe::Buffer::create(k == "shared" ? probe::BufferKind::Shared : probe::BufferKind::NoCopy, bytes);
            mem = buf->contents();
          }
        };
        keep(cb::run_with_metrics(
            "storage_cpu_touch", base("cpu_touch", k, bytes), cfg,
            [&](cb::Metrics& m) {
              m["first_write_ns"] = double(timed_ns([&] { std::memset(mem, 1, bytes); }));
              m["warm_write_ns"] = double(timed_ns([&] { std::memset(mem, 2, bytes); }));
              m["warm_read_ns"] = double(timed_ns([&] { g_sink = read_all(mem, bytes); }));
            },
            0, fresh));
      }
    }
  }

  // ---- gpu_first -------------------------------------------------------------
  if (wants(experiments, "gpu_first")) {
    for (size_t bytes : sizes) {
      for (const std::string& k : kinds_for({"shared", "private", "nocopy"})) {
        const probe::BufferKind kind = kind_from(k, src);
        // fresh:  new buffer (new memory) every call
        // rewrap: (nocopy only) new MTLBuffer wrapper every call over the same
        //         resident host memory -- exactly what metal::matmul_into does
        // reused: one buffer for every call
        std::vector<std::string> modes{"fresh", "reused"};
        if (kind == probe::BufferKind::NoCopy) modes.insert(modes.begin() + 1, "rewrap");
        for (const std::string& mode : modes) {
          std::optional<HostPages> host;
          if (mode == "rewrap") {
            host.emplace(bytes);
            std::memset(host->ptr, 1, bytes);
          }
          std::optional<probe::Buffer> buf;
          if (mode == "reused") buf = probe::Buffer::create(kind, bytes);
          auto params = base("gpu_first", k, bytes);
          params["buffer"] = mode;
          keep(cb::run_with_metrics(
              "storage_gpu_first", params, cfg, [&](cb::Metrics& m) { record_timing(m, probe::bind_dispatch(*buf)); },
              0,
              [&] {
                if (mode == "reused") return;
                buf.reset();
                buf = probe::Buffer::create(kind, bytes, host ? host->ptr : nullptr);
              }));
        }
      }
    }
  }

  // ---- gpu_bw ----------------------------------------------------------------
  if (wants(experiments, "gpu_bw")) {
    for (size_t bytes : sizes) {
      for (const std::string& k : kinds_for({"shared", "private", "nocopy"})) {
        const probe::BufferKind kind = kind_from(k, src);
        probe::Buffer a = probe::Buffer::create(kind, bytes);
        probe::Buffer b = probe::Buffer::create(kind, bytes);
        probe::gpu_stream(probe::Stream::Write, a);  // initialise (and make resident)
        for (probe::Stream s : {probe::Stream::Read, probe::Stream::Write, probe::Stream::Copy}) {
          auto params = base("gpu_bw", k, bytes);
          params["stream"] = probe::to_string(s);
          params["bytes_moved"] = std::to_string(s == probe::Stream::Copy ? 2 * bytes : bytes);
          keep(cb::run_with_metrics("storage_gpu_bw", params, cfg, [&](cb::Metrics& m) {
            record_timing(m, probe::gpu_stream(s, a, s == probe::Stream::Copy ? &b : nullptr));
          }));
        }
      }
    }
  }

  write_results(opt, env, results);
  return 0;
}
