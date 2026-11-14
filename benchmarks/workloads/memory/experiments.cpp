// Measurement logic of the storage_modes benchmark (moved verbatim from main.cpp; only
// printing goes through the Reporter).

#include "experiments.h"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
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

namespace computeedge::bench::storage_modes {

namespace cb = computeedge::bench;
namespace mtl = computeedge::metal;
namespace probe = computeedge::metal::probe;
using computeedge::now_ns;
using namespace computeedge::bench::harness;

int run(const Options& opt, harness::Reporter& report) {
  const char* src = kSource;
  const auto& sizes = opt.sizes;
  const auto& kinds = opt.kinds;
  const auto& experiments = opt.experiments;
  // Each experiment runs the requested kinds it supports, in this fixed order.
  auto kinds_for = [&](std::initializer_list<const char*> supported) {
    std::vector<std::string> out;
    for (const char* k : supported)
      if (wants(kinds, k)) out.push_back(k);
    return out;
  };
  prewarm(PrewarmWorkload::GpuStreamWritePrivate64MiB, opt.prewarm_ms, report);
  const cb::Config cfg = config(opt);
  auto keep = [&](cb::Result r) { report.result(std::move(r)); };
  auto base = [](const std::string& experiment, const std::string& kind, size_t bytes) {
    return std::map<std::string, std::string>{
        {"experiment", experiment}, {"kind", kind}, {"bytes", std::to_string(bytes)}, {"size", bytes_label(bytes)}};
  };

  // ---- create ----------------------------------------------------------------
  if (wants(experiments, "create")) {
    for (size_t bytes : sizes) {
      HostMemory host(bytes);
      std::memset(host.data(), 1, bytes);  // resident, so wrap/copy costs exclude host page faults
      for (const std::string& k : kinds_for({"shared", "private", "nocopy", "shared_copy"})) {
        const probe::BufferKind kind = parse_kind(k, src);
        keep(cb::run_with_metrics("storage_create", base("create", k, bytes), cfg, [&](cb::Metrics& m) {
          const uint64_t t0 = now_ns();
          probe::Buffer buf = probe::Buffer::create(kind, bytes, host.data());
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
        std::optional<HostMemory> host;
        std::optional<probe::Buffer> buf;
        void* mem = nullptr;
        auto fresh = [&] {  // untimed: new memory for every call
          buf.reset();
          host.reset();
          if (k == "host") {
            host.emplace(bytes);
            mem = host->data();
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
        const probe::BufferKind kind = parse_kind(k, src);
        // fresh:  new buffer (new memory) every call
        // rewrap: (nocopy only) new MTLBuffer wrapper every call over the same
        //         resident host memory -- exactly what metal::matmul_into does
        // reused: one buffer for every call
        std::vector<std::string> modes{"fresh", "reused"};
        if (kind == probe::BufferKind::NoCopy) modes.insert(modes.begin() + 1, "rewrap");
        for (const std::string& mode : modes) {
          std::optional<HostMemory> host;
          if (mode == "rewrap") {
            host.emplace(bytes);
            std::memset(host->data(), 1, bytes);
          }
          std::optional<probe::Buffer> buf;
          if (mode == "reused") buf = probe::Buffer::create(kind, bytes);
          auto params = base("gpu_first", k, bytes);
          params["buffer"] = mode;
          keep(cb::run_with_metrics(
              "storage_gpu_first", params, cfg,
              [&](cb::Metrics& m) { record_probe_timing(m, probe::bind_dispatch(*buf)); }, 0,
              [&] {
                if (mode == "reused") return;
                buf.reset();
                buf = probe::Buffer::create(kind, bytes, host ? host->data() : nullptr);
              }));
        }
      }
    }
  }

  // ---- gpu_bw ----------------------------------------------------------------
  if (wants(experiments, "gpu_bw")) {
    for (size_t bytes : sizes) {
      for (const std::string& k : kinds_for({"shared", "private", "nocopy"})) {
        const probe::BufferKind kind = parse_kind(k, src);
        probe::Buffer a = probe::Buffer::create(kind, bytes);
        probe::Buffer b = probe::Buffer::create(kind, bytes);
        probe::gpu_stream(probe::Stream::Write, a);  // initialise (and make resident)
        for (probe::Stream s : {probe::Stream::Read, probe::Stream::Write, probe::Stream::Copy}) {
          auto params = base("gpu_bw", k, bytes);
          params["stream"] = probe::to_string(s);
          params["bytes_moved"] = std::to_string(s == probe::Stream::Copy ? 2 * bytes : bytes);
          keep(cb::run_with_metrics("storage_gpu_bw", params, cfg, [&](cb::Metrics& m) {
            record_probe_timing(m, probe::gpu_stream(s, a, s == probe::Stream::Copy ? &b : nullptr));
          }));
        }
      }
    }
  }
  return 0;
}

}  // namespace computeedge::bench::storage_modes
