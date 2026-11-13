// M6 / RQ3: does GPU work slow the CPU's next pass over its own data?
//
// Every sample: warm CPU pass (untimed) -> interference phase (untimed) -> timed
// CPU pass. The CPU passes are
//   read  integer read-sum over W bytes (NoCopy buffer, behaves like host memory)
//   slc   16 MiB touched as one 128 B line per 8 KiB over 1 GiB, alternating direction:
//         the system-level-cache probe of Alpay & Basaran (arXiv 2606.27098)
//   sgemm Accelerate SGEMM 512^2 into a pre-allocated output (3 MiB of operands)
// Interference phases, each paired with a control of equal duration so clock /
// power-state effects (M5 finding 8: idling makes the next call slower) are not
// mistaken for cache effects. The CPU busy-spins through every phase, including
// GPU waits (polling the command buffer's status), so its clocks stay up:
//   idle_spin      control: plain spin for the median duration of gpu_write at this G
//   gpu_tiny       control: same submit + poll path, GPU rewrites a 16 KiB buffer for
//                  the same duration (GPU busy, negligible memory traffic)
//   gpu_write      GPU writes a separate G-byte buffer         -> eviction by GPU writes
//   gpu_read       GPU reads a separate G-byte buffer          -> eviction by GPU reads (no dirty lines)
//   cpu_read       this CPU thread reads a separate G buffer   -> reference: CPU-side eviction
//   idle_short     control for gpu_concurrent: spin 300 us
//   gpu_concurrent GPU keeps writing G while the timed pass runs -> bandwidth contention
//                  (300 us head start so the GPU is running; `overlap` metric checks it)
// Read pass only (handoff case: the CPU reads what the GPU just wrote):
//   idle_same      control: spin for the median duration of gpu_same
//   gpu_tiny_same  control: tiny GPU work of the same duration
//   gpu_same       GPU writes the CPU's own W buffer, then the CPU reads it
//
// Usage:
//   cache_interference [--working-sets 1M,4M,12M,64M] [--gpu-sizes 16M,64M,256M] [--passes read,slc,sgemm]
//                      + common flags (benchmarks/common/bench_common.h)

#include <pthread.h>
#include <sys/qos.h>

#include <algorithm>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "../common/bench_common.h"
#include "computeedge/computeedge.h"

using namespace ce_bench;
using computeedge::now_ns;

namespace {

volatile uint64_t g_sink;

uint64_t read_all(const void* p, size_t bytes) {
  const uint64_t* w = static_cast<const uint64_t*>(p);
  uint64_t acc = 0;
  for (size_t i = 0; i < bytes / sizeof(uint64_t); ++i) acc += w[i];
  return acc;
}

// Runs GPU work and busy-polls until it completes (keeps the CPU clocked up).
void gpu_spin(probe::Submission sub) {
  while (!sub.done()) {
  }
  sub.wait();
}

void spin_for(uint64_t ns) {
  const uint64_t until = now_ns() + ns;
  while (now_ns() < until) {
  }
}

bool wants(const std::vector<std::string>& list, const std::string& name) {
  return std::find(list.begin(), list.end(), name) != list.end();
}

// Median of a few runs of fn() (ns): calibrates control durations.
template <typename Fn>
uint64_t median_ns(Fn&& fn, int runs = 7) {
  std::vector<uint64_t> v;
  for (int i = 0; i < runs; ++i) {
    const uint64_t t0 = now_ns();
    fn();
    v.push_back(now_ns() - t0);
  }
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  const char* src = "benchmarks/cache/cache_interference.cpp";
  std::vector<size_t> working_sets = split_bytes("1M,4M,12M,64M");
  std::vector<size_t> gpu_sizes = split_bytes("16M,64M,256M");
  std::vector<std::string> passes{"read", "slc", "sgemm"};
  const CommonOptions opt = parse(argc, argv, src, [&](const std::string& f, const auto& next) {
    if (f == "--working-sets")
      working_sets = split_bytes(next());
    else if (f == "--gpu-sizes")
      gpu_sizes = split_bytes(next());
    else if (f == "--passes")
      passes = split(next());
    else
      return false;
    return true;
  });

  // Prefer the performance cores for the measuring thread.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

  const cb::Environment env = banner("ComputeEdge M6 / RQ3: CPU-GPU cache interference");
  prewarm_gpu(opt.prewarm_ms);
  const cb::Config cfg = config(opt);
  std::vector<cb::Result> results;
  probe::Buffer tiny = probe::Buffer::create(probe::BufferKind::Private, 16 << 10);
  const uint64_t tiny_ns =
      median_ns([&] { gpu_spin(probe::gpu_stream_async(probe::Stream::Write, tiny, nullptr, 64)); });
  // Repeats of the tiny write lasting about target_ns (time is not linear in
  // repeats at first, so refine a few times against the measurement).
  auto tiny_repeats = [&](uint64_t target_ns) {
    double reps = std::max(1.0, double(target_ns) * 64.0 / double(std::max<uint64_t>(tiny_ns, 1)));
    for (int i = 0; i < 4; ++i) {
      const uint64_t got =
          median_ns([&] { gpu_spin(probe::gpu_stream_async(probe::Stream::Write, tiny, nullptr, uint32_t(reps))); }, 5);
      reps = std::max(1.0, reps * double(target_ns) / double(std::max<uint64_t>(got, 1)));
    }
    return static_cast<uint32_t>(reps);
  };

  // Interference buffers, one pair per G (GPU target + CPU-written reference).
  struct Interferer {
    size_t bytes;
    probe::Buffer gpu;
    probe::Buffer cpu;
    uint64_t gpu_write_ns;  // calibrated median duration of the gpu_write phase
    uint32_t tiny_reps;     // gpu_tiny repeats lasting about as long
  };
  std::vector<Interferer> interferers;
  for (size_t g : gpu_sizes) {
    Interferer it{g, probe::Buffer::create(probe::BufferKind::Private, g),
                  probe::Buffer::create(probe::BufferKind::NoCopy, g), 0, 0};
    std::memset(it.cpu.contents(), 1, g);
    probe::gpu_stream(probe::Stream::Write, it.gpu);
    cb::trace("calibrate gpu_write g=" + std::to_string(g) + " runs=7 tiny_refinements=4 tiny_bytes=16384");
    it.gpu_write_ns = median_ns([&] { gpu_spin(probe::gpu_stream_async(probe::Stream::Write, it.gpu)); });
    it.tiny_reps = tiny_repeats(it.gpu_write_ns);
    const uint64_t tiny_check =
        median_ns([&] { gpu_spin(probe::gpu_stream_async(probe::Stream::Write, tiny, nullptr, it.tiny_reps)); });
    std::cout << "calibrated at " << bytes_label(g) << ": gpu_write " << it.gpu_write_ns / 1e3 << " us, gpu_tiny x"
              << it.tiny_reps << " " << tiny_check / 1e3 << " us\n";
    interferers.push_back(std::move(it));
  }
  std::cout << '\n';

  // A CPU pass: a name, a working-set label, and the timed operation.
  struct Pass {
    std::string name;
    std::string ws;
    std::function<void()> run;
    probe::Buffer* buffer;  // the W buffer for the read pass (gpu_same target), else nullptr
  };

  auto measure_pass = [&](Pass& pass) {
    auto params = [&](const std::string& phase, const std::string& g) {
      return std::map<std::string, std::string>{{"pass", pass.name}, {"ws", pass.ws}, {"phase", phase}, {"g", g}};
    };
    auto record = [&](const std::string& phase, const std::string& g, const std::function<void()>& interfere) {
      cb::Result r = cb::run_with_metrics(
          "cache_interference", params(phase, g), cfg,
          [&](cb::Metrics& m) { m["cpu_ns"] = double(median_ns(pass.run, 1)); }, 0,
          [&] {
            pass.run();  // warm: the working set is in cache
            interfere();
          });
      std::cout << cb::format_result(r) << std::endl;
      results.push_back(std::move(r));
    };

    for (Interferer& it : interferers) {
      const std::string g = bytes_label(it.bytes);
      record("idle_spin", g, [&] { spin_for(it.gpu_write_ns); });
      record("gpu_tiny", g,
             [&] { gpu_spin(probe::gpu_stream_async(probe::Stream::Write, tiny, nullptr, it.tiny_reps)); });
      record("gpu_write", g, [&] { gpu_spin(probe::gpu_stream_async(probe::Stream::Write, it.gpu)); });
      record("gpu_read", g, [&] { gpu_spin(probe::gpu_stream_async(probe::Stream::Read, it.gpu)); });
      record("cpu_read", g, [&] { g_sink = read_all(it.cpu.contents(), it.bytes); });

      // Concurrent: GPU runs long enough to cover the CPU pass several times over.
      const uint64_t pass_ns = median_ns(pass.run);
      const uint32_t repeats = static_cast<uint32_t>(
          std::max<uint64_t>(1, (std::max<uint64_t>(5'000'000, 4 * pass_ns) + it.gpu_write_ns - 1) / it.gpu_write_ns));
      std::optional<probe::Submission> sub;
      cb::Result r = cb::run_with_metrics(
          "cache_interference", params("gpu_concurrent", g), cfg,
          [&](cb::Metrics& m) {
            const uint64_t c0 = now_ns();
            pass.run();
            const uint64_t c1 = now_ns();
            const mtl::Timing t = sub->wait();
            m["cpu_ns"] = double(c1 - c0);
            m["overlap"] = (t.gpu_start_ns < c0 && t.gpu_end_ns > c1) ? 1.0 : 0.0;
          },
          0,
          [&] {
            pass.run();
            sub.emplace(probe::gpu_stream_async(probe::Stream::Write, it.gpu, nullptr, repeats));
            spin_for(300'000);
          });
      std::cout << cb::format_result(r) << std::endl;
      results.push_back(std::move(r));
    }
    record("idle_short", "-", [&] { spin_for(300'000); });

    if (pass.buffer) {
      probe::Buffer& w = *pass.buffer;
      const uint64_t same_ns = median_ns([&] { gpu_spin(probe::gpu_stream_async(probe::Stream::Write, w)); });
      const uint32_t same_tiny = tiny_repeats(same_ns);
      record("idle_same", pass.ws, [&] { spin_for(same_ns); });
      record("gpu_tiny_same", pass.ws,
             [&] { gpu_spin(probe::gpu_stream_async(probe::Stream::Write, tiny, nullptr, same_tiny)); });
      record("gpu_same", pass.ws, [&] { gpu_spin(probe::gpu_stream_async(probe::Stream::Write, w)); });
    }
  };

  if (wants(passes, "read")) {
    for (size_t ws : working_sets) {
      probe::Buffer w = probe::Buffer::create(probe::BufferKind::NoCopy, ws);
      std::memset(w.contents(), 1, ws);
      Pass pass{"read", bytes_label(ws), [&] { g_sink = read_all(w.contents(), ws); }, &w};
      measure_pass(pass);
    }
  }
  if (wants(passes, "slc")) {
    // Alpay & Basaran (arXiv 2606.27098) probe: 131,072 lines of 128 B, one every
    // 8 KiB across 1 GiB (16 MiB touched), direction alternating between passes.
    // The stride defeats prefetch and L2 set capacity, so the data lives in the
    // system-level cache shared with the GPU. Same controls as every other pass.
    constexpr size_t kStride = 8192, kLines = 131072, kSpan = kStride * kLines;
    probe::Buffer w = probe::Buffer::create(probe::BufferKind::NoCopy, kSpan);
    std::memset(w.contents(), 1, kSpan);
    const auto* base = static_cast<const unsigned char*>(w.contents());
    bool forward = true;
    Pass pass{"slc", "16M@8K",
              [&] {
                uint64_t acc = 0;
                if (forward) {
                  for (size_t i = 0; i < kLines; ++i) acc += *reinterpret_cast<const uint64_t*>(base + i * kStride);
                } else {
                  for (size_t i = kLines; i-- > 0;) acc += *reinterpret_cast<const uint64_t*>(base + i * kStride);
                }
                forward = !forward;
                g_sink = acc;
              },
              nullptr};
    measure_pass(pass);
  }
  if (wants(passes, "sgemm")) {
    using namespace computeedge;
    const Tensor a = Tensor::full({512, 512}, 0.5f), b = Tensor::full({512, 512}, 0.25f);
    Tensor c({512, 512});
    Pass pass{"sgemm", "3M", [&] { matmul_into(a, b, c); }, nullptr};
    measure_pass(pass);
  }

  write_results(opt, env, results);
  return 0;
}
