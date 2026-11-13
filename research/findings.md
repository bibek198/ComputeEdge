# Findings log

Running record of what the measurements show, one section per milestone. Each finding links to the committed
run it comes from and says how strong the evidence is. **Single-session results**: every number here comes from
one machine (MacBook Air M4, macOS 26.4.1, AC power, nominal thermal state) and, unless noted, one session.
Session-to-session variation is ~11% (§7.2 of the plan), so all of this must be re-measured across sessions in
M9 before it goes into a paper.

Evidence labels: **solid** (large effect, consistent across repeated runs or configurations), **likely**
(clear effect, one run), **open** (observed, not yet explained).

---

## M4: baselines (FP32 SGEMM)

Source: [`results/comparisons/20261113-091016`](../benchmarks/results/comparisons/20261113-091016/comparison.md)

1. **The CPU/GPU crossover is near 1024².** Accelerate wins below it (4.3 µs vs ~190 µs at 128²); GPU (MPS,
   MLX) wins from 2048² (~1.3×). At 1024² the 95% CIs overlap. *Solid* (large effects at both ends). The CPU
   series includes output allocation and the GPU series does not, so the CPU side is conservative.
2. **Our Metal call path costs the same as MLX's.** MPS through ComputeEdge ≈ MLX GPU stream at large sizes
   (57.6 vs 57.6 ms at 4096²) and within ~10% at small sizes. *Likely.*
3. **MLX CPU adds ~9 µs per call** over calling Accelerate directly at 128². *Likely.*

## M5: dispatch, synchronization, storage modes (Tracks A and B)

Source: [`results/reports/m5-20261113-174406`](../benchmarks/results/reports/m5-20261113-174406/report.md)
(runs listed in its `sources.json`)

### Track A: the fixed cost of a GPU round trip

4. **An empty GPU round trip costs ~85–120 µs, and almost none of it is CPU-side submission.** Encoding plus
   commit is ~4 µs; the rest is *queue* (commit → GPUStartTime, ~40–55 µs) and *wake* (GPUEndTime → CPU resumes,
   ~40–50 µs). GPU execution of the empty kernel is ~1.3 µs. *Solid.* This means a cheaper submission API
   cannot fix small-op GPU cost; the latency is in scheduling and completion signalling.
5. **A command buffer costs ~50× more than a dispatch.** Marginal cost: +0.58 µs per extra dispatch in an
   encoder, +0.70 µs per extra encoder, **+35 µs per extra command buffer** (of which ~21 µs GPU-side, ~5 µs
   encode). *Solid* (linear fits over 6–7 points each). Batching work into one command buffer is the main
   lever for small ops.
6. **Even busy-polling cannot get under ~39 µs of wake latency.** Spin on `MTLSharedEvent.signaledValue` gives
   wake ≈ 39 µs for an empty kernel, consistently over three runs, so the floor is GPU-completion → CPU
   visibility, not thread wake-up. Blocking methods (`waitUntilCompleted`, event wait, completion handler) add
   roughly 0–30 µs on top. Their ranking changes between runs (method-order confound), so **no blocking
   method is reliably best**. *Solid* for the floor; *open* for the ranking.
7. **Wake latency grows with the amount of GPU work.** Even spinning: ~39 µs (empty) → ~100 µs (4 MiB write) →
   ~110–118 µs (256 MiB write). Hypothesis: completion includes flushing GPU caches after the last kernel.
   *Open.* With the 256 MiB write, spin-on-event had the lowest wake in all three runs (110–118 µs vs
   130–165 µs for the blocking methods); at 4 MiB there was no consistent winner.
8. **An idle GPU makes the next round trip up to ~7× slower.** Empty round trip after an idle gap: 84 µs
   (0 ms) → 133 µs (0.1 ms) → 215 µs (1 ms) → 481 µs (10 ms) → 582 µs (100 ms), mostly queue time. CPU-side
   encode also rises from 2.5 to ~45 µs. *Likely* (one run; the spin-event series is non-monotonic between 10
   and 100 ms). **Consequence:** back-to-back benchmarks (all of M4) understate what sporadic GPU use costs in
   an application. The M8 dispatcher needs "time since the last GPU submission" as an input.
9. **Cold start:** compiling the probe library took 1.3 ms, with the OS shader cache likely warm. The first
   round trip in a process then took 1.5 ms, of which 1.2 ms was queueing. *One sample.*

### Track B: storage modes

10. **Storage mode does not change GPU streaming bandwidth.** Read/write/copy GB/s for Shared, Private and
    NoCopy are within ~5% at every size (reads ~100–120 GB/s from 4 MiB up). *Solid.* On M4 UMA, the
    differences between storage modes are in setup costs, not in kernel bandwidth.
11. **Every new MTLBuffer pays a size-proportional cost on its first GPU use, even over memory that is already
    resident.** First submission binding (not touching) a buffer:

    | size | reused | NoCopy, new wrapper over the same memory | new Shared/Private buffer |
    |---:|---:|---:|---:|
    | 64 MiB | ~0.1 ms | 0.56 ms | 1.2–1.3 ms |
    | 1 GiB | ~0.1 ms | 8.2 ms | 27–29 ms |

    New Shared/Private buffers cost ~3× more than re-wrapping, consistent with also zero-filling new pages.
    *Solid* (monotonic over six sizes, rewrap ≈ fresh NoCopy).
12. **This explains about half to 60% of the matmul queue growth seen in M3/M4.** `metal::matmul_into` wraps
    three new NoCopy buffers per call. Finding 11 predicts ~1.4 ms of queueing at 4096² (3 × 64 MiB) against
    2.4 ms measured, and ~0.43 ms at 2048² against 0.71 ms. *Likely.* **M7 lever:** cache the wrapper with
    the tensor's storage. The predicted saving is largest relative to call time near the crossover, e.g.
    ~0.17 ms of 1.32 ms (~12%) at 1024².
13. **Creating a buffer is cheap; filling it is not.** `newBufferWithLength` (Shared/Private) and
    `newBufferWithBytesNoCopy` take 1–25 µs from 16 KiB to 1 GiB, because the memory is backed lazily.
    `newBufferWithBytes` copies at ~18 GB/s (60 ms for 1 GiB), the same rate as first-touching a fresh Shared
    buffer from the CPU (~19–22 GB/s, page faults plus zero-fill). *Solid.*
14. **CPU writes into a Shared buffer's memory are slower than into malloc'd memory, even when warm.** Warm
    memset: 43–46 GB/s (Shared) vs 74–91 GB/s (host or NoCopy) at ≥64 MiB. Reads are equal (~65 GB/s). NoCopy
    over host memory behaves like host memory. *Open:* it needs a second write method (not memset) and a
    page-size/TLB check before it can be claimed.
15. **GPU writes slow down above ~256 MiB.** Write: 88 GB/s at 256 MiB, ~48 GB/s at 512 MiB, 36 GB/s at 1 GiB,
    with reads flat at ~100 GB/s and identical across storage modes. *Open*; M6 candidate (GPU TLB reach,
    memory compression or SLC behaviour).

### Measurement notes from M5

- The same configuration measured at different points in one run varied by up to ~30% (empty round trip:
  86, 104, 119 µs). Comparisons between configurations that differ by less than that need interleaved or
  repeated runs, as done for the synchronization methods (three runs, three orders).
- The CPU first-touch numbers for `host` and `nocopy` reuse memory the allocator has already faulted in, so
  they are not true first touches. Only the Shared numbers measure fresh pages.
- CPU bandwidth at 16 KiB is limited by timer resolution (~0.1–0.2 µs per operation) and should not be read
  as a bandwidth.
