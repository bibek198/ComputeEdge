# Findings log

Running record of what the measurements show, one section per milestone. Each finding links to the committed
run it comes from and says how strong the evidence is. **Single-session results**: every number here comes from
one machine (MacBook Air M4, macOS 26.4.1, AC power, nominal thermal state) and, unless noted, one session.
Session-to-session variation is ~11% (§7.2 of the plan), so all of this must be re-measured across sessions in
M9 before it goes into a paper.

Evidence labels: **solid** (large effect, consistent across repeated runs or configurations), **likely**
(clear effect, one run), **open** (observed, not yet explained).

**Confirmation status.** Each quantitative finding is now a claim (C1–C16) with a pass criterion fixed in
`benchmarks/analysis/confirm_claims.py`, and is re-checked in every confirmation session
(`scripts/run_session.sh`). The latest table is in `benchmarks/results/reports/confirm-*/claims.md`. As of
2026-11-13 (original runs plus session s0): 15 claims passed in every session where they could be evaluated; **C9
(no eviction by GPU traffic) is inconsistent**, because one of 18 cells came out at +10.6% against a 10% bound in s0.
The thermal claims C12–C13 rest on one valid run, because s0's thermal run is excluded (see
`benchmarks/results/exclusions.json`: the adapter was unplugged and a video call ran during it). Paper-ready means
confirmed in ≥3 sessions.

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
   **Mostly rejected in M6 (finding 22).** With the 256 MiB write, spin-on-event had the lowest wake in all three runs (110–118 µs vs
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
    over host memory behaves like host memory. **Resolved in M6 (finding 23): an artefact of timing the
    second write into a just-created buffer, not a property of Shared memory.**
15. **GPU writes slow down above ~256 MiB.** Write: 88 GB/s at 256 MiB, ~48 GB/s at 512 MiB, 36 GB/s at 1 GiB,
    with reads flat at ~100 GB/s and identical across storage modes. **Narrowed in M6 (finding 24): it is a
    per-buffer-size effect.**

### Measurement notes from M5

- The same configuration measured at different points in one run varied by up to ~30% (empty round trip:
  86, 104, 119 µs). Comparisons between configurations that differ by less than that need interleaved or
  repeated runs, as done for the synchronization methods (three runs, three orders).
- The CPU first-touch numbers for `host` and `nocopy` reuse memory the allocator has already faulted in, so
  they are not true first touches. Only the Shared numbers measure fresh pages.
- CPU bandwidth at 16 KiB is limited by timer resolution (~0.1–0.2 µs per operation) and should not be read
  as a bandwidth.

## M6: cache interference, thermal drift, and follow-ups

Source: [`results/reports/m6-20261113-191318`](../benchmarks/results/reports/m6-20261113-191318/report.md)
(runs at 7cf5c55, listed in its `sources.json`)

### RQ3: does GPU work slow the CPU?

Every GPU phase was compared against a control of equal duration. One control is a plain CPU spin. The other
runs the same submit-and-poll code path on a 16 KiB buffer, so the GPU is busy but moves almost no memory.
The second control turned out to be essential: having *any* GPU work in flight while the CPU polls slows the
next small CPU pass by +7–28%, regardless of what the GPU does. Without that control, this effect would have
been misread as cache eviction.

16. **GPU memory traffic does not evict the CPU's unrelated cached data.** The GPU wrote or read 16–256 MiB
    of a *different* buffer, then the CPU read its own 1–12 MiB working set. Change vs the tiny-GPU control:
    −5% to +6%, mostly within the CI. Accelerate SGEMM 512² shows the same (≤ ±1%). By contrast, the CPU
    reading 64–256 MiB itself slows the next pass by +39–47%. *Solid.* On M4, the CPU's L2 is effectively
    insulated from GPU streaming; RQ3's eviction hypothesis is **falsified** for unrelated data.
17. **Handing data from the GPU to the CPU costs ~45% of a cached pass.** When the GPU writes the CPU's own
    working set and the CPU then reads it, the pass is +42–48% slower for 1–12 MiB (CPU caches lose those
    lines). At 64 MiB the effect vanishes, because the data was not cache-resident anyway. *Solid.* This is
    the real interference cost in a CPU↔GPU pipeline, and it applies to every hand-off in M7–M8.
18. **Running at the same time is expensive.** While the GPU streams writes, a CPU read pass is 1.1–4.1×
    slower, and Accelerate SGEMM 512² is 1.2–2.6× slower. *Likely:* the magnitude varies a lot between
    configurations and some CIs are wide, and in one cell (W = 64M, G = 64M) the GPU overlapped the CPU pass
    in only 10% of samples. The direction is clear; the size needs repeated runs.
    **Consequence for M8:** placing work on the GPU to "free the CPU" can slow concurrent CPU work by more
    than it saves.

### RQ4: thermal drift (15 min, mixed CPU + GPU load, one run)

19. **The fanless Air reached thermal state `fair` after 1.5 min and stayed there.** It never reached
    `serious` in 15 minutes. *Likely* (one run).
20. **Under sustained mixed load, the CPU loses ~40% of its throughput and the GPU ~15–25%.** Accelerate
    1024²/2048²: ~1,150–1,500 → ~730–760 GFLOP/s, a sharp step at ~3.4 min. MPS 2048²: ~2,200 → ~1,850
    GFLOP/s, gradual; MPS 1024²: ~1,170 → ~880. *Likely.* Even at minute 0 the CPU is ~20% below the M4
    baseline: the heater runs right before each measurement and shares the power budget.
21. **The CPU/GPU crossover moves as the chip heats.** At 1024², CPU/GPU time goes 1.04 (tie) → 0.75–0.95
    (CPU ahead, minutes 1–3) → ~1.22 (GPU ahead, minute 4 onward). At 2048² the GPU's lead grows from 1.9× to
    2.6×. *Likely.* **RQ4's hypothesis is supported.** The large CPU step happens with *no* change in macOS
    thermal state (`fair` throughout), so `NSProcessInfo.thermalState` is too coarse to drive placement. The
    M8 dispatcher needs measured, recent performance rather than the thermal state alone.

### Follow-ups on the open M5 findings

22. **Finding 7 (wake latency grows with GPU work): the cache-writeback explanation is mostly rejected.**
    With equal sizes, read and write streams give about the same wake latency once GPU time is ≥ ~200 µs
    (16 MiB: 117 vs 110 µs; 256 MiB: 101 vs 110 µs). Only at 1–4 MiB do writes wake later (85 vs 44 µs at
    1 MiB). Wake rises from ~40 µs to a plateau of ~100–110 µs as the GPU stays busy longer. Mechanism:
    *open* (a GPU or firmware power-state effect is the leading candidate).
23. **Finding 14 resolved: the slow writes come from a just-created buffer, not Shared memory.** Warm,
    long-lived Shared buffers write at host speed (89.8 vs 89.3 GB/s; reads identical). A newly created
    Shared buffer needs two passes to reach full speed: write 1 at 20 GB/s (page faults and zero-fill),
    write 2 at 47 GB/s, write 3+ at 73 GB/s. M5 timed exactly write 2. *Solid.* Side result: write-combined
    Shared buffers are 20× slower to read from the CPU (3.4 GB/s), so they suit CPU→GPU upload only.
24. **Finding 15 narrowed: GPU writes slow down with the size of a single buffer, not the total bytes.**
    One buffer: ~97 GB/s up to 256 MiB, 88 GB/s at 384 MiB, 72 GB/s at 512 MiB, 36 GB/s at 1 GiB. The same
    bytes as 256 MiB buffers: 85–92 GB/s. A second pass in the same command buffer is no faster, so this is
    not first-touch. *Solid* (monotonic, large effect). Mechanism: *open*; GPU address-translation reach
    per buffer is the leading candidate, and Instruments would help here. **Practical rule:** keep GPU-written
    buffers ≤ 256–384 MiB.

### Decision gate (plan §8)

The characterization holds up on its own: a fixed GPU round-trip cost dominated by queueing and wake-up;
per-buffer-object mapping costs; an idle-gap penalty; and a crossover that drifts under heat. Several of
these are new and counter-intuitive (findings 6, 8, 11–12, 16–17, 21, 24). The pieces a 4-page
characterization paper still needs are repeated sessions (M9 protocol) and a second or longer thermal run.
