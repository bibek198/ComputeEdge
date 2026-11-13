# Claim confirmation across sessions

2 session(s): `orig` = `orig`, `s0` = `20261113-192541_s0_47a6278756c0`.
Criteria are fixed in `benchmarks/analysis/confirm_claims.py`. A claim is paper-ready when it is **confirmed in >= 3 sessions**. Runs in `benchmarks/results/exclusions.json` are left out (1 excluded).

| claim | finding | statement | criterion | orig | s0 | verdict |
|---|---:|---|---|---|---|---|
| C1 | 1 | CPU (Accelerate) beats the GPU (MPS) below ~1024², the GPU wins above | Like-for-like (both into pre-allocated outputs): CPU/GPU time < 1 at 128² and 512², > 1 at 2048² and 4096² | ✅ 0.857 | ✅ 0.793 | **confirmed (2/2)** |
| C2 | 2 | ComputeEdge's MPS path costs the same as MLX's GPU stream at large sizes | |ComputeEdge MPS / MLX GPU - 1| < 10% at 4096² | ✅ 1 | ✅ 1 | **confirmed (2/2)** |
| C3 | 4 | An empty GPU round trip costs ~85-120 µs; CPU-side encode + commit is a small share | Empty round trip (1 CB, 1 encoder, 1 dispatch) median in [50, 250] µs and encode+commit < 10% of it | ✅ 85.9 | ✅ 159 | **confirmed (2/2)** |
| C4 | 5 | An extra command buffer costs far more than an extra dispatch | Marginal total cost per command buffer > 20x the marginal cost per dispatch | ✅ 35.2 | ✅ 21.9 | **confirmed (2/2)** |
| C5 | 6 | Even busy-polling, the CPU sees GPU completion no sooner than ~39 µs after GPUEndTime | Median wake with spin-on-event, empty kernel >= 25 µs | ✅ 39 | ✅ 38.7 | **confirmed (2/2)** |
| C6 | 8 | An idle GPU makes the next round trip several times slower | Round trip after 100 ms idle >= 3x back-to-back (waitUntilCompleted) and after 10 ms > after 0 ms | ✅ 6.97 | ✅ 7.77 | **confirmed (2/2)** |
| C7 | 11 | A new MTLBuffer over already-resident memory pays a size-proportional cost on first GPU use | NoCopy rewrap / reused round trip >= 3x at 64 MiB and >= 20x at 1 GiB | ✅ 559 | ✅ 721 | **confirmed (2/2)** |
| C8 | 10 | Storage mode does not change GPU streaming bandwidth | At 64 MiB and 256 MiB, read and write GB/s of shared/private/nocopy all within ±10% of their mean | ✅ 5.22 | ✅ 8.29 | **confirmed (2/2)** |
| C9 | 16 | GPU traffic on other buffers does not evict the CPU's cached working set | |slowdown| of the read pass after GPU writes/reads G (vs tiny-GPU control) < 10% for every W <= 12 MiB and G | ✅ 6 | ❌ 10.6 | **inconsistent (1/2)** |
| C10 | 17 | GPU-to-CPU handoff of cache-resident data costs ~45% of a cached pass | Read-pass slowdown after the GPU wrote the CPU's own W (vs tiny control) >= 25% for W in 1-12 MiB | ✅ 41.7 | ✅ 39.5 | **confirmed (2/2)** |
| C11 | 18 | Concurrent GPU streaming slows CPU work | Median over valid cells (GPU overlapped >= 90% of samples) of concurrent-vs-short slowdown > +10% | ✅ 138 | ✅ 125 | **confirmed (2/2)** |
| C12 | 20 | Sustained load costs throughput on the heated unit(s) | Throughput in the last 5 minutes vs minute 0 drops >= 10% for the unit(s) the heater loads (mixed: both) | ✅ 32.3 | n/a | **confirmed (1/1)** |
| C13 | 21 | The CPU/GPU crossover drifts under sustained load, invisibly to macOS thermal state | CPU/GPU time ratio at 1024² changes by >= 10% between minute 0 and the last 5 minutes | ✅ 1.04 | n/a | **confirmed (1/1)** |
| C14 | 23 | A just-created Shared buffer is slow to write for its first two passes, then matches host memory | Shared write 2 < 0.8x write 4, and Shared write 4 within 10% of posix_memalign write 4 | ✅ 20.2 | ✅ 19.7 | **confirmed (2/2)** |
| C15 | 24 | GPU write bandwidth falls with the size of one buffer, not with total bytes | 1 GiB as one buffer < 0.6x the bandwidth of the same bytes as 256 MiB buffers | ✅ 36.3 | ✅ 37 | **confirmed (2/2)** |
| C16 | 22 | Wake latency after long GPU work does not depend on whether it wrote or read | |wake(write) / wake(read) - 1| < 20% at 64 MiB and 256 MiB | ✅ 8.94 | ✅ 10.8 | **confirmed (2/2)** |

## Headline values across sessions

| claim | value | orig | s0 | median | spread (max/min) |
|---|---|---:|---:|---:|---:|
| C1 | ratio_1024 | 0.857 | 0.793 | 0.825 | 1.08 |
| C1 | ratio_128 | 0.0152 | 0.0711 | 0.0431 | 4.68 |
| C1 | ratio_2048 | 1.2 | 1.21 | 1.2 | 1 |
| C2 | ratio | 1 | 1 | 1 | 1 |
| C3 | total_us | 85.9 | 159 | 122 | 1.85 |
| C3 | cpu_share_pct | 4.61 | 2.33 | 3.47 | 1.98 |
| C3 | queue_us | 41.5 | 88.8 | 65.1 | 2.14 |
| C3 | wake_us | 38.9 | 65.5 | 52.2 | 1.68 |
| C4 | per_cb_us | 35.2 | 21.9 | 28.5 | 1.61 |
| C4 | per_dispatch_us | 0.578 | 0.56 | 0.569 | 1.03 |
| C4 | ratio | 60.9 | 39.1 | 50 | 1.56 |
| C5 | wake_us | 39 | 38.7 | 38.8 | 1.01 |
| C6 | ratio_100ms | 6.97 | 7.77 | 7.37 | 1.12 |
| C6 | back_to_back_us | 83.5 | 124 | 104 | 1.49 |
| C6 | after_100ms_us | 582 | 964 | 773 | 1.66 |
| C7 | rewrap_64m_us | 559 | 721 | 640 | 1.29 |
| C7 | rewrap_1g_ms | 8.21 | 8.96 | 8.58 | 1.09 |
| C7 | reused_1g_us | 102 | 158 | 130 | 1.55 |
| C8 | max_dev_pct | 5.22 | 8.29 | 6.76 | 1.59 |
| C9 | max_abs_pct | 6 | 10.6 | 8.31 | 1.77 |
| C10 | min_pct | 41.7 | 39.5 | 40.6 | 1.06 |
| C10 | max_pct | 48.3 | 59 | 53.6 | 1.22 |
| C11 | median_pct | 138 | 125 | 132 | 1.1 |
| C11 | sgemm_median_pct | 64 | 87.8 | 75.9 | 1.37 |
| C12 | cpu_loss_pct | 32.3 | — | 32.3 | 1 |
| C12 | gpu_loss_pct | 14.2 | — | 14.2 | 1 |
| C13 | ratio_start | 1.04 | — | 1.04 | 1 |
| C13 | ratio_end | 1.21 | — | 1.21 | 1 |
| C14 | w1 | 20.2 | 19.7 | 19.9 | 1.02 |
| C14 | w2 | 47 | 49.7 | 48.4 | 1.06 |
| C14 | w4 | 73.5 | 73.4 | 73.4 | 1 |
| C15 | one_gbs | 36.3 | 37 | 36.7 | 1.02 |
| C15 | chunks_gbs | 90.2 | 85.6 | 87.9 | 1.05 |
| C16 | max_diff_pct | 8.94 | 10.8 | 9.86 | 1.2 |
