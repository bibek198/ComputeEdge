# Claim confirmation across sessions

4 session(s): `orig` = `orig`, `s0` = `20261113-192541_s0_47a6278756c0`, `s1` = `20261115-040830_s1_9f86e0f9e31d`, `s2` = `20261115-072142_s2_48c87d3ebb10`.
Criteria are fixed in `python/computeedge_analysis/claims/definitions.py`. A claim is paper-ready when it is **confirmed in >= 3 sessions**. Runs in `data/exclusions.json` are left out (2 excluded).

| claim | finding | statement | criterion | orig | s0 | s1 | s2 | verdict |
|---|---:|---|---|---|---|---|---|---|
| C1 | 1 | CPU (Accelerate) beats the GPU (MPS) below ~1024², the GPU wins above | Like-for-like (both into pre-allocated outputs): CPU/GPU time < 1 at 128² and 512², > 1 at 2048² and 4096² | ✅ 0.857 | ✅ 0.793 | ✅ 0.881 | ✅ 0.835 | **confirmed (4/4)** |
| C2 | 2 | ComputeEdge's MPS path costs the same as MLX's GPU stream at large sizes | |ComputeEdge MPS / MLX GPU - 1| < 10% at 4096² | ✅ 1 | ✅ 1 | ✅ 1.05 | ✅ 0.983 | **confirmed (4/4)** |
| C3 | 4 | An empty GPU round trip costs ~85-120 µs; CPU-side encode + commit is a small share | Empty round trip (1 CB, 1 encoder, 1 dispatch) median in [50, 250] µs and encode+commit < 10% of it | ✅ 85.9 | ✅ 159 | ✅ 158 | ✅ 119 | **confirmed (4/4)** |
| C4 | 5 | An extra command buffer costs far more than an extra dispatch | Marginal total cost per command buffer > 20x the marginal cost per dispatch | ✅ 35.2 | ✅ 21.9 | ✅ 21.9 | ✅ 23.2 | **confirmed (4/4)** |
| C5 | 6 | Even busy-polling, the CPU sees GPU completion no sooner than ~39 µs after GPUEndTime | Median wake with spin-on-event, empty kernel >= 25 µs | ✅ 39 | ✅ 38.7 | ✅ 40.8 | ✅ 31.8 | **confirmed (4/4)** |
| C6 | 8 | An idle GPU makes the next round trip several times slower | Round trip after 100 ms idle >= 3x back-to-back (waitUntilCompleted) and after 10 ms > after 0 ms | ✅ 6.97 | ✅ 7.77 | ✅ 6.8 | ✅ 4.23 | **confirmed (4/4)** |
| C7 | 11 | A new MTLBuffer over already-resident memory pays a size-proportional cost on first GPU use | NoCopy rewrap / reused round trip >= 3x at 64 MiB and >= 20x at 1 GiB | ✅ 559 | ✅ 721 | ✅ 565 | ✅ 543 | **confirmed (4/4)** |
| C8 | 10 | Storage mode does not change GPU streaming bandwidth | At 64 MiB and 256 MiB, read and write GB/s of shared/private/nocopy all within ±10% of their mean | ✅ 5.22 | ✅ 8.29 | ✅ 4.26 | ✅ 3.68 | **confirmed (4/4)** |
| C9 | 16 | GPU traffic on other buffers does not evict the CPU's cached working set | |slowdown| of the read pass after GPU writes/reads G (vs tiny-GPU control) < 10% for every W <= 12 MiB and G | ✅ 6 | ❌ 10.6 | ✅ 6.16 | ✅ 6.31 | **inconsistent (3/4)** |
| C10 | 17 | GPU-to-CPU handoff of cache-resident data costs ~45% of a cached pass | Read-pass slowdown after the GPU wrote the CPU's own W (vs tiny control) >= 25% for W in 1-12 MiB | ✅ 41.7 | ✅ 39.5 | ✅ 41.5 | ✅ 39 | **confirmed (4/4)** |
| C11 | 18 | Concurrent GPU streaming slows CPU work | Median over valid cells (GPU overlapped >= 90% of samples) of concurrent-vs-short slowdown > +10% | ✅ 138 | ✅ 125 | ✅ 166 | ✅ 171 | **confirmed (4/4)** |
| C12 | 20 | Sustained load costs throughput on the heated unit(s) | Throughput in the last 5 minutes vs minute 0 drops >= 10% for the unit(s) the heater loads (mixed: both) | ✅ 32.3 | n/a | ❌ 2.83 | ✅ 88.8 | **inconsistent (2/3)** |
| C13 | 21 | The CPU/GPU crossover drifts under sustained load, invisibly to macOS thermal state | CPU/GPU time ratio at 1024² changes by >= 10% between minute 0 and the last 5 minutes | ✅ 1.04 | n/a | ✅ 1.11 | ✅ 1.14 | **confirmed (3/3)** |
| C14 | 23 | A just-created Shared buffer is slow to write for its first two passes, then matches host memory | Shared write 2 < 0.8x write 4, and Shared write 4 within 10% of posix_memalign write 4 | ✅ 20.2 | ✅ 19.7 | ❌ 21.3 | ✅ 21.7 | **inconsistent (3/4)** |
| C15 | 24 | GPU write bandwidth falls with the size of one buffer, not with total bytes | 1 GiB as one buffer < 0.6x the bandwidth of the same bytes as 256 MiB buffers | ✅ 36.3 | ✅ 37 | ✅ 34.8 | ✅ 37.8 | **confirmed (4/4)** |
| C16 | 22 | Wake latency after long GPU work does not depend on whether it wrote or read | |wake(write) / wake(read) - 1| < 20% at 64 MiB and 256 MiB | ✅ 8.94 | ✅ 10.8 | ✅ 10.1 | ✅ 16.9 | **confirmed (4/4)** |
| C17 | 16 (scope) | GPU traffic displaces SLC-resident CPU data (reproduces arXiv 2606.27098 on M4, with clock controls) | Registered 2026-11-14 before any data: SLC-probe pass after the GPU writes 256 MiB is >= 15% slower than after the tiny-GPU control | n/a | n/a | ❌ 9.5 | ❌ 5.75 | **refuted (0/2)** |

## Headline values across sessions

| claim | value | orig | s0 | s1 | s2 | median | spread (max/min) |
|---|---|---:|---:|---:|---:|---:|---:|
| C1 | ratio_1024 | 0.857 | 0.793 | 0.881 | 0.835 | 0.846 | 1.11 |
| C1 | ratio_128 | 0.0152 | 0.0711 | 0.0147 | 0.0157 | 0.0154 | 4.85 |
| C1 | ratio_2048 | 1.2 | 1.21 | 1.24 | 1.29 | 1.22 | 1.08 |
| C2 | ratio | 1 | 1 | 1.05 | 0.983 | 1 | 1.07 |
| C3 | total_us | 85.9 | 159 | 158 | 119 | 139 | 1.85 |
| C3 | cpu_share_pct | 4.61 | 2.33 | 2.28 | 3.15 | 2.74 | 2.02 |
| C3 | queue_us | 41.5 | 88.8 | 87.9 | 58.4 | 73.1 | 2.14 |
| C3 | wake_us | 38.9 | 65.5 | 62.8 | 54 | 58.4 | 1.68 |
| C4 | per_cb_us | 35.2 | 21.9 | 21.9 | 23.2 | 22.6 | 1.61 |
| C4 | per_dispatch_us | 0.578 | 0.56 | 0.59 | 0.618 | 0.584 | 1.1 |
| C4 | ratio | 60.9 | 39.1 | 37.1 | 37.6 | 38.4 | 1.64 |
| C5 | wake_us | 39 | 38.7 | 40.8 | 31.8 | 38.8 | 1.28 |
| C6 | ratio_100ms | 6.97 | 7.77 | 6.8 | 4.23 | 6.88 | 1.84 |
| C6 | back_to_back_us | 83.5 | 124 | 160 | 163 | 142 | 1.95 |
| C6 | after_100ms_us | 582 | 964 | 1,086 | 688 | 826 | 1.87 |
| C7 | rewrap_64m_us | 559 | 721 | 565 | 543 | 562 | 1.33 |
| C7 | rewrap_1g_ms | 8.21 | 8.96 | 8.79 | 8.94 | 8.87 | 1.09 |
| C7 | reused_1g_us | 102 | 158 | 151 | 157 | 154 | 1.55 |
| C8 | max_dev_pct | 5.22 | 8.29 | 4.26 | 3.68 | 4.74 | 2.25 |
| C9 | max_abs_pct | 6 | 10.6 | 6.16 | 6.31 | 6.23 | 1.77 |
| C10 | min_pct | 41.7 | 39.5 | 41.5 | 39 | 40.5 | 1.07 |
| C10 | max_pct | 48.3 | 59 | 46.1 | 49.6 | 48.9 | 1.28 |
| C11 | median_pct | 138 | 125 | 166 | 171 | 152 | 1.36 |
| C11 | sgemm_median_pct | 64 | 87.8 | 2.18 | 5.43 | 34.7 | 40.3 |
| C12 | cpu_loss_pct | 32.3 | — | 2.83 | 88.8 | 32.3 | 31.4 |
| C12 | gpu_loss_pct | 14.2 | — | 6.09 | 75 | 14.2 | 12.3 |
| C13 | ratio_start | 1.04 | — | 1.11 | 1.14 | 1.11 | 1.1 |
| C13 | ratio_end | 1.21 | — | 0.769 | 2.93 | 1.21 | 3.81 |
| C14 | w1 | 20.2 | 19.7 | 21.3 | 21.7 | 20.7 | 1.1 |
| C14 | w2 | 47 | 49.7 | 48.4 | 50.5 | 49.1 | 1.07 |
| C14 | w4 | 73.5 | 73.4 | 73.4 | 73.4 | 73.4 | 1 |
| C15 | one_gbs | 36.3 | 37 | 34.8 | 37.8 | 36.7 | 1.08 |
| C15 | chunks_gbs | 90.2 | 85.6 | 83.3 | 87.4 | 86.5 | 1.08 |
| C16 | max_diff_pct | 8.94 | 10.8 | 10.1 | 16.9 | 10.5 | 1.89 |
| C17 | slowdown_256m_pct | — | — | 9.5 | 5.75 | 7.63 | 1.65 |
