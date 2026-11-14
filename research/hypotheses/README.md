# Pre-Registered Research Hypotheses (C1–C17)

This directory documents the 17 pre-registered research claims formulated in ComputeEdge to characterize CPU–GPU execution dynamics, unified memory behavior, and thermal drift on Apple Silicon.

---

## Claim Registry Overview

| Claim | Short Description | Verdict Target | Source Benchmarks |
|---|---|---|---|
| **C1** | CPU (Accelerate) beats the GPU (MPS) below ~1024², the GPU wins above | Confirmed | `cpu_matmul`, `metal_matmul` |
| **C2** | ComputeEdge's MPS path costs the same as MLX's GPU stream at large sizes | Confirmed | `metal_matmul`, `mlx_matmul` |
| **C3** | An empty GPU round trip costs ~85-120 µs; CPU-side encode + commit is a small share | Confirmed | `dispatch_overhead` |
| **C4** | An extra command buffer costs far more than an extra dispatch | Confirmed | `dispatch_overhead` |
| **C5** | Even busy-polling, the CPU sees GPU completion no sooner than ~39 µs after GPUEndTime | Confirmed | `sync_methods` |
| **C6** | An idle GPU makes the next round trip several times slower | Confirmed | `sync_methods` |
| **C7** | A new MTLBuffer over already-resident memory pays a size-proportional cost on first GPU use | Confirmed | `storage_modes` |
| **C8** | Storage mode does not change GPU streaming bandwidth | Confirmed | `storage_modes` |
| **C9** | GPU traffic on other buffers does not evict the CPU's cached working set | Inconsistent | `cache_interference` |
| **C10** | GPU-to-CPU handoff of cache-resident data costs ~45% of a cached pass | Confirmed | `cache_interference` |
| **C11** | Concurrent GPU streaming slows CPU work | Confirmed | `cache_interference` |
| **C12** | Sustained load costs throughput on the heated unit(s) | Confirmed | `thermal_drift` |
| **C13** | The CPU/GPU crossover drifts under sustained load, invisibly to macOS thermal state | Confirmed | `thermal_drift` |
| **C14** | A just-created Shared buffer is slow to write for its first two passes, then matches host memory | Confirmed | `followups` |
| **C15** | GPU write bandwidth falls with the size of one buffer, not with total bytes | Confirmed | `followups` |
| **C16** | Wake latency after long GPU work does not depend on whether it wrote or read | Confirmed | `followups` |
| **C17** | GPU traffic displaces SLC-resident CPU data (reproduces arXiv 2606.27098 on M4, with clock controls) | No Data / Open | `cache_interference` |

---

## Verification & Parity

All claim definitions, metrics, and bootstrap confidence intervals are implemented in [`analysis/src/computeedge_analysis/claims/definitions.py`](file:///Users/bibek/Developer/ComputeEdge/analysis/src/computeedge_analysis/claims/definitions.py) and evaluated across sessions via `ce-claims`.

Claim verdicts must match [`docs/verification/baseline/claims.json`](file:///Users/bibek/Developer/ComputeEdge/docs/verification/baseline/claims.json) identically.
