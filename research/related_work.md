# Related Work and Novelty Assessment

*Compiled November 2026 from a literature search across about 75 sources.*

> **Verify before citing.** Many 2026 arXiv items, the DaMoN '26 numbers, MLX issue details and venue facts were checked
> only through abstracts or search excerpts, not full texts. Each such item is marked **(unverified)** or **(snippet)**.
> Check every citation against the primary source before it goes into a manuscript.

---

## 1. Summary

The planned Phase-2 mechanisms are **not novel on their own**:

| Idea | Already covered by |
|---|---|
| Zero-copy shared buffers via `newBufferWithBytesNoCopy`, buffer cache | MLX Metal allocator (Shared + Untracked buffers, NoCopy wrap, size-keyed cache, residency sets) |
| Liveness-based buffer reuse / memory planning | XLA, TFLite, TVM, ggml-alloc; Pisarchyk & Lee 2020; TelaMalloc (ASPLOS '23); MODeL (ICML '23) |
| Size/cost-based CPU vs GPU placement | Qilin (MICRO '09), MNN (MLSys '20), CoDL (MobiSys '22); MLX docs show manual placement |
| CPU+GPU co-execution on Apple unified memory | FusionML (arXiv 2607.22785, 2026) |
| Apple shared-memory overheads, CPU–GPU co-processing | Beischl et al., DaMoN '26 |
| CPU vs GPU GEMM throughput/energy on M1–M4 | Hübner et al., "Apple vs. Oranges" (arXiv 2502.05317) |

**Candidate gaps.** These are hypotheses, not confirmed novelty:

1. A microbenchmark breakdown of Metal dispatch, command-buffer commit, `waitUntilCompleted` and `MTLSharedEvent` costs as a function of tensor size.
2. A controlled comparison of Shared, Private and NoCopy storage modes: allocation, first touch, CPU/GPU read/write, alignment.
3. Cache interference between CPU and GPU under zero-copy (see arXiv 2606.27098).
4. Safe reuse of buffers that both the CPU and GPU access on unified memory, where reuse depends on completion on each device rather than graph position. MLX issue #3461 is a real-world instance.
5. Per-op CPU/GPU crossover sizes measured against the M4's SME/AMX path in Accelerate, not a naive CPU baseline.
6. Sustained and thermal behaviour on a **fanless** MacBook Air. Prior work used Mac Studio, Pro or Max machines.

---

## 2. Apple Silicon characterization

| Work | Venue | Finding | Status |
|---|---|---|---|
| Hübner, Hu, Peng, Markidis — *Apple vs. Oranges: Evaluating the Apple Silicon M-Series SoCs for HPC* | arXiv 2502.05317, 2025 | M1–M4 bandwidth (~100 GB/s base chips), FP32 GEMM via Accelerate/MPS/Metal; M4 GPU ~2.9 TFLOPS; powermetrics energy | verified |
| Kenyon, Capano — *Apple Silicon Performance in Scientific Computing* | arXiv 2211.00720 (HPEC) | SHOC/OpenCL on M1, M1 Ultra vs V100/A100 | verified |
| Beischl, Morozov, Jungmair, Haussmann, Neumann — *Evaluating Apple Silicon for Data Processing* | DaMoN '26, doi:10.1145/3789237.3809127 | Significant default overhead for CPU–GPU shared allocation; coherence costs; CPU+GPU co-processing up to 5× | snippet |
| Remke, Breuer — *Hello SME! Generating Fast Matrix Multiplication Kernels Using the Scalable Matrix Extension* | SC-W 2024, arXiv 2409.18779 | M4 SME > 2.3 FP32 TFLOPS. The CPU path is a strong baseline. | abstract |
| Corrochano López, García Sánchez — *Evaluation of Domain-Specific Architectures … in Apple Silicon* | arXiv 2511.13450 | ANE GEMM 3.8 TFLOPS vs GPU 4.7 on M4 Pro at ~5 W vs ~24 W | verified |
| Kumaresan — *Rigel: Reverse-Engineering the Metal 4.1 Tensor Compute Path on M4 Max* | arXiv 2606.12765 | Metal 4.1 `matmul2d` runs on shader cores; FP8 emulated | verified |
| Bergach — *Beating vDSP: 138 GFLOPS FFT on Apple Silicon* | arXiv 2603.27569 | Threadgroup-memory access patterns are the bottleneck | verified |
| Gallego — *Metal-Sci* | arXiv 2605.09708 (DL4C @ ICML '26) | Metal scientific-kernel benchmark with roofline scoring | verified |
| *Residual GPU Cache State on Apple M4 Pro* | arXiv 2606.27098 | First CPU pass 31–63% slower after a 64–512 MiB GPU footprint. **Must-cite for zero-copy cost.** | unverified |
| *Characterizing WebGPU Dispatch Overhead* | arXiv 2604.02344 | ~71 µs Metal dispatch via wgpu on M2 | snippet |
| *When Quantization Is Free…* | arXiv 2605.05699 | Accelerate sends sgemm to AMX only above tile size | snippet |

Non-peer-reviewed references: corsix/amx (AMX reverse-engineering), philipturner/metal-benchmarks, tzakharko/m4-sme-exploration, Asahi GPU blog posts.

## 3. ML frameworks on Apple Silicon

| Work | Venue | Finding | Status |
|---|---|---|---|
| Feng, Xu, Wang, Lin — *Profiling Apple Silicon Performance for ML Training* | arXiv 2501.14925 | Gap to NVIDIA attributed to page faults, **kernel launch time**, power, BLAS; questions UMA benefit | verified |
| Rajesh et al. — *Production-Grade Local LLM Inference on Apple Silicon* (MLX, MLC-LLM, Ollama, llama.cpp, MPS) | arXiv 2511.05502 | MLX best sustained throughput; MPS memory-constrained | verified |
| Ajayi, Odunayo — *Benchmarking On-Device ML on Apple Silicon with MLX* | arXiv 2510.18921 | MLX vs PyTorch on encoder transformers | abstract |
| Rathnayaka et al. — *BaseRT* (native Metal LLM runtime) | arXiv 2607.00501; M5 follow-up 2607.19438 | Kernel fusion + unified-memory-aware dispatch; 1.35× decode vs MLX | verified |
| Benazir, Lin — *NPUMoE* | arXiv 2604.18788 | Dispatch/sync overhead of many small kernels is a key cost | verified |
| Javat, Kazakov — *Silicon Showdown* | arXiv 2605.00519 | Apple vs RTX 5090 tokens/J via MLX | verified |

**What MLX already does.** Checked against the docs and `mlx/backend/metal/allocator.cpp`:
- All buffers are `StorageModeShared | HazardTrackingModeUntracked`, and MLX does its own hazard tracking.
- External memory is wrapped no-copy.
- There is a page-granular `BufferCache`, a small-allocation `MTLHeap`, and residency sets tied to `set_wired_limit`.
- Evaluation is lazy, with explicit or implicit `eval()`.
- Operations are placed on CPU or GPU per stream, chosen manually by the user.

There is no MLX paper; cite it as software (Hannun et al., 2023).

Open MLX issues relevant to Track C **(unverified details)**:
- #3461: buffer freed while still in use under untracked mode
- #3350: pool growth despite the cache limit
- #3539: residency-set OOM

## 4. CPU–GPU co-execution on integrated / unified-memory SoCs

| Work | Venue | Relevance |
|---|---|---|
| Luk, Hong, Kim — *Qilin* | MICRO '09 | First adaptive, input-size-aware CPU/GPU split |
| Augonnet et al. — *StarPU* | Euro-Par '09 | Task runtime with history-based performance models |
| Topcuoglu et al. — *HEFT* | IEEE TPDS 2002 | Standard DAG placement baseline |
| Grewe, O'Boyle — static OpenCL partitioning | CC 2011 | ML-predicted CPU/GPU split |
| Kaleem et al. — *Adaptive Heterogeneous Scheduling for Integrated GPUs* | PACT '14 | Online profiling on integrated GPUs, within 3.2% of oracle |
| He, Lu, He — co-processing hash joins on coupled CPU-GPU | PVLDB '13 | Fine-grained split needed on shared-memory chips |
| Zhang et al. — co-running on integrated CPU/GPU | IEEE TPDS ~2017 | Co-running often loses because of shared bandwidth |
| Kim et al. — *μLayer* | EuroSys '19 | Channel-wise layer split across mobile CPU + GPU |
| Jiang et al. — *MNN* | MLSys '20 | Online pre-inference cost evaluation per backend |
| Zhang et al. — *nn-Meter* | MobiSys '21 (best paper) | Op-level latency prediction fails under fusion; predicts per kernel instead |
| Jia et al. — *CoDL* | MobiSys '22 | Data sharing + partition cost dominate; latency predictor; up to 4.93× |
| Jeong et al. — *Band* | MobiSys '22 | Multi-DNN subgraph scheduling across processors |
| Dagli, Belviranli — *HaX-CoNN* | PPoPP '24 | Shared-DRAM-contention-aware layer mapping |
| Li, Paolieri, Golubchik — fine-grained CPU-GPU co-execution | EPEW '25, arXiv 2510.21081 | OpenCL fine-grained SVM to reduce sync cost |
| Li, Paolieri, Golubchik — *Partition-Aware Scheduling* | arXiv 2609.14213 | Joint partition/assignment on shared-memory mobile SoCs |
| Khadem et al. — *Vector-Processing for Mobile Devices* | IISWC '23 | CPU vector units win on low-op-count kernels |
| Chen et al. — *HeteroInfer* | SOSP '25 | GPU–NPU sync ~400 µs, comparable to a kernel; fast UMA sync |
| Mohite — *FusionML* | arXiv 2607.22785, 2026 | **Closest prior work.** Apple CPU+GPU matmul split; MLX lazy graph serializes cross-stream work; decode bandwidth-bound |

**Zero-copy is not free (other platforms):**
- Otterness et al. (RTAS '17): Jetson TX1 zero-copy gives no performance benefit.
- Wang et al. (HotEdge '20): hidden unified-memory latency on Jetson.
- Wahlgren et al. (arXiv 2508.12743): AMD MI300A unified memory.
- NVIDIA *CUDA for Tegra* app note: cacheability differs by allocator.

## 5. Memory planning in ML runtimes

| System / paper | Mechanism |
|---|---|
| XLA buffer assignment | HLO live ranges, heap simulator (decreasing-size best fit) |
| TFLite ArenaPlanner | First/last-use intervals, best-fit, 64 B alignment |
| TVM Relax `StaticPlanBlockMemory` | Liveness-based storage-token reuse |
| ggml `gallocr` | Worst-case graph reservation, in-buffer reuse |
| PyTorch CUDA caching allocator | Pools, stream-aware frees (`record_stream`), expandable segments |
| ONNX Runtime | Memory-pattern record and replay, arena |
| Pisarchyk & Lee, arXiv 2001.03288 | Greedy-by-size/breadth sharing strategies |
| Checkmate (MLSys '20), MODeL (ICML '23), ROAM (arXiv 2310.19295) | Rematerialization and joint ordering/layout |
| TelaMalloc (ASPLOS '23), MiniMalloc (unverified venue) | Solver-based offline allocation |

Liveness-based reuse for static graphs is **solved engineering**. None of these planners model a buffer whose reuse depends on completion
on *two* devices that share pages. That is the Track C hypothesis.

## 6. Venue notes (verify CFPs)

| Venue | Format | Fit |
|---|---|---|
| IISWC short-paper track | 4 pp. incl. refs, archival | Strong, built for characterization |
| IEEE Computer Architecture Letters | 4 pp. incl. refs | Good for one sharp finding |
| EuroMLSys | 6 pp. | Good for engine + dispatcher |
| SEC Edge Intelligence workshop | 6 pp. | Good for edge framing |
| ISPASS | Full paper | Later, with open benchmark suite |
| SEC main track, tinyML/EDGE AI, IEEE ESL | — | Weak fit (full edge system, MCU focus, or "embedded" framing) |

SEC acceptance: 2018 ~36%, 2019 ~34%, 2020 ~22% (CORE submission data). Other rates were not found.

## 7. Must-read list

1. FusionML, arXiv 2607.22785
2. Beischl et al., DaMoN '26
3. CoDL, MobiSys '22
4. Hübner et al., arXiv 2502.05317
5. Feng et al., arXiv 2501.14925
6. Residual GPU Cache State, arXiv 2606.27098
7. MLX `mlx/backend/metal/allocator.cpp`
