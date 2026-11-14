# ADR 0005: Harness Error Handling, Early CLI Validation, and Timeout Safeguards

## Status
Accepted

## Context
Long-running hardware benchmark sessions on Apple Silicon are subject to environmental interruptions (e.g. loss of AC power, thermal state throttling, memory pressure) and edge cases in hardware synchronization (e.g. GPU deadlocks or hanging command buffer completions). Previously:
- Unexpected exceptions could terminate benchmarks without saving partial progress.
- Polling loops on GPU command buffers lacked timeouts, risking infinite hangs.
- Unchecked file output streams could fail silently when disk space or permissions failed.
- Invalid CLI arguments were parsed or rejected mid-execution after partial runs.

## Decision
1. **Standard `run_main()` Entry-Point Wrapper:**
   - Centralized top-level exception handling across all 9 benchmark binaries via `benchmarks/harness/run_main.h`.
   - Distinct, standardized process exit codes:
     - `0`: Success.
     - `1`: Runtime or hardware error (`computeedge::Error`, `std::exception`).
     - `2`: CLI usage / argument error (`computeedge::bench::harness::UsageError`).
     - `3`: Invalid benchmark run (`computeedge::bench::harness::InvalidRunError`, e.g. loss of AC power).
   - On runtime errors or invalid runs, partial results are flushed to disk and marked with a `PARTIAL` marker file.
2. **Early Option Validation:**
   - All benchmark CLI options, implementation modes, storage modes, and sweep arguments are validated in `options.cpp` prior to invoking warm-up, pre-warm, or measurement loops.
3. **Bounded GPU Synchronization Timeouts:**
   - Bounded timeouts (`kSyncTimeoutNs = 10s`) implemented for `SpinStatus` and `CompletionHandler` synchronization in Metal probes, preventing deadlocks or indefinite hanging.
4. **Thermal Drift Underflow Guards:**
   - Sleep intervals in sustained thermal runs are guarded against arithmetic underflow when cycle execution time exceeds the target interval.
5. **Checked Output Streams and Metal Guards:**
   - `ResultWriter` explicitly checks all `std::ofstream` operations for write failures (`badbit`, `failbit`).
   - Guarded Objective-C `UTF8String` conversions against `nil` strings and verified Metal devices, buffers, and command queues before dereference.

## Consequences
- Benchmark failures produce clean diagnostic messages, correct non-zero exit codes, and clearly marked partial artifacts.
- Long-running sweeps cannot hang indefinitely on GPU faults.
- Invalid arguments are caught immediately at zero cost before running hardware measurements.
