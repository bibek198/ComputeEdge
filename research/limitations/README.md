# Hardware Scope and Experimental Limitations

## 1. Physical Hardware Scope

All primary characterization measurements in ComputeEdge are conducted on:
- **Reference Device:** Apple Silicon MacBook Air (M4, 2024).
- **SoC Architecture:** Apple M4 (4 Performance cores, 6 Efficiency cores, 10-core GPU, 16-core Neural Engine).
- **Memory Subsystem:** 16 GB unified LPDDR5X memory (120 GB/s bandwidth).
- **Thermal Form Factor:** Fanless, passive thermal dissipation.

---

## 2. Experimental Boundaries & Control Conditions

1. **Power Controls:**
   - Benchmarks require active AC power (`pmset -g batt | grep 'AC Power'`).
   - Battery operation induces dynamic power throttling and alters CPU/GPU clock frequencies.
2. **Thermal State Controls:**
   - Baseline measurements enforce `--cooldown`, polling `NSProcessInfo.thermalState` until the system returns to `nominal` (state 0).
   - In sustained runs (`thermal_drift`), frequency throttling occurs prior to the kernel reporting non-nominal thermal state, necessitating direct hardware metric sampling via `powermetrics`.
3. **OS-Level Interference:**
   - Background operating system daemons and window compositors (WindowServer) share unified memory bandwidth.
   - Sessions sample background CPU load (`ps -A -o %cpu=`) and record environment snapshots (`env.json`) for every run to monitor background interference.

---

## 3. Generalizability Constraints

- **Active Cooling Differences:** Active-cooled systems (MacBook Pro, Mac mini, Mac Studio) exhibit higher sustained thermal thresholds and different crossover points than fanless devices.
- **Microarchitectural Differences:** M-series Pro/Max/Ultra configurations possess wider memory buses (up to 800+ GB/s) and larger System-Level Caches (SLC), which will shift the exact latency and bandwidth tipping points observed on base M4.
