#pragma once

// The benchmark environment: hardware, OS, power and thermal state, build
// provenance. Recorded once per run (env.json). Uses Foundation / IOKit / Metal.

#include <cstdint>
#include <string>

namespace computeedge::bench {

enum class ThermalState : int8_t { Nominal = 0, Fair = 1, Serious = 2, Critical = 3, Unknown = -1 };
const char* to_string(ThermalState state);

ThermalState thermal_state();

// Polls until the thermal state is at or below `target`, or the timeout expires.
// Returns true if the target was reached.
bool wait_for_thermal(ThermalState target, double timeout_s, double poll_s = 1.0);

struct Environment {
  std::string timestamp_utc;  // ISO 8601
  std::string model;          // hw.model, e.g. Mac16,12
  std::string chip;           // machdep.cpu.brand_string
  int performance_cores = 0;
  int efficiency_cores = 0;
  uint64_t memory_bytes = 0;
  std::string os_version;    // e.g. 26.4.1
  std::string os_build;      // kern.osversion
  std::string power_source;  // "AC Power", "Battery Power", or "unknown"
  bool low_power_mode = false;
  ThermalState thermal_state = ThermalState::Unknown;
  std::string gpu_name;                      // MTLDevice.name
  uint64_t gpu_recommended_working_set = 0;  // MTLDevice.recommendedMaxWorkingSetSize
  std::string compiler;
  std::string compiler_flags;  // CMAKE_CXX_FLAGS + build-type flags
  std::string build_type;
  std::string git_sha;  // captured at CMake configure time
  bool sanitizers = false;
  std::string veclib_maximum_threads;  // VECLIB_MAXIMUM_THREADS, empty if unset
};

Environment record_environment();

// "AC Power", "Battery Power" or "unknown" right now (cheap; for per-cycle checks
// in long runs, which are invalid if the source changes mid-run).
std::string current_power_source();
std::string to_json(const Environment& env);

}  // namespace computeedge::bench
