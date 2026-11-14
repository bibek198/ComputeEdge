#pragma once

// Command-line handling shared by the benchmark binaries.
//
// Each binary enables exactly the common flags it accepts (CommonFlag), so the
// set of accepted flags and their defaults are a per-binary contract:
//   --warmup N  --min-warmup-ms MS  --prewarm-ms MS  --reps N  --out DIR  --cooldown  --no-write
// Binary-specific flags are handled by an `extra` callback.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#include "computeedge/bench/runner.h"

namespace computeedge::bench::harness {

// Defaults are the pre-refactor values of every benchmark (docs/refactoring_verification.md).
struct CommonOptions {
  int warmup = 10;
  double min_warmup_ms = 200;
  double prewarm_ms = 1000;
  int reps = 30;
  std::string out = "data/scratch";
  bool cooldown = false;
  bool write = true;
};

enum CommonFlag : uint32_t {
  kWarmup = 1u << 0,
  kMinWarmupMs = 1u << 1,
  kPrewarmMs = 1u << 2,
  kReps = 1u << 3,
  kOut = 1u << 4,
  kCooldown = 1u << 5,
  kNoWrite = 1u << 6,
};
inline constexpr uint32_t kAllCommonFlags = kWarmup | kMinWarmupMs | kPrewarmMs | kReps | kOut | kCooldown | kNoWrite;

// Prints "error: <msg>" and a pointer to the binary's usage header, then exits with status 2.
[[noreturn]] void usage(const std::string& source, const std::string& msg);

// `extra(flag, next)` handles a binary-specific flag; returns false if `flag` is unknown.
using ExtraFlags = std::function<bool(const std::string& flag, const std::function<std::string()>& next)>;

// Parses argv. `source` is the binary's source path (shown by usage()); `flags`
// selects the common flags this binary accepts.
CommonOptions parse(int argc, char** argv, const std::string& source, uint32_t flags, const ExtraFlags& extra);

// The runner configuration implied by the common options.
Config config(const CommonOptions& o);

// "a,b,,c" -> {"a", "b", "c"}.
std::vector<std::string> split(const std::string& s);
// "16K,4M,1G" -> bytes (binary units).
std::vector<size_t> split_bytes(const std::string& s);
std::vector<double> split_doubles(const std::string& s);
// 65536 -> "64K", 3 * 2^30 -> "3G" (exact multiples only), otherwise the byte count.
std::string bytes_label(size_t bytes);
// Whether `name` is in `list`.
bool wants(const std::vector<std::string>& list, const std::string& name);

// Maps a string to the enumerator whose to_string() it equals; calls
// usage(source, "unknown <what> <s>") otherwise.
template <typename Enum>
Enum parse_enum(const std::string& s, std::initializer_list<Enum> values, const std::string& what,
                const std::string& source) {
  for (Enum e : values)
    if (s == to_string(e)) return e;
  usage(source, "unknown " + what + " " + s);
}

}  // namespace computeedge::bench::harness
