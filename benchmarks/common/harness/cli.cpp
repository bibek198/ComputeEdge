#include "harness/cli.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>

namespace computeedge::bench::harness {

void usage(const std::string& source, const std::string& msg) {
  std::fprintf(stderr, "error: %s\nsee the header of %s for usage\n", msg.c_str(), source.c_str());
  std::exit(2);
}

CommonOptions parse(int argc, char** argv, const std::string& source, uint32_t flags, const ExtraFlags& extra) {
  CommonOptions o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const std::function<std::string()> next = [&]() -> std::string {
      if (i + 1 >= argc) usage(source, "missing value for " + a);
      return argv[++i];
    };
    if ((flags & kWarmup) && a == "--warmup") {
      o.warmup = std::stoi(next());
    } else if ((flags & kMinWarmupMs) && a == "--min-warmup-ms") {
      o.min_warmup_ms = std::stod(next());
    } else if ((flags & kPrewarmMs) && a == "--prewarm-ms") {
      o.prewarm_ms = std::stod(next());
    } else if ((flags & kReps) && a == "--reps") {
      o.reps = std::stoi(next());
    } else if ((flags & kOut) && a == "--out") {
      o.out = next();
    } else if ((flags & kCooldown) && a == "--cooldown") {
      o.cooldown = true;
    } else if ((flags & kNoWrite) && a == "--no-write") {
      o.write = false;
    } else if (!extra || !extra(a, next)) {
      usage(source, "unknown argument " + a);
    }
  }
  return o;
}

Config config(const CommonOptions& o) {
  Config c;
  c.warmup = o.warmup;
  c.min_warmup_ns = static_cast<uint64_t>(o.min_warmup_ms * 1e6);
  c.repetitions = o.reps;
  c.cooldown = o.cooldown;
  return c;
}

std::vector<std::string> split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');)
    if (!item.empty()) out.push_back(item);
  return out;
}

std::vector<size_t> split_bytes(const std::string& s) {
  std::vector<size_t> out;
  for (std::string item : split(s)) {
    size_t mult = 1;
    const char suffix = item.back();
    if (suffix == 'K' || suffix == 'k') mult = size_t{1} << 10;
    if (suffix == 'M' || suffix == 'm') mult = size_t{1} << 20;
    if (suffix == 'G' || suffix == 'g') mult = size_t{1} << 30;
    if (mult != 1) item.pop_back();
    out.push_back(std::stoull(item) * mult);
  }
  return out;
}

std::vector<double> split_doubles(const std::string& s) {
  std::vector<double> out;
  for (const auto& item : split(s)) out.push_back(std::stod(item));
  return out;
}

std::string bytes_label(size_t bytes) {
  if (bytes >= (size_t{1} << 30) && bytes % (size_t{1} << 30) == 0) return std::to_string(bytes >> 30) + "G";
  if (bytes >= (size_t{1} << 20) && bytes % (size_t{1} << 20) == 0) return std::to_string(bytes >> 20) + "M";
  if (bytes >= (size_t{1} << 10) && bytes % (size_t{1} << 10) == 0) return std::to_string(bytes >> 10) + "K";
  return std::to_string(bytes);
}

bool wants(const std::vector<std::string>& list, const std::string& name) {
  for (const auto& item : list)
    if (item == name) return true;
  return false;
}

}  // namespace computeedge::bench::harness
