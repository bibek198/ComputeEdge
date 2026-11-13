// Environment recording (Objective-C++): hardware, OS, power and thermal state.

#import <Foundation/Foundation.h>
#include <IOKit/ps/IOPSKeys.h>
#include <IOKit/ps/IOPowerSources.h>
#import <Metal/Metal.h>
#include <sys/sysctl.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <thread>

#include "computeedge/bench/environment.h"

#ifndef CE_GIT_SHA
#define CE_GIT_SHA "unknown"
#endif
#ifndef CE_BUILD_TYPE
#define CE_BUILD_TYPE "unknown"
#endif
#ifndef CE_CXX_FLAGS
#define CE_CXX_FLAGS "unknown"
#endif
#ifndef CE_SANITIZE_ENABLED
#define CE_SANITIZE_ENABLED 0
#endif

namespace computeedge::bench {

namespace {

std::string sysctl_string(const char* name) {
  size_t len = 0;
  if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) return "";
  std::string value(len, '\0');
  if (sysctlbyname(name, value.data(), &len, nullptr, 0) != 0) return "";
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value;
}

template <typename T>
T sysctl_value(const char* name, T fallback) {
  T value{};
  size_t len = sizeof(value);
  if (sysctlbyname(name, &value, &len, nullptr, 0) != 0) return fallback;
  return value;
}

std::string power_source() {
  CFTypeRef info = IOPSCopyPowerSourcesInfo();
  if (!info) return "unknown";
  std::string result = "unknown";
  if (CFStringRef type = IOPSGetProvidingPowerSourceType(info)) {
    char buf[64];
    if (CFStringGetCString(type, buf, sizeof(buf), kCFStringEncodingUTF8)) result = buf;
  }
  CFRelease(info);
  return result;
}

std::string utc_timestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&now, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

std::string json_escape(const std::string& s) {
  std::ostringstream os;
  for (char c : s) {
    switch (c) {
      case '"': os << "\\\""; break;
      case '\\': os << "\\\\"; break;
      case '\n': os << "\\n"; break;
      case '\t': os << "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          os << buf;
        } else {
          os << c;
        }
    }
  }
  return os.str();
}

}  // namespace

const char* to_string(ThermalState state) {
  switch (state) {
    case ThermalState::Nominal: return "nominal";
    case ThermalState::Fair: return "fair";
    case ThermalState::Serious: return "serious";
    case ThermalState::Critical: return "critical";
    case ThermalState::Unknown: return "unknown";
  }
  return "unknown";
}

ThermalState thermal_state() {
  @autoreleasepool {
    switch ([[NSProcessInfo processInfo] thermalState]) {
      case NSProcessInfoThermalStateNominal: return ThermalState::Nominal;
      case NSProcessInfoThermalStateFair: return ThermalState::Fair;
      case NSProcessInfoThermalStateSerious: return ThermalState::Serious;
      case NSProcessInfoThermalStateCritical: return ThermalState::Critical;
    }
    return ThermalState::Unknown;
  }
}

bool wait_for_thermal(ThermalState target, double timeout_s, double poll_s) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s);
  while (true) {
    const ThermalState s = thermal_state();
    if (s != ThermalState::Unknown && static_cast<int>(s) <= static_cast<int>(target)) return true;
    if (std::chrono::steady_clock::now() >= deadline) return false;
    std::this_thread::sleep_for(std::chrono::duration<double>(poll_s));
  }
}

std::string current_power_source() { return power_source(); }

Environment record_environment() {
  Environment env;
  env.timestamp_utc = utc_timestamp();
  env.model = sysctl_string("hw.model");
  env.chip = sysctl_string("machdep.cpu.brand_string");
  env.performance_cores = sysctl_value<int>("hw.perflevel0.physicalcpu", 0);
  env.efficiency_cores = sysctl_value<int>("hw.perflevel1.physicalcpu", 0);
  env.memory_bytes = sysctl_value<uint64_t>("hw.memsize", 0);
  env.os_build = sysctl_string("kern.osversion");
  @autoreleasepool {
    NSProcessInfo* info = [NSProcessInfo processInfo];
    NSOperatingSystemVersion v = [info operatingSystemVersion];
    env.os_version =
        std::to_string(v.majorVersion) + "." + std::to_string(v.minorVersion) + "." + std::to_string(v.patchVersion);
    env.low_power_mode = [info isLowPowerModeEnabled];
  }
  @autoreleasepool {
    if (id<MTLDevice> device = MTLCreateSystemDefaultDevice()) {
      env.gpu_name = [[device name] UTF8String];
      env.gpu_recommended_working_set = [device recommendedMaxWorkingSetSize];
    }
  }
  env.power_source = power_source();
  env.thermal_state = thermal_state();
#ifdef __clang_version__
  env.compiler = std::string("clang ") + __clang_version__;
#endif
  env.compiler_flags = CE_CXX_FLAGS;
  env.build_type = CE_BUILD_TYPE;
  env.git_sha = CE_GIT_SHA;
  env.sanitizers = CE_SANITIZE_ENABLED != 0;
  if (const char* t = std::getenv("VECLIB_MAXIMUM_THREADS")) env.veclib_maximum_threads = t;
  return env;
}

std::string to_json(const Environment& env) {
  std::ostringstream os;
  auto str = [&](const char* key, const std::string& value, bool last = false) {
    os << "  \"" << key << "\": \"" << json_escape(value) << "\"" << (last ? "\n" : ",\n");
  };
  os << "{\n";
  str("timestamp_utc", env.timestamp_utc);
  str("model", env.model);
  str("chip", env.chip);
  os << "  \"performance_cores\": " << env.performance_cores << ",\n";
  os << "  \"efficiency_cores\": " << env.efficiency_cores << ",\n";
  os << "  \"memory_bytes\": " << env.memory_bytes << ",\n";
  str("os_version", env.os_version);
  str("os_build", env.os_build);
  str("power_source", env.power_source);
  os << "  \"low_power_mode\": " << (env.low_power_mode ? "true" : "false") << ",\n";
  str("thermal_state", to_string(env.thermal_state));
  str("gpu_name", env.gpu_name);
  os << "  \"gpu_recommended_working_set\": " << env.gpu_recommended_working_set << ",\n";
  str("compiler", env.compiler);
  str("compiler_flags", env.compiler_flags);
  str("build_type", env.build_type);
  str("git_sha", env.git_sha);
  os << "  \"sanitizers\": " << (env.sanitizers ? "true" : "false") << ",\n";
  str("veclib_maximum_threads", env.veclib_maximum_threads, true);
  os << "}\n";
  return os.str();
}

}  // namespace computeedge::bench
