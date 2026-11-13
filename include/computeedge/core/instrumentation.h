#pragma once

// Event logging for ComputeEdge.
//
// Every allocation, free, dispatch and synchronization point can emit an Event.
// The log is process-global and disabled by default; when disabled, emit() costs
// a single relaxed atomic load. Timestamps use CLOCK_UPTIME_RAW (nanoseconds),
// the same timebase that Metal GPU timestamps can later be correlated against.

#include <atomic>
#include <cstdint>
#include <iosfwd>
#include <mutex>
#include <string>
#include <vector>

namespace computeedge {

enum class EventType : uint8_t { Alloc, Free, Dispatch, Submit, Sync, Begin, End, Reuse };
enum class Backend : uint8_t { None, CPU, Metal };

const char* to_string(EventType type);
const char* to_string(Backend backend);

// Monotonic nanoseconds (CLOCK_UPTIME_RAW).
uint64_t now_ns();

// Returns a new, process-unique execution id (never 0).
uint64_t next_exec_id();

struct Event {
  uint64_t t_ns = 0;
  EventType type = EventType::Begin;
  uint64_t id = 0;       // buffer id or operation id
  uint64_t exec_id = 0;  // groups the events of one operation (0 = none)
  uint64_t bytes = 0;    // where applicable
  Backend backend = Backend::None;
  const char* label = nullptr;  // must point to a string with static storage duration
};

class EventLog {
 public:
  static EventLog& global();

  void set_enabled(bool enabled) { enabled_.store(enabled, std::memory_order_relaxed); }
  bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

  void record(const Event& event);
  void clear();
  std::vector<Event> snapshot() const;
  size_t size() const;

  // CSV columns: t_ns,type,id,exec_id,bytes,backend,label
  void write_csv(std::ostream& out) const;
  void write_csv(const std::string& path) const;

 private:
  std::atomic<bool> enabled_{false};
  mutable std::mutex mutex_;
  std::vector<Event> events_;
};

// Fast-path helper: records only when the global log is enabled.
inline void emit(EventType type, uint64_t id, uint64_t bytes = 0, Backend backend = Backend::None, uint64_t exec_id = 0,
                 const char* label = nullptr) {
  EventLog& log = EventLog::global();
  if (!log.enabled()) return;
  log.record(Event{now_ns(), type, id, exec_id, bytes, backend, label});
}

// RAII: enables the global log for a scope and restores the previous state.
class ScopedEventLogging {
 public:
  explicit ScopedEventLogging(bool clear_first = true);
  ~ScopedEventLogging();
  ScopedEventLogging(const ScopedEventLogging&) = delete;
  ScopedEventLogging& operator=(const ScopedEventLogging&) = delete;

 private:
  bool previous_;
};

}  // namespace computeedge
