#include "computeedge/core/instrumentation.h"

#include <time.h>

#include <fstream>
#include <ostream>

#include "computeedge/core/error.h"

namespace computeedge {

const char* to_string(EventType type) {
  switch (type) {
    case EventType::Alloc: return "alloc";
    case EventType::Free: return "free";
    case EventType::Dispatch: return "dispatch";
    case EventType::Submit: return "submit";
    case EventType::Sync: return "sync";
    case EventType::Begin: return "begin";
    case EventType::End: return "end";
    case EventType::Reuse: return "reuse";
  }
  return "unknown";
}

const char* to_string(Backend backend) {
  switch (backend) {
    case Backend::None: return "none";
    case Backend::CPU: return "cpu";
    case Backend::Metal: return "metal";
  }
  return "unknown";
}

uint64_t now_ns() { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

uint64_t next_exec_id() {
  static std::atomic<uint64_t> counter{0};
  return counter.fetch_add(1, std::memory_order_relaxed) + 1;
}

EventLog& EventLog::global() {
  static EventLog log;
  return log;
}

void EventLog::record(const Event& event) {
  std::lock_guard<std::mutex> lock(mutex_);
  events_.push_back(event);
}

void EventLog::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  events_.clear();
}

std::vector<Event> EventLog::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return events_;
}

size_t EventLog::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return events_.size();
}

void EventLog::write_csv(std::ostream& out) const {
  const std::vector<Event> events = snapshot();
  out << "t_ns,type,id,exec_id,bytes,backend,label\n";
  for (const Event& e : events) {
    out << e.t_ns << ',' << to_string(e.type) << ',' << e.id << ',' << e.exec_id << ',' << e.bytes << ','
        << to_string(e.backend) << ',' << (e.label ? e.label : "") << '\n';
  }
}

void EventLog::write_csv(const std::string& path) const {
  std::ofstream file(path);
  if (!file) throw Error("EventLog: cannot open " + path);
  write_csv(file);
}

ScopedEventLogging::ScopedEventLogging(bool clear_first) : previous_(EventLog::global().enabled()) {
  if (clear_first) EventLog::global().clear();
  EventLog::global().set_enabled(true);
}

ScopedEventLogging::~ScopedEventLogging() { EventLog::global().set_enabled(previous_); }

}  // namespace computeedge
