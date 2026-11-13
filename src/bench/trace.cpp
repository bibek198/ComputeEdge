#include "computeedge/bench/trace.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace computeedge::bench {

namespace {

struct TraceSink {
  std::mutex mutex;
  std::FILE* file = nullptr;
};

TraceSink& trace_sink() {
  static TraceSink* sink = [] {
    auto* t = new TraceSink;  // intentionally leaked: lives for the process
    if (const char* path = std::getenv("CE_TRACE"); path && *path) t->file = std::fopen(path, "a");
    return t;
  }();
  return *sink;
}

}  // namespace

bool trace_enabled() { return trace_sink().file != nullptr; }

void trace(const std::string& line) {
  TraceSink& t = trace_sink();
  if (!t.file) return;
  std::lock_guard<std::mutex> lock(t.mutex);
  std::fprintf(t.file, "%s\n", line.c_str());
  std::fflush(t.file);
}

}  // namespace computeedge::bench
