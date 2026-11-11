#pragma once

#include <cstdint>

#include "computeedge/instrumentation.h"

namespace computeedge {

// RAII timer. On construction it emits a Begin event; on destruction it emits End
// and, if a sink is given, writes the elapsed nanoseconds into it.
//
//   uint64_t ns = 0;
//   { ScopedTimer t("matmul", &ns); ... }
class ScopedTimer {
 public:
  explicit ScopedTimer(const char* label, uint64_t* sink_ns = nullptr, uint64_t exec_id = 0,
                       Backend backend = Backend::None)
      : label_(label), sink_(sink_ns), exec_id_(exec_id), backend_(backend) {
    emit(EventType::Begin, exec_id_, 0, backend_, exec_id_, label_);
    start_ = now_ns();
  }

  ~ScopedTimer() {
    const uint64_t end = now_ns();
    if (sink_) *sink_ = end - start_;
    emit(EventType::End, exec_id_, 0, backend_, exec_id_, label_);
  }

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

  uint64_t elapsed_ns() const { return now_ns() - start_; }

 private:
  const char* label_;
  uint64_t* sink_;
  uint64_t exec_id_;
  Backend backend_;
  uint64_t start_ = 0;
};

}  // namespace computeedge
