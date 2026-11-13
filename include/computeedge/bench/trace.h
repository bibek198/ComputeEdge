#pragma once

// Execution trace for refactoring verification (docs/refactoring_verification.md).

#include <string>

namespace computeedge::bench {

// Inert unless the environment variable CE_TRACE names a file. When it does,
// every trace() call appends one line to that file describing *what* is executed
// (benchmark names, parameters, warm-up / calibration / repetition policies,
// pre-warm and heater workloads), never timings. Used to prove that a refactor
// leaves benchmark behaviour unchanged (docs/refactoring_verification.md).
bool trace_enabled();
void trace(const std::string& line);

}  // namespace computeedge::bench
