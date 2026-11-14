#pragma once

#include <stdexcept>
#include <string>

namespace computeedge {

// Thrown for invalid shapes, out-of-range indices and other API misuse.
class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& what) : std::runtime_error(what) {}
};

// Usage error (exit code 2).
class UsageError : public Error {
 public:
  explicit UsageError(const std::string& what) : Error(what) {}
};

// Invalid benchmark run (exit code 3), e.g. power source changed during run.
class InvalidRunError : public Error {
 public:
  explicit InvalidRunError(const std::string& what) : Error(what) {}
};

}  // namespace computeedge
