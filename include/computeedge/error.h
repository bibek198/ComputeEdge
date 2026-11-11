#pragma once

#include <stdexcept>
#include <string>

namespace computeedge {

// Thrown for invalid shapes, out-of-range indices and other API misuse.
class Error : public std::runtime_error {
 public:
  explicit Error(const std::string& what) : std::runtime_error(what) {}
};

}  // namespace computeedge
