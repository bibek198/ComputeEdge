#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "computeedge/core/allocator.h"
#include "computeedge/core/error.h"

namespace computeedge {

enum class DType : uint8_t { Float32 };

constexpr size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::Float32: return 4;
  }
  return 0;
}

using Shape = std::vector<size_t>;
using Strides = std::vector<size_t>;  // in elements, not bytes

size_t shape_numel(const Shape& shape);
Strides contiguous_strides(const Shape& shape);

// An owned block of memory. Freed (and a Free event logged) when the last
// Tensor referring to it goes away.
class Storage {
 public:
  Storage(size_t bytes, Allocator& allocator);
  ~Storage();
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  void* data() const { return block_.ptr; }
  size_t bytes() const { return block_.bytes; }
  size_t requested_bytes() const { return block_.requested; }
  uint64_t id() const { return block_.id; }
  Allocator& allocator() const { return *allocator_; }

 private:
  Block block_;
  Allocator* allocator_;
};

// An N-dimensional FP32 view onto a Storage. Copying a Tensor is cheap and
// shares storage; views (reshape, transpose) share storage too.
class Tensor {
 public:
  Tensor() = default;

  // Zero-initialised tensor.
  explicit Tensor(Shape shape, Allocator& allocator = default_allocator());
  Tensor(Shape shape, std::initializer_list<float> values, Allocator& allocator = default_allocator());
  Tensor(Shape shape, const std::vector<float>& values, Allocator& allocator = default_allocator());

  static Tensor zeros(Shape shape, Allocator& allocator = default_allocator());
  static Tensor full(Shape shape, float value, Allocator& allocator = default_allocator());

  const Shape& shape() const { return shape_; }
  const Strides& strides() const { return strides_; }
  size_t ndim() const { return shape_.size(); }
  size_t numel() const { return shape_numel(shape_); }
  size_t offset() const { return offset_; }
  DType dtype() const { return DType::Float32; }
  size_t nbytes() const { return numel() * dtype_size(dtype()); }
  bool defined() const { return storage_ != nullptr; }
  bool is_contiguous() const;

  const std::shared_ptr<Storage>& storage() const { return storage_; }

  // Pointer to the first element of this view.
  float* data();
  const float* data() const;

  float& at(std::initializer_list<size_t> index);
  float at(std::initializer_list<size_t> index) const;

  Tensor reshape(Shape new_shape) const;
  Tensor transpose(size_t dim0 = 0, size_t dim1 = 1) const;
  Tensor contiguous() const;
  Tensor clone() const;

  // Elements in row-major logical order.
  std::vector<float> to_vector() const;

  void print(std::ostream& out) const;
  // The same text as print(), as a string.
  std::string to_string() const;

 private:
  Tensor(std::shared_ptr<Storage> storage, Shape shape, Strides strides, size_t offset);
  size_t linear_index(std::initializer_list<size_t> index) const;
  void fill_from(const float* values, size_t count);

  std::shared_ptr<Storage> storage_;
  Shape shape_;
  Strides strides_;
  size_t offset_ = 0;
};

// Writes t.print(out) to the stream.
std::ostream& operator<<(std::ostream& out, const Tensor& t);

}  // namespace computeedge
