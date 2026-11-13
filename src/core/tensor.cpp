#include "computeedge/core/tensor.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

namespace computeedge {

namespace {

std::string shape_str(const Shape& shape) {
  std::ostringstream os;
  os << '[';
  for (size_t i = 0; i < shape.size(); ++i) os << (i ? ", " : "") << shape[i];
  os << ']';
  return os.str();
}

}  // namespace

size_t shape_numel(const Shape& shape) {
  size_t n = 1;
  for (size_t d : shape) {
    if (d != 0 && n > std::numeric_limits<size_t>::max() / d) throw Error("shape too large: " + shape_str(shape));
    n *= d;
  }
  return n;
}

Strides contiguous_strides(const Shape& shape) {
  Strides strides(shape.size(), 1);
  size_t acc = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    strides[i] = acc;
    acc *= shape[i];
  }
  return strides;
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

Storage::Storage(size_t bytes, Allocator& allocator) : block_(allocator.allocate(bytes)), allocator_(&allocator) {}

Storage::~Storage() { allocator_->deallocate(block_); }

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

Tensor::Tensor(Shape shape, Allocator& allocator)
    : storage_(std::make_shared<Storage>(shape_numel(shape) * dtype_size(DType::Float32), allocator)),
      shape_(std::move(shape)),
      strides_(contiguous_strides(shape_)) {
  std::memset(storage_->data(), 0, nbytes());
}

Tensor::Tensor(Shape shape, std::initializer_list<float> values, Allocator& allocator)
    : Tensor(std::move(shape), allocator) {
  fill_from(values.begin(), values.size());
}

Tensor::Tensor(Shape shape, const std::vector<float>& values, Allocator& allocator)
    : Tensor(std::move(shape), allocator) {
  fill_from(values.data(), values.size());
}

Tensor::Tensor(std::shared_ptr<Storage> storage, Shape shape, Strides strides, size_t offset)
    : storage_(std::move(storage)), shape_(std::move(shape)), strides_(std::move(strides)), offset_(offset) {}

Tensor Tensor::zeros(Shape shape, Allocator& allocator) { return Tensor(std::move(shape), allocator); }

Tensor Tensor::full(Shape shape, float value, Allocator& allocator) {
  Tensor t(std::move(shape), allocator);
  std::fill_n(t.data(), t.numel(), value);
  return t;
}

void Tensor::fill_from(const float* values, size_t count) {
  if (count != numel()) {
    throw Error("Tensor: " + std::to_string(count) + " values given for shape " + shape_str(shape_) + " (" +
                std::to_string(numel()) + " elements)");
  }
  if (count) std::memcpy(data(), values, count * sizeof(float));
}

bool Tensor::is_contiguous() const {
  size_t expected = 1;
  for (size_t i = shape_.size(); i-- > 0;) {
    if (shape_[i] == 1) continue;
    if (strides_[i] != expected) return false;
    expected *= shape_[i];
  }
  return true;
}

float* Tensor::data() {
  if (!storage_) throw Error("Tensor: undefined tensor");
  return static_cast<float*>(storage_->data()) + offset_;
}

const float* Tensor::data() const {
  if (!storage_) throw Error("Tensor: undefined tensor");
  return static_cast<const float*>(storage_->data()) + offset_;
}

size_t Tensor::linear_index(std::initializer_list<size_t> index) const {
  if (index.size() != shape_.size()) {
    throw Error("Tensor::at: expected " + std::to_string(shape_.size()) + " indices, got " +
                std::to_string(index.size()));
  }
  size_t linear = 0;
  size_t dim = 0;
  for (size_t i : index) {
    if (i >= shape_[dim]) {
      throw Error("Tensor::at: index " + std::to_string(i) + " out of range for dim " + std::to_string(dim) +
                  " of size " + std::to_string(shape_[dim]));
    }
    linear += i * strides_[dim];
    ++dim;
  }
  return linear;
}

float& Tensor::at(std::initializer_list<size_t> index) { return data()[linear_index(index)]; }

float Tensor::at(std::initializer_list<size_t> index) const { return data()[linear_index(index)]; }

Tensor Tensor::reshape(Shape new_shape) const {
  if (shape_numel(new_shape) != numel()) {
    throw Error("Tensor::reshape: cannot reshape " + shape_str(shape_) + " to " + shape_str(new_shape));
  }
  if (!is_contiguous()) return contiguous().reshape(std::move(new_shape));
  Strides strides = contiguous_strides(new_shape);
  return Tensor(storage_, std::move(new_shape), std::move(strides), offset_);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
  if (dim0 >= ndim() || dim1 >= ndim()) {
    throw Error("Tensor::transpose: dims out of range for shape " + shape_str(shape_));
  }
  Shape shape = shape_;
  Strides strides = strides_;
  std::swap(shape[dim0], shape[dim1]);
  std::swap(strides[dim0], strides[dim1]);
  return Tensor(storage_, std::move(shape), std::move(strides), offset_);
}

std::vector<float> Tensor::to_vector() const {
  const size_t n = numel();
  std::vector<float> out(n);
  if (n == 0) return out;
  if (is_contiguous()) {
    std::memcpy(out.data(), data(), n * sizeof(float));
    return out;
  }
  // Strided gather with an odometer over the logical index.
  const float* base = data();
  std::vector<size_t> idx(ndim(), 0);
  size_t src = 0;
  for (size_t k = 0; k < n; ++k) {
    out[k] = base[src];
    for (size_t d = ndim(); d-- > 0;) {
      ++idx[d];
      src += strides_[d];
      if (idx[d] < shape_[d]) break;
      src -= idx[d] * strides_[d];
      idx[d] = 0;
    }
  }
  return out;
}

Tensor Tensor::contiguous() const {
  if (is_contiguous()) return *this;
  return clone();
}

Tensor Tensor::clone() const {
  return Tensor(shape_, to_vector(), storage_ ? storage_->allocator() : default_allocator());
}

void Tensor::print(std::ostream& out) const {
  if (!storage_) {
    out << "Tensor(undefined)\n";
    return;
  }
  const std::vector<float> v = to_vector();
  if (ndim() == 2) {
    for (size_t r = 0; r < shape_[0]; ++r) {
      for (size_t c = 0; c < shape_[1]; ++c) out << (c ? " " : "") << v[r * shape_[1] + c];
      out << '\n';
    }
    return;
  }
  out << "Tensor" << shape_str(shape_) << ":";
  for (float x : v) out << ' ' << x;
  out << '\n';
}

void Tensor::print() const { print(std::cout); }

}  // namespace computeedge
