#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "computeedge/tensor.h"

using namespace computeedge;

namespace {

Tensor random_tensor(Shape shape, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(shape_numel(shape));
  for (float& x : v) x = dist(rng);
  return Tensor(std::move(shape), v);
}

// Double-precision reference on logical (row-major) element order.
std::vector<double> reference_matmul(const Tensor& a, const Tensor& b) {
  const size_t M = a.shape()[0], K = a.shape()[1], N = b.shape()[1];
  const std::vector<float> A = a.to_vector(), B = b.to_vector();
  std::vector<double> C(M * N, 0.0);
  for (size_t i = 0; i < M; ++i)
    for (size_t k = 0; k < K; ++k)
      for (size_t j = 0; j < N; ++j) C[i * N + j] += double(A[i * K + k]) * double(B[k * N + j]);
  return C;
}

void expect_close(const Tensor& got, const std::vector<double>& ref, double rel_tol = 1e-4) {
  const std::vector<float> g = got.to_vector();
  ASSERT_EQ(g.size(), ref.size());
  for (size_t i = 0; i < g.size(); ++i) {
    const double err = std::abs(double(g[i]) - ref[i]);
    ASSERT_LE(err, rel_tol * (std::abs(ref[i]) + 1.0)) << "at flat index " << i;
  }
}

}  // namespace

TEST(Matmul, TwoByTwo) {
  Tensor a({2, 2}, {1, 2, 3, 4});
  Tensor b({2, 2}, {5, 6, 7, 8});
  Tensor c = matmul(a, b);
  EXPECT_EQ(c.shape(), (Shape{2, 2}));
  EXPECT_EQ(c.to_vector(), (std::vector<float>{19, 22, 43, 50}));
}

TEST(Matmul, RectangularRandomMatchesReference) {
  Tensor a = random_tensor({64, 128}, 1);
  Tensor b = random_tensor({128, 96}, 2);
  Tensor c = matmul(a, b);
  EXPECT_EQ(c.shape(), (Shape{64, 96}));
  expect_close(c, reference_matmul(a, b));
}

TEST(Matmul, NonContiguousInputs) {
  Tensor a = random_tensor({32, 16}, 3).transpose();  // 16 x 32 view
  Tensor b = random_tensor({8, 32}, 4).transpose();   // 32 x 8 view
  ASSERT_FALSE(a.is_contiguous());
  Tensor c = matmul(a, b);
  EXPECT_EQ(c.shape(), (Shape{16, 8}));
  expect_close(c, reference_matmul(a, b));
}

TEST(Matmul, IdentityLeavesMatrixUnchanged) {
  Tensor a = random_tensor({5, 5}, 5);
  Tensor eye({5, 5});
  for (size_t i = 0; i < 5; ++i) eye.at({i, i}) = 1.0f;
  EXPECT_EQ(matmul(a, eye).to_vector(), a.to_vector());
}

TEST(Matmul, VectorShapes) {
  Tensor row({1, 3}, {1, 2, 3});
  Tensor col({3, 1}, {4, 5, 6});
  EXPECT_EQ(matmul(row, col).to_vector(), (std::vector<float>{32}));
  EXPECT_EQ(matmul(col, row).shape(), (Shape{3, 3}));
}

TEST(Matmul, ShapeErrors) {
  Tensor a({2, 3});
  Tensor b({4, 2});
  EXPECT_THROW(matmul(a, b), Error);
  EXPECT_THROW(matmul(Tensor({2, 2, 2}), Tensor({2, 2})), Error);
  EXPECT_THROW(matmul(Tensor(), Tensor({2, 2})), Error);
}

TEST(Matmul, OutputUsesInputAllocator) {
  HostAllocator alloc;
  Tensor a({4, 4}, alloc);
  Tensor b({4, 4}, alloc);
  const uint64_t before = alloc.stats().alloc_count;
  Tensor c = matmul(a, b);
  EXPECT_EQ(&c.storage()->allocator(), &alloc);
  EXPECT_EQ(alloc.stats().alloc_count, before + 1) << "contiguous inputs must not be copied";
}
