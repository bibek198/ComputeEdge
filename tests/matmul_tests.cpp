#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "computeedge/ops.h"

using namespace computeedge;

namespace {

Tensor random_tensor(Shape shape, uint32_t seed, Allocator& alloc = default_allocator()) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(shape_numel(shape));
  for (float& x : v) x = dist(rng);
  return Tensor(std::move(shape), v, alloc);
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

class MatmulImpls : public ::testing::TestWithParam<MatmulImpl> {};

INSTANTIATE_TEST_SUITE_P(All, MatmulImpls, ::testing::Values(MatmulImpl::Naive, MatmulImpl::Accelerate),
                         [](const auto& info) { return std::string(to_string(info.param)); });

TEST_P(MatmulImpls, TwoByTwo) {
  Tensor a({2, 2}, {1, 2, 3, 4});
  Tensor b({2, 2}, {5, 6, 7, 8});
  Tensor c = matmul(a, b, GetParam());
  EXPECT_EQ(c.shape(), (Shape{2, 2}));
  EXPECT_EQ(c.to_vector(), (std::vector<float>{19, 22, 43, 50}));
}

TEST_P(MatmulImpls, RectangularRandomMatchesReference) {
  Tensor a = random_tensor({64, 128}, 1);
  Tensor b = random_tensor({128, 96}, 2);
  Tensor c = matmul(a, b, GetParam());
  EXPECT_EQ(c.shape(), (Shape{64, 96}));
  expect_close(c, reference_matmul(a, b));
}

TEST_P(MatmulImpls, OddSizes) {
  Tensor a = random_tensor({17, 33}, 7);
  Tensor b = random_tensor({33, 5}, 8);
  expect_close(matmul(a, b, GetParam()), reference_matmul(a, b));
}

TEST_P(MatmulImpls, TransposedInputs) {
  Tensor a = random_tensor({32, 16}, 3).transpose();  // 16 x 32 view
  Tensor b = random_tensor({8, 32}, 4).transpose();   // 32 x 8 view
  ASSERT_FALSE(a.is_contiguous());
  Tensor c = matmul(a, b, GetParam());
  EXPECT_EQ(c.shape(), (Shape{16, 8}));
  expect_close(c, reference_matmul(a, b));
}

TEST_P(MatmulImpls, MixedLayouts) {
  Tensor a = random_tensor({20, 12}, 9);              // row-major 20 x 12
  Tensor b = random_tensor({7, 12}, 10).transpose();  // 12 x 7 view
  expect_close(matmul(a, b, GetParam()), reference_matmul(a, b));
}

TEST_P(MatmulImpls, IdentityLeavesMatrixUnchanged) {
  Tensor a = random_tensor({5, 5}, 5);
  Tensor eye({5, 5});
  for (size_t i = 0; i < 5; ++i) eye.at({i, i}) = 1.0f;
  EXPECT_EQ(matmul(a, eye, GetParam()).to_vector(), a.to_vector());
}

TEST_P(MatmulImpls, VectorShapes) {
  Tensor row({1, 3}, {1, 2, 3});
  Tensor col({3, 1}, {4, 5, 6});
  EXPECT_EQ(matmul(row, col, GetParam()).to_vector(), (std::vector<float>{32}));
  EXPECT_EQ(matmul(col, row, GetParam()).to_vector(), (std::vector<float>{4, 8, 12, 5, 10, 15, 6, 12, 18}));
}

TEST_P(MatmulImpls, ZeroInnerDimensionGivesZeros) {
  Tensor a({3, 0});
  Tensor b({0, 2});
  Tensor out = Tensor::full({3, 2}, 5.0f);
  matmul_into(a, b, out, GetParam());
  for (float v : out.to_vector()) EXPECT_EQ(v, 0.0f);
}

TEST_P(MatmulImpls, MatmulIntoOverwritesOutput) {
  Tensor a({2, 2}, {1, 2, 3, 4});
  Tensor b({2, 2}, {5, 6, 7, 8});
  Tensor out = Tensor::full({2, 2}, 100.0f);
  matmul_into(a, b, out, GetParam());
  EXPECT_EQ(out.to_vector(), (std::vector<float>{19, 22, 43, 50}));
}

TEST_P(MatmulImpls, MatmulIntoDoesNotAllocate) {
  HostAllocator alloc;
  Tensor a = random_tensor({8, 8}, 11, alloc);
  Tensor b = random_tensor({8, 8}, 12, alloc);
  Tensor out({8, 8}, alloc);
  const uint64_t before = alloc.stats().alloc_count;
  matmul_into(a, b, out, GetParam());
  EXPECT_EQ(alloc.stats().alloc_count, before);
}

TEST(Matmul, AccelerateMatchesNaiveOnLargerProblem) {
  Tensor a = random_tensor({200, 300}, 13);
  Tensor b = random_tensor({300, 150}, 14);
  const std::vector<float> fast = matmul(a, b, MatmulImpl::Accelerate).to_vector();
  const std::vector<float> ref = matmul(a, b, MatmulImpl::Naive).to_vector();
  for (size_t i = 0; i < ref.size(); ++i) ASSERT_NEAR(fast[i], ref[i], 1e-4 * (std::abs(ref[i]) + 1.0f));
}

TEST(Matmul, DefaultImplIsAccelerate) {
  Tensor a({2, 2}, {1, 2, 3, 4});
  Tensor b({2, 2}, {5, 6, 7, 8});
  EXPECT_EQ(matmul(a, b).to_vector(), (std::vector<float>{19, 22, 43, 50}));
}

TEST(Matmul, AccelerateUsesTransposedViewsWithoutCopy) {
  HostAllocator alloc;
  Tensor a = random_tensor({32, 16}, 3, alloc).transpose();
  Tensor b = random_tensor({8, 32}, 4, alloc).transpose();
  Tensor out({16, 8}, alloc);
  const uint64_t before = alloc.stats().alloc_count;
  matmul_into(a, b, out, MatmulImpl::Accelerate);
  EXPECT_EQ(alloc.stats().alloc_count, before) << "transposed views should map to CblasTrans";
}

TEST(Matmul, ShapeErrors) {
  Tensor a({2, 3});
  Tensor b({4, 2});
  EXPECT_THROW(matmul(a, b), Error);
  EXPECT_THROW(matmul(Tensor({2, 2, 2}), Tensor({2, 2})), Error);
  EXPECT_THROW(matmul(Tensor(), Tensor({2, 2})), Error);
}

TEST(Matmul, MatmulIntoValidatesOutput) {
  Tensor a({2, 3});
  Tensor b({3, 4});
  Tensor wrong({4, 2});
  EXPECT_THROW(matmul_into(a, b, wrong), Error);
  Tensor strided = Tensor({4, 2}).transpose();  // right shape, not contiguous
  EXPECT_THROW(matmul_into(a, b, strided), Error);
  Tensor undefined;
  EXPECT_THROW(matmul_into(a, b, undefined), Error);
}

TEST(Matmul, MatmulIntoRejectsAliasing) {
  Tensor a({3, 3});
  EXPECT_THROW(matmul_into(a, a, a), Error);
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
