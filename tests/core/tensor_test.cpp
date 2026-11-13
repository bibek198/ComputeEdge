#include <gtest/gtest.h>

#include <sstream>
#include <vector>

#include "computeedge/core/tensor.h"

using namespace computeedge;

TEST(Tensor, ShapeStridesNumel) {
  Tensor t({2, 3, 4});
  EXPECT_EQ(t.ndim(), 3u);
  EXPECT_EQ(t.numel(), 24u);
  EXPECT_EQ(t.nbytes(), 96u);
  EXPECT_EQ(t.strides(), (Strides{12, 4, 1}));
  EXPECT_TRUE(t.is_contiguous());
  EXPECT_EQ(t.dtype(), DType::Float32);
}

TEST(Tensor, ZeroInitialised) {
  Tensor t({3, 3});
  for (float v : t.to_vector()) EXPECT_EQ(v, 0.0f);
}

TEST(Tensor, FromValuesAndIndexing) {
  Tensor t({2, 2}, {1, 2, 3, 4});
  EXPECT_EQ(t.at({0, 0}), 1.0f);
  EXPECT_EQ(t.at({0, 1}), 2.0f);
  EXPECT_EQ(t.at({1, 0}), 3.0f);
  EXPECT_EQ(t.at({1, 1}), 4.0f);
  t.at({1, 1}) = 9.0f;
  EXPECT_EQ(t.to_vector(), (std::vector<float>{1, 2, 3, 9}));
}

TEST(Tensor, FullAndVectorCtor) {
  Tensor f = Tensor::full({2, 3}, 1.5f);
  for (float v : f.to_vector()) EXPECT_EQ(v, 1.5f);
  std::vector<float> vals{1, 2, 3, 4, 5, 6};
  Tensor t({3, 2}, vals);
  EXPECT_EQ(t.to_vector(), vals);
}

TEST(Tensor, WrongValueCountThrows) { EXPECT_THROW(Tensor({2, 2}, {1, 2, 3}), Error); }

TEST(Tensor, BadIndexThrows) {
  Tensor t({2, 2});
  EXPECT_THROW(t.at({2, 0}), Error);
  EXPECT_THROW(t.at({0}), Error);
}

TEST(Tensor, CopySharesStorage) {
  Tensor a({2}, {1, 2});
  Tensor b = a;
  b.at({0}) = 7;
  EXPECT_EQ(a.at({0}), 7.0f);
  EXPECT_EQ(a.storage(), b.storage());
}

TEST(Tensor, CloneIsDeep) {
  Tensor a({2}, {1, 2});
  Tensor b = a.clone();
  b.at({0}) = 7;
  EXPECT_EQ(a.at({0}), 1.0f);
  EXPECT_NE(a.storage(), b.storage());
}

TEST(Tensor, ReshapeIsView) {
  Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor b = a.reshape({3, 2});
  EXPECT_EQ(b.storage(), a.storage());
  EXPECT_EQ(b.at({2, 1}), 6.0f);
  EXPECT_THROW(a.reshape({4, 2}), Error);
}

TEST(Tensor, TransposeIsStridedView) {
  Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor t = a.transpose();
  EXPECT_EQ(t.shape(), (Shape{3, 2}));
  EXPECT_EQ(t.strides(), (Strides{1, 3}));
  EXPECT_FALSE(t.is_contiguous());
  EXPECT_EQ(t.storage(), a.storage());
  EXPECT_EQ(t.at({2, 1}), 6.0f);
  EXPECT_EQ(t.at({0, 1}), 4.0f);
  EXPECT_EQ(t.to_vector(), (std::vector<float>{1, 4, 2, 5, 3, 6}));
  EXPECT_THROW(a.transpose(0, 2), Error);
}

TEST(Tensor, ContiguousMaterialisesTranspose) {
  Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor c = a.transpose().contiguous();
  EXPECT_TRUE(c.is_contiguous());
  EXPECT_NE(c.storage(), a.storage());
  EXPECT_EQ(c.to_vector(), (std::vector<float>{1, 4, 2, 5, 3, 6}));
  Tensor same = a.contiguous();
  EXPECT_EQ(same.storage(), a.storage()) << "contiguous() on a contiguous tensor must not copy";
}

TEST(Tensor, ReshapeOfNonContiguousCopies) {
  Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor r = a.transpose().reshape({6});
  EXPECT_EQ(r.to_vector(), (std::vector<float>{1, 4, 2, 5, 3, 6}));
}

TEST(Tensor, ThreeDimTransposeGather) {
  std::vector<float> v(24);
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i);
  Tensor a({2, 3, 4}, v);
  Tensor t = a.transpose(0, 2);  // shape {4, 3, 2}
  EXPECT_EQ(t.at({3, 2, 1}), a.at({1, 2, 3}));
  std::vector<float> g = t.to_vector();
  EXPECT_EQ(g[0], 0.0f);
  EXPECT_EQ(g[1], 12.0f);
  EXPECT_EQ(g[2], 4.0f);
}

TEST(Tensor, EmptyTensor) {
  Tensor t({0, 3});
  EXPECT_EQ(t.numel(), 0u);
  EXPECT_TRUE(t.to_vector().empty());
}

TEST(Tensor, Print2D) {
  Tensor t({2, 2}, {19, 22, 43, 50});
  std::ostringstream os;
  t.print(os);
  EXPECT_EQ(os.str(), "19 22\n43 50\n");
}

TEST(Tensor, ToStringAndStreamMatchPrint) {
  Tensor t({2, 2}, {1, 2, 3, 4});
  std::ostringstream printed, streamed;
  t.print(printed);
  streamed << t;
  EXPECT_EQ(t.to_string(), printed.str());
  EXPECT_EQ(streamed.str(), printed.str());
}

TEST(Tensor, UndefinedTensor) {
  Tensor t;
  EXPECT_FALSE(t.defined());
  EXPECT_THROW(t.data(), Error);
}
