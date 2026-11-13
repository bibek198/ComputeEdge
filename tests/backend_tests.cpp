// Metal backend tests: correctness against Accelerate in every storage mode,
// edge shapes, timing invariants and the shared CPU/GPU timeline.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include "computeedge/computeedge.h"
#include "computeedge/metal.h"

using namespace computeedge;
namespace mtl = computeedge::metal;

namespace {

Tensor random_tensor(Shape shape, uint32_t seed, Allocator& alloc = default_allocator()) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(shape_numel(shape));
  for (float& x : v) x = dist(rng);
  return Tensor(std::move(shape), v, alloc);
}

void expect_matches_accelerate(const Tensor& got, const Tensor& a, const Tensor& b) {
  const std::vector<float> g = got.to_vector();
  const std::vector<float> r = matmul(a, b, MatmulImpl::Accelerate).to_vector();
  ASSERT_EQ(g.size(), r.size());
  const double tol = 1e-5 * std::sqrt(double(std::max<size_t>(a.shape()[1], 1)));
  for (size_t i = 0; i < g.size(); ++i) {
    ASSERT_LE(std::abs(double(g[i]) - double(r[i])), tol * (std::abs(double(r[i])) + 1.0)) << "at " << i;
  }
}

class MetalTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!mtl::available()) GTEST_SKIP() << "Metal not available";
  }
};

// Every storage mode x every kernel.
class MetalStorage : public ::testing::TestWithParam<std::tuple<mtl::StorageMode, mtl::Kernel>> {
 protected:
  void SetUp() override {
    if (!mtl::available()) GTEST_SKIP() << "Metal not available";
  }
  mtl::StorageMode storage() const { return std::get<0>(GetParam()); }
  mtl::MatmulOptions opts() const { return {std::get<0>(GetParam()), std::get<1>(GetParam())}; }
};

}  // namespace

INSTANTIATE_TEST_SUITE_P(All, MetalStorage,
                         ::testing::Combine(::testing::Values(mtl::StorageMode::NoCopy, mtl::StorageMode::Shared,
                                                             mtl::StorageMode::Private),
                                            ::testing::Values(mtl::Kernel::Tiled16, mtl::Kernel::MPS)),
                         [](const auto& info) {
                           return std::string(mtl::to_string(std::get<0>(info.param))) + "_" +
                                  mtl::to_string(std::get<1>(info.param));
                         });

TEST_F(MetalTest, DeviceInfo) {
  const mtl::DeviceInfo info = mtl::device_info();
  EXPECT_FALSE(info.name.empty());
  EXPECT_TRUE(info.has_unified_memory) << "Apple Silicon GPUs have unified memory";
  EXPECT_GT(info.recommended_max_working_set_bytes, 0u);
}

TEST_F(MetalTest, CompileIsOneTime) {
  mtl::warmup();
  const uint64_t compiled = mtl::compile_ns();
  EXPECT_GT(compiled, 0u);
  EXPECT_EQ(mtl::warmup(), 0u) << "second warmup must not recompile";
  Tensor a({4, 4}), b({4, 4});
  mtl::matmul(a, b);
  EXPECT_EQ(mtl::compile_ns(), compiled);
}

TEST_P(MetalStorage, TwoByTwo) {
  Tensor a({2, 2}, {1, 2, 3, 4});
  Tensor b({2, 2}, {5, 6, 7, 8});
  EXPECT_EQ(mtl::matmul(a, b, opts()).to_vector(), (std::vector<float>{19, 22, 43, 50}));
}

TEST_P(MetalStorage, RandomSquareMatchesAccelerate) {
  for (size_t n : {16, 64, 256}) {
    Tensor a = random_tensor({n, n}, 1), b = random_tensor({n, n}, 2);
    expect_matches_accelerate(mtl::matmul(a, b, opts()), a, b);
  }
}

TEST_P(MetalStorage, NonMultipleOfTileShapes) {
  Tensor a = random_tensor({17, 33}, 3), b = random_tensor({33, 5}, 4);
  expect_matches_accelerate(mtl::matmul(a, b, opts()), a, b);
  Tensor row = random_tensor({1, 100}, 5), col = random_tensor({100, 1}, 6);
  expect_matches_accelerate(mtl::matmul(row, col, opts()), row, col);
  expect_matches_accelerate(mtl::matmul(col, row, opts()), col, row);
}

TEST_P(MetalStorage, TransposedInputs) {
  Tensor a = random_tensor({40, 24}, 7).transpose();  // 24 x 40
  Tensor b = random_tensor({8, 40}, 8).transpose();   // 40 x 8
  expect_matches_accelerate(mtl::matmul(a, b, opts()), a, b);
}

TEST_P(MetalStorage, ZeroInnerDimension) {
  Tensor a({3, 0}), b({0, 2});
  Tensor out = Tensor::full({3, 2}, 5.0f);
  mtl::matmul_into(a, b, out, opts());
  for (float v : out.to_vector()) EXPECT_EQ(v, 0.0f);
}

TEST_P(MetalStorage, EmptyOutput) {
  Tensor a({0, 4}), b({4, 3});
  Tensor c = mtl::matmul(a, b, opts());
  EXPECT_EQ(c.shape(), (Shape{0, 3}));
}

TEST_P(MetalStorage, MatmulIntoOverwrites) {
  Tensor a({2, 2}, {1, 2, 3, 4}), b({2, 2}, {5, 6, 7, 8});
  Tensor out = Tensor::full({2, 2}, -1.0f);
  mtl::matmul_into(a, b, out, opts());
  EXPECT_EQ(out.to_vector(), (std::vector<float>{19, 22, 43, 50}));
}

TEST_P(MetalStorage, TimingIsConsistent) {
  mtl::warmup();
  Tensor a = random_tensor({128, 128}, 9), b = random_tensor({128, 128}, 10);
  Tensor out({128, 128});
  mtl::Timing t;
  const uint64_t before = now_ns();
  mtl::matmul_into(a, b, out, opts(), &t);
  const uint64_t after = now_ns();

  EXPECT_GT(t.total_ns, 0u);
  EXPECT_GT(t.wait_ns, 0u);
  EXPECT_GT(t.gpu_ns(), 0u);
  EXPECT_LE(t.encode_ns + t.commit_ns + t.wait_ns, t.total_ns);
  // GPU timestamps share the CPU timebase: GPU work lies inside the call and
  // finishes before the CPU observes completion.
  EXPECT_GT(t.gpu_start_ns, before);
  EXPECT_LE(t.gpu_start_ns, t.gpu_end_ns);
  EXPECT_LE(t.gpu_end_ns, t.wait_end_ns);
  EXPECT_LT(t.wait_end_ns, after);
  EXPECT_GE(t.wake_ns(), 0);
  if (storage() == mtl::StorageMode::NoCopy) {
    EXPECT_EQ(t.copy_out_ns, 0u);
  } else {
    EXPECT_GT(t.copy_out_ns, 0u);
  }
}

TEST_F(MetalTest, NoCopyDoesNotAllocateHostMemory) {
  HostAllocator alloc;
  Tensor a = random_tensor({32, 32}, 11, alloc), b = random_tensor({32, 32}, 12, alloc);
  Tensor out({32, 32}, alloc);
  const uint64_t before = alloc.stats().alloc_count;
  mtl::matmul_into(a, b, out, {mtl::StorageMode::NoCopy});
  EXPECT_EQ(alloc.stats().alloc_count, before);
  expect_matches_accelerate(out, a, b);
}

TEST_F(MetalTest, EmitsDispatchSubmitSyncAndGpuSpan) {
  mtl::warmup();
  Tensor a({16, 16}), b({16, 16}), out({16, 16});
  ScopedEventLogging logging;
  mtl::matmul_into(a, b, out);
  const auto events = EventLog::global().snapshot();
  auto find = [&](EventType type, const char* label) {
    return std::find_if(events.begin(), events.end(), [&](const Event& e) {
      return e.type == type && (label == nullptr || std::string(e.label ? e.label : "") == label);
    });
  };
  const auto dispatch = find(EventType::Dispatch, nullptr);
  const auto submit = find(EventType::Submit, nullptr);
  const auto sync = find(EventType::Sync, nullptr);
  const auto gpu_begin = find(EventType::Begin, "gpu_exec");
  const auto gpu_end = find(EventType::End, "gpu_exec");
  ASSERT_NE(dispatch, events.end());
  ASSERT_NE(submit, events.end());
  ASSERT_NE(sync, events.end());
  ASSERT_NE(gpu_begin, events.end());
  ASSERT_NE(gpu_end, events.end());
  EXPECT_EQ(dispatch->backend, Backend::Metal);
  for (auto it : {submit, sync, gpu_begin, gpu_end}) EXPECT_EQ(it->exec_id, dispatch->exec_id);
  EXPECT_LE(dispatch->t_ns, submit->t_ns);
  EXPECT_LE(gpu_begin->t_ns, gpu_end->t_ns);
  EXPECT_LE(gpu_end->t_ns, sync->t_ns);
}

TEST_F(MetalTest, MpsAndTiledAgree) {
  Tensor a = random_tensor({300, 200}, 21), b = random_tensor({200, 100}, 22);
  const std::vector<float> tiled = mtl::matmul(a, b, {mtl::StorageMode::NoCopy, mtl::Kernel::Tiled16}).to_vector();
  const std::vector<float> mps = mtl::matmul(a, b, {mtl::StorageMode::NoCopy, mtl::Kernel::MPS}).to_vector();
  for (size_t i = 0; i < tiled.size(); ++i) ASSERT_NEAR(tiled[i], mps[i], 1e-4f * (std::abs(tiled[i]) + 1.0f));
}

TEST_F(MetalTest, MpsWithOffsetView) {
  // A reshaped view keeps offset 0, but exercise MPS on transposed (materialised) inputs too.
  Tensor a = random_tensor({64, 48}, 23).transpose();
  Tensor b = random_tensor({64, 32}, 24);
  expect_matches_accelerate(mtl::matmul(a, b, {mtl::StorageMode::NoCopy, mtl::Kernel::MPS}), a, b);
}

TEST_F(MetalTest, ValidatesArguments) {
  Tensor a({2, 3}), b({4, 2});
  EXPECT_THROW(mtl::matmul(a, b), Error);
  Tensor ok_b({3, 4});
  Tensor strided = Tensor({4, 2}).transpose();
  EXPECT_THROW(mtl::matmul_into(a, ok_b, strided), Error);
  Tensor sq({3, 3});
  EXPECT_THROW(mtl::matmul_into(sq, sq, sq), Error);
}
