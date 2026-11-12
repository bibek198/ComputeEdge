#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "computeedge/allocator.h"
#include "computeedge/instrumentation.h"
#include "computeedge/scoped_timer.h"
#include "computeedge/ops.h"

using namespace computeedge;

namespace {

size_t count_type(const std::vector<Event>& events, EventType type) {
  return static_cast<size_t>(std::count_if(events.begin(), events.end(), [&](const Event& e) { return e.type == type; }));
}

}  // namespace

TEST(Instrumentation, ClockIsMonotonic) {
  const uint64_t a = now_ns();
  const uint64_t b = now_ns();
  EXPECT_LE(a, b);
}

TEST(Instrumentation, ExecIdsAreUniqueAndNonZero) {
  const uint64_t a = next_exec_id();
  const uint64_t b = next_exec_id();
  EXPECT_NE(a, 0u);
  EXPECT_NE(a, b);
}

TEST(Instrumentation, DisabledLogRecordsNothing) {
  EventLog::global().set_enabled(false);
  EventLog::global().clear();
  emit(EventType::Dispatch, 1);
  { Tensor t({8}); }
  EXPECT_EQ(EventLog::global().size(), 0u);
}

TEST(Instrumentation, ScopedLoggingRestoresState) {
  EventLog::global().set_enabled(false);
  {
    ScopedEventLogging on;
    EXPECT_TRUE(EventLog::global().enabled());
  }
  EXPECT_FALSE(EventLog::global().enabled());
}

TEST(Instrumentation, AllocAndFreeEventsMatchBufferIds) {
  ScopedEventLogging on;
  uint64_t id = 0;
  size_t bytes = 0;
  {
    Tensor t({10, 10});
    id = t.storage()->id();
    bytes = t.storage()->bytes();
  }
  const auto events = EventLog::global().snapshot();
  ASSERT_EQ(events.size(), 2u);
  EXPECT_EQ(events[0].type, EventType::Alloc);
  EXPECT_EQ(events[1].type, EventType::Free);
  EXPECT_EQ(events[0].id, id);
  EXPECT_EQ(events[1].id, id);
  EXPECT_EQ(events[0].bytes, bytes);
  EXPECT_LE(events[0].t_ns, events[1].t_ns);
}

TEST(Instrumentation, TimelineReconstructsLiveBytes) {
  ScopedEventLogging on;
  {
    Tensor a({64, 64});
    Tensor b({64, 64});
    Tensor c = matmul(a, b);
  }
  // Replay alloc/free events: live bytes must return to zero and never go negative.
  std::map<uint64_t, uint64_t> live;
  int64_t current = 0;
  for (const Event& e : EventLog::global().snapshot()) {
    if (e.type == EventType::Alloc) {
      EXPECT_EQ(live.count(e.id), 0u);
      live[e.id] = e.bytes;
      current += static_cast<int64_t>(e.bytes);
    } else if (e.type == EventType::Free) {
      ASSERT_EQ(live.count(e.id), 1u) << "free without alloc";
      current -= static_cast<int64_t>(live[e.id]);
      live.erase(e.id);
    }
    EXPECT_GE(current, 0);
  }
  EXPECT_TRUE(live.empty());
  EXPECT_EQ(current, 0);
}

TEST(Instrumentation, MatmulEmitsDispatchAndTimedSpan) {
  ScopedEventLogging on;
  Tensor a({2, 3});
  Tensor b({3, 4});
  EventLog::global().clear();
  Tensor c = matmul(a, b);
  const auto events = EventLog::global().snapshot();
  ASSERT_EQ(count_type(events, EventType::Dispatch), 1u);
  ASSERT_EQ(count_type(events, EventType::Begin), 1u);
  ASSERT_EQ(count_type(events, EventType::End), 1u);
  const auto dispatch = *std::find_if(events.begin(), events.end(), [](const Event& e) { return e.type == EventType::Dispatch; });
  EXPECT_EQ(dispatch.backend, Backend::CPU);
  EXPECT_NE(dispatch.exec_id, 0u);
  for (const Event& e : events) {
    if (e.type == EventType::Begin || e.type == EventType::End) EXPECT_EQ(e.exec_id, dispatch.exec_id);
  }
}

TEST(Instrumentation, ScopedTimerWritesSink) {
  uint64_t ns = 0;
  {
    ScopedTimer t("sleep", &ns);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  EXPECT_GE(ns, 1'000'000u);
}

TEST(Instrumentation, CsvHasHeaderAndRows) {
  ScopedEventLogging on;
  emit(EventType::Sync, 42, 128, Backend::Metal, 7, "test");
  std::ostringstream os;
  EventLog::global().write_csv(os);
  const std::string csv = os.str();
  EXPECT_EQ(csv.rfind("t_ns,type,id,exec_id,bytes,backend,label\n", 0), 0u);
  EXPECT_NE(csv.find(",sync,42,7,128,metal,test\n"), std::string::npos);
}

TEST(Instrumentation, ConcurrentRecordingIsSafe) {
  ScopedEventLogging on;
  constexpr int kThreads = 8, kPerThread = 1000;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([] {
      for (int i = 0; i < kPerThread; ++i) emit(EventType::Dispatch, static_cast<uint64_t>(i));
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(EventLog::global().size(), static_cast<size_t>(kThreads * kPerThread));
}
