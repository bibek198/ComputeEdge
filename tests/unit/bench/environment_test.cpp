#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "computeedge/bench/environment.h"
#include "computeedge/core/error.h"
#include "computeedge/core/instrumentation.h"

#include <unistd.h>
#include <algorithm>

namespace cb = computeedge::bench;

TEST(Environment, RecordsHardwareAndState) {
  const cb::Environment env = cb::record_environment();
  EXPECT_FALSE(env.model.empty());
  EXPECT_FALSE(env.chip.empty());
  EXPECT_GT(env.memory_bytes, 0u);
  EXPECT_GT(env.performance_cores, 0);
  EXPECT_FALSE(env.os_version.empty());
  EXPECT_FALSE(env.os_build.empty());
  EXPECT_FALSE(env.timestamp_utc.empty());
  EXPECT_NE(env.thermal_state, cb::ThermalState::Unknown);

  const std::string json = cb::to_json(env);
  EXPECT_EQ(json.front(), '{');
  EXPECT_NE(json.find("\"chip\": \""), std::string::npos);
  EXPECT_NE(json.find("\"thermal_state\""), std::string::npos);
}
