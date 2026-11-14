# Build and feature options for ComputeEdge

option(CE_BUILD_TESTS "Build unit tests" ON)
option(CE_BUILD_EXAMPLES "Build examples" ON)
option(CE_BUILD_BENCHMARKS "Build benchmarks" ON)
option(CE_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(CE_WARNINGS_AS_ERRORS "Treat warnings in ComputeEdge targets as errors" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Git revision recorded in benchmark output (captured at configure time; re-run
# cmake after committing to refresh it).
execute_process(
  COMMAND git describe --always --dirty --abbrev=12
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE CE_GIT_SHA
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(NOT CE_GIT_SHA)
  set(CE_GIT_SHA "unknown")
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" CE_BUILD_TYPE_UPPER)
string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CE_BUILD_TYPE_UPPER}}" CE_CXX_FLAGS)
if(CE_SANITIZE)
  string(APPEND CE_CXX_FLAGS " -fsanitize=address,undefined")
endif()
