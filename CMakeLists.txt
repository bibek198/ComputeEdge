cmake_minimum_required(VERSION 3.24)
project(ComputeEdge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CE_BUILD_TESTS "Build unit tests" ON)
option(CE_BUILD_EXAMPLES "Build examples" ON)
option(CE_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CE_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(computeedge STATIC
  src/core/allocator.cpp
  src/core/instrumentation.cpp
  src/core/tensor.cpp
  src/cpu/matmul_naive.cpp
)
add_library(computeedge::computeedge ALIAS computeedge)

target_include_directories(computeedge PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(computeedge PRIVATE -Wall -Wextra -Wpedantic -Wshadow)

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(CE_BUILD_EXAMPLES)
  add_executable(matrix_multiplication examples/matrix_multiplication.cpp)
  target_link_libraries(matrix_multiplication PRIVATE computeedge)
  set_target_properties(matrix_multiplication PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CE_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(computeedge_tests
    tests/allocator_tests.cpp
    tests/instrumentation_tests.cpp
    tests/tensor_tests.cpp
    tests/matmul_tests.cpp
  )
  target_link_libraries(computeedge_tests PRIVATE computeedge GTest::gtest_main)
  target_compile_options(computeedge_tests PRIVATE -Wall -Wextra)

  include(GoogleTest)
  gtest_discover_tests(computeedge_tests)
endif()
