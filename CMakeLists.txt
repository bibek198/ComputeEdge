cmake_minimum_required(VERSION 3.24)
project(ComputeEdge VERSION 0.1.0 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CE_BUILD_TESTS "Build unit tests" ON)
option(CE_BUILD_EXAMPLES "Build examples" ON)
option(CE_BUILD_BENCHMARKS "Build benchmarks" ON)
option(CE_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CE_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=address,undefined)
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

set(CE_WARNINGS -Wall -Wextra -Wpedantic -Wshadow)

# Library layout (docs/plan_and_architecture.md §3.2):
#   computeedge_core   tensor, allocator, instrumentation; portable C++, no Apple frameworks
#   computeedge_cpu    CPU backend (Accelerate)              -> core
#   computeedge_metal  Metal backend (Objective-C++)         -> core
#   computeedge_bench  benchmark harness + env recorder      -> core
#   computeedge        convenience target linking core + cpu + metal

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
add_library(computeedge_core STATIC
  src/core/allocator.cpp
  src/core/instrumentation.cpp
  src/core/tensor.cpp
  src/core/validate.cpp
)
target_include_directories(computeedge_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(computeedge_core PRIVATE ${CE_WARNINGS})

# ---------------------------------------------------------------------------
# CPU backend
# ---------------------------------------------------------------------------
add_library(computeedge_cpu STATIC
  src/cpu/matmul.cpp
  src/cpu/matmul_naive.cpp
  src/cpu/matmul_accelerate.cpp
)
target_link_libraries(computeedge_cpu PUBLIC computeedge_core "-framework Accelerate")
target_compile_options(computeedge_cpu PRIVATE ${CE_WARNINGS})

# ---------------------------------------------------------------------------
# Metal backend. Kernels are compiled at runtime (newLibraryWithSource:), so
# no offline Metal toolchain is needed; the .metal sources are embedded here.
# ---------------------------------------------------------------------------
set(CE_METAL_KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels/matmul.metal)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CE_METAL_KERNELS})
file(READ src/metal/kernels/matmul.metal CE_MSL_MATMUL)
configure_file(src/metal/kernel_sources.h.in ${CMAKE_BINARY_DIR}/generated/computeedge/kernel_sources.h @ONLY)

add_library(computeedge_metal STATIC src/metal/metal_backend.mm)
target_link_libraries(computeedge_metal PUBLIC computeedge_core "-framework Metal" "-framework Foundation")
target_include_directories(computeedge_metal PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_compile_options(computeedge_metal PRIVATE ${CE_WARNINGS} -fobjc-arc)

add_library(computeedge INTERFACE)
target_link_libraries(computeedge INTERFACE computeedge_core computeedge_cpu computeedge_metal)
add_library(computeedge::computeedge ALIAS computeedge)

# ---------------------------------------------------------------------------
# Benchmark harness (Foundation / IOKit / Metal for environment recording)
# ---------------------------------------------------------------------------
add_library(computeedge_bench STATIC
  src/benchmark/benchmark.cpp
  src/benchmark/stats.cpp
  src/benchmark/env_recorder.mm
)
target_link_libraries(computeedge_bench PUBLIC computeedge_core
  "-framework Foundation" "-framework IOKit" "-framework Metal")
target_compile_options(computeedge_bench PRIVATE ${CE_WARNINGS})
set_source_files_properties(src/benchmark/env_recorder.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
target_compile_definitions(computeedge_bench PRIVATE
  CE_GIT_SHA="${CE_GIT_SHA}"
  CE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  CE_CXX_FLAGS="${CE_CXX_FLAGS}"
  CE_SANITIZE_ENABLED=$<BOOL:${CE_SANITIZE}>)

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(CE_BUILD_EXAMPLES)
  add_executable(matrix_multiplication examples/matrix_multiplication.cpp)
  target_link_libraries(matrix_multiplication PRIVATE computeedge)
  set_target_properties(matrix_multiplication PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(CE_BUILD_BENCHMARKS)
  function(ce_benchmark name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE computeedge_bench ${ARGN})
    target_compile_options(${name} PRIVATE ${CE_WARNINGS})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)
  endfunction()

  ce_benchmark(cpu_matmul benchmarks/baseline/cpu_matmul.cpp computeedge_cpu)
  ce_benchmark(metal_matmul benchmarks/baseline/metal_matmul.cpp computeedge_cpu computeedge_metal)
  ce_benchmark(event_log_overhead benchmarks/overhead/event_log.cpp)
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
    tests/benchmark_tests.cpp
    tests/backend_tests.cpp
  )
  target_link_libraries(computeedge_tests PRIVATE computeedge computeedge_bench GTest::gtest_main)
  target_compile_options(computeedge_tests PRIVATE -Wall -Wextra)

  include(GoogleTest)
  gtest_discover_tests(computeedge_tests)

  # Layering checks (plan §3.2): core links with no Apple frameworks, and public
  # headers expose no Objective-C.
  add_executable(core_only_check tests/core_only_check.cpp)
  target_link_libraries(core_only_check PRIVATE computeedge_core)
  add_test(NAME layering.core_links_without_frameworks COMMAND core_only_check)
  add_test(NAME layering.no_objc_in_public_headers
    COMMAND sh -c "! grep -rnE '#import|@interface|@protocol|id<|NSObject' '${CMAKE_CURRENT_SOURCE_DIR}/include'")
endif()
