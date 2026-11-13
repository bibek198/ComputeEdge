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
# CI turns warnings into errors for ComputeEdge's own targets only; third-party
# code (googletest via FetchContent) never receives these flags.
option(CE_WARNINGS_AS_ERRORS "Treat warnings in ComputeEdge targets as errors" OFF)
if(CE_WARNINGS_AS_ERRORS)
  list(APPEND CE_WARNINGS -Werror)
endif()

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
set(CE_METAL_KERNELS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels/matmul.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels/probes.metal)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CE_METAL_KERNELS})
file(READ src/metal/kernels/matmul.metal CE_MSL_MATMUL)
file(READ src/metal/kernels/probes.metal CE_MSL_PROBES)
configure_file(src/metal/kernel_sources.h.in ${CMAKE_BINARY_DIR}/generated/computeedge/kernel_sources.h @ONLY)

add_library(computeedge_metal STATIC src/metal/metal_backend.mm src/metal/probes.mm)
target_link_libraries(computeedge_metal PUBLIC computeedge_core
  "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation")
target_include_directories(computeedge_metal PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_compile_options(computeedge_metal PRIVATE ${CE_WARNINGS} -fobjc-arc)

add_library(computeedge INTERFACE)
target_link_libraries(computeedge INTERFACE computeedge_core computeedge_cpu computeedge_metal)
add_library(computeedge::computeedge ALIAS computeedge)

# ---------------------------------------------------------------------------
# Benchmark harness (Foundation / IOKit / Metal for environment recording)
# ---------------------------------------------------------------------------
add_library(computeedge_bench STATIC
  src/bench/environment.mm
  src/bench/result_writer.cpp
  src/bench/runner.cpp
  src/bench/stats.cpp
  src/bench/trace.cpp
)
target_link_libraries(computeedge_bench PUBLIC computeedge_core
  "-framework Foundation" "-framework IOKit" "-framework Metal")
target_compile_options(computeedge_bench PRIVATE ${CE_WARNINGS})
set_source_files_properties(src/bench/environment.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
target_compile_definitions(computeedge_bench PRIVATE
  CE_GIT_SHA="${CE_GIT_SHA}"
  CE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  CE_CXX_FLAGS="${CE_CXX_FLAGS}"
  CE_SANITIZE_ENABLED=$<BOOL:${CE_SANITIZE}>)

# ---------------------------------------------------------------------------
# Benchmark harness: code shared by the benchmark binaries (benchmarks/harness/).
# The GPU part is a separate library so CPU-only binaries do not link Metal.
# ---------------------------------------------------------------------------
add_library(computeedge_bench_harness STATIC
  benchmarks/harness/cli.cpp
  benchmarks/harness/console.cpp
  benchmarks/harness/correctness.cpp
  benchmarks/harness/host_memory.cpp
  benchmarks/harness/results.cpp
)
target_include_directories(computeedge_bench_harness PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
target_link_libraries(computeedge_bench_harness PUBLIC computeedge_bench computeedge_core)
target_compile_options(computeedge_bench_harness PRIVATE ${CE_WARNINGS})

add_library(computeedge_bench_harness_gpu STATIC benchmarks/harness/gpu.cpp)
target_link_libraries(computeedge_bench_harness_gpu PUBLIC computeedge_bench_harness computeedge_metal)
target_compile_options(computeedge_bench_harness_gpu PRIVATE ${CE_WARNINGS})

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
    target_link_libraries(${name} PRIVATE ${ARGN})   # each binary lists exactly what it links
    target_compile_options(${name} PRIVATE ${CE_WARNINGS})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)
  endfunction()

  ce_benchmark(cpu_matmul benchmarks/cpu_matmul/main.cpp computeedge_bench_harness computeedge_cpu)
  ce_benchmark(metal_matmul benchmarks/metal_matmul/main.cpp computeedge_bench_harness_gpu computeedge_cpu)
  ce_benchmark(event_log_overhead benchmarks/event_log_overhead/main.cpp computeedge_bench_harness)
  ce_benchmark(ce_env tools/ce_env/main.cpp computeedge_bench)
  # M5 characterization (Tracks A and B)
  ce_benchmark(dispatch_overhead benchmarks/dispatch_overhead/main.cpp computeedge_bench_harness_gpu)
  ce_benchmark(sync_methods benchmarks/sync_methods/main.cpp computeedge_bench_harness_gpu)
  ce_benchmark(storage_modes benchmarks/storage_modes/main.cpp computeedge_bench_harness_gpu)
  # M6: cache interference and thermal drift
  ce_benchmark(cache_interference benchmarks/cache_interference/main.cpp computeedge_bench_harness_gpu computeedge_cpu)
  ce_benchmark(thermal_drift benchmarks/thermal_drift/main.cpp computeedge_bench_harness_gpu computeedge_cpu)
  ce_benchmark(followups benchmarks/followups/main.cpp computeedge_bench_harness_gpu)
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
    tests/core/allocator_test.cpp
    tests/core/instrumentation_test.cpp
    tests/core/tensor_test.cpp
    tests/cpu/matmul_test.cpp
    tests/metal/matmul_test.cpp
    tests/metal/probes_test.cpp
    tests/bench/stats_test.cpp
    tests/bench/environment_test.cpp
    tests/bench/runner_test.cpp
    tests/bench/result_writer_test.cpp
    tests/harness/cli_test.cpp
    tests/harness/gpu_test.cpp
    tests/harness/memory_and_correctness_test.cpp
  )
  target_link_libraries(computeedge_tests PRIVATE computeedge computeedge_bench_harness_gpu
  GTest::gtest_main)
  target_compile_options(computeedge_tests PRIVATE -Wall -Wextra)

  include(GoogleTest)
  gtest_discover_tests(computeedge_tests)

  # Layering checks (plan §3.2): core links with no Apple frameworks, and public
  # headers expose no Objective-C.
  add_executable(core_only_check tests/layering/core_only_check.cpp)
  target_link_libraries(core_only_check PRIVATE computeedge_core)
  add_test(NAME layering.core_links_without_frameworks COMMAND core_only_check)
  add_test(NAME data.results_unchanged
    COMMAND sh "${CMAKE_CURRENT_SOURCE_DIR}/scripts/results_manifest.sh" check)
  add_test(NAME layering.no_objc_in_public_headers
    COMMAND sh -c "! grep -rnE '#import|@interface|@protocol|id<|NSObject' '${CMAKE_CURRENT_SOURCE_DIR}/include'")
endif()
