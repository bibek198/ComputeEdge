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

# Target-specific compile options (no global add_compile_options).
# Automatically provides ARC flags for Objective-C++ translation units.
add_library(computeedge_compile_options INTERFACE)
target_compile_features(computeedge_compile_options INTERFACE cxx_std_20)
target_compile_options(computeedge_compile_options INTERFACE
  ${CE_WARNINGS}
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)
if(CE_SANITIZE)
  target_compile_options(computeedge_compile_options INTERFACE
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
  target_link_options(computeedge_compile_options INTERFACE -fsanitize=address,undefined)
endif()

# Library layout (docs/plan_and_architecture.md §3.2):
#   computeedge::core   tensor, allocator, instrumentation; portable C++, no Apple frameworks
#   computeedge::cpu    CPU backend (Accelerate)              -> core
#   computeedge::metal  Metal backend (Objective-C++)         -> core
#   computeedge::bench  benchmark harness + env recorder      -> core
#   computeedge         convenience target linking core + cpu + metal

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
target_link_libraries(computeedge_core PRIVATE computeedge_compile_options)
add_library(computeedge::core ALIAS computeedge_core)

# ---------------------------------------------------------------------------
# CPU backend
# ---------------------------------------------------------------------------
add_library(computeedge_cpu STATIC
  src/cpu/matmul.cpp
  src/cpu/matmul_naive.cpp
  src/cpu/matmul_accelerate.cpp
)
target_link_libraries(computeedge_cpu PUBLIC computeedge::core "-framework Accelerate" PRIVATE computeedge_compile_options)
add_library(computeedge::cpu ALIAS computeedge_cpu)

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
target_link_libraries(computeedge_metal PUBLIC computeedge::core
  "-framework Metal" "-framework MetalPerformanceShaders" "-framework Foundation"
  PRIVATE computeedge_compile_options)
target_include_directories(computeedge_metal PRIVATE ${CMAKE_BINARY_DIR}/generated)
add_library(computeedge::metal ALIAS computeedge_metal)

add_library(computeedge INTERFACE)
target_link_libraries(computeedge INTERFACE computeedge::core computeedge::cpu computeedge::metal)
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
  src/bench/windows.cpp
)
target_link_libraries(computeedge_bench PUBLIC computeedge::core
  "-framework Foundation" "-framework IOKit" "-framework Metal"
  PRIVATE computeedge_compile_options)
target_compile_definitions(computeedge_bench PRIVATE
  CE_GIT_SHA="${CE_GIT_SHA}"
  CE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  CE_CXX_FLAGS="${CE_CXX_FLAGS}"
  CE_SANITIZE_ENABLED=$<BOOL:${CE_SANITIZE}>)
add_library(computeedge::bench ALIAS computeedge_bench)

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
target_link_libraries(computeedge_bench_harness PUBLIC computeedge::bench computeedge::core PRIVATE computeedge_compile_options)
add_library(computeedge::harness ALIAS computeedge_bench_harness)

add_library(computeedge_bench_harness_gpu STATIC benchmarks/harness/gpu.cpp)
target_link_libraries(computeedge_bench_harness_gpu PUBLIC computeedge::harness computeedge::metal PRIVATE computeedge_compile_options)
add_library(computeedge::harness_gpu ALIAS computeedge_bench_harness_gpu)

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(CE_BUILD_EXAMPLES)
  add_executable(matrix_multiplication examples/matrix_multiplication.cpp)
  target_link_libraries(matrix_multiplication PRIVATE computeedge::computeedge computeedge_compile_options)
  set_target_properties(matrix_multiplication PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)
endif()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(CE_BUILD_BENCHMARKS)
  # ce_benchmark(<name> <dir> <libs...>): <dir>/main.cpp plus, when present,
  # <dir>/options.cpp (CLI) and <dir>/experiments.cpp (measurement logic).
  function(ce_benchmark name dir)
    set(sources ${dir}/main.cpp)
    foreach(part options experiments)
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/${part}.cpp)
        list(APPEND sources ${dir}/${part}.cpp)
      endif()
    endforeach()
    add_executable(${name} ${sources})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
    target_link_libraries(${name} PRIVATE ${ARGN} computeedge_compile_options)   # each binary lists exactly what it links
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)
  endfunction()

  ce_benchmark(cpu_matmul benchmarks/cpu_matmul computeedge::harness computeedge::cpu)
  ce_benchmark(metal_matmul benchmarks/metal_matmul computeedge::harness_gpu computeedge::cpu)
  ce_benchmark(event_log_overhead benchmarks/event_log_overhead computeedge::harness)
  ce_benchmark(ce_env tools/ce_env computeedge::bench)
  # M5 characterization (Tracks A and B)
  ce_benchmark(dispatch_overhead benchmarks/dispatch_overhead computeedge::harness_gpu)
  ce_benchmark(sync_methods benchmarks/sync_methods computeedge::harness_gpu)
  ce_benchmark(storage_modes benchmarks/storage_modes computeedge::harness_gpu)
  # M6: cache interference and thermal drift
  ce_benchmark(cache_interference benchmarks/cache_interference computeedge::harness_gpu computeedge::cpu)
  ce_benchmark(thermal_drift benchmarks/thermal_drift computeedge::harness_gpu computeedge::cpu)
  ce_benchmark(followups benchmarks/followups computeedge::harness_gpu)
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
    tests/bench/windows_test.cpp
    tests/harness/cli_test.cpp
    tests/harness/gpu_test.cpp
    tests/harness/memory_and_correctness_test.cpp
    tests/harness/run_main_test.cpp
    tests/benchmarks/options_test.cpp
    # Each binary's real option parser, so its CLI contract is unit-tested.
    benchmarks/cache_interference/options.cpp
    benchmarks/cpu_matmul/options.cpp
    benchmarks/dispatch_overhead/options.cpp
    benchmarks/event_log_overhead/options.cpp
    benchmarks/followups/options.cpp
    benchmarks/metal_matmul/options.cpp
    benchmarks/storage_modes/options.cpp
    benchmarks/sync_methods/options.cpp
    benchmarks/thermal_drift/options.cpp
  )
  target_link_libraries(computeedge_tests PRIVATE computeedge::computeedge computeedge::harness_gpu
    GTest::gtest_main computeedge_compile_options)

  include(GoogleTest)
  gtest_discover_tests(computeedge_tests)

  # Self-contained headers test: ensure each public header compiles by itself
  file(GLOB_RECURSE CE_ALL_PUBLIC_HEADERS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
  set(CE_HEADER_CHECK_SRCS "")
  foreach(hdr ${CE_ALL_PUBLIC_HEADERS})
    file(RELATIVE_PATH rel "${CMAKE_CURRENT_SOURCE_DIR}/include" ${hdr})
    string(REGEX REPLACE "[^a-zA-Z0-9]" "_" safe "${rel}")
    set(src "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${safe}.cpp")
    file(GENERATE OUTPUT ${src} CONTENT "#include \"${rel}\"\nint main_${safe}() { return 0; }\n")
    list(APPEND CE_HEADER_CHECK_SRCS ${src})
  endforeach()
  add_library(computeedge_header_checks OBJECT ${CE_HEADER_CHECK_SRCS})
  target_include_directories(computeedge_header_checks PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(computeedge_header_checks PRIVATE computeedge::computeedge computeedge::bench computeedge_compile_options)

  # Layering checks (plan §3.2): core links with no Apple frameworks, and public
  # headers expose no Objective-C.
  add_executable(core_only_check tests/layering/core_only_check.cpp)
  target_link_libraries(core_only_check PRIVATE computeedge::core computeedge_compile_options)
  add_test(NAME layering.core_links_without_frameworks COMMAND core_only_check)
  add_test(NAME data.results_unchanged
    COMMAND sh "${CMAKE_CURRENT_SOURCE_DIR}/scripts/results_manifest.sh" check)
  add_test(NAME layering.no_objc_in_public_headers
    COMMAND sh -c "! grep -rnE '#import|@interface|@protocol|id<|NSObject' '${CMAKE_CURRENT_SOURCE_DIR}/include'")
endif()
