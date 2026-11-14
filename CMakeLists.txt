cmake_minimum_required(VERSION 3.24)
project(ComputeEdge VERSION 0.1.0 LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Modular CMake configuration modules
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ComputeEdgeOptions)
include(CompilerWarnings)
include(Sanitizers)
include(Dependencies)
include(Metal)
include(Packaging)

# Library layout:
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
# Metal backend
# ---------------------------------------------------------------------------
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
# Benchmark framework (Foundation / IOKit / Metal for environment recording)
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
# Benchmark harness: code shared by the benchmark binaries (benchmarks/common/harness/).
# The GPU part is a separate library so CPU-only binaries do not link Metal.
# ---------------------------------------------------------------------------
add_library(computeedge_bench_harness STATIC
  benchmarks/common/harness/cli.cpp
  benchmarks/common/harness/console.cpp
  benchmarks/common/harness/correctness.cpp
  benchmarks/common/harness/host_memory.cpp
  benchmarks/common/harness/results.cpp
)
target_include_directories(computeedge_bench_harness PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/common
  ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
)
target_link_libraries(computeedge_bench_harness PUBLIC computeedge::bench computeedge::core PRIVATE computeedge_compile_options)
add_library(computeedge::harness ALIAS computeedge_bench_harness)

add_library(computeedge_bench_harness_gpu STATIC benchmarks/common/harness/gpu.cpp)
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

  ce_benchmark(cpu_matmul benchmarks/workloads/matmul/cpu computeedge::harness computeedge::cpu)
  ce_benchmark(metal_matmul benchmarks/workloads/matmul/metal computeedge::harness_gpu computeedge::cpu)
  ce_benchmark(event_log_overhead benchmarks/workloads/instrumentation computeedge::harness)
  ce_benchmark(ce_env tools/ce_env computeedge::bench)
  # M5 characterization (Tracks A and B)
  ce_benchmark(dispatch_overhead benchmarks/workloads/dispatch computeedge::harness_gpu)
  ce_benchmark(sync_methods benchmarks/workloads/synchronization computeedge::harness_gpu)
  ce_benchmark(storage_modes benchmarks/workloads/memory computeedge::harness_gpu)
  # M6: cache interference and thermal drift
  ce_benchmark(cache_interference benchmarks/workloads/cache_interference computeedge::harness_gpu computeedge::cpu)
  ce_benchmark(thermal_drift benchmarks/workloads/thermal computeedge::harness_gpu computeedge::cpu)
  ce_benchmark(followups benchmarks/workloads/followups computeedge::harness_gpu)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CE_BUILD_TESTS)
  ce_setup_googletest()

  enable_testing()
  add_executable(computeedge_tests
    tests/unit/core/allocator_test.cpp
    tests/unit/core/instrumentation_test.cpp
    tests/unit/core/tensor_test.cpp
    tests/unit/cpu/matmul_test.cpp
    tests/unit/metal/matmul_test.cpp
    tests/unit/metal/probes_test.cpp
    tests/unit/bench/stats_test.cpp
    tests/unit/bench/environment_test.cpp
    tests/unit/bench/runner_test.cpp
    tests/unit/bench/result_writer_test.cpp
    tests/unit/bench/windows_test.cpp
    tests/unit/harness/cli_test.cpp
    tests/unit/harness/gpu_test.cpp
    tests/unit/harness/memory_and_correctness_test.cpp
    tests/unit/harness/run_main_test.cpp
    tests/integration/benchmarks/options_test.cpp
    # Each binary's real option parser, so its CLI contract is unit-tested.
    benchmarks/workloads/cache_interference/options.cpp
    benchmarks/workloads/matmul/cpu/options.cpp
    benchmarks/workloads/dispatch/options.cpp
    benchmarks/workloads/instrumentation/options.cpp
    benchmarks/workloads/followups/options.cpp
    benchmarks/workloads/matmul/metal/options.cpp
    benchmarks/workloads/memory/options.cpp
    benchmarks/workloads/synchronization/options.cpp
    benchmarks/workloads/thermal/options.cpp
  )
  target_link_libraries(computeedge_tests PRIVATE computeedge::computeedge computeedge::harness_gpu
    GTest::gtest_main computeedge_compile_options)

  include(GoogleTest)
  gtest_discover_tests(computeedge_tests)

  # Automated public header self-containment checks
  ce_setup_header_checks()

  # Layering checks (plan §3.2): core links with no Apple frameworks, and public
  # headers expose no Objective-C.
  add_executable(core_only_check tests/architecture/core_only_check.cpp)
  target_link_libraries(core_only_check PRIVATE computeedge::core computeedge_compile_options)
  add_test(NAME layering.core_links_without_frameworks COMMAND core_only_check)
  add_test(NAME data.results_unchanged
    COMMAND sh "${CMAKE_CURRENT_SOURCE_DIR}/scripts/results_manifest.sh" check)
  add_test(NAME layering.no_objc_in_public_headers
    COMMAND sh -c "! grep -rnE '#import|@interface|@protocol|id<|NSObject' '${CMAKE_CURRENT_SOURCE_DIR}/include'")
endif()
