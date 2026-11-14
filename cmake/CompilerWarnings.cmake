# Target-scoped compiler warning flags (no global add_compile_options).

set(CE_WARNINGS -Wall -Wextra -Wpedantic -Wshadow)

# CI turns warnings into errors for ComputeEdge's own targets only; third-party
# code (googletest via FetchContent) never receives these flags.
if(CE_WARNINGS_AS_ERRORS)
  list(APPEND CE_WARNINGS -Werror)
endif()

# Core interface library for target-specific compile options.
if(NOT TARGET computeedge_compile_options)
  add_library(computeedge_compile_options INTERFACE)
  target_compile_features(computeedge_compile_options INTERFACE cxx_std_20)
  target_compile_options(computeedge_compile_options INTERFACE
    ${CE_WARNINGS}
    $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
  )
endif()
