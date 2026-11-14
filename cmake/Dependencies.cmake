# External framework and testing dependencies

# Accelerate framework (CPU BLAS)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

# GoogleTest (for testing)
macro(ce_setup_googletest)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endmacro()
