# Apple Metal runtime shader embedding and framework configuration.
# Kernels are compiled at runtime (newLibraryWithSource:), so no offline
# Metal toolchain is needed; the .metal sources are embedded via kernel_sources.h.

set(CE_METAL_KERNELS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels/matmul.metal
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernels/probes.metal)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CE_METAL_KERNELS})

file(READ src/metal/kernels/matmul.metal CE_MSL_MATMUL)
file(READ src/metal/kernels/probes.metal CE_MSL_PROBES)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/kernel_sources.h.in
  ${CMAKE_BINARY_DIR}/generated/computeedge/kernel_sources.h
  @ONLY
)
