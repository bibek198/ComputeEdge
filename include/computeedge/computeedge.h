#pragma once

// Umbrella header for the ComputeEdge core and CPU libraries. The Metal
// backend (computeedge/metal/*) and the benchmark harness (computeedge/bench/*)
// are included separately.

#include "computeedge/core/allocator.h"
#include "computeedge/core/error.h"
#include "computeedge/core/instrumentation.h"
#include "computeedge/core/scoped_timer.h"
#include "computeedge/core/tensor.h"
#include "computeedge/core/validate.h"
#include "computeedge/cpu/matmul.h"
