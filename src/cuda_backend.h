#pragma once

#include "BSPOT.h"
#include "telemetry.h"

namespace BSPOT::cuda_backend {

bool enabled();

bool projectDots(const scalar* points, int dim, int cols, const int* ids, int count, const scalar* dir, scalar* out,
                 telemetry::Recorder* rec = nullptr);

} // namespace BSPOT::cuda_backend
