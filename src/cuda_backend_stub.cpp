#include "cuda_backend.h"

namespace BSPOT::cuda_backend {

bool enabled() { return false; }

bool projectDots(const scalar*, int, int, const int*, int, const scalar*, scalar*, telemetry::Recorder*) {
    return false;
}

} // namespace BSPOT::cuda_backend
