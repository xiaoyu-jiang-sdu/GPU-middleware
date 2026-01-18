#include "dcu_context.h"
#include "dcu_common.h"

namespace dcu {

DCUContext::DCUContext() {
    CHECK_HIP(hipStreamCreate(&stream_));
    miopenCreate(&miopen_);
    miopenSetStream(miopen_, stream_);
    rocblas_create_handle(&rocblas_);
    rocblas_set_stream(rocblas_, stream_);
}

DCUContext::~DCUContext() {
    rocblas_destroy_handle(rocblas_);
    miopenDestroy(miopen_);
    CHECK_HIP(hipStreamDestroy(stream_));
}

} // namespace dcu