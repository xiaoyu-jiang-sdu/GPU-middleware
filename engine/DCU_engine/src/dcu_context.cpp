//
// Created by 31437 on 26-1-16.
//
#include "dcu/dcu_context.h"
#include <iostream>

namespace dcu {

DCUContext::DCUContext() {
    hipStreamCreate(&stream_);
    miopenCreate(&miopen_);
    miopenSetStream(miopen_, stream_);
    rocblas_create_handle(&rocblas_);
    rocblas_set_stream(rocblas_, stream_);
}

DCUContext::~DCUContext() {
    rocblas_destroy_handle(rocblas_);
    miopenDestroy(miopen_);
    hipStreamDestroy(stream_);
}

} // namespace dcu
