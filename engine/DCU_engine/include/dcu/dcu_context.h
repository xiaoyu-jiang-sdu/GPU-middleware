//
// Created by 31437 on 26-1-16.
//

#ifndef DCU_CONTEXT_H
#define DCU_CONTEXT_H

#endif //DCU_CONTEXT_H
#pragma once
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <rocblas/rocblas.h>

namespace dcu {

class DCUContext {
public:
    DCUContext();
    ~DCUContext();

    miopenHandle_t& get_handle() { return miopen_; }
    rocblas_handle& get_rocblas() { return rocblas_; }

private:
    hipStream_t stream_;
    miopenHandle_t miopen_;
    rocblas_handle rocblas_;
};

} // namespace dcu
