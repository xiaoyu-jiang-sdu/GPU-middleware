#pragma once
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <rocblas/rocblas.h>
#include "context.h"
#include "dcu_common.h"

namespace dcu {

class DCUContext : public engine::Context {
public:
    DCUContext();
    ~DCUContext();

    void sync() override { CHECK_HIP(hipStreamSynchronize(stream_)); }

    hipStream_t& get_stream() { return stream_; }
    miopenHandle_t& get_miopen() { return miopen_; }
    rocblas_handle& get_rocblas() { return rocblas_; }

private:
    hipStream_t stream_;
    miopenHandle_t miopen_;
    rocblas_handle rocblas_;
};

} // namespace dcu