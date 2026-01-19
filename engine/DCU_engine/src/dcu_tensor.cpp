#include "dcu_tensor.h"
#include "dcu_common.h"
#include <hip/hip_runtime.h>

namespace dcu {

void DCUTensor::allocate()
{
    size_t bytes = size() * sizeof(float);
    CHECK_HIP(hipMalloc(&data_, bytes));
}

DCUTensor::DCUTensor(const std::vector<int>& shape): shape_(shape)
{
    allocate();
}

DCUTensor::DCUTensor(const float* host_data,
                     const std::vector<int>& shape): shape_(shape)
{
    allocate();
    CHECK_HIP(hipMemcpy(data_, host_data,
                            size() * sizeof(float),
                            hipMemcpyHostToDevice));
}

DCUTensor::~DCUTensor()
{
    CHECK_HIP(hipFree(data_));
}

} // namespace dcu