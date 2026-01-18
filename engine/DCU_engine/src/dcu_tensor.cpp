#include "dcu_tensor.h"
#include "dcu_common.h"
#include <hip/hip_runtime.h>

namespace dcu {

DCUTensor::DCUTensor(const std::vector<int>& shape) : shape_(shape) {
    size_t bytes = sizeof(float);
    for (int d : shape_) bytes *= d;
    CHECK_HIP(hipMalloc(&data_, bytes));
}

DCUTensor::~DCUTensor() {
    CHECK_HIP(hipFree(data_));
}

} // namespace dcu