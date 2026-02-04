#include "dcu_tensor.h"
#include "dcu_common.h"
#include <hip/hip_runtime.h>
#include <iostream>

namespace dcu {
void DCUTensor::allocate()
{
    // 根据 bytes 分配内存
    CHECK_HIP(hipMalloc(&data_, bytes()));
}

bool DCUTensor::is_contiguous() const
{
    return strides_ == compute_contiguous_strides(shape_);
}

void DCUTensor::init_contiguous_strides()
{
    // Tensor 基类方法，计算连续的strides
    strides_ = compute_contiguous_strides(shape_);
}

// 分配 GPU 上 shape / strides
void DCUTensor::allocate_device_metadata() const
{
    if (!d_shape_) {
        CHECK_HIP(hipMalloc(&d_shape_, s_size()));
        CHECK_HIP(hipMemcpy(d_shape_, shape_.data(), s_size(), hipMemcpyHostToDevice));
    }
    if (!d_strides_) {
        CHECK_HIP(hipMalloc(&d_strides_, s_size()));
        CHECK_HIP(hipMemcpy(d_strides_, strides_.data(), s_size(), hipMemcpyHostToDevice));
    }
}

DCUTensor::DCUTensor(const std::vector<int>& shape, engine::DataType dtype)
    : shape_(shape),
      dtype_(dtype),
      owns_data_(true)
{
    init_contiguous_strides();
    allocate();
}

DCUTensor::DCUTensor(const void* host_data,
                     const std::vector<int>& shape,
                     engine::DataType dtype)
    : shape_(shape),
      dtype_(dtype),
      owns_data_(true)
{
    init_contiguous_strides();
    allocate();
    CHECK_HIP(hipMemcpy(data_, host_data, bytes(), hipMemcpyHostToDevice));
}

DCUTensor::DCUTensor(void* host_data,
                     const std::vector<int>& shape,
                     const std::vector<int>& strides,
                     engine::DataType dtype,
                     bool owns_data,
                     std::shared_ptr<DCUTensor> base)
    : data_(host_data), // 只传指针
      shape_(shape),
      strides_(strides),
      dtype_(base -> dtype()),
      owns_data_(owns_data),
      base_(base)
{
}

DCUTensor::~DCUTensor()
{
    if (owns_data_ && !base_ && data_) {
        CHECK_HIP(hipFree(data_));
    }
    if (d_shape_) CHECK_HIP(hipFree(d_shape_));
    if (d_strides_) CHECK_HIP(hipFree(d_strides_));
}

size_t DCUTensor::size() const {
    size_t s = 1;
    for (int d : shape_) s *= d;
    return s;
}

size_t DCUTensor::offset(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::runtime_error("offset: index dimension mismatch");
    }
    size_t off = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::runtime_error("offset: index out of bounds");
        }
        off += indices[i] * strides_[i];
    }
    return off;
}

// 返回 GPU 上 shape / strides
int* DCUTensor::d_shape() const {
    allocate_device_metadata();
    return d_shape_;
}

int* DCUTensor::d_strides() const {
    allocate_device_metadata();
    return d_strides_;
}

void DCUTensor::debug_info(const std::string& tag) const {
    std::cout << "\n=== DCUTensor Debug Info [" << tag << "] ===" << std::endl;

    // 基本信息
    std::cout << "Data pointer: " << data_ << std::endl;
    std::cout << "Owns data: " << (owns_data_ ? "true" : "false") << std::endl;
    std::cout << "Is contiguous: " << (is_contiguous() ? "true" : "false") << std::endl;
    std::cout << "Total elements: " << size() << std::endl;
    std::cout << "Total bytes: " << bytes() << " bytes" << std::endl;

    // 形状信息
    std::cout << "Shape (" << shape_.size() << "D): [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 步幅信息
    std::cout << "Strides (elements): [";
    for (size_t i = 0; i < strides_.size(); ++i) {
        std::cout << strides_[i];
        if (i < strides_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // 设备元数据指针
    std::cout << "Device shape pointer: " << d_shape_ << std::endl;
    std::cout << "Device strides pointer: " << d_strides_ << std::endl;

    // 连续性检查
    if (!shape_.empty()) {
        std::cout << "Contiguity check:" << std::endl;
        int expected_stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            std::cout << "  dim " << i << ": shape=" << shape_[i]
                      << ", stride=" << strides_[i]
                      << ", expected=" << expected_stride;
            if (strides_[i] != expected_stride) {
                std::cout << " MISMATCH";
            } else {
                std::cout << " OK";
            }
            std::cout << std::endl;
            expected_stride *= shape_[i];
        }
    }
    std::cout << "=====================================" << std::endl;
}

} // namespace dcu