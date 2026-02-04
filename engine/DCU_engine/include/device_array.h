#pragma once
#include "dcu_common.h"
#include <vector>
#include <hip/hip_runtime.h>
#include <stdexcept>

namespace dcu {

template <typename T>
class DeviceArray {
public:
    DeviceArray() = default;

    DeviceArray(size_t n, hipStream_t stream = nullptr)
        : size_(n)
    {
        if (n == 0) return;
        CHECK_HIP(hipMalloc(&data_, n * sizeof(T)));
    }

    DeviceArray(const std::vector<T>& host,
                hipStream_t stream = nullptr)
        : DeviceArray(host.size(), stream)
    {
        if (size_ > 0) {
            CHECK_HIP(hipMemcpyAsync(
                data_, host.data(),
                size_ * sizeof(T),
                hipMemcpyHostToDevice,
                stream
            ));
        }
    }

    ~DeviceArray() {
        if (data_) CHECK_HIP(hipFree(data_));
    }

    // 禁止拷贝
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    // 允许移动
    DeviceArray(DeviceArray&& other) noexcept {
        move_from(other);
    }

    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            cleanup();
            move_from(other);
        }
        return *this;
    }

    T* data() const { return data_; }
    size_t size() const { return size_; }

private:
    T* data_ = nullptr;
    size_t size_ = 0;

    void cleanup() {
        if (data_) hipFree(data_);
        data_ = nullptr;
        size_ = 0;
    }

    void move_from(DeviceArray& other) {
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
};

} // namespace dcu
