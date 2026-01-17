#pragma once
#include <vector>

namespace engine {

enum class DeviceType {
    DCU
};

class Tensor {
public:
    virtual ~Tensor() = default;

    // 返回设备数据指针
    virtual void* data() const = 0;

    // 返回张量形状
    virtual const std::vector<int>& shape() const = 0;

    // 返回设备类型
    virtual DeviceType device() const = 0;
};

} // namespace engine
