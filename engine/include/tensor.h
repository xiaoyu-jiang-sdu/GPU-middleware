#pragma once
#include <vector>

namespace engine {

enum class DeviceType {
    CPU,
    DCU
};

class Tensor {
public:
    virtual ~Tensor() = default;

    // 返回设备数据指针
    virtual void* data() const = 0;

    // 返回张量形状
    virtual const std::vector<int>& shape() const = 0;

    // strides
    virtual const std::vector<int>& strides() const = 0;

    // 返回设备类型
    virtual DeviceType device() const = 0;

    // 内存是否连续分配
    virtual bool is_contiguous() const = 0;

    // 计算连续分配的strides
    static std::vector<int> compute_contiguous_strides(const std::vector<int>& shape)
{
    std::vector<int> strides(shape.size());
    int stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

};

} // namespace engine
