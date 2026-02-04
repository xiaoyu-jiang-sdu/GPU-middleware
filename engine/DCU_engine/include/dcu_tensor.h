#pragma once
#include <vector>
#include <memory>
#include <hip/hip_runtime.h>
#include "tensor.h"
#include "dtype.h"

namespace dcu {

class DCUTensor : public engine::Tensor {
public:
    // 只传入shape构造tensor
    explicit DCUTensor(const std::vector<int>& shape,
                       engine::DataType dtype = engine::DataType::FLOAT32);
    // 传入host_data 和 shape构造tensor, cp host_data
    explicit DCUTensor(const void* host_data,
                       const std::vector<int>& shape,
                       engine::DataType dtype = engine::DataType::FLOAT32);
    // view 构造 不分配内存）
    explicit DCUTensor(void* host_data,
              const std::vector<int>& shape,
              const std::vector<int>& strides,
              engine::DataType dtype,
              bool owns_data,
              std::shared_ptr<DCUTensor> base);

    ~DCUTensor();

    void* data() const override { return data_; }

    const std::vector<int>& shape() const override { return shape_; }
    const std::vector<int>& strides() const override { return strides_; }

    // GPU 上 shape / strides
    int* d_shape() const;
    int* d_strides() const;

    engine::DeviceType device() const override { return device_; }

    // 是否内存连续分配
    bool is_contiguous() const override;

    // 元素类型、大小
    engine::DataType dtype() const { return dtype_; }
    size_t element_size() const { return engine::dtype_size(dtype_); }

    // 总元素个数
    size_t size() const;

    // 总元素占据空间大小
    size_t bytes() const { return size() * element_size();}

    // shape 和 stride 的 size
    size_t s_size() const { return sizeof(int) * shape_.size();}

    // 根据索引计算偏移
    size_t offset(const std::vector<int>& indices) const;

    // n维
    int ndim() const { return static_cast<int>(shape_.size());}

    // 调试输出
    void debug_info(const std::string& tag = "") const;
private:
    std::vector<int> shape_;
    std::vector<int> strides_; // 单位为元素

    mutable int* d_shape_{nullptr};
    mutable int* d_strides_{nullptr};

    void* data_{nullptr};
    bool owns_data_{true};

    engine::DeviceType device_{engine::DeviceType::DCU};
    engine::DataType dtype_; // data type

    std::shared_ptr<DCUTensor> base_{nullptr};  // 保证 data_ 不会提前释放

    void allocate();
    void init_contiguous_strides();
    void allocate_device_metadata() const; // GPU 上分配shape、strides
};

} // namespace dcu
