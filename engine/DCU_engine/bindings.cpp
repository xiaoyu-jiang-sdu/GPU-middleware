#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "dcu_context.h"
#include "dcu_tensor.h"
#include "dcu_engine.h"
#include "dcu_common.h"

namespace py = pybind11;
namespace dcu {
py::array DCUTensor_to_numpy(const dcu::DCUTensor &t) {
    size_t numel = t.size();
    size_t ndim  = t.shape().size();

    if (numel == 0) return py::array(); // 空 tensor

    // 分配连续 CPU buffer
    auto host_data = std::make_shared<std::vector<float>>(numel);

    // 拷贝 GPU 数据到 CPU（连续 buffer）
    CHECK_HIP(hipMemcpy(host_data->data(), t.data(), numel * sizeof(float), hipMemcpyDeviceToHost));

    // 构造 shape 和 strides
    std::vector<ssize_t> shape(t.shape().begin(), t.shape().end());
    std::vector<ssize_t> strides_bytes(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        strides_bytes[i] = static_cast<ssize_t>(t.strides()[i] * sizeof(float));
    }

    // heap 分配 shared_ptr 交给 capsule 管理生命周期
    auto sp_for_capsule = new std::shared_ptr<std::vector<float>>(host_data);
    py::capsule free_when_done(sp_for_capsule, [](void* f){
        auto sp = static_cast<std::shared_ptr<std::vector<float>>*>(f);
        delete sp;
    });

    // 构造 NumPy array
    return py::array(
        py::buffer_info(
            host_data->data(),                 // data pointer
            sizeof(float),                     // 每个元素大小
            py::format_descriptor<float>::format(), // 数据类型
            shape.size(),                      // ndim
            shape,                             // shape
            strides_bytes                       // strides（字节为单位）
        ),
        free_when_done
    );
}

PYBIND11_MODULE(dcu, m) {
    m.doc() = "DCU Engine Python Bindings";

    py::class_<DCUContext>(m, "Context")
        .def(py::init<>());

    // 枚举tensor 数据类型
    py::enum_<engine::DataType>(m, "DataType")
        .value("FLOAT32", engine::DataType::FLOAT32)
        .value("INT32",   engine::DataType::INT32)
        .value("INT64",   engine::DataType::INT64)
        .value("BOOL",    engine::DataType::BOOL)
        .export_values();

    py::class_<DCUTensor, std::shared_ptr<DCUTensor>>(m, "DCUTensor")
        // 原始 shape 构造
        .def(py::init<const std::vector<int>&,
                      engine::DataType>(),
             py::arg("shape"),
             py::arg("dtype") = engine::DataType::FLOAT32
        )
        // 属性
        .def_property_readonly("dtype", &DCUTensor::dtype) // 数据类型
        .def_property_readonly("ndim", &DCUTensor::ndim) // n维
        .def_property_readonly("size", &DCUTensor::size) // 元素总数
        .def_property_readonly("bytes", &DCUTensor::bytes) // 总字节数
        .def_property_readonly("is_contiguous", &DCUTensor::is_contiguous) //是否连续
        .def_property_readonly("shape",
                [](const DCUTensor& self) -> py::tuple {
                    auto& s = self.shape();
                    return py::cast(s);
        })
        //
        .def("to_numpy", &DCUTensor_to_numpy)
        // numpy array 构造 lambda
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            // 获取 shape
            std::vector<int> shape(arr.ndim());
            for (ssize_t i = 0; i < arr.ndim(); ++i)
                shape[i] = static_cast<int>(arr.shape()[i]);

            // 调用现有 DCUTensor(host_data, shape) 构造
            return new DCUTensor(static_cast<const void*>(arr.data()), shape);
        }));

    py::class_<DCUEngine, std::shared_ptr<DCUEngine>>(m, "Engine")
        .def(py::init<>())
        // 二元运算符
        .def("add", &DCUEngine::add)
        .def("sub", &DCUEngine::sub)
        .def("mul", &DCUEngine::mul)
        .def("div", &DCUEngine::div)
        .def("pow", &DCUEngine::pow)
        .def("mod", &DCUEngine::mod)
        .def("mul_scalar", &DCUEngine::mul_scalar)
        // nn 算子
        .def("matmul", &DCUEngine::matmul)
        .def("conv2d", &DCUEngine::conv2d)
        // 激活函数
        .def("relu", &DCUEngine::relu)
        .def("erf", &DCUEngine::erf)
        .def("sqrt", &DCUEngine::sqrt)
        // 池化算子
        .def("max_pool2d", &DCUEngine::max_pool2d)
        .def("global_avg_pool", &DCUEngine::global_avg_pool)
        // 归一化
        .def("batch_norm_2d", &DCUEngine::batch_norm_2d)
        //transform
        .def("flatten", &DCUEngine::flatten)
        .def("concat", &DCUEngine::concat)
        // shape view
        .def("transpose", [](std::shared_ptr<DCUEngine> self,
                                std::shared_ptr<DCUTensor> x,
                                const std::vector<int>& perm,
                                DCUContext& ctx) {
                return self->transpose(x, perm, &ctx);
        })
        .def("unsqueeze", [](std::shared_ptr<DCUEngine> self,
                                std::shared_ptr<DCUTensor> x,
                                const std::vector<int>& axes,
                                DCUContext& ctx) {
            return self->unsqueeze(x, axes, &ctx);
        })
        .def("reshape", [](std::shared_ptr<DCUEngine> self,
                                std::shared_ptr<DCUTensor> x,
                                const std::vector<int>& shape,
                                DCUContext& ctx) {
            return self->reshape(x, shape, &ctx);
        });
}

} // namespace dcu
