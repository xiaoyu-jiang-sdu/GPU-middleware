#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "dcu_context.h"
#include "dcu_tensor.h"
#include "dcu_engine.h"
#include "dcu_common.h"
#include "dispatch/dispatch.h"

namespace py = pybind11;
namespace dcu {
template <typename T>
py::array DCUTensor_to_numpy_impl(const dcu::DCUTensor& t)
{
    size_t numel = t.size();
    size_t ndim  = t.ndim();

    if (numel == 0) {
        return py::array(py::dtype::of<T>());
    }

    // 分配 host buffer
    auto host_data = std::make_shared<std::vector<T>>(numel);

    CHECK_HIP(hipMemcpy(
        host_data->data(),
        t.data(),
        numel * dtype_size(t.dtype()),
        hipMemcpyDeviceToHost
    ));

    if constexpr (std::is_same_v<T, uint8_t>) {
        for (size_t i = 0; i < numel; ++i) {
            (*host_data)[i] = ((*host_data)[i] != 0) ? 1 : 0;
        }
    }

    // shape
    std::vector<ssize_t> shape(t.shape().begin(), t.shape().end());

    // strides (bytes)
    std::vector<ssize_t> strides_bytes(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        strides_bytes[i] = static_cast<ssize_t>(t.strides()[i] * sizeof(T));
    }

    // capsule 管理生命周期
    auto* sp_for_capsule =
        new std::shared_ptr<std::vector<T>>(host_data);

    py::capsule free_when_done(sp_for_capsule, [](void* f) {
        delete static_cast<std::shared_ptr<std::vector<T>>*>(f);
    });

    return py::array(
        py::buffer_info(
            host_data->data(),
            sizeof(T),
            py::format_descriptor<T>::format(),
            ndim,
            shape,
            strides_bytes
        ),
        free_when_done
    );
}

py::array DCUTensor_to_numpy(const dcu::DCUTensor& t)
{
    py::array result;

    DISPATCH_ALL_TYPE(t.dtype(), [&](auto dtype_enum) {
        using T = typename engine::CType<decltype(dtype_enum)::value>::type;

        result = DCUTensor_to_numpy_impl<T>(t);
    });

    return result;
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
        .value("UINT8",   engine::DataType::UINT8)
        .export_values();

    py::class_<DCUTensor, std::shared_ptr<DCUTensor>>(m, "DCUTensor")
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
        // to_np cp到cpu上
        .def("to_numpy", &DCUTensor_to_numpy)
        // 构造函数
        // 原始 shape 构造
        .def(py::init<const std::vector<int>&,
                      engine::DataType>(),
             py::arg("shape"),
             py::arg("dtype") = engine::DataType::FLOAT32
        )
        .def(py::init([](py::array arr, engine::DataType dtype = engine::DataType::FLOAT32) {
            // 推断 shape
            std::vector<int> shape(arr.ndim());
            for (ssize_t i = 0; i < arr.ndim(); ++i)
                shape[i] = static_cast<int>(arr.shape()[i]);
            // 创建 DCUTensor
            return new DCUTensor(static_cast<const void*>(arr.data()), shape, dtype);
        }),
            py::arg("array"),
            py::arg("dtype") = engine::DataType::FLOAT32
        );

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
        //logical 算子
        .def("equal", &DCUEngine::equal)
        .def("not_equal", &DCUEngine::not_equal)
        .def("where", &DCUEngine::where)
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
        // type cast
        .def("cast", &DCUEngine::cast)
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
