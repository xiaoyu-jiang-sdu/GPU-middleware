#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "dcu_context.h"
#include "dcu_tensor.h"
#include "dcu_engine.h"
#include "dcu_common.h"
#include <hip/hip_runtime.h>

namespace py = pybind11;
namespace dcu {

py::array_t<float> DCUTensor_to_numpy(const DCUTensor &t) {
    std::vector<float> host_data(t.size());
    CHECK_HIP(hipMemcpy(host_data.data(), t.data(), sizeof(float) * t.size(), hipMemcpyDeviceToHost));
    std::vector<ssize_t> shape(t.shape().begin(), t.shape().end());
    return py::array_t<float>(shape, host_data.data());
}
PYBIND11_MODULE(dcu, m) {
    m.doc() = "DCU Engine Python Bindings";

    py::class_<DCUContext>(m, "Context")
        .def(py::init<>());

    py::class_<DCUTensor>(m, "DCUTensor")
        // 原始 shape 构造
        .def(py::init<const std::vector<int>&>())
        // size & shape
        .def("size", &DCUTensor::size)
        .def("shape", &DCUTensor::shape_list)
        .def("to_numpy", &DCUTensor_to_numpy)
        // numpy array 构造 lambda
        .def(py::init([](py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
            // 获取 shape
            std::vector<int> shape(arr.ndim());
            for (ssize_t i = 0; i < arr.ndim(); ++i)
                shape[i] = static_cast<int>(arr.shape()[i]);

            // 调用现有 DCUTensor(host_data, shape) 构造
            return new DCUTensor(static_cast<const float*>(arr.data()), shape);
        }));

    py::class_<DCUEngine>(m, "Engine")
        .def(py::init<>())
        .def("add", &DCUEngine::add)
        .def("matmul", &DCUEngine::matmul)
        .def("relu", &DCUEngine::relu)
        .def("transpose", &DCUEngine::transpose)
        .def("mul_scalar", &DCUEngine::mul_scalar)
        .def("conv2d", &DCUEngine::conv2d)
        .def("max_pool2d", &DCUEngine::max_pool2d)
        .def("global_avg_pool", &DCUEngine::global_avg_pool)
        .def("flatten", &DCUEngine::flatten)
        .def("batch_norm_2d", &DCUEngine::batch_norm_2d);
}

} // namespace dcu
