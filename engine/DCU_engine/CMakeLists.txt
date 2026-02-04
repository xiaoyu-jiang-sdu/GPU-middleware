cmake_minimum_required(VERSION 3.20)
project(dcu_engine LANGUAGES CXX)

# 强制定义 HIP 平台
add_compile_definitions(__HIP_PLATFORM_AMD__=1)

# 用 hipcc 编译
set(CMAKE_C_COMPILER /opt/dtk/bin/hipcc)
set(CMAKE_CXX_COMPILER /opt/dtk/bin/hipcc)

# 设置 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# DTK 根目录
set(DTK_ROOT /opt/dtk-25.04.4)

# include & lib
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${DTK_ROOT}/include
)

link_directories(${DTK_ROOT}/lib)

# 收集所有 cpp 文件
file(GLOB_RECURSE ENGINE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)
# 查找 pybind11
find_package(pybind11 REQUIRED)

# 设置生成共享库用于 Python
add_library(dcu_engine SHARED ${ENGINE_SOURCES})

# 公共 include 路径
target_include_directories(dcu_engine
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${DTK_ROOT}/include
)

# 公共 DTK 库链接
set(DTK_LIBS
    ${DTK_ROOT}/lib/libamdhip64.so
    ${DTK_ROOT}/lib/libMIOpen.so
    ${DTK_ROOT}/lib/librocblas.so
    ${DTK_ROOT}/lib/libhipfft.so
    ${DTK_ROOT}/lib/libhiprand.so
    ${DTK_ROOT}/lib/libhipblas.so
    ${DTK_ROOT}/lib/libhipblaslt.so
)

target_link_libraries(dcu_engine
    PUBLIC
    ${DTK_LIBS}
)

# 添加 bindings.cpp
set(BINDINGS_SRC bindings.cpp)

# 生成 pybind11 模块
pybind11_add_module(dcu ${BINDINGS_SRC})

# 设置输出在modules
set_target_properties(dcu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/modules
)

# 链接 DCU Engine 库和 DTK
target_link_libraries(dcu PRIVATE dcu_engine ${DTK_LIBS})

# 设置 include 路径
target_include_directories(dcu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${DTK_ROOT}/include
)