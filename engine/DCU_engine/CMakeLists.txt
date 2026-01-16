cmake_minimum_required(VERSION 3.20)
project(dcu_engine LANGUAGES CXX)

# 强制定义 HIP 平台
add_compile_definitions(__HIP_PLATFORM_AMD__=1)

# 用 hipcc 编译
set(CMAKE_C_COMPILER /opt/dtk/bin/hipcc)
set(CMAKE_CXX_COMPILER /opt/dtk/bin/hipcc)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# DTK 根目录
set(DTK_ROOT /opt/dtk-25.04.4)

# include & lib
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${DTK_ROOT}/include
)

# 收集所有 cpp 文件
file(GLOB_RECURSE ENGINE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# 静态库 dcu_engine
add_library(dcu_engine STATIC ${ENGINE_SOURCES})

# 链接 DTK 库（使用绝对路径，确保找到库）
target_link_libraries(dcu_engine
    PRIVATE
    ${DTK_ROOT}/lib/libamdhip64.so
    ${DTK_ROOT}/lib/libMIOpen.so
    ${DTK_ROOT}/lib/librocblas.so
    ${DTK_ROOT}/lib/libhipfft.so
    ${DTK_ROOT}/lib/libhiprand.so
    ${DTK_ROOT}/lib/libhipblas.so
)

# 测试可执行文件
add_executable(test_engine src/test_engine.cpp)

# 链接 dcu_engine 静态库和 DTK 库
target_link_libraries(test_engine
    PRIVATE
    dcu_engine
    ${DTK_ROOT}/lib/libamdhip64.so
    ${DTK_ROOT}/lib/libMIOpen.so
    ${DTK_ROOT}/lib/librocblas.so
    ${DTK_ROOT}/lib/libhipfft.so
    ${DTK_ROOT}/lib/libhiprand.so
    ${DTK_ROOT}/lib/libhipblas.so
)

# 确保编译器能找到头文件
target_include_directories(test_engine
    PRIVATE
    include
    ${DTK_ROOT}/include
)

# 设置运行时库路径
set_target_properties(test_engine PROPERTIES
    BUILD_RPATH "${DTK_ROOT}/lib"
)
