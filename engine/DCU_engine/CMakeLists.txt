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

# 创建静态库 dcu_engine
add_library(dcu_engine STATIC ${ENGINE_SOURCES})

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
)

target_link_libraries(dcu_engine
    PUBLIC
    ${DTK_LIBS}
)

# 测试可执行文件
add_executable(test_engine src/test_engine.cpp)

# 测试程序链接 dcu_engine（自动继承 include & lib）
target_link_libraries(test_engine
    PRIVATE
    dcu_engine
)

# 设置运行时库路径
set_target_properties(test_engine PROPERTIES
    BUILD_RPATH "${DTK_ROOT}/lib"
)
