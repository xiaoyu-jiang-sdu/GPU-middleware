cmake_minimum_required(VERSION 3.20)
project(engine LANGUAGES CXX)

# 设置 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加公共 include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加 DCU_engine 子目录
add_subdirectory(DCU_engine)
