cmake_minimum_required(VERSION 3.5)

project(main LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include")

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

SET(CUDA_PATH ${PROJECT_SOURCE_DIR}/src/cuda/build)
LINK_LIBRARIES(${CUDA_PATH}/libvoxel_render.so)

add_executable(main ${SOURCES})