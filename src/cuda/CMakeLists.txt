project(dibr_gpu)
cmake_minimum_required(VERSION 3.5)

# packages
find_package(CUDA)

add_definitions("-DTimeEventRecord")

INCLUDE_DIRECTORIES("../../include")

# nvcc flags -g for debug
#set(CUDA_NVCC_FLAGS -O3;-G;-g)

# set(CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75;--generate-line-info)
set(CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86;)
# set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
# set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g;-rdc=true)

file(GLOB_RECURSE CURRENT_HEADERS  *.h *.hpp *.cuh)
file(GLOB CURRENT_SOURCES  *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES}) 

cuda_add_library(voxel_render SHARED ${CURRENT_HEADERS} ${CURRENT_SOURCES})
# cuda_add_library(dibr_gpu STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})