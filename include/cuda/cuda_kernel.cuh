#pragma once
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <thrust/extrema.h>
#include "cuda_kernel.h"
#include "device_launch_parameters.h"
#include "utils.h"

#define HANDLE_ERROR() (HandleError(__FILE__, __LINE__))

static void HandleError(const char *file, int line) {
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)  {
        fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(cudaStatus), cudaGetErrorString(cudaStatus), file, line);
        exit(int(cudaStatus));
    }
}

__global__ void dequantization_kernel(context* ctx, uint8_t *input_depth, float *output_depth);
__global__ void integrate_kernel(context* ctx);
__global__ void depth_to_world_pcd(context* ctx, int cam_idx);
