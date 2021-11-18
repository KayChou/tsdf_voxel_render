#pragma once
#include "typedef.cuh"

#define HANDLE_ERROR() (HandleError(__FILE__, __LINE__))

__global__ void dequantization_kernel(context* ctx, uint8_t *input_depth, float *output_depth);
__global__ void integrate_L0_kernel(context* ctx, Lock *lock);
__global__ void integrate_L1_kernel(context* ctx, Lock *lock);
__global__ void depth_to_world_pcd(context* ctx, int cam_idx);

void reset_context(context* ctx);

static void HandleError(const char *file, int line) {
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)  {
        fprintf(stderr, "Error %d: \"%s\" in %s at line %d\n", int(cudaStatus), cudaGetErrorString(cudaStatus), file, line);
        exit(int(cudaStatus));
    }
}