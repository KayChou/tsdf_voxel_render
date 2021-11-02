#include <cuda_runtime_api.h>
#include <cuda.h>
#include "cuda_kernel.h"

// ====================================================================
// create context, it contains all data that gpu needs to use
// ====================================================================
context* init_context()
{
    cudaSetDevice(0);
    context* ctx;
    
    cudaMallocManaged((void**)&ctx, sizeof(context));

    ctx->resolution[0] = 500;
    ctx->resolution[1] = 500;
    ctx->resolution[2] = 500;

    ctx->volume_origin[0] = 0;
    ctx->volume_origin[1] = 0;
    ctx->volume_origin[2] = 0;

    ctx->voxel_size = 0.1;
    ctx->trunc_margin = 5 * ctx->voxel_size;

    ctx->tsdf_threshold = 1.0f;

    int voxel_num = ctx->resolution[0] * ctx->resolution[1] * ctx->resolution[2];

    cudaMalloc((void**)&ctx->tsdf_voxel, voxel_num * sizeof(float));
    cudaMalloc((void**)&ctx->weight_voxel, voxel_num * sizeof(float));
    cudaMalloc((void**)&ctx->in_buf_depth, WIDTH * HEIGHT * sizeof(uint8_t));
    cudaMalloc((void**)&ctx->depth, WIDTH * HEIGHT * sizeof(float));

    cudaMemset(ctx->tsdf_voxel, 1, voxel_num * sizeof(float));
    cudaMemset(ctx->weight_voxel, 0, voxel_num * sizeof(float));

    return ctx;
}


__global__ void dequantization_kernel(context* ctx, uint8_t *input_depth, float *output_depth)
{
    int x = blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y * blockDim.x * gridDim.x;

    float maxdisp = fB / ctx->min_depth;
    float mindisp = fB / ctx->max_depth;
    output_depth[idx] = (float)((float)input_depth[idx] / 255.f * (maxdisp - mindisp) + mindisp);
    output_depth[idx] = (float)(fB / (float)output_depth[idx]);
}


// ====================================================================
// dequantization is necessary since depth has been quantizated to 0-255
// ====================================================================
void dequantization(context* ctx, uint8_t *input_depth, float *output_depth)
{
    int width = ctx->width;
    int height = ctx->height;

    dim3 blocks(width / 32, height / 24);
    dim3 threads(32, 24);

    cudaMemset(output_depth, 0, width * height * sizeof(float));
    dequantization_kernel<<<blocks, threads>>>(ctx, input_depth, output_depth);
}


__global__ void integrate_kernel(context* ctx, int cam_idx)
{
    float fx = ctx->krt[cam_idx].fx;
    float fy = ctx->krt[cam_idx].fy;
    float cx = ctx->krt[cam_idx].cx;
    float cy = ctx->krt[cam_idx].cy;

    float* R = ctx->krt[cam_idx].R;
    float* T = ctx->krt[cam_idx].T;

    int z_voxel = blockIdx.x;
    int y_voxel = threadIdx.x;

    int dim_x = ctx->resolution[0];
    int dim_y = ctx->resolution[1];
    int dim_z = ctx->resolution[2];

    float world_x, world_y, world_z;
    float camera_x, camera_y, camera_z;
    int pix_x, pix_y;    

    for(int x_voxel = 0; x_voxel < ctx->resolution[2]; x_voxel++) {
        // convert voxel index to world points position
        world_x = x_voxel * ctx->voxel_size;
        world_y = y_voxel * ctx->voxel_size;
        world_z = z_voxel * ctx->voxel_size;

        // convert point from world to camera coordinate
        world_x -= T[0];
        world_y -= T[1];
        world_z -= T[2];
        camera_x = R[0] * world_x + R[1] * world_y + R[2] * world_z;
        camera_y = R[3] * world_x + R[4] * world_y + R[5] * world_z;
        camera_z = R[6] * world_x + R[7] * world_y + R[8] * world_z;

        if(camera_z <= 0) {
            continue;
        }

        // convert point from camera to pixel coorinate
        pix_x = roundf(fx * camera_x / camera_z + cx);
        pix_y = roundf(fy * camera_y / camera_z + cy);

        if(pix_x < 0 || pix_x >= WIDTH || pix_y < 0 || pix_y >= HEIGHT) {
            continue;
        }

        float depth_value = ctx->depth[pix_y * WIDTH + pix_x];

        float diff = depth_value - camera_z;
        if (diff <= -ctx->trunc_margin) {
            continue;
        }

        int voxel_idx = z_voxel * dim_y * dim_x + y_voxel * dim_x + x_voxel;
        float dist = fmin(1.0f, diff / ctx->trunc_margin);

        float weight = ctx->weight_voxel[voxel_idx];
        ctx->tsdf_voxel[voxel_idx] = (ctx->tsdf_voxel[voxel_idx] * weight + dist) / (weight + 1.0f);
        ctx->weight_voxel[voxel_idx] += 1.0f;
    }
}


// ====================================================================
// core function, integrate an depth frame to volume
// ====================================================================
void Integrate(context* ctx, int cam_idx, uint8_t *in_buf_depth)
{
#ifdef TimeEventRecord
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
#endif

    cudaMemcpy(ctx->in_buf_depth, in_buf_depth, WIDTH * HEIGHT * sizeof(uint8_t), cudaMemcpyHostToDevice);
    dequantization(ctx, ctx->in_buf_depth, ctx->depth);

    integrate_kernel<<<ctx->resolution[2], ctx->resolution[1]>>>(ctx, cam_idx);

#ifdef TimeEventRecord
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float millisecond = 0;
    cudaEventElapsedTime(&millisecond, start, end);
    printf("\t Integrate time = %f ms\n", millisecond);
#endif
}


void memcpy_volume_to_cpu(context* ctx, float* tsdf_out, float* weight_out)
{
    int voxel_num = ctx->resolution[0] * ctx->resolution[1] * ctx->resolution[2];

    printf("voxel num: %d\n", voxel_num);
    cudaMemcpy(tsdf_out, ctx->tsdf_voxel, voxel_num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weight_out, ctx->weight_voxel, voxel_num * sizeof(float), cudaMemcpyDeviceToHost);
}


// ====================================================================
// release memory in GPU
// ====================================================================
void release_context(context* ctx)
{
    cudaFree(ctx->tsdf_voxel);
    cudaFree(ctx->weight_voxel);
    cudaFree(ctx->in_buf_depth);
    cudaFree(ctx->depth);
    cudaFree(ctx);
}
