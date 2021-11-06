#include <cuda_runtime_api.h>
#include <cuda.h>
#include <thrust/extrema.h>
#include "cuda_kernel.h"
#include "device_launch_parameters.h"

// ====================================================================
// create context, it contains all data that gpu needs to use
// ====================================================================
context* init_context()
{
    cudaSetDevice(0);
    context* ctx;
    
    cudaMallocManaged((void**)&ctx, sizeof(context));

    ctx->resolution[0] = DIM_X;
    ctx->resolution[1] = DIM_Y;
    ctx->resolution[2] = DIM_Z;

    ctx->voxel_size = VOXEL_SIZE;
    ctx->trunc_margin = 5 * ctx->voxel_size;

    ctx->tsdf_threshold = TSDF_THRESHOLD;
    ctx->weight_threshhold = WEIGHT_THRESHOLD;

    int voxel_num = ctx->resolution[0] * ctx->resolution[1] * ctx->resolution[2];

    cudaMalloc((void**)&ctx->tsdf_voxel, voxel_num * sizeof(float));
    cudaMalloc((void**)&ctx->weight_voxel, voxel_num * sizeof(float));
    cudaMalloc((void**)&ctx->color_voxel, voxel_num * sizeof(uint8_t) * 3);

    cudaMalloc((void**)&ctx->in_buf_depth, WIDTH * HEIGHT * sizeof(uint8_t));
    cudaMalloc((void**)&ctx->in_buf_color, WIDTH * HEIGHT * sizeof(uint8_t) * 3);
    cudaMalloc((void**)&ctx->depth, WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void**)&ctx->pcd, 3 * WIDTH * HEIGHT * sizeof(float));

    cudaMemset(ctx->tsdf_voxel, 1, voxel_num * sizeof(float));
    cudaMemset(ctx->weight_voxel, 0, voxel_num * sizeof(float));
    cudaMemset(ctx->color_voxel, 0, voxel_num * sizeof(uint8_t) * 3);

    return ctx;
}


__global__ void dequantization_kernel(context* ctx, uint8_t *input_depth, float *output_depth)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y * blockDim.x * gridDim.x;

    if(input_depth[idx] == 0) {
        output_depth[idx] = 0;
        return;
    }

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

    float world_x, world_y, world_z;
    float camera_x, camera_y, camera_z;
    int pix_x, pix_y;
    float old_r, old_g, old_b;
    float new_r, new_g, new_b; 

    for(int x_voxel = 0; x_voxel < ctx->resolution[2]; x_voxel++) {
        // convert voxel index to world points position
        world_x = world_x0 + x_voxel * ctx->voxel_size;
        world_y = world_y0 + y_voxel * ctx->voxel_size;
        world_z = world_z0 + z_voxel * ctx->voxel_size;

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
        int pix_idx = pix_y * WIDTH + pix_x;

        if(pix_x < 0 || pix_x >= WIDTH || pix_y < 0 || pix_y >= HEIGHT) {
            continue;
        }

        float depth_value = ctx->depth[pix_idx];
        new_r = ctx->in_buf_color[3 * pix_idx + 0];
        new_g = ctx->in_buf_color[3 * pix_idx + 1];
        new_b = ctx->in_buf_color[3 * pix_idx + 2];
        if(depth_value == 0 || new_r == 0 || new_g == 0 || new_b == 0) {
            continue;
        }

        float diff = depth_value - camera_z;
        if (diff <= -ctx->trunc_margin) {
            continue;
        }

        int voxel_idx = z_voxel * dim_y * dim_x + y_voxel * dim_x + x_voxel;
        float dist = fmin(1.0f, diff / ctx->trunc_margin);

        // update TSDF and weight
        float weight = ctx->weight_voxel[voxel_idx];
        ctx->tsdf_voxel[voxel_idx] = (ctx->tsdf_voxel[voxel_idx] * weight + dist) / (weight + 1.0f);
        ctx->weight_voxel[voxel_idx] += 1.0f;


        // update color
        old_r = ctx->color_voxel[3 * voxel_idx + 0];
        old_g = ctx->color_voxel[3 * voxel_idx + 1];
        old_b = ctx->color_voxel[3 * voxel_idx + 2];

        ctx->color_voxel[3 * voxel_idx + 0] = (uint8_t)fminf((float)(old_r * weight + new_r * 1.0f) / (weight + 1.0f), 255);
        ctx->color_voxel[3 * voxel_idx + 1] = (uint8_t)fminf((float)(old_g * weight + new_g * 1.0f) / (weight + 1.0f), 255);
        ctx->color_voxel[3 * voxel_idx + 2] = (uint8_t)fminf((float)(old_b * weight + new_b * 1.0f) / (weight + 1.0f), 255);
    }
}


// ====================================================================
// core function, integrate an depth frame to volume
// ====================================================================
void Integrate(context* ctx, int cam_idx, uint8_t *in_buf_depth, uint8_t* in_buf_color)
{
#ifdef TimeEventRecord
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
#endif

    cudaMemcpy(ctx->in_buf_depth, in_buf_depth, WIDTH * HEIGHT * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->in_buf_color, in_buf_color, 3 * WIDTH * HEIGHT * sizeof(uint8_t), cudaMemcpyHostToDevice);
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


__global__ void depth_to_world_pcd(context* ctx, int cam_idx) 
{
    float fx = ctx->krt[cam_idx].fx;
    float fy = ctx->krt[cam_idx].fy;
    float cx = ctx->krt[cam_idx].cx;
    float cy = ctx->krt[cam_idx].cy;

    float* R = ctx->krt[cam_idx].R;
    float* T = ctx->krt[cam_idx].T;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y * blockDim.x * gridDim.x;

    float depth_val = ctx->depth[idx];
    if(depth_val == 0) {
        return;
    }

    float camera_x = (x - cx) * depth_val / fx;
    float camera_y = (y - cy) * depth_val / fy;
    float camera_z = depth_val;

    float world_x = R[0] * camera_x + R[3] * camera_y + R[6] * camera_z + T[0];
    float world_y = R[1] * camera_x + R[4] * camera_y + R[7] * camera_z + T[1];
    float world_z = R[2] * camera_x + R[5] * camera_y + R[8] * camera_z + T[2];

    ctx->pcd[3 * idx + 0] = world_x;
    ctx->pcd[3 * idx + 1] = world_y;
    ctx->pcd[3 * idx + 2] = world_z;
}


void get_pcd_in_world(context* ctx, uint8_t *in_buf_depth, float *pcd, int cam_idx)
{
    cudaMemcpy(ctx->in_buf_depth, in_buf_depth, WIDTH * HEIGHT * sizeof(uint8_t), cudaMemcpyHostToDevice);
    dequantization(ctx, ctx->in_buf_depth, ctx->depth);

    dim3 blocks(WIDTH / 32, HEIGHT / 24);
    dim3 threads(32, 24);

    depth_to_world_pcd<<<blocks, threads>>>(ctx, cam_idx);

    cudaMemcpy(pcd, ctx->pcd, 3 * WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
}


void memcpy_volume_to_cpu(context* ctx, float* tsdf_out, float* weight_out, uint8_t* rgb_out)
{
    int voxel_num = ctx->resolution[0] * ctx->resolution[1] * ctx->resolution[2];

    printf("voxel num: %d\n", voxel_num);
    cudaMemcpy(tsdf_out, ctx->tsdf_voxel, voxel_num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(weight_out, ctx->weight_voxel, voxel_num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(rgb_out, ctx->color_voxel, voxel_num * sizeof(uint8_t) * 3, cudaMemcpyDeviceToHost);
}


// ====================================================================
// release memory in GPU
// ====================================================================
void release_context(context* ctx)
{
    cudaFree(ctx->tsdf_voxel);
    cudaFree(ctx->weight_voxel);
    cudaFree(ctx->in_buf_depth);
    cudaFree(ctx->in_buf_color);
    cudaFree(ctx->depth);
    cudaFree(ctx->pcd);
    cudaFree(ctx);
}
