#include "cuda_kernel.cuh"

// ====================================================================
// create context, it contains all data that gpu needs to use
// ====================================================================
context* init_context()
{
    cudaSetDevice(1);
    context* ctx;
    
    cudaMallocManaged((void**)&ctx, sizeof(context));

    ctx->resolution[0] = DIM_X;
    ctx->resolution[1] = DIM_Y;
    ctx->resolution[2] = DIM_Z;

    ctx->trunc_margin = 5 * VOXEL_SIZE;

    ctx->weight_threshhold = WEIGHT_THRESHOLD;
    ctx->L0_voxel_num = 0;
    ctx->L2_voxel_num = 0;

    int voxel_num = ctx->resolution[0] * ctx->resolution[1] * ctx->resolution[2];

    cudaMalloc((void**)&ctx->tsdf_voxel, voxel_num * sizeof(float));
    cudaMalloc((void**)&ctx->color_voxel, voxel_num * sizeof(uint8_t) * 3);
    cudaMalloc((void**)&ctx->valid_voxel, voxel_num * sizeof(baseVoxel));
    cudaMalloc((void**)&ctx->in_buf_depth, CAM_NUM * WIDTH * HEIGHT * sizeof(uint8_t));
    cudaMalloc((void**)&ctx->in_buf_color, CAM_NUM * WIDTH * HEIGHT * sizeof(uint8_t) * 3);
    cudaMalloc((void**)&ctx->depth, CAM_NUM * WIDTH * HEIGHT * sizeof(float));
    cudaMalloc((void**)&ctx->pcd, 3 * WIDTH * HEIGHT * sizeof(float));

    cudaMemset(ctx->tsdf_voxel, 1, voxel_num * sizeof(float));
    cudaMemset(ctx->color_voxel, 0, voxel_num * sizeof(uint8_t) * 3);
    HANDLE_ERROR();

    return ctx;
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


// ====================================================================
// core function, integrate an depth frame to volume
// ====================================================================
void Integrate(context* ctx, uint8_t *in_buf_depth, uint8_t* in_buf_color)
{
#ifdef TimeEventRecord
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
#endif
    reset_context(ctx);
    cudaMemcpy(ctx->in_buf_depth, in_buf_depth, CAM_NUM * WIDTH * HEIGHT * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->in_buf_color, in_buf_color, CAM_NUM * 3 * WIDTH * HEIGHT * sizeof(uint8_t), cudaMemcpyHostToDevice);

    for(int i = 0; i < CAM_NUM; i++) {
        dequantization(ctx, ctx->in_buf_depth + WIDTH * HEIGHT * i, ctx->depth + WIDTH * HEIGHT * i);
        HANDLE_ERROR();
    }

    Lock *lock;
    cudaMallocManaged((void**)&lock, sizeof(Lock));

    integrate_L0_kernel<<<dim3(DIM_Z / 32, DIM_Y / 32), dim3(32, 32)>>>(ctx, lock);
    cudaDeviceSynchronize(); // force cpu to wait util kernel finish

    int block_config = (ctx->L0_voxel_num + 255) / 256;
    integrate_L1_kernel<<<block_config, 256>>>(ctx, lock);
    
    HANDLE_ERROR();

#ifdef TimeEventRecord
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float millisecond = 0;
    cudaEventElapsedTime(&millisecond, start, end);
    printf("\t Integrate time = %f ms\n", millisecond);
#endif
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


void memcpy_volume_to_cpu(context* ctx, baseVoxel* voxel_out, int &voxel_num)
{
    voxel_num = ctx->L0_voxel_num + ctx->L1_voxel_num + ctx->L2_voxel_num;
    voxel_num = ctx->L1_voxel_num;
    cudaMemcpy(voxel_out, ctx->valid_voxel + ctx->L0_voxel_num, voxel_num * sizeof(baseVoxel), cudaMemcpyDeviceToHost);
}


void reset_context(context* ctx)
{
    ctx->L0_voxel_num = 0;
    ctx->L1_voxel_num = 0;
    ctx->L2_voxel_num = 0;
}


// ====================================================================
// release memory in GPU
// ====================================================================
void release_context(context* ctx)
{
    cudaFree(ctx->tsdf_voxel);
    cudaFree(ctx->color_voxel);
    cudaFree(ctx->valid_voxel);
    cudaFree(ctx->in_buf_depth);
    cudaFree(ctx->in_buf_color);
    cudaFree(ctx->depth);
    cudaFree(ctx->pcd);
    cudaFree(ctx);
}
