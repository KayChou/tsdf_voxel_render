#include "cuda_kernel.cuh"

// input xyz in world, output tsdf and color
__device__ void integrate_one_voxel(const float x, 
                                    const float y, 
                                    const float z,
                                    context* ctx,
                                    KRT* Krt,
                                    float &weight,
                                    float &tsdf,
                                    uint8_t &r,
                                    uint8_t &g, 
                                    uint8_t &b) {
    uint8_t new_r, new_g, new_b;
    float camera_x, camera_y, camera_z;
    int pix_x, pix_y;

    tsdf = 0;
    weight = 0;
    r = 0;
    g = 0;
    b = 0;

    for(int cam_idx = 0; cam_idx < CAM_NUM; cam_idx++) {
        float fx = Krt[cam_idx].fx;
        float fy = Krt[cam_idx].fy;
        float cx = Krt[cam_idx].cx;
        float cy = Krt[cam_idx].cy;
        float* R = Krt[cam_idx].R;
        float* T = Krt[cam_idx].T;

        // convert point from world to camera coordinate
        camera_x = R[0] * (x - T[0]) + R[1] * (y - T[1]) + R[2] * (z - T[2]);
        camera_y = R[3] * (x - T[0]) + R[4] * (y - T[1]) + R[5] * (z - T[2]);
        camera_z = R[6] * (x - T[0]) + R[7] * (y - T[1]) + R[8] * (z - T[2]);

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

        float depth_value = ctx->depth[WIDTH * HEIGHT * cam_idx + pix_idx];
        if(depth_value == 0) {
            continue;
        }

        float diff = depth_value - camera_z;
        if (diff <= -ctx->trunc_margin) {
            continue;
        }
        float dist = fmin(1.0f, diff / ctx->trunc_margin);

        // update TSDF
        tsdf = (tsdf * weight + dist) / (weight + 1.0f);

        // update color
        new_r = ctx->in_buf_color[WIDTH * HEIGHT * 3 * cam_idx + 3 * pix_idx + 0];
        new_g = ctx->in_buf_color[WIDTH * HEIGHT * 3 * cam_idx + 3 * pix_idx + 1];
        new_b = ctx->in_buf_color[WIDTH * HEIGHT * 3 * cam_idx + 3 * pix_idx + 2];

        r = (uint8_t)fminf((float)(r * weight + new_r * 1.0f) / (weight + 1.0f), 255);
        g = (uint8_t)fminf((float)(g * weight + new_g * 1.0f) / (weight + 1.0f), 255);
        b = (uint8_t)fminf((float)(b * weight + new_b * 1.0f) / (weight + 1.0f), 255);

        // update weight
        weight += 1.0f;
    }
    return;
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


__global__ void integrate_L0_kernel(context* ctx, Lock *lock)
{
    int z_voxel = threadIdx.x + blockIdx.x * blockDim.x;
    int y_voxel = threadIdx.y + blockIdx.y * blockDim.y;
    int cnt;

    float world_x, world_y, world_z;
    float weight;
    baseVoxel L0_voxel;

    __shared__ int L0_cnt[32][32]; // cnt L1 voxels' num in one block(32 * 32 threads)
    __shared__ int L0_idx[32][32][8]; // 32 * 32 * DIM_X voxel has 32 * DIM_X * surfaces at max
    __shared__ KRT cam_pose[CAM_NUM];

    if(threadIdx.x == 0 && threadIdx.y == 0) {
        memcpy(cam_pose, ctx->krt, CAM_NUM * sizeof(KRT));
    }

    // each cuda thread handles one volume line(x axis), with batch size 8
    // (because shared memory is limited to 48KB)
    for(int batch = 0; batch < DIM_X / 8; batch++) {
        int startIdx = 8 * batch;
        L0_cnt[threadIdx.x][threadIdx.y] = 0;
        __syncthreads();
        for(int x_voxel = startIdx; x_voxel < startIdx + 8; x_voxel++) {
            weight = 0;
            int voxel_idx = z_voxel * DIM_Y * DIM_X + y_voxel * DIM_X + x_voxel;
            
            // convert voxel index to world points position
            world_x = world_x0 + x_voxel * VOXEL_SIZE;
            world_y = world_y0 + y_voxel * VOXEL_SIZE;
            world_z = world_z0 + z_voxel * VOXEL_SIZE;

            // compute one voxel's tsdf, weight and color
            integrate_one_voxel(world_x, world_y, world_z, 
                                ctx,
                                cam_pose,
                                weight,
                                L0_voxel.tsdf,
                                L0_voxel.rgb[0],
                                L0_voxel.rgb[1],
                                L0_voxel.rgb[2]);

            // copy tsdf and color from shared memory to global memory
            float tsdf_tmp = (weight < WEIGHT_THRESHOLD) ? 2 * TSDF_THRESHOLD_L0 : L0_voxel.tsdf;
            ctx->tsdf_voxel[voxel_idx] = tsdf_tmp;
            ctx->color_voxel[3 * voxel_idx + 0] = L0_voxel.rgb[0];
            ctx->color_voxel[3 * voxel_idx + 1] = L0_voxel.rgb[1];
            ctx->color_voxel[3 * voxel_idx + 2] = L0_voxel.rgb[2];

            if(abs(tsdf_tmp) < TSDF_THRESHOLD_L0) {
                L0_idx[threadIdx.x][threadIdx.y][L0_cnt[threadIdx.x][threadIdx.y]] = voxel_idx;
                L0_cnt[threadIdx.x][threadIdx.y] += 1;
            }
            __syncthreads();
        }

        __syncthreads(); // wait all threads finish

        // count L1 voxels num, if no L1 voxel, then no need to lock since lock is expensive
        cnt = 0;
        for(int i = 0; i < blockDim.x; i++) {
            for(int j = 0; j < blockDim.y; j++) {
                cnt += L0_cnt[i][j];
            }
        }
        __syncthreads();

        if(cnt > 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            lock->lock(); // lock, make sure each block runs in serial
            __threadfence(); // this is critical to make sure it is locked
            
            for(int i = 0; i < blockDim.x; i++) {
                for(int j = 0; j < blockDim.y; j++) {
                    for(int k = 0; k < L0_cnt[i][j]; k++) {
                        int voxel_idx = L0_idx[i][j][k];
                        int ptr = ctx->L0_voxel_num;
                        
                        int x_idx = voxel_idx % DIM_X;
                        int y_idx = ((voxel_idx - x_idx) / DIM_X) % DIM_Y;
                        int z_idx = (voxel_idx - x_idx - y_idx * DIM_X) / (DIM_Y * DIM_X);

                        ctx->valid_voxel[ptr].tsdf = ctx->tsdf_voxel[voxel_idx];
                        ctx->valid_voxel[ptr].x = world_x0 + x_idx * VOXEL_SIZE;
                        ctx->valid_voxel[ptr].y = world_y0 + y_idx * VOXEL_SIZE;
                        ctx->valid_voxel[ptr].z = world_z0 + z_idx * VOXEL_SIZE;
                        ctx->valid_voxel[ptr].rgb[0] = ctx->color_voxel[3 * voxel_idx + 0];
                        ctx->valid_voxel[ptr].rgb[1] = ctx->color_voxel[3 * voxel_idx + 1];
                        ctx->valid_voxel[ptr].rgb[2] = ctx->color_voxel[3 * voxel_idx + 2];
                        ctx->L0_voxel_num += 1;
                    }
                }
            }
            lock->unlock();
        }
        __syncthreads();
    }
}


__global__ void integrate_L1_kernel(context* ctx, Lock *lock)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= ctx->L0_voxel_num) {
        return;
    }

    float weight;
    int tid = threadIdx.x;

    __shared__ KRT cam_pose[CAM_NUM];

    if(threadIdx.x == 0 && threadIdx.y == 0) {
        memcpy(cam_pose, ctx->krt, CAM_NUM * sizeof(KRT));
    }

    __shared__ baseVoxel L1_voxel[256][8];
    __shared__ int cnt;
    cnt = 0;

    __syncthreads();

    for(int i = 0; i < 8; i++) { // each L0 voxel has 2^3 = 8 L1 voxels
        L1_voxel[tid][i].x = ctx->valid_voxel[idx].x + (2*(i/4) - 1) * VOXEL_SIZE / 2.0f;
        L1_voxel[tid][i].y = ctx->valid_voxel[idx].y + (2*((i % 4) / 2) - 1) * VOXEL_SIZE / 2.0f;
        L1_voxel[tid][i].z = ctx->valid_voxel[idx].z + (2*(i % 2) - 1) * VOXEL_SIZE / 2.0f;

        // compute one voxel's tsdf, weight and color
        integrate_one_voxel(L1_voxel[tid][i].x,
                            L1_voxel[tid][i].y,
                            L1_voxel[tid][i].z,
                            ctx,
                            cam_pose,
                            weight,
                            L1_voxel[tid][i].tsdf,
                            L1_voxel[tid][i].rgb[0],
                            L1_voxel[tid][i].rgb[1],
                            L1_voxel[tid][i].rgb[2]);
        // copy tsdf and color from shared memory to global memory
        float tsdf_tmp = (weight < WEIGHT_THRESHOLD) ? 2 * TSDF_THRESHOLD_L0 : L1_voxel[tid][i].tsdf;
        if(tsdf_tmp < TSDF_THRESHOLD_L1) {
            atomicAdd(&cnt, 1);
        }
        L1_voxel[tid][i].tsdf = tsdf_tmp;
    }
    __syncthreads();

    if(cnt > 0 && threadIdx.x == 0) {
        lock->lock();
        __threadfence();
        for(int i = 0; i < blockDim.x; i++) {
            if(idx < ctx->L0_voxel_num) {
                for(int j = 0; j < 8; j++) {
                    float tsdf_tmp = L1_voxel[i][j].tsdf;
                    if(abs(tsdf_tmp) < TSDF_THRESHOLD_L1) {
                        int ptr = ctx->L0_voxel_num + ctx->L1_voxel_num;

                        ctx->valid_voxel[ptr].tsdf = L1_voxel[i][j].tsdf;
                        ctx->valid_voxel[ptr].x = isnan(L1_voxel[i][j].x) ? world_x0 : L1_voxel[i][j].x;
                        ctx->valid_voxel[ptr].y = isnan(L1_voxel[i][j].y) ? world_y0 : L1_voxel[i][j].y;
                        ctx->valid_voxel[ptr].z = isnan(L1_voxel[i][j].z) ? world_z0 : L1_voxel[i][j].z;
                        ctx->valid_voxel[ptr].rgb[0] = L1_voxel[i][j].rgb[0];
                        ctx->valid_voxel[ptr].rgb[1] = L1_voxel[i][j].rgb[1];
                        ctx->valid_voxel[ptr].rgb[2] = L1_voxel[i][j].rgb[2];
                        ctx->L1_voxel_num += 1;
                    }
                }
            }
        }
        lock->unlock();
    }
}


__global__ void integrate_L2_kernel(context* ctx, Lock *lock)
{
    int idx = ctx->L0_voxel_num + threadIdx.x + blockIdx.x * blockDim.x;

    if(idx >= ctx->L0_voxel_num + ctx->L1_voxel_num) {
        return;
    }

    float weight;
    int tid = threadIdx.x;

    __shared__ KRT cam_pose[CAM_NUM];

    if(threadIdx.x == 0) {
        memcpy(cam_pose, ctx->krt, CAM_NUM * sizeof(KRT));
    }

    __shared__ baseVoxel L2_voxel[256][8];
    __shared__ int cnt;
    cnt = 0;

    __syncthreads();

    for(int i = 0; i < 8; i++) { // each L0 voxel has 2^3 = 8 L1 voxels
        L2_voxel[tid][i].x = ctx->valid_voxel[idx].x + (2*(i/4) - 1) * VOXEL_SIZE / 2.0f;
        L2_voxel[tid][i].y = ctx->valid_voxel[idx].y + (2*((i % 4) / 2) - 1) * VOXEL_SIZE / 2.0f;
        L2_voxel[tid][i].z = ctx->valid_voxel[idx].z + (2*(i % 2) - 1) * VOXEL_SIZE / 2.0f;

        // compute one voxel's tsdf, weight and color
        integrate_one_voxel(L2_voxel[tid][i].x,
                            L2_voxel[tid][i].y,
                            L2_voxel[tid][i].z,
                            ctx,
                            cam_pose,
                            weight,
                            L2_voxel[tid][i].tsdf,
                            L2_voxel[tid][i].rgb[0],
                            L2_voxel[tid][i].rgb[1],
                            L2_voxel[tid][i].rgb[2]);
        // copy tsdf and color from shared memory to global memory
        float tsdf_tmp = (weight < WEIGHT_THRESHOLD) ? 2 * TSDF_THRESHOLD_L0 : L2_voxel[tid][i].tsdf;
        if(tsdf_tmp < TSDF_THRESHOLD_L1) {
            atomicAdd(&cnt, 1);
        }
        L2_voxel[tid][i].tsdf = tsdf_tmp;
    }
    __syncthreads();

    if(cnt > 0 && threadIdx.x == 0) {
        lock->lock();
        __threadfence();
        for(int i = 0; i < blockDim.x; i++) {
            if(idx < ctx->L0_voxel_num + ctx->L1_voxel_num) {
                for(int j = 0; j < 8; j++) {
                    float tsdf_tmp = L2_voxel[i][j].tsdf;
                    if(abs(tsdf_tmp) < TSDF_THRESHOLD_L1) {
                        int ptr = ctx->L0_voxel_num + ctx->L1_voxel_num + ctx->L2_voxel_num;

                        ctx->valid_voxel[ptr].tsdf = L2_voxel[i][j].tsdf;
                        ctx->valid_voxel[ptr].x = isnan(L2_voxel[i][j].x) ? world_x0 : L2_voxel[i][j].x;
                        ctx->valid_voxel[ptr].y = isnan(L2_voxel[i][j].y) ? world_y0 : L2_voxel[i][j].y;
                        ctx->valid_voxel[ptr].z = isnan(L2_voxel[i][j].z) ? world_z0 : L2_voxel[i][j].z;
                        ctx->valid_voxel[ptr].rgb[0] = L2_voxel[i][j].rgb[0];
                        ctx->valid_voxel[ptr].rgb[1] = L2_voxel[i][j].rgb[1];
                        ctx->valid_voxel[ptr].rgb[2] = L2_voxel[i][j].rgb[2];
                        ctx->L2_voxel_num += 1;
                    }
                }
            }
        }
        lock->unlock();
    }
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
