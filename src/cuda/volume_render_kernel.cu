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


__global__ void integrate_kernel(context* ctx)
{
    int z_voxel = threadIdx.x + blockIdx.x * blockDim.x;
    int y_voxel = threadIdx.y + blockIdx.y * blockDim.y;

    float world_x, world_y, world_z;
    float weight;

    __shared__ int L1_cnt[32][32]; // if 1: need to be spilt to L1 voxel
    L1_cnt[threadIdx.x][threadIdx.y] = 0;

    __shared__ baseVoxel L0_voxel[32][32];
    __shared__ KRT cam_pose[CAM_NUM];
    if(threadIdx.x == 0 && threadIdx.y == 0) {
        memcpy(cam_pose, ctx->krt, CAM_NUM * sizeof(KRT));
    }
    __syncthreads();

    // each cuda thread handles one volume line(x axis)
    for(int x_voxel = 0; x_voxel < DIM_X; x_voxel++) {
        weight = 0;
        int voxel_idx = z_voxel * DIM_Y * DIM_X + y_voxel * DIM_X + x_voxel;
        
        // convert voxel index to world points position
        world_x = world_x0 + x_voxel * ctx->voxel_size;
        world_y = world_y0 + y_voxel * ctx->voxel_size;
        world_z = world_z0 + z_voxel * ctx->voxel_size;

        integrate_one_voxel(world_x, world_y, world_z, 
                            ctx,
                            cam_pose,
                            weight,
                            L0_voxel[threadIdx.x][threadIdx.y].tsdf,
                            L0_voxel[threadIdx.x][threadIdx.y].rgb[0],
                            L0_voxel[threadIdx.x][threadIdx.y].rgb[1],
                            L0_voxel[threadIdx.x][threadIdx.y].rgb[2]);

        // copy tsdf and color from shared memory to global memory
        ctx->tsdf_voxel[voxel_idx] = (weight < WEIGHT_THRESHOLD) ? 2 * TSDF_THRESHOLD_L0 : L0_voxel[threadIdx.x][threadIdx.y].tsdf;
        ctx->color_voxel[3 * voxel_idx + 0] = L0_voxel[threadIdx.x][threadIdx.y].rgb[0];
        ctx->color_voxel[3 * voxel_idx + 1] = L0_voxel[threadIdx.x][threadIdx.y].rgb[1];
        ctx->color_voxel[3 * voxel_idx + 2] = L0_voxel[threadIdx.x][threadIdx.y].rgb[2];
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
