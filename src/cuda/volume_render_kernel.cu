#include "cuda_kernel.cuh"

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
    float camera_x, camera_y, camera_z;
    int pix_x, pix_y;
    float old_r, old_g, old_b;
    float new_r, new_g, new_b;
    float weight;

    __shared__ baseVoxel L0_voxel[32][32];

    // each cuda thread handles one volume line(x axis)
    for(int x_voxel = 0; x_voxel < ctx->resolution[2]; x_voxel++) {
        weight = 0;
        int voxel_idx = z_voxel * DIM_Y * DIM_X + y_voxel * DIM_X + x_voxel;
        // for each voxel, loop for all views
        for(int cam_idx = 0; cam_idx < CAM_NUM; cam_idx++) {
            float fx = ctx->krt[cam_idx].fx;
            float fy = ctx->krt[cam_idx].fy;
            float cx = ctx->krt[cam_idx].cx;
            float cy = ctx->krt[cam_idx].cy;

            float* R = ctx->krt[cam_idx].R;
            float* T = ctx->krt[cam_idx].T;
        
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

            float depth_value = ctx->depth[WIDTH * HEIGHT * cam_idx + pix_idx];
            new_r = ctx->in_buf_color[WIDTH * HEIGHT * 3 * cam_idx + 3 * pix_idx + 0];
            new_g = ctx->in_buf_color[WIDTH * HEIGHT * 3 * cam_idx + 3 * pix_idx + 1];
            new_b = ctx->in_buf_color[WIDTH * HEIGHT * 3 * cam_idx + 3 * pix_idx + 2];
            if(depth_value == 0 || new_r == 0 || new_g == 0 || new_b == 0) {
                continue;
            }

            float diff = depth_value - camera_z;
            if (diff <= -ctx->trunc_margin) {
                continue;
            }

            float dist = fmin(1.0f, diff / ctx->trunc_margin);

            // update TSDF and weight
            L0_voxel[threadIdx.x][threadIdx.y].tsdf = (L0_voxel[threadIdx.x][threadIdx.y].tsdf * weight + dist) / (weight + 1.0f);
            // ctx->tsdf_voxel[voxel_idx] = (ctx->tsdf_voxel[voxel_idx] * weight + dist) / (weight + 1.0f);
            weight += 1.0f;

            // update color
            old_r = L0_voxel[threadIdx.x][threadIdx.y].rgb[0];
            old_g = L0_voxel[threadIdx.x][threadIdx.y].rgb[1];
            old_b = L0_voxel[threadIdx.x][threadIdx.y].rgb[2];

            L0_voxel[threadIdx.x][threadIdx.y].rgb[0] = (uint8_t)fminf((float)(old_r * weight + new_r * 1.0f) / (weight + 1.0f), 255);
            L0_voxel[threadIdx.x][threadIdx.y].rgb[1] = (uint8_t)fminf((float)(old_g * weight + new_g * 1.0f) / (weight + 1.0f), 255);
            L0_voxel[threadIdx.x][threadIdx.y].rgb[2] = (uint8_t)fminf((float)(old_b * weight + new_b * 1.0f) / (weight + 1.0f), 255);
        }

        // copy tsdf and color from shared memory to global memory
        if(weight < WEIGHT_THRESHOLD) {
            ctx->tsdf_voxel[voxel_idx] = 2 * TSDF_THRESHOLD;
        }
        else {
            ctx->tsdf_voxel[voxel_idx] = L0_voxel[threadIdx.x][threadIdx.y].tsdf;
        }
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
