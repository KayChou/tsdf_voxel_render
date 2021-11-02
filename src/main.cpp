#include <stdio.h>
#include "utils.h"
#include "cuda_kernel.h"


int main(int argc, char** argv)
{
    context* ctx = init_context(); // create ctx in unified memory which can be accessed both in cpu and gpu
    load_camera_params(ctx);

    char depth_filename[200];
    char color_filename[200];
    uint8_t *in_buf_depth = new uint8_t[WIDTH * HEIGHT * sizeof(uint8_t)];
    uint8_t *in_buf_color = new uint8_t[WIDTH * HEIGHT * sizeof(uint8_t) * 3];

    int voxel_num = ctx->resolution[0] * ctx->resolution[1] * ctx->resolution[2];
    float* tsdf_cpu = new float[voxel_num];
    float* weight_cpu = new float[voxel_num];

    for(int i = 0; i < camera_num; i++) {
        // read an depth image
        sprintf(depth_filename, "%s/depth_yuv/%d.yuv", data_path, i);
        sprintf(color_filename, "%s/image_yuv/%d.yuv", data_path, i);
        printf("Integrating %2d/%d: %s\n", i + 1, camera_num, color_filename);

        FILE* depth_fp = fopen(depth_filename, "rb");
        FILE* color_fp = fopen(color_filename, "rb");
        fread(in_buf_depth, 1, WIDTH * HEIGHT, depth_fp);
        fread(in_buf_color, 1, WIDTH * HEIGHT * 3, depth_fp);

        // integrate current frame to volume in GPU
        for(int k = 0; k<1; k++) {
            Integrate(ctx, i, in_buf_depth);
        }
    }

    // memcpy volume to CPU
    memcpy_volume_to_cpu(ctx, tsdf_cpu, weight_cpu);

    char ply_filename[200] = "../fusion.ply";
    save_volume_to_ply(ctx, ply_filename, tsdf_cpu, weight_cpu);
    


    // write point cloud to local .ply file
    release_context(ctx);
    delete [] in_buf_depth;
    delete [] in_buf_color;
    delete [] tsdf_cpu;
    delete [] weight_cpu;
    return 0;
}