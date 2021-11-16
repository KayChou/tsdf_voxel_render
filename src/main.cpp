#include <stdio.h>
#include "utils.h"
#include "cuda_kernel.h"


int main(int argc, char** argv)
{
    context* ctx = init_context(); // create ctx in unified memory which can be accessed both in cpu and gpu
    load_camera_params(ctx);

    char depth_filename[200];
    char color_filename[200];
    uint8_t *in_buf_depth = new uint8_t[CAM_NUM * WIDTH * HEIGHT * sizeof(uint8_t)];
    uint8_t *in_buf_color = new uint8_t[CAM_NUM * WIDTH * HEIGHT * sizeof(uint8_t) * 3];

    int voxel_num = ctx->resolution[0] * ctx->resolution[1] * ctx->resolution[2];
    float* tsdf_cpu = new float[voxel_num];
    uint8_t* color_cpu = new uint8_t[voxel_num * 3];

    float* point_cloud = new float[WIDTH * HEIGHT * 3];

    // read RGB-D from all views to in_buf_color and in_buf_depth
    for(int i = 0; i < CAM_NUM; i++) {
        // read RGBD image
        sprintf(depth_filename, "%s/depth_yuv/%d.yuv", data_path, i);
        sprintf(color_filename, "%s/image_yuv/%d.yuv", data_path, i);
        // printf("Integrating %2d/%d: %s\n", i + 1, CAM_NUM, color_filename);

        FILE* depth_fp = fopen(depth_filename, "rb");
        FILE* color_fp = fopen(color_filename, "rb");
        fread(in_buf_depth + WIDTH * HEIGHT * i, 1, WIDTH * HEIGHT, depth_fp);
        fread(in_buf_color + WIDTH * HEIGHT * 3 * i, 1, WIDTH * HEIGHT * 3, color_fp);

#if GEN_PCD_OF_EACH_CAM
        // convert depth to world coordinate and save
        get_pcd_in_world(ctx, in_buf_depth, point_cloud, i);
        char pcd_filename[200];
        sprintf(pcd_filename, "../results/camera_%d.ply", i);
        save_pcd_as_ply(pcd_filename, point_cloud);
#endif

        fclose(depth_fp);
        fclose(color_fp);
    }


    // integrate current frame to volume in GPU
    printf("begin to integrate\n");
    for(int i = 0; i < 10; i++) {
        Integrate(ctx, in_buf_depth, in_buf_color);
    }

    // memcpy volume to CPU
    memcpy_volume_to_cpu(ctx, tsdf_cpu, color_cpu);
    save_volume_to_ply(ctx, "../results/fusion.ply", tsdf_cpu, color_cpu);

    release_context(ctx);
    delete [] in_buf_depth;
    delete [] in_buf_color;
    delete [] tsdf_cpu;
    delete [] color_cpu;
    delete [] point_cloud;
    return 0;
}