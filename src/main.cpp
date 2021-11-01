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

    float* volume = new float[ctx->resolution[0] * ctx->resolution[1], ctx->resolution[2]];

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
        for(int k = 0; k<100; k++) {
            Integrate(ctx, i, in_buf_depth, volume);
        }
    }

    // memcpy volume to CPU
    


    // write point cloud to local .ply file


    release_context(ctx);
    delete [] in_buf_depth;
    delete [] in_buf_color;
    delete [] volume;
    return 0;
}