#include "utils.h"

void load_camera_params(context *ctx)
{
    printf("loading camera params: %s\n", param_path);
    
    FILE *file = fopen(param_path, "r");
    char temps[100];
    int camID;
    fscanf(file, "%f %f\n", &ctx->min_depth, &ctx->max_depth);

    for (int cam_idx = 0; cam_idx < camera_num; cam_idx++)
    {
        fscanf(file, "%s %d\n", temps, &camID);
        fscanf(file, "%s %d %d\n", temps, &ctx->width, &ctx->height);
        assert(ctx->width == WIDTH);
        assert(ctx->height == HEIGHT);

        // K matrix
        fscanf(file, "%s %f %f %f %f\n", temps, &ctx->krt[cam_idx].fx, // fx
                                                &ctx->krt[cam_idx].fy, // fy
                                                &ctx->krt[cam_idx].cx, // x0
                                                &ctx->krt[cam_idx].cy);// y0

        // R matrix
        fscanf(file, "%s", temps);
        for (int i = 0; i < 9; i++) {
            fscanf(file, " %f", &ctx->krt[cam_idx].R[i]);
        }
        fgetc(file);

        // world position
        fscanf(file, "%s %f %f %f\n", temps, &ctx->krt[cam_idx].T[0], 
                                             &ctx->krt[cam_idx].T[1], 
                                             &ctx->krt[cam_idx].T[2]);
    }
    fclose(file);
}