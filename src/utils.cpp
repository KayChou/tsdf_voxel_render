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

        printf("world position: %f %f %f\n", ctx->krt[cam_idx].T[0], ctx->krt[cam_idx].T[1], ctx->krt[cam_idx].T[2]);
    }
    fclose(file);
}


void save_volume_to_ply(context *ctx, char *filename, float* tsdf, float* weight)
{
    int dim_x = ctx->resolution[0];
    int dim_y = ctx->resolution[1];
    int dim_z = ctx->resolution[2];
    int voxel_num = ctx->resolution[0] * ctx->resolution[1] * ctx->resolution[2];

    printf("threshold: %f %f\n", ctx->tsdf_threshold, ctx->weight_threshhold);

    // count total num of points in point cloud
    int num_pts = 0;
    for(int i = 0; i < voxel_num; i++) {
        if(std::abs(tsdf[i]) < ctx->tsdf_threshold && weight[i] > ctx->weight_threshhold) {
            num_pts++;
        }
    }

    printf("valid points: %d\n", num_pts);

    // Create header for .ply file
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %d\n", num_pts);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "end_header\n");

    for(int i = 0; i < voxel_num; i++) {
        if(std::abs(tsdf[i]) < ctx->tsdf_threshold && weight[i] > ctx->weight_threshhold) {
            int z = floor(i / (dim_x * dim_y));
            int y = floor((i - (z * dim_x * dim_y)) / dim_x);
            int x = i - (z * dim_x * dim_y) - y * dim_x;
            
            float pt_x = (float)x * ctx->voxel_size;
            float pt_y = (float)y * ctx->voxel_size;
            float pt_z = (float)z * ctx->voxel_size;

            fwrite(&pt_x, sizeof(float), 1, fp);
            fwrite(&pt_y, sizeof(float), 1, fp);
            fwrite(&pt_z, sizeof(float), 1, fp);
        }
    }
    fclose(fp);
}