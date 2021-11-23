#include "utils.h"

void load_camera_params(context *ctx)
{
    printf("loading camera params: %s\n", param_path);
    
    FILE *file = fopen(param_path, "r");
    char temps[100];
    int camID;
    fscanf(file, "%f %f\n", &ctx->min_depth, &ctx->max_depth);

    for (int cam_idx = 0; cam_idx < CAM_NUM; cam_idx++)
    {
        fscanf(file, "%s %d\n", temps, &camID);
        fscanf(file, "%s %d %d\n", temps, &ctx->width, &ctx->height);
        assert(ctx->width == WIDTH);
        assert(ctx->height == HEIGHT);

        // K matrix
        fscanf(file, "%s %f %f %f %f\n", temps, &ctx->krt[camID].fx, // fx
                                                &ctx->krt[camID].fy, // fy
                                                &ctx->krt[camID].cx, // x0
                                                &ctx->krt[camID].cy);// y0

        // R matrix
        fscanf(file, "%s", temps);
        for (int i = 0; i < 9; i++) {
            fscanf(file, " %f", &ctx->krt[camID].R[i]);
        }
        fgetc(file);

        // world position
        fscanf(file, "%s %f %f %f\n", temps, &ctx->krt[camID].T[0], 
                                             &ctx->krt[camID].T[1], 
                                             &ctx->krt[camID].T[2]);

        // printf("world position: %f %f %f\n", ctx->krt[cam_idx].T[0], ctx->krt[cam_idx].T[1], ctx->krt[cam_idx].T[2]);
    }
    fclose(file);
}


void save_volume_to_ply(context *ctx, char *filename, baseVoxel* voxels, int num_pts)
{
    int dim_x = ctx->resolution[0];
    int dim_y = ctx->resolution[1];
    int dim_z = ctx->resolution[2];
    int voxel_num = ctx->resolution[0] * ctx->resolution[1] * ctx->resolution[2];

    printf("threshold: %f %f\n", TSDF_THRESHOLD_L0, ctx->weight_threshhold);

    // count total num of points in point cloud
    printf("valid points: %d\n", num_pts);

    // Create header for .ply file
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %d\n", num_pts);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "property uchar red\n");
    fprintf(fp, "property uchar green\n");
    fprintf(fp, "property uchar blue\n");
    fprintf(fp, "end_header\n");

    for(int i = 0; i < num_pts; i++) {
        if(std::abs(voxels[i].tsdf) < TSDF_THRESHOLD_L0) {
            int z = floor(i / (dim_x * dim_y));
            int y = floor((i - (z * dim_x * dim_y)) / dim_x);
            int x = i - (z * dim_x * dim_y) - y * dim_x;
            
            float pt_x = std::fmax(world_x0 - VOXEL_SIZE * DIM_X, std::fmin((float)(voxels[i].x), world_x0 + VOXEL_SIZE * DIM_X));
            float pt_y = std::fmax(world_y0 - VOXEL_SIZE * DIM_Y, std::fmin((float)(voxels[i].y), world_y0 + VOXEL_SIZE * DIM_Y));
            float pt_z = std::fmax(world_z0 - VOXEL_SIZE * DIM_Z, std::fmin((float)(voxels[i].z), world_z0 + VOXEL_SIZE * DIM_Z));

            uint8_t r = voxels[i].rgb[0];
            uint8_t g = voxels[i].rgb[1];
            uint8_t b = voxels[i].rgb[2];

            fprintf(fp, "%f ", pt_x);
            fprintf(fp, "%f ", pt_y);
            fprintf(fp, "%f ", pt_z);
            fprintf(fp, "%d ", r);
            fprintf(fp, "%d ", g);
            fprintf(fp, "%d\n", b);
        }
    }
    fclose(fp);
}


void save_pcd_as_ply(char* filename, float* pcd)
{
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "ply\n");
    fprintf(fp, "format binary_little_endian 1.0\n");
    fprintf(fp, "element vertex %d\n", WIDTH * HEIGHT);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "end_header\n");

    float min_x = 1000, min_y = 1000, min_z = 1000;
    float max_x = -1000, max_y = -1000, max_z = -1000;

    for(int i = 0; i < WIDTH * HEIGHT; i++) {
        float x = pcd[3 * i + 0];
        float y = pcd[3 * i + 1];
        float z = pcd[3 * i + 2];

        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        min_z = std::min(min_z, z);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
        max_z = std::max(max_z, z);

        fwrite(&x, sizeof(float), 1, fp);
        fwrite(&y, sizeof(float), 1, fp);
        fwrite(&z, sizeof(float), 1, fp);
    }

    printf("bbox: (%f %f %f) -- (%f %f %f)\n", min_x, min_y, min_z, max_x, max_y, max_z);
    fclose(fp);
}