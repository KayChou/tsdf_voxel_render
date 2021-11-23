#pragma once
#include "typedef.h"

void load_camera_params(context *ctx);

void save_volume_to_ply(context *ctx, char *filename, baseVoxel* voxels, int num_pts);

void save_pcd_as_ply(char* filename, float* pcd);