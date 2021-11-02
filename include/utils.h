#pragma once
#include "config.h"
#include "assert.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>

typedef struct KRT
{
    float fx, fy, cx, cy;
    float R[3 * 3];
    float T[3];
} KRT;


typedef struct context
{
    int width;
    int height;
    KRT krt[camera_num];
    float min_depth, max_depth;

    int resolution[3];

    float* tsdf_voxel;
    float* weight_voxel;

    float* pcd;

    float voxel_size;
    float trunc_margin;

    float tsdf_threshold = 0.2f;
    float weight_threshhold = 0.0f;

    uint8_t* in_buf_depth; // input depth image
    float* depth; // depth after dequantization

} Krt;


void load_camera_params(context *ctx);

void save_volume_to_ply(context *ctx, char *filename, float* tsdf, float* weight);

void save_pcd_as_ply(char* filename, float* pcd);