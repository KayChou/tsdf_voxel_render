#pragma once
#include "config.h"
#include "assert.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <math.h>

typedef struct baseVoxel
{
    float tsdf;
    float x;
    float y;
    float z;
    uint8_t rgb[3];
} baseVoxel;


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
    KRT krt[CAM_NUM];
    float min_depth, max_depth;

    int resolution[3];

    float* tsdf_voxel;
    uint8_t* color_voxel;

    baseVoxel* valid_voxel;

    float* pcd;

    float voxel_size;
    float trunc_margin;
    float weight_threshhold = 0.0f;

    uint8_t* in_buf_depth; // input depth image
    uint8_t* in_buf_color;
    float* depth; // depth after dequantization

    int L1_voxel_num;
    int *L1_voxel_idx;

    int L2_voxel_num;
} context;