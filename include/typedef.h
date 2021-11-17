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
    uint8_t rgb[3];
} baseVoxel;


// one baseVoxel contains 3x3x3 L1Voxels
typedef struct L1Voxel
{
    int baseIdx; // indicate which baseVoxel belongs to
    float tsdf;
    uint8_t rgb[3];
} L1Voxel;


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

    float* pcd;

    float voxel_size;
    float trunc_margin;

    float weight_threshhold = 0.0f;

    uint8_t* in_buf_depth; // input depth image
    uint8_t* in_buf_color;
    float* depth; // depth after dequantization
} context;