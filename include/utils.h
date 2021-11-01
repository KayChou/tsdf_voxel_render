#pragma once
#include "config.h"
#include "assert.h"
#include <stdint.h>
#include <stdio.h>

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
    float volume_origin[3];

    float* tsdf_voxel;
    float* weight_voxel;

    float voxel_size;
    float trunc_margin;

    uint8_t* in_buf_depth; // input depth image
    float* depth; // depth after dequantization

} Krt;


void load_camera_params(context *ctx);