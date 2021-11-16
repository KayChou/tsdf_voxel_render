#pragma once
#include "utils.h"

extern "C" context* init_context();

extern "C" void release_context(context* ctx);

extern "C" void Integrate(context* ctx, uint8_t *in_buf_depth, uint8_t* in_buf_color);

extern "C" void memcpy_volume_to_cpu(context* ctx, float* tsdf_out, float* weight_out, uint8_t* rgb_out);

extern "C" void get_pcd_in_world(context* ctx, uint8_t *in_buf_depth, float *pcd, int cam_idx);