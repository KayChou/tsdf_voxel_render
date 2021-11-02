#pragma once
#include "utils.h"

extern "C" context* init_context();

extern "C" void release_context(context* ctx);

extern "C" void Integrate(context* ctx, int cam_idx, uint8_t *in_buf_depth);

extern "C" void memcpy_volume_to_cpu(context* ctx, float* tsdf_out, float* weight_out);

extern "C" void get_pcd_in_world(context* ctx, uint8_t *in_buf_depth, float *pcd, int cam_idx);