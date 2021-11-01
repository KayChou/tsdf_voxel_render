#pragma once
#include "utils.h"

extern "C" context* init_context();
extern "C" void release_context(context* ctx);

extern "C" void Integrate(context* ctx, int cam_idx, uint8_t *in_buf_depth, float* out_Volume);