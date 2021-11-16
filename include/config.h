#pragma once

// #define data_path (char *)"../data/frame_0"
// #define param_path (char *)"../data/para_cameras.txt"

#define data_path (char *)"/SSD1/zk/freeview/medialab_20210924/basketball/frames/frame_0"
#define param_path (char *)"/SSD1/zk/freeview/medialab_20210924/created/para_cameras.txt"

#define CAM_NUM 4
#define WIDTH 1920
#define HEIGHT 1080

#define world_x0 -20
#define world_y0 -20
#define world_z0 40

#define DIM_X 800
#define DIM_Y 800
#define DIM_Z 800
#define VOXEL_SIZE 0.05

#define TSDF_THRESHOLD 0.5
#define WEIGHT_THRESHOLD 0.2 * CAM_NUM

#define GEN_PCD_OF_EACH_CAM 0
#define fB 32504.0