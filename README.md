# Volumetric TSDF Fusion

## 项目说明

基于 CPU + GPU 实现的体素渲染的一个初级版本。输入数据包括：
- 多个相机的深度图
- 对应的相机参数（外参 + 内参），形式： Xc = R(Xw  - T)。其中 Xc 是相机坐标系下的三维点坐标，Xw 是世界坐标系下的点坐标。
  
输出三维体素模型。

## 算法步骤简述

1. 建立体素
   - 建立长方体包围盒（volume），能够包围待重建的物体
   - 对包围盒进行 n 等分，划分为网格像素（体素， voxel）

2. 计算 TSDF 与权重
   - 对于立方体中的每个体素g，将g转化为世界坐标系下的三维坐标点p
   - 对于体素中的 p(x, y, z)，根据相机内外参数计算其在深度图中的对应像素点 x, 对应的深度为 depth(x), 点 v 到相机坐标原点的距离为 distance(v)
   - 符号距离函数 sdf(p) = depth(x) - distance(v)
   - 截断距离：当 sdf 在截断距离 u 内，tsdf(p) = sdf(p) / u，否则如果sdf > 0， tsdf = 1, sdf < 0, tsdf = -1
   - 计算权重


## 编译

### Step 1: 编译cuda函数

```bash
cd src/cuda
mkdir build
cd build
cmake ..
make
```

此处编译要求安装显卡驱动和cuda，在已经安装的情况下，需要根据显卡型号修改 src/cuda/CMakeLists.txt 第 15 行的计算功能集版本，例如 3090 显卡为compute_86， 2080Ti 为compute_75。其他显卡需自行查看 NVIDIA 官网。


### Step 2: 编译Cpp

在项目根目录执行：
```bash
mkdir build
cd build
cmake ..
make
```

编译成功后运行 ./main 即可运行。运行后会在项目根目录下生成 fusion.ply 文件，使用 meshlab 即可查看体素渲染后的点云模型。

## TODO
1. 给体素添加颜色信息
2. 表面重建（通俗的讲就是用三角形面元连接体素表面，得到 mesh 模型）
3. 内存优化（存储所有体素是非常耗费显存的，这个可以不做）

看到比较好的效果来自文章： [FusionMLS](https://link.springer.com/article/10.1007/s41095-018-0121-0)