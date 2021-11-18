#pragma once
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <thrust/extrema.h>
#include "cuda_kernel.h"
#include "device_launch_parameters.h"
#include "utils.h"

typedef struct Lock {
    int mutex;
    Lock(void) {
        mutex = 0;
    }
    __device__ void lock(void) {
        while (atomicCAS(&mutex, 0, 1) != 0) {
            __nanosleep(100000);
        }
    }
    __device__ void unlock(void) {
        atomicExch(&mutex, 0);
    }
} Lock;
