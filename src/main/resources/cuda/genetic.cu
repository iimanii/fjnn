#include <cuda.h>

#include "util.h"

extern "C"
__global__ void cross_over_mutate(float* a0, float* a1, float* r, long size, 
                                  float min, float max, double mutation,
                                  float* rng_crossover, float* rng_mutate, float* rng_pool) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i >= size)
        return;
    
    float w0 = a0[i];
    float w1 = a1[i];
    
    float w = rng_crossover[i] < 0.5f ? w0 : w1;

    if(rng_mutate[i] < mutation)
        w += rng_pool[i] * (max - min) + min;
    
    r[i] = w;
}

extern "C"
__global__ void clip_weights(float* weights, long size, float min, float max) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i >= size)
        return;
    
    float w = weights[i];
    
    if(w > max)
        weights[i] = max;
    else if(w < min)
        weights[i] = min;
}