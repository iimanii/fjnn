#include <cuda.h>

#include "util.h"

extern "C"
__global__ void crossOverMutate(float* a0, float* a1, float* r, long size, double mutation, 
                                float* crossover, float* mutate, float* gaussian) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(i >= size)
        return;
    
    float w0 = a0[i];
    float w1 = a1[i];
    
    float w = crossover[i] < 0.5 ? w0 : w1;

    if(mutate[i] < mutation)
        w = w + gaussian[i];
    
    r[i] = w;
}