/*
 * The MIT License
 *
 * Copyright 2018 Ahmed Tarek.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "util.h"

extern "C"
__global__ void accumulate_vector(float* v, long size, float* r) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int width = gridDim.x * blockDim.x * 2;
    cache[threadIdx.x] = 0;

    for(int i=0; i < size; i+=width) {
        int i0 = i + threadIdx.x + blockIdx.x * blockDim.x * 2;;
        int i1 = i0 + blockDim.x;

        if(i0 < size) {
            float a = v[i0];
            float b = 0;

            if(i1 < size)
                b = v[i1];

            cache[threadIdx.x] += a + b;
        }
    }
    
    __syncthreads();

    sumBlock(cache, min((long)blockDim.x, size));
    
    if(threadIdx.x == 0)
        r[blockIdx.x] = cache[0];
}

extern "C"
__global__ void accumulate_vector_int(int* v, long size, int* r) {
    __shared__ int cache[THREADS_PER_BLOCK];
    
    int width = gridDim.x * blockDim.x * 2;
    cache[threadIdx.x] = 0;
    
    for(int i=0; i < size; i+=width) {
        int i0 = i + threadIdx.x + blockIdx.x * blockDim.x * 2;
        int i1 = i0 + blockDim.x;

        if(i0 < size) {
            int a = v[i0];
            int b = 0;

            if(i1 < size)
                b = v[i1];

            cache[threadIdx.x] += a + b;
        }
    }
    
    __syncthreads();

    sumBlock(cache, min((long)blockDim.x, size));
    
    if(threadIdx.x == 0)
        r[blockIdx.x] = cache[0];
}


extern "C"
__global__ void sum_abs_difference_int(int* a, int* b, long size, int* r) {
    __shared__ int cache[THREADS_PER_BLOCK];
    
    int width = gridDim.x * blockDim.x;
    cache[threadIdx.x] = 0;

    for(int i=0; i < size; i+=width) {
        int i0 = i + threadIdx.x + blockIdx.x * blockDim.x;

        if(i0 < size) {
            cache[threadIdx.x] += abs(a[i0] - b[i0]);
        }
    }
    
    __syncthreads();

    sumBlock(cache, min((long)blockDim.x, size));
    
    if(threadIdx.x == 0)
        r[blockIdx.x] = cache[0];
}

extern "C"
__global__ void sum_abs_difference(float* a, float* b, long size, float* r) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int width = gridDim.x * blockDim.x;
    cache[threadIdx.x] = 0;

    for(int i=0; i < size; i+=width) {
        int i0 = i + threadIdx.x + blockIdx.x * blockDim.x;

        if(i0 < size) {
            cache[threadIdx.x] += fabsf(a[i0] - b[i0]);
        }
    }
    
    __syncthreads();

    sumBlock(cache, min((long)blockDim.x, size));
    
    if(threadIdx.x == 0)
        r[blockIdx.x] = cache[0];
}
