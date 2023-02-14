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

/**
 * Rectifier Linear Unit
 */
extern "C"
__global__ void ReLU(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size && v[row] < 0.0f)
        v[row] = 0.0f;
}

/**
 * Leaky Rectifier Linear Unit
 */
extern "C"
__global__ void LeakyReLU(float* v, long size, float alpha) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size && v[row] < 0.0f)
        v[row] = v[row] * alpha;
}

/**
 * Conditional Rectifier Linear Unit
 */
extern "C"
__global__ void ReLU_Conditional(float* v, unsigned char* c, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int index = row + blockIdx.y * size;
    
    if(row < size && c[row])
        v[index] = max(0.0f, v[row]);
}
/**
 * Rectifier Linear Unit
 */
extern "C"
__global__ void multi_ReLU(float* v, long size, long pitch) {
    int row  = blockDim.x * blockIdx.x + threadIdx.x;
    int index = blockIdx.y * pitch + row;
        
    if(row < size)
        v[index] = max(0.0f, v[index]);
}


/**
 * Sigmoid
 */
extern "C"
__global__ void Sigmoid(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        v[row] = 1.0 / (1 + safe_exp(-v[row]));
}

/**
 * Conditional Sigmoid
 */
extern "C"
__global__ void Sigmoid_Conditional(float* v, unsigned char* c, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int index = row + blockIdx.y * size;
    
    if(row < size && c[row])
        v[index] = 1.0 / (1 + safe_exp(-v[index]));
}

/**
 * Sigmoid
 */
extern "C"
__global__ void multi_Sigmoid(float* v, long size, long pitch) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int index = blockIdx.y * pitch + row;
        
    if(row < size)
        v[index] = 1.0 / (1 + safe_exp(-v[index]));
}

/**
 * Sigmoid
 */
extern "C"
__global__ void multi_Sigmoid_Conditional(float* v, unsigned char* c, long size, long pitch) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int index = blockIdx.y * pitch + row;
        
    if(row < size && c[row])
        v[index] = 1.0 / (1 + safe_exp(-v[index]));
}


/**
 * Sin
 */
extern "C"
__global__ void Sin(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        v[row] = sin(v[row]);
}
/**
 * Conditional Sin
 */
extern "C"
__global__ void Sin_Conditional(float* v, unsigned char* c, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int index = row + blockIdx.y * size;
    
    if(row < size && c[row])
        v[index] = sin(v[index]);
}

/**
 * Sin
 */
extern "C"
__global__ void multi_Sin(float* v, long size, long pitch) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int index = blockIdx.y * pitch + row;    
    
    if(row < size)
        v[index] = sin(v[index]);
}


/**
 * Tanh
 */
extern "C"
__global__ void Tanh(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        v[row] = tanh(v[row]);
}

/**
 * Step
 */
extern "C"
__global__ void Step(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        v[row] = v[row] >= 0 ? 1 : 0;
}

/**
 * SoftMax
 */
extern "C"
__global__ void SoftMax_1(float* v, long size, float* sums) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row >= size) 
        return;
    
    cache[threadIdx.x] = safe_exp(v[row]);
    
    v[row] = cache[threadIdx.x];
    
    __syncthreads();
    
    sumBlock(cache, min((long)blockDim.x, size - blockDim.x * blockIdx.x));

    /* each block creates a sum */
    if(threadIdx.x == 0)
       sums[blockIdx.x] = cache[0];
}

extern "C"
__global__ void SoftMax_2(float* v, long size, float* sums, long sums_size) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int loops = calcIterations(blockDim.x, sums_size);
    
    cache[threadIdx.x] = 0;
    
    /* 
     * Calculate total sum from "sums" array
     * should loop only once unless size > THREADS_PER_BLOCK^2
     */
    for(int i=0; i < loops; i++) {
        int index = i * blockDim.x + threadIdx.x;
        
        if(index < sums_size)
            cache[threadIdx.x] += sums[index];
    }
    
    __syncthreads();
        
    sumBlock(cache, min((long)blockDim.x, sums_size));

    __syncthreads();
        
    double sum = cache[0];
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        if(sum == 0) {
            v[row] = 1.0 / size;
        } else {
            v[row] /= sum;
        }
}


/**
 * Tanh
 */
extern "C"
__global__ void multi_Tanh(float* v, long size, long pitch) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int index = blockIdx.y * pitch + row;
    
    if(row < size)
        v[index] = tanh(v[index]);
}

/**
 * Step
 */
extern "C"
__global__ void multi_Step(float* v, long size, long pitch) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int index = blockIdx.y * pitch + row;
    
    if(row < size)
        v[index] = v[index] >= 0 ? 1 : 0;
}

/**
 * SoftMax
 */
extern "C"
__global__ void multi_SoftMax_1(float* v, long size, long pitch_v, float* sums, long pitch_s) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int index = blockIdx.y * pitch_v + row;
    
    if(row >= size) 
        return;
    
    cache[threadIdx.x] = safe_exp(v[index]);
    
    v[index] = cache[threadIdx.x];
    
    __syncthreads();
    
    sumBlock(cache, min((long)blockDim.x, size - blockDim.x * blockIdx.x));

    /* each block creates a sum */
    if(threadIdx.x == 0)
       sums[blockIdx.y * pitch_s + blockIdx.x] = cache[0];
}

extern "C"
__global__ void multi_SoftMax_2(float* v, long size, long pitch_v, float* sums, long sums_size, long pitch_s) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int loops = calcIterations(blockDim.x, sums_size);
    
    cache[threadIdx.x] = 0;
    
    /* 
     * Calculate total sum from "sums" array
     * should loop only once unless size > THREADS_PER_BLOCK^2
     */
    int base = blockIdx.y * pitch_s;
    
    for(int i=0; i < loops; i++) {
        int index = i * blockDim.x + threadIdx.x;
        
        if(index < sums_size)
            cache[threadIdx.x] += sums[base + index];
    }
    
    __syncthreads();
        
    sumBlock(cache, min((long)blockDim.x, sums_size));

    __syncthreads();
        
    double sum = cache[0];
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int index = blockIdx.y * pitch_v + row;
    
    if(row < size)
        if(sum == 0) {
            v[index] = 1.0 / size;
        } else {
            v[index] /= sum;
        }
}