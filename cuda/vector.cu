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
 * c[x] = a[x] + b[x]
 */
extern "C"
__global__ void add(float* a, float* b, float* c, long size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size)
        c[idx] = a[idx] + b[idx];
}

/**
 * c[x] = a[x] + alpha * b[x]
 */
extern "C"
__global__ void add_multiply(float* a, float* b, float* c, float alpha, long size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size)
        c[idx] = a[idx] + alpha * b[idx];
}

/**
 * c[x] = a[x] * b[x]
 */
extern "C"
__global__ void multiply(float* a, float* b, float* c, long size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(i < size)
        c[i] = a[i] * b[i];
}

/**
 * c[x] *= factor
 */
extern "C"
__global__ void scale(float* data, float factor, long size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < size) {
        data[tid] *= factor;
    }
}

/**
 * array[x] = array[x] + y;
 */
extern "C"
__global__ void add_scalar(float* array, float scalar, long size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < size) {
        array[idx] += scalar;
    }
}

/**
 * c[x + stride * stride_len] = a[x + stride * stride_len] + b[x]
 * 
 * each block calculates 1 or more strides
 */
extern "C"
__global__ void add_stride(float* a, float* b, float* c, long stride_size, long total_size) {
    int b_index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(b_index < stride_size) {
        int stride_id = threadIdx.y + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y;
        int a_index = b_index + stride_id * stride_size;
        
        if(a_index < total_size)
            c[a_index] = a[a_index] + b[b_index];
    }
}

/**
 * c[x + stride * stride_len] = alpha * a[x + stride * stride_len] * b[x]
 * 
 * each block calculates 1 or more strides
 */
extern "C"
__global__ void multiply_stride(float* a, float* b, float* c, float alpha, long stride_size, long total_size) {
   int b_index = threadIdx.x + blockIdx.x * blockDim.x;
   
   if(b_index < stride_size) {
       int stride_id = threadIdx.y + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y;
       int a_index = b_index + stride_id * stride_size;
       
       if(a_index < total_size)
           c[a_index] = alpha * a[a_index] * b[b_index];
   }
}

extern "C"
__global__ void reduce_stride(float* a, float* b, float* c, long sizeA, long sizeB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sizeB) {
        // Add elements from a and b if idx < sizeB
        c[idx] = a[idx] + b[idx];
    } else if (idx < sizeA) {
        // Copy elements from a if idx >= sizeB
        c[idx] = a[idx];
    }
}

extern "C"
__global__ void reduce_stride_in_place(float* a, float* b, long sizeB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sizeB) {
        // Add elements from b to a if idx < sizeB
        a[idx] = a[idx] + b[idx];
    }
}

extern "C"
__global__ void optimized_reduce_stride(float* a, float* b, float* c, long sizeA, long sizeB, long iterations) {    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = 0; i < iterations; i++) {
        int index = i * stride + idx;
        
        if(index < sizeB)
            c[index] = a[index] + b[index];
        else if(index < sizeA)
            c[index] = a[index];
    }
}

extern "C" 
__global__ void threshold(float* mask, float rate, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        mask[idx] = mask[idx] > rate ? 1.0f : 0.0f;
    }
}
