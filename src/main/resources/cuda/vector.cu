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
__global__ void scale(float* data, float factor, int size) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < size) {
        data[tid] *= factor;
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
__global__ void reduce_stride_in_place(float* a, float* b, int sizeB) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < sizeB) {
        // Add elements from b to a if idx < sizeB
        a[idx] = a[idx] + b[idx];
    }
}