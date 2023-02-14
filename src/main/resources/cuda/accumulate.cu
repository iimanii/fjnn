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
 * a[x + stride * stride_len] += b[x]
 * 
 * each block calculates 1 or more strides
 */
extern "C"
__global__ void add_stride(float* a, float* b, int stride_size, long total_size) {
    int b_index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(b_index < stride_size) {
        int stride_id = threadIdx.y + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y;
        int a_index = b_index + stride_id * stride_size;
        
        if(a_index < total_size)
            a[a_index] += b[b_index];
    }
}

extern "C"
__global__ void add_stride_vectorized(float* a, float* b, int stride_size, long strides_count) {
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int b_index = threadIndex * 4;
    
    int rem = stride_size % 4;
    
    if(threadIndex < stride_size / 4) {
        int stride_id = threadIdx.y + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y;

        if(stride_id < strides_count) {
            float4 load_b = reinterpret_cast<float4*>(&b[b_index])[0];
        
            int a_index = b_index + stride_id * stride_size;
            /* aligned access */
            if(rem == 0) {
                float4 load_a = reinterpret_cast<float4*>(&a[a_index])[0];
                load_a.x += load_b.x;
                load_a.y += load_b.y;
                load_a.z += load_b.z;
                load_a.w += load_b.w;                
                reinterpret_cast<float4*>(&a[a_index])[0] = load_a;
            } else {
                /* non-aligned access */
                a[a_index] += load_b.x;
                a[a_index+1] += load_b.y;
                a[a_index+2] += load_b.z;
                a[a_index+3] += load_b.w;  
            }
        }
    }
    
    /* calculate remaining stride_size % 4 */
    if(threadIdx.x >= 4 || blockIdx.x > 0)
        return;
    
    if(threadIdx.x >= rem)
        return;
    
    b_index = stride_size - rem + threadIdx.x;
    float load_b = b[b_index];

    int stride_id = threadIdx.y + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y;

    if(stride_id < strides_count) {
        int a_index = b_index + stride_id * stride_size;
        a[a_index] += load_b;
//        printf("3- %d %d %d %d %d %d\n", a_index, stride_id, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
    }
}

/**
 * a[x + stride * stride_len] += b[x]
 * 
 * each block loads 1 or more strides
 * each block loops over multiple strides of a
 * 
 */
extern "C"
__global__ void add_stride_vectorized_loop(float* a, float* b, int stride_size, long strides_count, int iterations_per_load) {
    int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int b_index = threadIndex * 4;
    
    int stride_group = blockDim.y * gridDim.y * gridDim.z;
    
    int rem = stride_size % 4;
    
    if(threadIndex < stride_size / 4) {
        float4 load_b = reinterpret_cast<float4*>(&b[b_index])[0];

//        printf("1- %d %d %d %d %d %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, threadIndex, stride_size);
        
        for(int i=0; i < iterations_per_load; i++) {
            int stride_id = stride_group * i;
            stride_id += threadIdx.y + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y;
            
            if(stride_id < strides_count) {                
                int a_index = b_index + stride_id * stride_size;
                
                /* aligned access */
                if(rem == 0) {
                    float4 load_a = reinterpret_cast<float4*>(&a[a_index])[0];
                    load_a.x += load_b.x;
                    load_a.y += load_b.y;
                    load_a.z += load_b.z;
                    load_a.w += load_b.w;                
                    reinterpret_cast<float4*>(&a[a_index])[0] = load_a;
                } else {
                    /* non-aligned access */
                    a[a_index] += load_b.x;
                    a[a_index+1] += load_b.y;
                    a[a_index+2] += load_b.z;
                    a[a_index+3] += load_b.w;  
                }
            }
        }
    } 
//    else
//        printf("%d %d %d %d %d %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, threadIndex, stride_size);
    
    /* calculate remaining stride_size % 4 */
    if(threadIdx.x >= 4 || blockIdx.x > 0)
        return;
    
    if(threadIdx.x >= rem)
        return;
    
    b_index = stride_size - rem + threadIdx.x;
    float load_b = b[b_index];

    for(int i=0; i < iterations_per_load; i++) {
        int stride_id = stride_group * i;
        stride_id += threadIdx.y + (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y;
        
        if(stride_id < strides_count) {
            int a_index = b_index + stride_id * stride_size;
            a[a_index] += load_b;
//            printf("3- %d %d %d %d %d %d\n", a_index, stride_id, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
        }        
    }
}

/**
 * a[x + stride * stride_len] += b[x]
 * 
 * each block calculates 1 or more strides
 */
extern "C"
__global__ void add_stride_loop(float* a, float* b, int stride_size, int strides_per_block, int iterations, long total_size) {
    for(int i=0; i < iterations; i++) {
        int b_index = threadIdx.x + i * blockDim.x;

        if(b_index < stride_size) {
            int stride_id = threadIdx.y + (blockIdx.y * gridDim.x + blockIdx.x) * strides_per_block;
            int a_index = b_index + stride_id * stride_size;

            if(a_index < total_size)
                a[a_index] += b[b_index];
        }
    }
}

/**
 * a[x + stride * stride_len] += b[x]
 * 
 * each block calculates a full stride
 */
extern "C"
__global__ void add_stride_old(float* a, float* b, long stride_size, long total_size) {
    int b_index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(b_index < stride_size) {
        int a_index = b_index + (blockIdx.y + blockIdx.z * gridDim.y) * stride_size;
        
        if(a_index < total_size)
            a[a_index] += b[b_index];
    }
}

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
