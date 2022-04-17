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
 * Matrix x Vector
 * 
 * Each block calculates a single row in the matrix
 */

extern "C"
__global__ void matrix_mul_vector(float* m, float* v, float* r, long columns) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int row = blockIdx.x;
    
    /* starting index of the row to calculate */
    long i_row = row * columns;
    
    int loop = calcIterations(blockDim.x, columns);
    
    cache[threadIdx.x] = 0;
    
    /* make sure we cover the whole vector */
    for(int i=0; i < loop; i++) {
        int j = i * blockDim.x + threadIdx.x;
        
        if(j < columns)
            cache[threadIdx.x] += m[i_row + j] * v[j];
    }

    __syncthreads();

    sumBlock(cache, min((long)blockDim.x, columns));
    
    if(threadIdx.x == 0) {
        r[row] = cache[0];
//        if(blockIdx.x == 0)
//            printf("%d %d\n", blockDim.x, blockDim.y);            
//            printf("%f %d\n", v[v_size-1], blockIdx.x);
    }
}

extern "C"
__global__ void vector_mul_matrix(float* v, float* m, float* r, long rows, long cols) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int column = blockIdx.x;
        
    int loop = calcIterations(blockDim.x, rows);
    
    cache[threadIdx.x] = 0;
    
    /* make sure we cover the whole vector */
    for(int i=0; i < loop; i++) {
        int j = i * blockDim.x + threadIdx.x;
        
        if(j < rows)
            cache[threadIdx.x] += m[column + j * cols] * v[j];
    }

    __syncthreads();

    sumBlock(cache, min((long)blockDim.x, rows));
    
    if(threadIdx.x == 0) {
        r[column] = cache[0];
//        if(blockIdx.x == 0)
//            printf("%d %d\n", blockDim.x, blockDim.y);            
//            printf("%f %d\n", v[v_size-1], blockIdx.x);
    }
}

extern "C"
__global__ void memset_single_float(float* a, long i, float v) {
    a[i] = v;
}

extern "C"
__global__ void accumulate_vector(float* a, float* b, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int index = row + blockIdx.y * size;
    
    if(row < size)
        a[index] += b[row];
}

extern "C"
__global__ void matrix_mul_vector_slow(float* m, float* v, float* r, long m_row, long v_size) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    for(int row=0; row < m_row; row++) {
        /* starting index of the row to calculate */
        long i_row = row * v_size;

        int loop = calcIterations(blockDim.x, v_size);

        cache[threadIdx.x] = 0;

        /* make sure we cover the whole vector */
        for(int i=0; i < loop; i++) {
            int j = i * blockDim.x + threadIdx.x;

            if(j < v_size)
                cache[threadIdx.x] += m[i_row + j] * v[j];
        }

        __syncthreads();

        sumBlock(cache, min((long)blockDim.x, v_size));

        if(threadIdx.x == 0) {
            r[row] = cache[0];
        }

        __syncthreads();
    }
}

/**
 * Matrix x Vector
 * 
 * Each block calculates a single row in the matrix
 */
extern "C"
__global__ void multi_matrix_mul_vector(float* m, long pitch_m, float* v, long pitch_v, float* r, long pitch_r, long columns) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int row = blockIdx.x;
    
    /* starting index of the row to calculate */
    long i_row = row * columns;
    
    long m_base = blockIdx.y * pitch_m;
    long v_base = blockIdx.y * pitch_v;
    
    int loop = calcIterations(blockDim.x, columns);
    
    cache[threadIdx.x] = 0;
    
    /* make sure we cover the whole vector */
    for(int i=0; i < loop; i++) {
        int j = i * blockDim.x + threadIdx.x;
        
        if(j < columns)
            cache[threadIdx.x] += m[m_base + i_row + j] * v[v_base + j];
    }

    __syncthreads();

    sumBlock(cache, min((long)blockDim.x, columns));
        
    long r_base = blockIdx.y * pitch_r;
    
    if (threadIdx.x == 0)
        r[r_base + row] = cache[0];   
}

extern "C"
__global__ void multi_accumulate_vector(float* a, float* b, long size, size_t pitch) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int index = blockIdx.y * pitch + row;
    
    if(row < size)
        a[index] += b[index];
}


/* 
 * each block calculates one element 
 * NOTE: matrix is represented by 1D memory for performance reasons
 * 
 * [rows_m0 x cols_m0] x [rows_m1 x cols_m1] = [rows_m0 x cols_m1] 
 */
extern "C"
__global__ void matrix_mul_matrix(float* m0, float* m1, float* r, int cols_m0, int cols_m1) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int loop = calcIterations(blockDim.x, cols_m0);
    
    cache[threadIdx.x] = 0;

    int row = blockIdx.x;
    int col = blockIdx.y;
    
    /* make sure we cover the whole vector */
    for(int i=0; i < loop; i++) {
        int j = i * blockDim.x + threadIdx.x;
        if(j < cols_m0) {
            int r0 = row;
            int c0 = j;
            
            int r1 = j;
            int c1 = col;

            int i_m0 = index_2d(r0, c0, cols_m0);
            int i_m1 = index_2d(r1, c1, cols_m1);

            cache[threadIdx.x] += m0[i_m0] * m1[i_m1];
        }
    }

    __syncthreads();

    sumBlock(cache, min(blockDim.x, cols_m0));
    
    if(threadIdx.x == 0) {
        int i_r = index_2d(row, col, cols_m1);
        r[i_r] = cache[0];
    }
}


extern "C"
__global__ void matrix_mul_matrix_transpose(float* m0, float* m1, float* r, int cols_m0, int cols_m1) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int loop = calcIterations(blockDim.x, cols_m0);
    
//    printf("%d", loop);
    cache[threadIdx.x] = 0;

    /* make sure we cover the whole vector */
    for(int i=0; i < loop; i++) {
        int j = i * blockDim.x + threadIdx.x;
        
        if(j < cols_m0) {
            int i_m0 = blockIdx.x * cols_m0 + j;
            int i_m1 = blockIdx.y * cols_m1 + j;
        
            cache[threadIdx.x] += m0[i_m0] * m1[i_m1];
        }
    }

    __syncthreads();

    sumBlock(cache, min(blockDim.x, cols_m0));
    
    if(threadIdx.x == 0) {
        int i_r = blockIdx.x * cols_m1 + blockIdx.y;
        r[i_r] = cache[0];
    }
}


extern "C"
__global__ void matrix_mul_matrix_2(float* m0, float* m1, float* r, int cols_m0, int cols_m1) {
    __shared__ float m0_cache[32][32];
    __shared__ float m1_cache[32][32];
  
    int loops = cols_m0 / 32;
    float sum = 0;
    
    int row_r = (blockIdx.x * blockDim.y + threadIdx.x);
    int col_r = (blockIdx.y * blockDim.x + threadIdx.y);
    int i = 0;
    // main loop
    for(int i=0; i < loops; i++) {        
        int row_m0 = row_r;
        int col_m0 = i * blockDim.x + threadIdx.y;
        
        int row_m1 = i * blockDim.y + threadIdx.x;
        int col_m1 = col_r;

//        m0_cache[threadIdx.x][threadIdx.y] = m0[index_2d(row_m0, col_m0, cols_m0)];
//        m1_cache[threadIdx.x][threadIdx.y] = m1[index_2d(row_m1, col_m1, cols_m1)];

        __syncthreads();

#pragma unroll
        for(int k=0; k < 32; k++) {
            sum += m0_cache[threadIdx.x][(threadIdx.y+k) % 32] * m1_cache[threadIdx.x][(threadIdx.y+k) % 32];
        }

        __syncthreads();
    }
    
//    if(false) {        
//        int row_m0 = row_r;
//        int col_m0 = col_r;
//        int row_m1 = col_r;
//        int col_m1 = row_r;
//
//        m0_cache[threadIdx.x][threadIdx.y] = m0[index_2d(row_m0, col_m0, cols_m0)];
//        m1_cache[threadIdx.y][threadIdx.x] = m1[index_2d(row_m1, col_m1, cols_m1)];
//
//        __syncthreads();
//
//        for(int k=0; k < blockDim.x; k++)
//            sum += m0_cache[threadIdx.x][threadIdx.y] * m1_cache[threadIdx.y][threadIdx.x];
//
//        __syncthreads();
//    }

    r[index_2d(row_r, col_r, cols_m1)] = sum;
}
