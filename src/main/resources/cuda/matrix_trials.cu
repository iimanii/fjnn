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

            int i_m0 = index2d(r0, c0, cols_m0);
            int i_m1 = index2d(r1, c1, cols_m1);

            cache[threadIdx.x] += m0[i_m0] * m1[i_m1];
        }
    }

    __syncthreads();

    sumBlock(cache, min(blockDim.x, cols_m0));
    
    if(threadIdx.x == 0) {
        int i_r = index2d(row, col, cols_m1);
        r[i_r] = cache[0];
    }
}

/**
 * each thread calculates 1 value in output
 * x -> column index
 * y -> row index
 **/
extern "C"
__global__ void matrix_mul_matrix_1(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x >= cols_b || y >= rows_a)
        return;
    
    float sum = 0;
    
    for(int i=0; i < cols_a; i++) {
        int ia = index2d(y, i, cols_a);
        int ib = index2d(i, x, cols_b);
//        sum = __fmaf_rn(a[y * cols_a + i], b[i * cols_b + x], sum);
//        sum += a[y * cols_a + i] * b[i * cols_b + x];
        sum = __fadd_rn(__fmul_rn(a[ia], b[ib]), sum);
    }
    
//    r[y * cols_b + x] = sum + alpha * r[y * cols_b + x];
//    r[y * cols_b + x] = __fmaf_rn(alpha, r[y * cols_b + x], sum);
    int ir = index2d(y, x, cols_b);
    r[ir] = __fadd_rn(__fmul_rn(alpha, r[ir]), sum);
}

/**
 * each thread calculates 1 value in output
 * x -> column index
 * y -> row index
 **/
extern "C"
__global__ void matrix_mul_matrix_2(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x >= cols_b || y >= rows_a)
        return;
    
    __shared__ float a_cache[32][32];
    __shared__ float b_cache[32][32];
    
    int loops = cols_a / 32;
    
    float sum = 0;
    
    for(int i=0; i < loops; i++) {
        a_cache[threadIdx.y][threadIdx.x] = a[index2d(y, i * 32 + threadIdx.x, cols_a)];
        b_cache[threadIdx.y][threadIdx.x] = b[index2d(i * 32 + threadIdx.y, x, cols_b)];
//        a_cache[threadIdx.y][threadIdx.x] = a[index2d(y, i * 32 + threadIdx.x, cols_a)];
//        b_cache[threadIdx.x][threadIdx.y] = b[index2d(i * 32 + threadIdx.x, threadIdx.y + blockIdx.x * blockDim.x, cols_b)];
        
        __syncthreads();
        
        for(int j=0; j < 32; j++)
            sum += a_cache[threadIdx.y][j] * b_cache[j][threadIdx.x];
//            sum = __fadd_rn(__fmul_rn(a_cache[threadIdx.y][j], b_cache[j][threadIdx.x]), sum);
        
        __syncthreads();
    }

    int ir = index2d(y, x, cols_b);
//    r[ir] = __fadd_rn(__fmul_rn(alpha, r[ir]), sum);
    r[ir] = alpha * r[ir] + sum;
}

__device__ void print2d(float* a, int r, int c) {
    for(int w=0; w < r; w++) {
        for(int z=0; z < c; z++) {
            printf("%.0f ", a[w * c + z]);
        }
        printf("\n");
    }    
    printf("\n");
}

/**
 * each thread calculates 1 value in output
 * x -> column index
 * y -> row index
 **/

#define BLOCK_SIZE 16

extern "C"
__global__ void matrix_mul_matrix_3(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 4;
    
    if(x >= cols_b || y >= rows_a)
        return;
    
    __shared__ float a_cache[64][64];
    __shared__ float b_cache[64][64];
    
    const int loops = cols_a / 64;
    
    float sum[4][4] = {0.0f};
    
    for(int i=0; i < loops; i++) {
        for(int j=0; j < 4; j++) {
            reinterpret_cast<float4*>(a_cache[threadIdx.y * 4 + j])[threadIdx.x] = 
                    reinterpret_cast<float4*>(&a[index2d(y + j, (i * blockDim.x + threadIdx.x) * 4, cols_a)])[0];
            
            reinterpret_cast<float4*>(b_cache[threadIdx.y * 4 + j])[threadIdx.x] = 
                    reinterpret_cast<float4*>(&b[index2d((i * blockDim.y + threadIdx.y) * 4 + j, x, cols_b)])[0];
        }
        
        __syncthreads();
        
        for(int y0=0; y0 < 4; y0++)
            for(int x0=0; x0 < 4; x0++)
                for(int j=0; j < 64; j++)
                    sum[y0][x0] += a_cache[threadIdx.y*4+y0][j] * b_cache[j][threadIdx.x*4+x0];
        
        __syncthreads();
    }
    
    for(int y0=0; y0 < 4; y0++) {
        int ir = index2d(y+y0, x, cols_b);
        float4 loaded = reinterpret_cast<float4*>(&r[ir])[0];
        loaded.x = loaded.x * alpha + sum[y0][0];
        loaded.y = loaded.y * alpha + sum[y0][1];
        loaded.z = loaded.z * alpha + sum[y0][2];
        loaded.w = loaded.w * alpha + sum[y0][3];
        reinterpret_cast<float4*>(&r[ir])[0] = loaded;
    }
}

/**
 * each thread calculates 1 value in output
 * x -> column index
 * y -> row index
 **/

#define BLOCK_SIZE_X_8x8 16
#define BLOCK_SIZE_Y_8x8 8
#define LOAD_SIZE_8x8 64
#define ROWS_PER_THREAD_8x8 (LOAD_SIZE_8x8 / BLOCK_SIZE_Y_8x8)
#define COLS_PER_THREAD_8x8 (LOAD_SIZE_8x8 / BLOCK_SIZE_X_8x8)
//
//__device__ void matrix_mul_matrix_16x8_edges_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha, 
//                                                       float a_cache[LOAD_SIZE][LOAD_SIZE], float b_cache[LOAD_SIZE][LOAD_SIZE],
//                                                       int remainX, int remainY) {
//    const int loops = cols_a / LOAD_SIZE;
//    
//    int loopsX = (remainX - 1) / blockDim.x + 1;
//    int loopsY = (remainY - 1) / blockDim.y + 1;
//        
//    int x = (threadIdx.x + blockIdx.x * blockDim.x) * COLS_PER_THREAD;
//    int y = (threadIdx.y + blockIdx.y * blockDim.y) * ROWS_PER_THREAD;
//    
//    float sum[ROWS_PER_THREAD][COLS_PER_THREAD] = {0.0f};
//    
//    for(int i=0; i < loops; i++) {
//        for(int j=0; j < ROWS_PER_THREAD; j++) {
//            for(int k=0; k < COLS_PER_THREAD; k++) {
//                int i_ax = (i * blockDim.x + threadIdx.x) * COLS_PER_THREAD + k;
//                int i_ay = y + j;
//                
//                if(i_ax < cols_a && i_ay < rows_a)
//                    a_cache[threadIdx.y * ROWS_PER_THREAD + j][threadIdx.x + k] = a[index2d(i_ay, i_ax, cols_a)];
//                
//                int i_bx = x + k;
//                int i_by = (i * blockDim.y + threadIdx.y) * ROWS_PER_THREAD + j;
//                
//                if(i_bx < cols_b && i_by < cols_a)
//                    b_cache[threadIdx.y * ROWS_PER_THREAD + j][threadIdx.x + k] = b[index2d(i_by, i_bx, cols_b)];
//            }
//        }
//        
//        __syncthreads();
//        
////        for(int y0=0; y0 < loopsY; y0++) {
////            for(int x0=0; x0 < loopsX; x0++) {
////                int ix = threadIdx.x + blockDim.x * x0;
////                int iy = threadIdx.y + blockDim.y * y0;
////
////                if(iy < remainY && ix < remainX) {
////                    for(int j=0; j < LOAD_SIZE; j++)
////                        sum[y0][x0] += a_cache[iy][j] * b_cache[j][ix];
////                }
////            }
////        }
//        
//        __syncthreads();
//    }
//    
//    const int rem = cols_a % LOAD_SIZE;
//    
//    if(rem > 0) {
////        for(int y0=0; y0 < ROWS_PER_THREAD; y0++)
////            for(int x0=0; x0 < COLS_PER_THREAD; x0++)
////                for(int j=0; j < rem; j++)
////                    sum[y0][x0] += a_cache[threadIdx.y * ROWS_PER_THREAD + y0][j] * b_cache[j][threadIdx.x * COLS_PER_THREAD + x0];
////        
////        __syncthreads();        
//    }    
//    
////    for(int y0=0; y0 < loopsY; y0++) {
////        for(int x0=0; x0 < loopsX; x0++) {
////            int ix = x + blockDim.x * x0;
////            int iy = y + blockDim.y * y0;
////
////            if(iy < rows_a && ix < cols_b) {
////                int ir = index2d(iy, ix, cols_b);
////                r[ir] = r[ir] * alpha + sum[y0][x0];
////            }
////        }
////    }
//}

extern "C"
__global__ void matrix_mul_matrix_8x8_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    __shared__ float a_cache[LOAD_SIZE_8x8][LOAD_SIZE_8x8];
    __shared__ float b_cache[LOAD_SIZE_8x8][LOAD_SIZE_8x8];

//    int rowsPerBlock = blockDim.y * ROWS_PER_THREAD_8x8;
//    int remainY = rows_a - blockIdx.y * rowsPerBlock;
//    
//    int colsPerBlock = blockDim.x * COLS_PER_THREAD_8x8;
//    int remainX = cols_b - blockIdx.x * rowsPerBlock;
    
//    if(remainY < rowsPerBlock || remainX < colsPerBlock) {
////        printf("unaligned edge");
////        matrix_mul_matrix_16x8_edges_unaligned(a, b, r, rows_a, cols_a, cols_b, alpha, a_cache, b_cache, remainX, remainY);
//        return;
//    }
    
    const int loops = cols_a / LOAD_SIZE_8x8;
//    const int loops = (cols_a - 1) / LOAD_SIZE_8x8 + 1;
    const int rows_b = cols_a;
    
    int blockX = (blockIdx.x * blockDim.x) * COLS_PER_THREAD_8x8;
    int blockY = (blockIdx.y * blockDim.y) * ROWS_PER_THREAD_8x8;
    
    float sum[ROWS_PER_THREAD_8x8][COLS_PER_THREAD_8x8] = {0.0f};
    
//    if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1 && blockIdx.y == 1) {
//        for(int i=0; i < LOAD_SIZE_8x8; i++)
//            for(int j=0; j < LOAD_SIZE_8x8; j++) {
//                a_cache[i][j] = 0;
//                b_cache[i][j] = 0;
//            }
//    }
//    
//    __syncthreads();
                
    for(int i=0; i < loops; i++) {
        for(int j=0; j < ROWS_PER_THREAD_8x8; j++) {
            int iy = blockY + threadIdx.y * ROWS_PER_THREAD_8x8 + j;
            
            if(iy >= rows_a)
                break;
            
            for(int k=0; k < COLS_PER_THREAD_8x8; k++) {
                int ix = i * blockDim.x * COLS_PER_THREAD_8x8 + k * blockDim.x + threadIdx.x;
                
                if(ix >= cols_a)
                    break;
                
                a_cache[threadIdx.y * ROWS_PER_THREAD_8x8 + j][threadIdx.x + k * blockDim.x] = a[index2d(iy, ix, cols_a)];
            }
        }
        
        for(int j=0; j < ROWS_PER_THREAD_8x8; j++) {
            int iy = (i * blockDim.y + threadIdx.y) * ROWS_PER_THREAD_8x8 + j;
            
            if(iy >= rows_b)
                break;
            
            for(int k=0; k < COLS_PER_THREAD_8x8; k++) {
                int ix = blockX + k * blockDim.x + threadIdx.x;
                
                if(ix >= cols_b)
                    break;
                
                b_cache[threadIdx.y * ROWS_PER_THREAD_8x8 + j][threadIdx.x + k * blockDim.x] = b[index2d(iy, ix, cols_b)];
            }
        }
        
        __syncthreads();
////            
//        if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1 && blockIdx.y == 1) {
//            printf("loops: %d %d %d %d\n", loops, cols_a / LOAD_SIZE_8x8, gridDim.x, gridDim.y);
//            print2d(&a_cache[0][0], LOAD_SIZE_8x8, LOAD_SIZE_8x8);
//            print2d(&b_cache[0][0], LOAD_SIZE_8x8, LOAD_SIZE_8x8);
//        }
        int loaded = min(LOAD_SIZE_8x8, cols_a - loops * blockDim.y * ROWS_PER_THREAD_8x8);
        
        for(int y0=0; y0 < ROWS_PER_THREAD_8x8; y0++) {
            int iy = threadIdx.y + y0 * blockDim.y;
            
            if(blockY + iy >= rows_a)
                break;
            
            for(int x0=0; x0 < COLS_PER_THREAD_8x8; x0++) {
                int ix = threadIdx.x + x0 * blockDim.x;
                
                if(blockX + ix >= cols_b) 
                    break;
                
                for(int j=0; j < LOAD_SIZE_8x8; j++)
                    sum[y0][x0] += a_cache[iy][j] * b_cache[j][ix];
            }
        }
        
        __syncthreads();
    }
//    
//    const int rem = cols_a % LOAD_SIZE;
//    
//    if(rem > 0) {
////        for(int y0=0; y0 < ROWS_PER_THREAD; y0++)
////            for(int x0=0; x0 < COLS_PER_THREAD; x0++)
////                for(int j=0; j < rem; j++)
////                    sum[y0][x0] += a_cache[threadIdx.y * ROWS_PER_THREAD + y0][j] * b_cache[j][threadIdx.x * COLS_PER_THREAD + x0];
////        
////        __syncthreads();        
//    }    
//    
    for(int y0=0; y0 < ROWS_PER_THREAD_8x8; y0++) {
        int ry = blockIdx.y * blockDim.y * ROWS_PER_THREAD_8x8 + threadIdx.y + y0 * blockDim.y;
        
        if(ry >= rows_a)
            break;
        
        for(int x0=0; x0 < COLS_PER_THREAD_8x8; x0++) {
            int rx = blockIdx.x * blockDim.x * COLS_PER_THREAD_8x8 + threadIdx.x + x0 * blockDim.x;
            
            if(rx >= cols_b)
                break;
            
            int ir = index2d(ry, rx, cols_b);
            r[ir] = r[ir] * alpha + sum[y0][x0];
        }
    }
}


#define BLOCK_SIZE_X_16x8 16
#define BLOCK_SIZE_Y_16x8 8
#define LOAD_SIZE_16x8 (BLOCK_SIZE_X_16x8 * 4)
#define ROWS_PER_THREAD_16x8 (LOAD_SIZE_16x8 / BLOCK_SIZE_Y_16x8)
#define COLS_PER_THREAD_16x8 (LOAD_SIZE_16x8 / BLOCK_SIZE_X_16x8)

extern "C"
__global__ void matrix_mul_matrix_16x8(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    __shared__ float a_cache[LOAD_SIZE_16x8][LOAD_SIZE_16x8];
    __shared__ float b_cache[LOAD_SIZE_16x8][LOAD_SIZE_16x8];
    
    int rowsPerBlock = blockDim.y * ROWS_PER_THREAD_16x8;
    int remainY = rows_a - blockIdx.y * rowsPerBlock;
    
    int colsPerBlock = blockDim.x * COLS_PER_THREAD_16x8;
    int remainX = cols_b - blockIdx.x * rowsPerBlock;
    
//    if(remainY < rowsPerBlock || remainX < colsPerBlock) {
//        matrix_mul_matrix_16x8_edges_unaligned(a, b, r, rows_a, cols_a, cols_b, alpha, a_cache, b_cache, remainX, remainY);
//        return;
//    }

    const int loops = cols_a / LOAD_SIZE_16x8;
    
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * COLS_PER_THREAD_16x8;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * ROWS_PER_THREAD_16x8;
    
    float sum[ROWS_PER_THREAD_16x8][COLS_PER_THREAD_16x8] = {0.0f};
    
    for(int i=0; i < loops; i++) {
        for(int j=0; j < ROWS_PER_THREAD_16x8; j++) {
            reinterpret_cast<float4*>(a_cache[threadIdx.y * ROWS_PER_THREAD_16x8 + j])[threadIdx.x] = 
                    reinterpret_cast<float4*>(&a[index2d(y + j, (i * blockDim.x + threadIdx.x) * COLS_PER_THREAD_16x8, cols_a)])[0];
            
            reinterpret_cast<float4*>(b_cache[threadIdx.y * ROWS_PER_THREAD_16x8 + j])[threadIdx.x] = 
                    reinterpret_cast<float4*>(&b[index2d((i * blockDim.y + threadIdx.y) * ROWS_PER_THREAD_16x8 + j, x, cols_b)])[0];
        }
        
        __syncthreads();
//        
//
//        float a[ROWS_PER_THREAD_16x8][COLS_PER_THREAD_16x8];
//        float b[COLS_PER_THREAD_16x8][COLS_PER_THREAD_16x8];
//        
//        const int window = LOAD_SIZE_16x8 / COLS_PER_THREAD_16x8;
//        
//        for(int m0 = 0; m0 < window; m0++) {
//            /* load all */
//            for(int y0=0; y0 < ROWS_PER_THREAD_16x8; y0++) {
//                for(int x0=0; x0 < COLS_PER_THREAD_16x8; x0++) {
//                    a[y0][x0] = a_cache[threadIdx.y * ROWS_PER_THREAD_16x8 + y0][m0 * COLS_PER_THREAD_16x8 + x0];
////                    if((threadIdx.y * ROWS_PER_THREAD_16x8 + y0) >= 64 ||
////                       (m0 * COLS_PER_THREAD_16x8 + x0 >= 64) ||
////                       (threadIdx.x * ROWS_PER_THREAD_16x8 + y0) >= 64)
////                        printf("%d %d %d %d %d %d %d %d\n",
////                                m0, y0, x0,
////                                threadIdx.x, threadIdx.y,
////                                threadIdx.y * ROWS_PER_THREAD_16x8 + y0, 
////                                m0 * COLS_PER_THREAD_16x8 + x0,
////                                threadIdx.x * ROWS_PER_THREAD_16x8 + y0);
//                }
//            }
//
//            for(int y0=0; y0 < COLS_PER_THREAD_16x8; y0++) {
//                for(int x0=0; x0 < COLS_PER_THREAD_16x8; x0++) {
//                    b[y0][x0] = b_cache[m0 * COLS_PER_THREAD_16x8 + y0][threadIdx.x * COLS_PER_THREAD_16x8 + x0];
//                }
//            }
//            
//            /* multiply all */
//            for(int y0=0; y0 < ROWS_PER_THREAD_16x8; y0++) {
//                for(int x0=0; x0 < COLS_PER_THREAD_16x8; x0++) {
//                    for(int k0=0; k0 < COLS_PER_THREAD_16x8; k0++)
//                        sum[y0][x0] += a[y0][k0] * b[k0][x0];
//                }
//            }
//        }
        for(int y0=0; y0 < ROWS_PER_THREAD_16x8; y0++)
            for(int x0=0; x0 < COLS_PER_THREAD_16x8; x0++)
                for(int j=0; j < LOAD_SIZE_16x8; j++)
                    sum[y0][x0] += a_cache[threadIdx.y * ROWS_PER_THREAD_16x8 + y0][j] * b_cache[j][threadIdx.x * COLS_PER_THREAD_16x8 + x0];
        
        __syncthreads();
    }
    
    const int rem = cols_a % LOAD_SIZE_16x8;
    
    if(rem > 0) {
//        for(int y0=0; y0 < ROWS_PER_THREAD; y0++)
//            for(int x0=0; x0 < COLS_PER_THREAD; x0++)
//                for(int j=0; j < rem; j++)
//                    sum[y0][x0] += a_cache[threadIdx.y * ROWS_PER_THREAD + y0][j] * b_cache[j][threadIdx.x * COLS_PER_THREAD + x0];
//        
//        __syncthreads();        
    }    
    
    for(int y0=0; y0 < ROWS_PER_THREAD_16x8; y0++) {
        int ir = index2d(y+y0, x, cols_b);
        float4 loaded = reinterpret_cast<float4*>(&r[ir])[0];
        loaded.x = loaded.x * alpha + sum[y0][0];
        loaded.y = loaded.y * alpha + sum[y0][1];
        loaded.z = loaded.z * alpha + sum[y0][2];
        loaded.w = loaded.w * alpha + sum[y0][3];
        reinterpret_cast<float4*>(&r[ir])[0] = loaded;
    }
}


//
//__device__ void load_row_unaligned(float* src, float* dst, int cols_src, int cols_dst, int x, int y, int length) {
//    int x0 = index2d(x, y, stride);    
//    int rem = x0 % 4;    
//    int aligned_x0 = x0 + rem;
//    int loads = length / 4;
//    int loops = (loads - 1) / blockDim.x + 1;
//    
//    for(int i=0; i < loops; i++) {
//        int inc = loops * 4 * blockDim.x;
//        
//        int i_src = aligned_x0 + inc;
//        int i_dst = threadIdx.x * 4 + rem + inc;
//        
//        if(x + i_dst + 4 > cols_src)
//            break;
//        
//        reinterpret_cast<float4*>(&dst[i_dst])[0] = reinterpret_cast<float4*>(&src[i_src])[0];
//    }
//    
//    if(threadIdx.x >= 4)
//        return;
//    
//}


#define LOAD_SIZE_64 64
#define ROWS_PER_THREAD 8
#define COLS_PER_THREAD 4

__device__ void load_row_aligned_64(float* src, float* dst, int length) {
//    int loads = length / 4;
//    int loops = (loads - 1) / blockDim.x + 1;
    
    
//        printf("%d %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
//    for(int i=0; i < loops; i++) {
//        int x = (threadIdx.x + i * blockDim.x) * 4;
        
    if(threadIdx.x >= length /4)
        return;

    reinterpret_cast<float4*>(dst)[threadIdx.x] = reinterpret_cast<float4*>(src)[threadIdx.x];
//    }
    
    int rem = length % 4;
    
    if(threadIdx.x >= rem)
        return;
    
    int i_dst = length - rem + threadIdx.x;
    dst[i_dst] = src[i_dst];
}


extern "C"
__global__ void matrix_mul_matrix_64x64(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    __shared__ float a_cache[LOAD_SIZE_16x8][LOAD_SIZE_16x8];
    __shared__ float b_cache[LOAD_SIZE_16x8][LOAD_SIZE_16x8];
    
    int rowsPerBlock = blockDim.y * ROWS_PER_THREAD_16x8;
    int remainY = rows_a - blockIdx.y * rowsPerBlock;
    
    int colsPerBlock = blockDim.x * COLS_PER_THREAD_16x8;
    int remainX = cols_b - blockIdx.x * rowsPerBlock;
    
//    if(remainY < rowsPerBlock || remainX < colsPerBlock) {
//        matrix_mul_matrix_16x8_edges_unaligned(a, b, r, rows_a, cols_a, cols_b, alpha, a_cache, b_cache, remainX, remainY);
//        return;
//    }

    const int loops = cols_a / LOAD_SIZE_16x8;
    const int rows_b = cols_a;
    
    int blockX = blockIdx.x * LOAD_SIZE_16x8;
    int blockY = blockIdx.y * LOAD_SIZE_16x8;
    
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * COLS_PER_THREAD_16x8;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * ROWS_PER_THREAD_16x8;
    
    float sum[ROWS_PER_THREAD_16x8][COLS_PER_THREAD_16x8] = {0.0f};
    
    for(int i=0; i < loops; i++) {
        for(int j=0; j < ROWS_PER_THREAD_16x8; j++) {
            int y = threadIdx.y * ROWS_PER_THREAD + j;
            int y_a = blockY + y;
            
            if(y_a < rows_a) {
                int x = i * LOAD_SIZE_16x8;
                int len = min(cols_a - x, LOAD_SIZE_16x8);
                load_row_aligned_64(&a[index2d(y_a, x, cols_a)], &a_cache[y][0], len);
            }
            
            int y_b = i * LOAD_SIZE_16x8 + y;
            
            if(y_b < rows_b) {
                int x = blockX;
                int len = min(cols_b - x, LOAD_SIZE_16x8);
                load_row_aligned_64(&b[index2d(y_b, x, cols_b)], &b_cache[y][0], len);
            }
        }
        
        __syncthreads();
        
        float a[ROWS_PER_THREAD_16x8][COLS_PER_THREAD_16x8];
        float b[COLS_PER_THREAD_16x8][COLS_PER_THREAD_16x8];
        
        const int window = LOAD_SIZE_16x8 / COLS_PER_THREAD_16x8;
        
        for(int m0 = 0; m0 < window; m0++) {
            /* load all */
            for(int y0=0; y0 < ROWS_PER_THREAD_16x8; y0++) {
                for(int x0=0; x0 < COLS_PER_THREAD_16x8; x0++) {
                    a[y0][x0] = a_cache[threadIdx.y * ROWS_PER_THREAD_16x8 + y0][m0 * COLS_PER_THREAD_16x8 + x0];
//                    if((threadIdx.y * ROWS_PER_THREAD_16x8 + y0) >= 64 ||
//                       (m0 * COLS_PER_THREAD_16x8 + x0 >= 64) ||
//                       (threadIdx.x * ROWS_PER_THREAD_16x8 + y0) >= 64)
//                        printf("%d %d %d %d %d %d %d %d\n",
//                                m0, y0, x0,
//                                threadIdx.x, threadIdx.y,
//                                threadIdx.y * ROWS_PER_THREAD_16x8 + y0, 
//                                m0 * COLS_PER_THREAD_16x8 + x0,
//                                threadIdx.x * ROWS_PER_THREAD_16x8 + y0);
                }
            }

            for(int y0=0; y0 < COLS_PER_THREAD_16x8; y0++) {
                for(int x0=0; x0 < COLS_PER_THREAD_16x8; x0++) {
                    b[y0][x0] = b_cache[m0 * COLS_PER_THREAD_16x8 + y0][threadIdx.x * COLS_PER_THREAD_16x8 + x0];
                }
            }
            
            /* multiply all */
            for(int y0=0; y0 < ROWS_PER_THREAD_16x8; y0++) {
                for(int x0=0; x0 < COLS_PER_THREAD_16x8; x0++) {
                    for(int k0=0; k0 < COLS_PER_THREAD_16x8; k0++)
                        sum[y0][x0] += a[y0][k0] * b[k0][x0];
                }
            }
        }
        
//        for(int y0=0; y0 < ROWS_PER_THREAD_16x8; y0++)
//            for(int x0=0; x0 < COLS_PER_THREAD_16x8; x0++)
//                for(int j=0; j < LOAD_SIZE_16x8; j++)
//                    sum[y0][x0] += a_cache[threadIdx.y * ROWS_PER_THREAD_16x8 + y0][j] * b_cache[j][threadIdx.x * COLS_PER_THREAD_16x8 + x0];
        
        __syncthreads();
    }
    
    const int rem = cols_a % LOAD_SIZE_16x8;
    
    if(rem > 0) {
//        for(int y0=0; y0 < ROWS_PER_THREAD; y0++)
//            for(int x0=0; x0 < COLS_PER_THREAD; x0++)
//                for(int j=0; j < rem; j++)
//                    sum[y0][x0] += a_cache[threadIdx.y * ROWS_PER_THREAD + y0][j] * b_cache[j][threadIdx.x * COLS_PER_THREAD + x0];
//        
//        __syncthreads();        
    }    
    
    for(int y0=0; y0 < ROWS_PER_THREAD_16x8; y0++) {
        int ir = index2d(y+y0, x, cols_b);
        float4 loaded = reinterpret_cast<float4*>(&r[ir])[0];
        loaded.x = loaded.x * alpha + sum[y0][0];
        loaded.y = loaded.y * alpha + sum[y0][1];
        loaded.z = loaded.z * alpha + sum[y0][2];
        loaded.w = loaded.w * alpha + sum[y0][3];
        reinterpret_cast<float4*>(&r[ir])[0] = loaded;
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
__global__ void matrix_mul_matrix_23(float* m0, float* m1, float* r, int cols_m0, int cols_m1) {
//    __shared__ float m0_cache[32][32];
//    __shared__ float m1_cache[32][32];
//  
//    int loops = cols_m0 / 32;
//    float sum = 0;
//    
//    int row_r = (blockIdx.x * blockDim.y + threadIdx.x);
//    int col_r = (blockIdx.y * blockDim.x + threadIdx.y);
//    int i = 0;
    // main loop
//    for(int i=0; i < loops; i++) {        
//        int row_m0 = row_r;
//        int col_m0 = i * blockDim.x + threadIdx.y;
//        
//        int row_m1 = i * blockDim.y + threadIdx.x;
//        int col_m1 = col_r;

//        m0_cache[threadIdx.x][threadIdx.y] = m0[index2d(row_m0, col_m0, cols_m0)];
//        m1_cache[threadIdx.x][threadIdx.y] = m1[index2d(row_m1, col_m1, cols_m1)];

//        __syncthreads();
//
//#pragma unroll
//        for(int k=0; k < 32; k++) {
//            sum += m0_cache[threadIdx.x][(threadIdx.y+k) % 32] * m1_cache[threadIdx.x][(threadIdx.y+k) % 32];
//        }
//
//        __syncthreads();
//    }
    
//    if(false) {        
//        int row_m0 = row_r;
//        int col_m0 = col_r;
//        int row_m1 = col_r;
//        int col_m1 = row_r;
//
//        m0_cache[threadIdx.x][threadIdx.y] = m0[index2d(row_m0, col_m0, cols_m0)];
//        m1_cache[threadIdx.y][threadIdx.x] = m1[index2d(row_m1, col_m1, cols_m1)];
//
//        __syncthreads();
//
//        for(int k=0; k < blockDim.x; k++)
//            sum += m0_cache[threadIdx.x][threadIdx.y] * m1_cache[threadIdx.y][threadIdx.x];
//
//        __syncthreads();
//    }

//    r[index2d(row_r, col_r, cols_m1)] = sum;
}
