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


__device__ void print2d(float* a, int r, int c) {
    for(int w=0; w < r; w++) {
        for(int z=0; z < c; z++) {
            printf("%.0f ", a[w * c + z]);
        }
        printf("\n");
    }    
    printf("\n");
}

#define BLOCK_SIZE 128
#define CACHE_SIZE_ROWS 64
#define CACHE_SIZE_COLS 32

#define LOCAL_MEMORY_PER_THREAD 2

//#define LOAD_WINDOW (CACHE_SIZE_COLS * LOCAL_MEMORY_PER_THREAD)

#define CALC_WINDOW_ROWS 128
#define CALC_WINDOW_COLS 64

#define ROWS_PER_THREAD 8
#define COLS_PER_THREAD 4

extern "C"
__global__ void matrix_mul_matrix_6(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    __shared__ float a_cache[CACHE_SIZE_ROWS][CACHE_SIZE_COLS];
    __shared__ float b_cache[CACHE_SIZE_COLS][CACHE_SIZE_ROWS];
    
    float4 a_local[LOCAL_MEMORY_PER_THREAD];
    float4 b_local[LOCAL_MEMORY_PER_THREAD];

    const int loops = cols_a / CACHE_SIZE_COLS;
    
    int blockX = blockIdx.x * CACHE_SIZE_ROWS;
    int blockY = blockIdx.y * CACHE_SIZE_ROWS;
    
    int aIdx = threadIdx.x % (CACHE_SIZE_COLS / 4);
    int aIdy = threadIdx.x / (CACHE_SIZE_COLS / 4);
    
    int bIdx = threadIdx.x % (CACHE_SIZE_ROWS / 4);
    int bIdy = threadIdx.x / (CACHE_SIZE_ROWS / 4);
    
    int cIdx = threadIdx.x % (CALC_WINDOW_COLS / COLS_PER_THREAD);
    int cIdy = threadIdx.x / (CALC_WINDOW_ROWS / ROWS_PER_THREAD);
    
    float sum[ROWS_PER_THREAD][COLS_PER_THREAD] = {0.0f};
    
    for(int i=0; i < loops; i++) {
        /* load all into local memory */
            int a_y0 = blockY + aIdy;
            int a_x0 = aIdx * 4 + i * CACHE_SIZE_COLS;
//            reinterpret_cast<float4*>(a_local)[j] = reinterpret_cast<float4*>(&a[index2d(a_y0, a_x0, cols_a)])[0];
            reinterpret_cast<float4*>(&a_cache[aIdy][aIdx * 4])[0] = reinterpret_cast<float4*>(&a[index2d(a_y0, a_x0, cols_a)])[0];
            a_y0 += 64;
            reinterpret_cast<float4*>(&a_cache[aIdy][aIdx * 4])[0] = reinterpret_cast<float4*>(&a[index2d(a_y0, a_x0, cols_a)])[0];
            
            int b_y0 = bIdy + i * CACHE_SIZE_COLS;
            int b_x0 = blockX + bIdx * 4;
//            reinterpret_cast<float4*>(b_local)[j] = reinterpret_cast<float4*>(&b[index2d(b_y0, b_x0, cols_b)])[0];
            reinterpret_cast<float4*>(&b_cache[bIdy][bIdx * 4])[0] = reinterpret_cast<float4*>(&b[index2d(b_y0, b_x0, cols_b)])[0];
            b_y0 += 4;
            reinterpret_cast<float4*>(&a_cache[aIdy][aIdx * 4])[0] = reinterpret_cast<float4*>(&a[index2d(a_y0, a_x0, cols_a)])[0];
//        }
        
        /* save each round to shared memory then compute */
//        for(int j=0; j < LOCAL_MEMORY_PER_THREAD; j++) {
//            reinterpret_cast<float4*>(&a_cache[aIdy][aIdx * 4])[0] = reinterpret_cast<float4*>(a_local)[j];
//            reinterpret_cast<float4*>(&b_cache[bIdy][bIdx * 4])[0] = reinterpret_cast<float4*>(b_local)[j];
            
            __syncthreads();

            /* calculate */
            float a[ROWS_PER_THREAD][COLS_PER_THREAD];
            float b[COLS_PER_THREAD][COLS_PER_THREAD];
            
            const int window = CACHE_SIZE_COLS / COLS_PER_THREAD;
            
            for(int m0=0; m0 < window; m0++) {
                /* load all */
                for(int y0=0; y0 < ROWS_PER_THREAD; y0++) {
                    for(int x0=0; x0 < COLS_PER_THREAD; x0++) {
                        int y1 = (y0 + cIdy) % ROWS_PER_THREAD;
                        a[y1][x0] = a_cache[cIdy * ROWS_PER_THREAD + y1][m0 * COLS_PER_THREAD + x0];
                    }
                }
    
                for(int y0=0; y0 < COLS_PER_THREAD; y0++) {
                    for(int x0=0; x0 < COLS_PER_THREAD; x0++) {
                        int x1 = (x0 + cIdx) % COLS_PER_THREAD;
                        b[y0][x1] = b_cache[m0 * COLS_PER_THREAD + y0][cIdx * COLS_PER_THREAD + x1];
                    }
                }
                
                /* multiply all */
                for(int y0=0; y0 < ROWS_PER_THREAD; y0++) {
                    for(int x0=0; x0 < COLS_PER_THREAD; x0++) {
                        for(int k0=0; k0 < COLS_PER_THREAD; k0++)
                            sum[y0][x0] += a[y0][k0] * b[k0][x0];
                    }
                }
            }

            __syncthreads();
//        }
    }
    
    int y = blockY + cIdy * ROWS_PER_THREAD;
    int x = blockX + cIdx * COLS_PER_THREAD;
    
    for(int y0=0; y0 < ROWS_PER_THREAD; y0++) {
        int ir = index2d(y + y0, x, cols_b);
        float4 loaded = reinterpret_cast<float4*>(&r[ir])[0];
        loaded.x = loaded.x * alpha + sum[y0][0];
        loaded.y = loaded.y * alpha + sum[y0][1];
        loaded.z = loaded.z * alpha + sum[y0][2];
        loaded.w = loaded.w * alpha + sum[y0][3];
        reinterpret_cast<float4*>(&r[ir])[0] = loaded;
    }
}


__forceinline__
__device__ void compute(float sum[ROWS_PER_THREAD][COLS_PER_THREAD], 
                        float a_cache[CACHE_SIZE_ROWS][CACHE_SIZE_COLS], 
                        float b_cache[CACHE_SIZE_COLS][CACHE_SIZE_ROWS], int cIdx, int cIdy) {
    float a[ROWS_PER_THREAD][COLS_PER_THREAD];
    float b[COLS_PER_THREAD][COLS_PER_THREAD];

    const int window = CACHE_SIZE_COLS / COLS_PER_THREAD;

    for(int m0 = 0; m0 < window; m0++) {
        /* load all */
        for(int y0=0; y0 < ROWS_PER_THREAD; y0++) {
            for(int x0=0; x0 < COLS_PER_THREAD; x0++) {
                a[y0][x0] = a_cache[cIdy * ROWS_PER_THREAD + y0][m0 * COLS_PER_THREAD + x0];
            }
        }

        for(int y0=0; y0 < COLS_PER_THREAD; y0++) {
            for(int x0=0; x0 < COLS_PER_THREAD; x0++) {
                b[y0][x0] = b_cache[m0 * COLS_PER_THREAD + y0][cIdx * COLS_PER_THREAD + x0];
            }
        }

        /* multiply all */
        for(int y0=0; y0 < ROWS_PER_THREAD; y0++) {
            for(int x0=0; x0 < COLS_PER_THREAD; x0++) {
                for(int k0=0; k0 < COLS_PER_THREAD; k0++)
                    sum[y0][x0] += a[y0][k0] * b[k0][x0];
            }
        }
    }
}

extern "C"
__global__ void matrix_mul_matrix_7(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    __shared__ float a_cache[CACHE_SIZE_ROWS][CACHE_SIZE_COLS];
    __shared__ float b_cache[CACHE_SIZE_COLS][CACHE_SIZE_ROWS];
    
    float4 a_local_0, a_local_1, a_local_2, a_local_3;
    float4 b_local_0, b_local_1, b_local_2, b_local_3;

    const int loops = cols_a / 32;
    
    int blockX = blockIdx.x * CACHE_SIZE_ROWS;
    int blockY = blockIdx.y * CACHE_SIZE_ROWS;
    
    int aIdx = threadIdx.x % (CACHE_SIZE_COLS / 4);
    int aIdy = threadIdx.x / (CACHE_SIZE_COLS / 4);
    
    int bIdx = threadIdx.x % (CACHE_SIZE_ROWS / 4);
    int bIdy = threadIdx.x / (CACHE_SIZE_ROWS / 4);
    
    int cIdx = threadIdx.x % (CACHE_SIZE_ROWS / 4);
    int cIdy = threadIdx.x / (CACHE_SIZE_ROWS / 4);
    
    float sum[ROWS_PER_THREAD][COLS_PER_THREAD] = {0.0f};
    
    for(int i=0; i < loops; i++) {
        int a_y0 = blockY + aIdy;
        int a_x0 = aIdx * 4 + i * 32;
        /* load all into local memory */
//        for(int j=0; j < LOCAL_MEMORY_PER_THREAD; j++) {
//        a_local_0 = reinterpret_cast<float4*>(&a[index2d(a_y0, a_x0 + 0 * CACHE_SIZE_COLS, cols_a)])[0];
//        a_local_1 = reinterpret_cast<float4*>(&a[index2d(a_y0, a_x0 + 1 * CACHE_SIZE_COLS, cols_a)])[0];
//        a_local_2 = reinterpret_cast<float4*>(&a[index2d(a_y0, a_x0 + 2 * CACHE_SIZE_COLS, cols_a)])[0];
//        a_local_3 = reinterpret_cast<float4*>(&a[index2d(a_y0, a_x0 + 3 * CACHE_SIZE_COLS, cols_a)])[0];
            //reinterpret_cast<float4*>(&a[index2d(a_y0, a_x0, cols_a)])[0];
            
        int b_y0 = bIdy + i * 32;
        int b_x0 = blockX + bIdx * 4;
//        b_local_0 = reinterpret_cast<float4*>(&b[index2d(b_y0 + 0 * CACHE_SIZE_COLS, b_x0, cols_b)])[0];
//        b_local_1 = reinterpret_cast<float4*>(&b[index2d(b_y0 + 1 * CACHE_SIZE_COLS, b_x0, cols_b)])[0];
//        b_local_2 = reinterpret_cast<float4*>(&b[index2d(b_y0 + 2 * CACHE_SIZE_COLS, b_x0, cols_b)])[0];
//        b_local_3 = reinterpret_cast<float4*>(&b[index2d(b_y0 + 3 * CACHE_SIZE_COLS, b_x0, cols_b)])[0];
//        reinterpret_cast<float4*>(b_local)[j] = reinterpret_cast<float4*>(&b[index2d(b_y0, b_x0, cols_b)])[0];
//        }
        
        /* save each round to shared memory then compute */
//        for(int j=0; j < LOCAL_MEMORY_PER_THREAD; j++) {
//            if(threadIdx.x == 0 && threadIdx.y == 0) {
//                for(int i=0; i < CACHE_SIZE_ROWS; i++)
//                    for(int j=0; j < CACHE_SIZE_COLS; j++) {
//                        a_cache[i][j] = 0;
//                        b_cache[j][i] = 0;
//                    }
//            }
//
//            __syncthreads();

            reinterpret_cast<float4*>(&a_cache[aIdy][aIdx * 4])[0] = 
                    reinterpret_cast<float4*>(&a[index2d(a_y0, a_x0, cols_a)])[0];//a_local_0;//reinterpret_cast<float4*>(a_local)[j];
            reinterpret_cast<float4*>(&a_cache[aIdy + 16][aIdx * 4])[0] = 
                    reinterpret_cast<float4*>(&a[index2d(a_y0 + 16, a_x0, cols_a)])[0];//a_local_0;//reinterpret_cast<float4*>(a_local)[j];
            reinterpret_cast<float4*>(&a_cache[aIdy + 32][aIdx * 4])[0] = 
                    reinterpret_cast<float4*>(&a[index2d(a_y0 + 32, a_x0, cols_a)])[0];            
            reinterpret_cast<float4*>(&a_cache[aIdy + 48][aIdx * 4])[0] = 
                    reinterpret_cast<float4*>(&a[index2d(a_y0 + 48, a_x0, cols_a)])[0];
                    
            reinterpret_cast<float4*>(&b_cache[bIdy][bIdx * 4])[0] = 
                    reinterpret_cast<float4*>(&b[index2d(b_y0, b_x0, cols_b)])[0];//b_local_0;//reinterpret_cast<float4*>(b_local)[j];       
            reinterpret_cast<float4*>(&b_cache[bIdy + 8][bIdx * 4])[0] = 
                    reinterpret_cast<float4*>(&b[index2d(b_y0 + 8, b_x0, cols_b)])[0];//b_local_0;//reinterpret_cast<float4*>(b_local)[j];
            reinterpret_cast<float4*>(&b_cache[bIdy + 16][bIdx * 4])[0] = 
                    reinterpret_cast<float4*>(&b[index2d(b_y0 + 16, b_x0, cols_b)])[0];//b_local_0;//reinterpret_cast<float4*>(b_local)[j];
            reinterpret_cast<float4*>(&b_cache[bIdy + 24][bIdx * 4])[0] = 
                    reinterpret_cast<float4*>(&b[index2d(b_y0 + 24, b_x0, cols_b)])[0];//b_local_0;//reinterpret_cast<float4*>(b_local)[j];
                    
//            /* double buffering */
//            a_x0 += 16;
//            a_local_0 = reinterpret_cast<float4*>(&a[index2d(a_y0, a_x0, cols_a)])[0];
//            a_local_1 = reinterpret_cast<float4*>(&a[index2d(a_y0 + 32, a_x0, cols_a)])[0];
//            
//            b_y0 += 16;            
//            b_local_0 = reinterpret_cast<float4*>(&b[index2d(b_y0, b_x0, cols_b)])[0];
//            b_local_1 = reinterpret_cast<float4*>(&b[index2d(b_y0 + 8, b_x0, cols_b)])[0];
            
            __syncthreads();
            compute(sum, a_cache, b_cache, cIdx, cIdy);           
            __syncthreads(); 
//            
//            reinterpret_cast<float4*>(&a_cache[aIdy][aIdx * 4])[0] = a_local_0;
//            reinterpret_cast<float4*>(&a_cache[aIdy + 32][aIdx * 4])[0] = a_local_1;
//            
//            reinterpret_cast<float4*>(&b_cache[bIdy][bIdx * 4])[0] = b_local_0;
//            reinterpret_cast<float4*>(&b_cache[bIdy + 8][bIdx * 4])[0] = b_local_1;
//            __syncthreads();
//            compute(sum, a_cache, b_cache, cIdx, cIdy);           
//            __syncthreads(); 
//            
//            reinterpret_cast<float4*>(&a_cache[aIdy][aIdx * 4])[0] = a_local_1;//reinterpret_cast<float4*>(a_local)[j];
//            reinterpret_cast<float4*>(&b_cache[bIdy][bIdx * 4])[0] = b_local_1;//reinterpret_cast<float4*>(b_local)[j];
//            __syncthreads();
//            compute(sum, a_cache, b_cache, cIdx, cIdy);           
//            __syncthreads(); 
//            
//            reinterpret_cast<float4*>(&a_cache[aIdy][aIdx * 4])[0] = a_local_2;//reinterpret_cast<float4*>(a_local)[j];
//            reinterpret_cast<float4*>(&b_cache[bIdy][bIdx * 4])[0] = b_local_2;//reinterpret_cast<float4*>(b_local)[j];
//            __syncthreads();
//            compute(sum, a_cache, b_cache, cIdx, cIdy);           
//            __syncthreads(); 
//            
//            reinterpret_cast<float4*>(&a_cache[aIdy][aIdx * 4])[0] = a_local_3;//reinterpret_cast<float4*>(a_local)[j];
//            reinterpret_cast<float4*>(&b_cache[bIdy][bIdx * 4])[0] = b_local_3;//reinterpret_cast<float4*>(b_local)[j];
//            __syncthreads();
//            compute(sum, a_cache, b_cache, cIdx, cIdy);           
//            __syncthreads(); 
                        
//            if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
//                printf("loops: %d %d %d %d\n", loops, cols_a / LOAD_SIZE_8x8, gridDim.x, gridDim.y);
//                print2d(&a_cache[0][0], CACHE_SIZE_ROWS, CACHE_SIZE_COLS);
//                printf("###########\n");
//                print2d(&b_cache[0][0], CACHE_SIZE_COLS, CACHE_SIZE_ROWS);  
//                printf("###########################\n");          
//            }

//            for(int y0=0; y0 < ROWS_PER_THREAD; y0++)
//                for(int x0=0; x0 < COLS_PER_THREAD; x0++)
//                    for(int j=0; j < CACHE_SIZE_COLS; j++)
//                        sum[y0][x0] += a_cache[cIdy * ROWS_PER_THREAD + y0][j] * b_cache[j][cIdx * COLS_PER_THREAD + x0];
//            /* calculate */
            

//            __syncthreads();
//        }
    }
    
//    const int rem = cols_a % LOAD_SIZE_16x8;
    
//    if(rem > 0) {
//        for(int y0=0; y0 < ROWS_PER_THREAD; y0++)
//            for(int x0=0; x0 < COLS_PER_THREAD; x0++)
//                for(int j=0; j < rem; j++)
//                    sum[y0][x0] += a_cache[threadIdx.y * ROWS_PER_THREAD + y0][j] * b_cache[j][threadIdx.x * COLS_PER_THREAD + x0];
//        
//        __syncthreads();        
//    }    
    
    int y = blockY + cIdy * ROWS_PER_THREAD;
    int x = blockX + cIdx * COLS_PER_THREAD;
    
    for(int y0=0; y0 < ROWS_PER_THREAD; y0++) {
        int ir = index2d(y + y0, x, cols_b);
        float4 loaded = reinterpret_cast<float4*>(&r[ir])[0];
        loaded.x = loaded.x * alpha + sum[y0][0];
        loaded.y = loaded.y * alpha + sum[y0][1];
        loaded.z = loaded.z * alpha + sum[y0][2];
        loaded.w = loaded.w * alpha + sum[y0][3];
        reinterpret_cast<float4*>(&r[ir])[0] = loaded;
    }
}
