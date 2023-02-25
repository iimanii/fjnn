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
#include "matrix.h"

__device__ void print2d(float* a, int r, int c) {
    for(int w=0; w < r; w++) {
        for(int z=0; z < c; z++) {
            printf("%.0f ", a[w * c + z]);
        }
        printf("\n");
    }
}

template<const int W_A_ROWS, const int W_A_COLS, const int W_B_COLS>
__device__ void computeMatrix(float a_cache[CACHE_SIZE_ROWS_A][CACHE_SIZE_COLS_A],
                              float b_cache[CACHE_SIZE_COLS_A][CACHE_SIZE_COLS_B],
                              float sum[ROWS_PER_THREAD][COLS_PER_THREAD],
                              int idx, int idy) {    
    float a[W_A_ROWS][W_A_COLS];
    float b[W_A_COLS][W_B_COLS];
    
    for(int wy=0; wy < ROWS_PER_THREAD / W_A_ROWS; wy++) {
        for(int wx=0; wx < COLS_PER_THREAD / W_B_COLS; wx++) {            
            for(int wk=0; wk < CACHE_SIZE_COLS_A / W_A_COLS; wk++) {
                int a_y0 = idy + wy * W_A_ROWS;
                int a_x0 = wk * W_A_COLS;
                
                /* load window A */
                for(int y0=0; y0 < W_A_ROWS; y0++) {
                    for(int x0=0; x0 < W_A_COLS; x0++) {                        
                        a[y0][x0] = a_cache[a_y0 + y0][a_x0 + x0];
                    }
                }
                
                int b_x0 = idx + wx * W_B_COLS;
                int b_y0 = wk * W_A_COLS;
                
                /* load window B */
                for(int y0=0; y0 < W_A_COLS; y0++) {
                    for(int x0=0; x0 < W_B_COLS; x0++) {
//                        if(b_y0 + y0 > CACHE_SIZE_COLS_A || b_x0 + x0 > CACHE_SIZE_COLS_B)
//                            printf("B %d %d %d %d %d %d %d %d %d %d\n", threadIdx.x, blockIdx.x, blockIdx.y, b_y0, y0, b_x0, x0, wy, wx, wk);
                        
                        b[y0][x0] = b_cache[b_y0 + y0][b_x0 + x0];
                    }
                }

#if DEBUG == 1
                if(threadIdx.x == 60 && blockIdx.x == 0 && blockIdx.y == 0) {
                    print2d(&a[0][0], W_A_ROWS, W_A_COLS);
                    printf("A ############## %d %d\n", a_x0, a_y0);
                }
                
                if(threadIdx.x == 60 && blockIdx.x == 0 && blockIdx.y == 0) {
                    print2d(&b[0][0], W_A_COLS, W_B_COLS);
                    printf("B ############## %d %d\n", b_x0, b_y0);
                }
#endif

                int c_y0 = wy * W_A_ROWS;
                int c_x0 = wx * W_B_COLS;
                
                /* multiply all */
                for(int y0=0; y0 < W_A_ROWS; y0++) {
                    for(int x0=0; x0 < W_B_COLS; x0++) {
                        for(int k0=0; k0 < W_A_COLS; k0++)
                            sum[c_y0 + y0][c_x0 + x0] += a[y0][k0] * b[k0][x0];
                    }
                }
                
#if DEBUG == 1
                if(threadIdx.x == 60 && blockIdx.x == 0 && blockIdx.y == 0) {
                    print2d(&sum[0][0], ROWS_PER_THREAD, COLS_PER_THREAD);
                    printf("#########################\n");
                }
#endif
            }
        }
    }
}

template<const int THREADS, const int ROWS, const int COLS>
__device__ void load(float* src, float cache[ROWS][COLS], int y, int x, int stride) {
    const int iterations = (ROWS * COLS) / (4 * THREADS);
    const int idx = threadIdx.x % (COLS / 4);
    const int idy = threadIdx.x / (COLS / 4);
    const int rowsPerIteration = (4 * THREADS) / COLS;
//    
//    if(threadIdx.x == 0)
//        printf("%d %d %d %d\n", iterations, (ROWS * COLS), (4 * THREADS), rowsPerIteration);
    for(int i=0; i < iterations; i++) {
        reinterpret_cast<float4*>(&cache[idy + i * rowsPerIteration][idx * 4])[0] =
                reinterpret_cast<float4*>(&src[index2d(y + idy + i * rowsPerIteration, x + idx * 4, stride)])[0];
//        {(float)threadIdx.x, (float)threadIdx.x, (float)0, (float)0};
    }
}

template<const int ROWS, const int COLS>
__device__ void zero(float cache[ROWS][COLS]) {
    if(threadIdx.x == 0) {
        for(int i=0; i < ROWS; i++) {
            for(int j=0; j < COLS; j++) {
                cache[i][j] = 0;
            }
        }
    }
}

extern "C"
//__launch_bounds__(128, 5)
__global__ void matrix_mul_matrix(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    __shared__ float a_cache[CACHE_SIZE_ROWS_A][CACHE_SIZE_COLS_A];
    __shared__ float b_cache[CACHE_SIZE_COLS_A][CACHE_SIZE_COLS_B];
    
    float4 a_local_0, a_local_1, a_local_2, a_local_3;
    float4 b_local_0, b_local_1, b_local_2, b_local_3;

    const int loops = cols_a / CACHE_SIZE_COLS_A;
    
    int blockX = blockIdx.x * CACHE_SIZE_COLS_B;
    int blockY = blockIdx.y * CACHE_SIZE_ROWS_A;
    
    int cIdx = COLS_PER_THREAD * (threadIdx.x % (CACHE_SIZE_COLS_B / COLS_PER_THREAD));
    int cIdy = ROWS_PER_THREAD * (threadIdx.x / (CACHE_SIZE_COLS_B / COLS_PER_THREAD));
    
    float sum[ROWS_PER_THREAD][COLS_PER_THREAD] = {0.0f};
    
    zero<CACHE_SIZE_ROWS_A, CACHE_SIZE_COLS_A>(a_cache);
    zero<CACHE_SIZE_COLS_A, CACHE_SIZE_COLS_B>(b_cache);
    __syncthreads();
    
    for(int i=0; i < loops; i++) {
        load<BLOCK_SIZE, CACHE_SIZE_ROWS_A, CACHE_SIZE_COLS_A>(a, a_cache, blockY, i * CACHE_SIZE_COLS_A, cols_a);
        load<BLOCK_SIZE, CACHE_SIZE_COLS_A, CACHE_SIZE_COLS_B>(b, b_cache, i * CACHE_SIZE_COLS_A, blockX, cols_b);
     
        __syncthreads();

#if DEBUG == 1
        if(threadIdx.x == 60 && blockIdx.x == 0 && blockIdx.y == 0) {
            printf("loops: %d %d %d %d %d\n", loops, cIdx, cIdy, blockX, blockY);
            print2d(&a_cache[0][0], CACHE_SIZE_ROWS_A, CACHE_SIZE_COLS_A);
            printf("###########\n");
            print2d(&b_cache[0][0], CACHE_SIZE_COLS_A, CACHE_SIZE_COLS_B);  
            printf("###########################\n");          
        }
#endif
//        int a_y0 = blockY + aIdy;
//        int a_x0 = aIdx * 4 + i * CACHE_SIZE_COLS_A;
//        const int threadsPerRowA = CACHE_SIZE_COLS_A / 4;
//        
//        for(int j=0; j < CACHE_SIZE_ROWS_A; j+=(BLOCK_SIZE / threadsPerRowA))
//            reinterpret_cast<float4*>(&a_cache[aIdy + j][aIdx * 4])[0] = 
//                    reinterpret_cast<float4*>(&a[index2d(a_y0 + j, a_x0, cols_a)])[0];
//        
//        int b_y0 = bIdy + i * CACHE_SIZE_COLS_A;
//        int b_x0 = blockX + bIdx * 4;
//        const int threadsPerRowB = CACHE_SIZE_COLS_B / 4;
//        for(int j=0; j < CACHE_SIZE_COLS_A; j+=(BLOCK_SIZE / threadsPerRowB))
//            reinterpret_cast<float4*>(&b_cache[bIdy + j][bIdx * 4])[0] = 
//                    reinterpret_cast<float4*>(&b[index2d(b_y0 + j, b_x0, cols_b)])[0];

//            /* double buffering */
//            a_x0 += 16;
//            a_local_0 = reinterpret_cast<float4*>(&a[index2d(a_y0, a_x0, cols_a)])[0];
//            a_local_1 = reinterpret_cast<float4*>(&a[index2d(a_y0 + 32, a_x0, cols_a)])[0];
//            
//            b_y0 += 16;            
//            b_local_0 = reinterpret_cast<float4*>(&b[index2d(b_y0, b_x0, cols_b)])[0];
//            b_local_1 = reinterpret_cast<float4*>(&b[index2d(b_y0 + 8, b_x0, cols_b)])[0];       
        computeMatrix<CALC_WINDOW_A_ROWS, CALC_WINDOW_A_COLS, CALC_WINDOW_B_COLS>(a_cache, b_cache, sum, cIdx, cIdy);
        __syncthreads();
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
    
    int y = blockY + cIdy;
    int x = blockX + cIdx;
    
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
