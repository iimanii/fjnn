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
#include "matrix_unaligned.h"

extern "C"
__global__ void matrix_mul_matrix_default_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    matrix_mul_matrix_unaligned<DEFAULT_BLOCK_SIZE,
        DEFAULT_CACHE_SIZE_ROWS_A,
        DEFAULT_CACHE_SIZE_COLS_A,
        DEFAULT_CACHE_SIZE_COLS_B,
        
        DEFAULT_CALC_WINDOW_A_ROWS, 
        DEFAULT_CALC_WINDOW_A_COLS,
        DEFAULT_CALC_WINDOW_B_COLS>(a, b, r, rows_a, cols_a, cols_b, alpha);
}


/* ############################################ */
/* #################### 128 ################### */
/* ############################################ */


extern "C"
__global__ void matrix_mul_matrix_128x16x64_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {    
    matrix_mul_matrix_unaligned<128, 128, 16, 64, 16, 8, 1>(a, b, r, rows_a, cols_a, cols_b, alpha);
}

extern "C"
__global__ void matrix_mul_matrix_128x16x32_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    matrix_mul_matrix_unaligned<128, 128, 16, 32, 1, 1, 1>(a, b, r, rows_a, cols_a, cols_b, alpha);
}

extern "C"
__global__ void matrix_mul_matrix_128x16x16_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    matrix_mul_matrix_unaligned<128, 128, 16, 16, 1, 1, 1>(a, b, r, rows_a, cols_a, cols_b, alpha);
}

/* ############################################ */
/* #################### 64 #################### */
/* ############################################ */

extern "C"
__global__ void matrix_mul_matrix_64x16x64_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    matrix_mul_matrix_unaligned<128, 64, 16, 64, 1, 1, 1>(a, b, r, rows_a, cols_a, cols_b, alpha);
}

extern "C"
__global__ void matrix_mul_matrix_64x16x32_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    matrix_mul_matrix_unaligned<128, 64, 16, 32, 1, 1, 1>(a, b, r, rows_a, cols_a, cols_b, alpha);
}


extern "C"
__global__ void matrix_mul_matrix_64x16x16_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    matrix_mul_matrix_unaligned<128, 64, 16, 16, 1, 1, 1>(a, b, r, rows_a, cols_a, cols_b, alpha);
}

/* ############################################ */
/* #################### 32 #################### */
/* ############################################ */

extern "C"
__global__ void matrix_mul_matrix_32x16x64_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    matrix_mul_matrix_unaligned<128, 32, 16, 64, 1, 1, 1>(a, b, r, rows_a, cols_a, cols_b, alpha);
}

extern "C"
__global__ void matrix_mul_matrix_32x16x32_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    matrix_mul_matrix_unaligned<128, 32, 16, 32, 1, 1, 1>(a, b, r, rows_a, cols_a, cols_b, alpha);
}

extern "C"
__global__ void matrix_mul_matrix_32x16x16_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    matrix_mul_matrix_unaligned<128, 32, 16, 16, 1, 1, 1>(a, b, r, rows_a, cols_a, cols_b, alpha);
}

/* ############################################ */
/* #################### 16 #################### */
/* ############################################ */
extern "C"
__global__ void matrix_mul_matrix_16x16x16_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    matrix_mul_matrix_unaligned<256, 16, 16, 16, 1, 16, 1>(a, b, r, rows_a, cols_a, cols_b, alpha);
}


/**
 * each thread calculates 1 value in output
 * x -> column index
 * y -> row index
 **/
extern "C"
__global__ void matrix_mul_matrix_small_unaligned(float* a, float* b, float* r, int rows_a, int cols_a, int cols_b, float alpha) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(x >= cols_b || y >= rows_a)
        return;
    
    float sum = 0;
    
    for(int i=0; i < cols_a; i++) {
        int ia = index2d(y, i, cols_a);
        int ib = index2d(i, x, cols_b);
//        sum = __fmaf_rn(a[y * cols_a + i], b[i * cols_b + x], sum);
        sum += a[y * cols_a + i] * b[i * cols_b + x];
//        sum = __fadd_rn(__fmul_rn(a[ia], b[ib]), sum);
    }
    
//    r[y * cols_b + x] = __fmaf_rn(alpha, r[y * cols_b + x], sum);
    int ir = index2d(y, x, cols_b);
    r[ir] = sum + alpha * r[ir];
//    r[ir] = __fadd_rn(__fmul_rn(alpha, r[ir]), sum);
}