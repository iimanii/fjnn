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
 * Mean Square Error derivative
 */
extern "C"
__global__ void MeanSquareErrorDerivative(float* output, float* expected, float* result, long size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    float multiplier = 2.0f / size;
    
    if(i < size)
        result[i] = multiplier * (output[i] - expected[i]);
}

/**
 * Binary Cross Entropy derivative
 */
extern "C"
__global__ void BinaryCrossEntropyDerivative(float* output, float* expected, float* result, float alpha, float beta, long size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < size) {
        float clipped = fmaxf(1e-7f, fminf(1.0f - 1e-7f, output[i]));
        
        float weight = (expected[i] == 1.0f) ? alpha : beta;
        result[i] = weight * (clipped - expected[i]) / (clipped * (1.0f - clipped));
    }
}
/**
 * Weighted Mean Square Error derivative
 */
extern "C"
__global__ void WeightedMeanSquareErrorPrime(float* output, float* expected, float* weights, float* result, long size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < size)
        result[i] = weights[i] * (output[i] - expected[i]);
}
