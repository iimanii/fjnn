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

#define EPSILON 1e-5f;

/**
 * LayerNorm
 */
template<const int BLOCK_SIZE>
__device__ float getMean(float* src, int size, float shared[BLOCK_SIZE]) {
    const int iterations = (size - 1) / BLOCK_SIZE + 1;
    
    // Calculate mean
    float sum = 0;
    for(int i=0; i < iterations; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        if(index < size) {
            sum += src[index];
        }
    }
    
    // Get mean over block
    return reduceBlockSum<BLOCK_SIZE>(shared, sum) / size;
}

template<const int BLOCK_SIZE>
__device__ float getVariance(float* src, int size, float mean, float shared[BLOCK_SIZE]) {
    const int iterations = (size - 1) / BLOCK_SIZE + 1;
    
    float sum = 0;    
    for(int i=0; i < iterations; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        if(index < size) {
            float diff = src[index] - mean;
            sum += diff * diff;
        }
    }

    // Get variance over block
    return reduceBlockSum<BLOCK_SIZE>(shared, sum) / size;
}

template<const int BLOCK_SIZE>
__device__ void LayerNormalizer(float* pre_norm, float* norm, float* post_norm, float* stds, float* gamma, float* beta, long neurons) {
    __shared__ float temp[BLOCK_SIZE];
    
    int blockIndex = blockIdx.x * neurons;
    
    float* input = &pre_norm[blockIndex];
    const int iterations = (neurons - 1) / BLOCK_SIZE + 1;
    
    // Get mean over block
    float mean = getMean<BLOCK_SIZE>(input, neurons, temp);
//    
//    if(threadIdx.x == 0) 
//        means[blockIdx.x] = mean;
    
    // Get variance over block
    float variance = getVariance<BLOCK_SIZE>(input, neurons, mean, temp) + EPSILON;
    float std = sqrtf(variance);
    
    if(threadIdx.x == 0) 
        stds[blockIdx.x] = std;
    
    // Normalize values
    for(int i=0; i < iterations; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        if(index < neurons) {
            float normalized = (input[index] - mean) / std;
            norm[blockIndex + index] = normalized;
            post_norm[blockIndex + index] = gamma[index] * normalized + beta[index];
        }
    }
}

template<const int BLOCK_SIZE>
__device__ void LayerNormalizerDerivative(float* norm, float* stds, float* gamma, float* gradients, float* deltaLoss, long neurons) {
    __shared__ float shared[BLOCK_SIZE];
    
    int blockIndex = blockIdx.x * neurons;
    
    const int iterations = (neurons - 1) / BLOCK_SIZE + 1;
    
    float sumDx = 0;                        // Sum of dxHat
    float sumDxXHat = 0;                    // Sum of dxHat * normalized values
    
    for(int i=0; i < iterations; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        if(index < neurons) {
            float dxHat = gradients[blockIndex + index] * gamma[i];

            // Accumulate sums for later use
            sumDx += dxHat;
            sumDxXHat += dxHat * norm[blockIndex + index];
        }
    }
    
    // Get mean over block
    float meanDx = reduceBlockSum<BLOCK_SIZE>(shared, sumDx) / neurons;
    float meanDxHat = reduceBlockSum<BLOCK_SIZE>(shared, sumDxXHat) / neurons;    
    float invStd = 1.0f / stds[blockIdx.x];
    
    // Normalize values
    for(int i=0; i < iterations; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        if(index < neurons) {
            float xHat = norm[blockIndex + index];
            float dxHat = gradients[blockIndex + index] * gamma[i];
            
            deltaLoss[blockIndex + index] = invStd * (dxHat - meanDx - xHat * meanDxHat);
        }
    }
}

/* 
 * create a list of LayerNorm_X
 */
#define LAYER_NORMALIZER_KERNELS(SIZE) \
extern "C" \
__global__ void LayerNormalizer_##SIZE(float* pre_norm, float* norm, float* post_norm, float* stds, float* gamma, float* beta, long neurons)  { \
    LayerNormalizer<SIZE>(pre_norm, norm, post_norm, stds, gamma, beta, neurons); \
} \
extern "C" \
__global__ void LayerNormalizerDerivative_##SIZE(float* norm, float* stds, float* gamma, float* gradients, float* deltaLoss, long neurons)  { \
    LayerNormalizerDerivative<SIZE>(norm, stds, gamma, gradients, deltaLoss, neurons); \
}

LAYER_NORMALIZER_KERNELS(32)
LAYER_NORMALIZER_KERNELS(64)
LAYER_NORMALIZER_KERNELS(128)
LAYER_NORMALIZER_KERNELS(256)
LAYER_NORMALIZER_KERNELS(512)