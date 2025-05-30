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

extern "C"
__global__ void updateWeightsWithDecay(
    float* weights,             // weights to update
    const float* gradients,     // gradients
    float learningRate,         // learning rate
    float weightDecay,          // weight decay parameter
    int size                    // total number of weights
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Combine gradient and weight decay in one update
        weights[idx] -= learningRate * (gradients[idx] + weightDecay * weights[idx]);
    }
}

/**
 * Adam weight update kernel for Connection class
 */
extern "C"
__global__ void adamUpdateConnectionWeights(float* weights,
                                            float* gradients,
                                            float* momentum,
                                            float* velocity,
                                            float learningRate,
                                            float beta1,
                                            float beta2,
                                            float epsilon,
                                            float beta1Power,
                                            float beta2Power,
                                            float weightDecay,
                                            long size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx];
        
        // Update momentum: m = beta1 * m + (1 - beta1) * grad
        momentum[idx] = beta1 * momentum[idx] + (1.0f - beta1) * grad;
        
        // Update velocity: v = beta2 * v + (1 - beta2) * grad^2
        velocity[idx] = beta2 * velocity[idx] + (1.0f - beta2) * grad * grad;
        
        // Bias correction
        float mHat = momentum[idx] / (1.0f - beta1Power);
        float vHat = velocity[idx] / (1.0f - beta2Power);
        
        // Update weight: w = w - lr * mHat / (sqrt(vHat) + epsilon)
        weights[idx] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
        
        // Apply weight decay after Adam update (similar to CPU implementation)
        if (weightDecay > 0.0f) {
            weights[idx] -= learningRate * weightDecay * weights[idx];
        }
    }
}

/**
 * Adam bias update kernel for Connection class
 */
extern "C"
__global__ void adamUpdateConnectionBiases(float* biases,
                                           float* gradients,
                                           float* momentum,
                                           float* velocity,
                                           float learningRate,
                                           float beta1,
                                           float beta2,
                                           float epsilon,
                                           float beta1Power,
                                           float beta2Power,
                                           long size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx];
        
        // Update momentum: m = beta1 * m + (1 - beta1) * grad
        momentum[idx] = beta1 * momentum[idx] + (1.0f - beta1) * grad;
        
        // Update velocity: v = beta2 * v + (1 - beta2) * grad^2
        velocity[idx] = beta2 * velocity[idx] + (1.0f - beta2) * grad * grad;
        
        // Bias correction
        float mHat = momentum[idx] / (1.0f - beta1Power);
        float vHat = velocity[idx] / (1.0f - beta2Power);
        
        // Update bias: b = b - lr * mHat / (sqrt(vHat) + epsilon)
        biases[idx] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
        
        // Note: No weight decay applied to biases (consistent with CPU implementation)
    }
}