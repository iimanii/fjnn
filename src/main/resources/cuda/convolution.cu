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
 * Transforms 1D input data into im2col format for convolution
 */
extern "C"
__global__ void im2col(float* input, float* im2colOutput, int inputDim, int outputDim, int kernelWidth, int unitSize, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = batchSize * outputDim * kernelWidth;
    
    if (idx >= totalSize) 
        return;
    
    // Calculate position in im2col output
    int kernelPos = idx % kernelWidth;
    int outputPos = (idx / kernelWidth) % outputDim;
    int batchIdx = idx / (outputDim * kernelWidth);
    
    // Calculate corresponding position in input
    int inputPos = batchIdx * inputDim + outputPos * unitSize + kernelPos;
    
    // Copy the value
    im2colOutput[idx] = input[inputPos];
}

/**
 * Extracts channel data from interleaved input
 * Each thread handles one output element
 */
extern "C"
__global__ void extractChannel(float* input, float* output, 
                              int inputStride, int unitSize, int channelOffset,
                              int totalChunks) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int totalElements = unitSize * totalChunks;
    
    if (idx < totalElements) {
        // Decode output position: chunk, unit_pos
        int chunk = idx / unitSize;
        int unit_pos = idx % unitSize;
        
        // Calculate input index
        int src_idx = chunk * inputStride + channelOffset + unit_pos;
        
        output[idx] = input[src_idx];
    }
}

/**
 * Distributes gradients from a single channel back to interleaved input positions
 * This reverses the channel extraction process during backpropagation
 */
extern "C"
__global__ void distributeChannelGradients(float* channelGradients,
                                          float* inputGradients,
                                          int inputStride,
                                          int unitSize,
                                          int channelOffset,
                                          int totalChunks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = unitSize * totalChunks;
    
    if (idx < totalElements) {
        // Calculate which chunk and which position within unit
        int chunkIdx = idx / unitSize;
        int unitPos = idx % unitSize;
        
        // Calculate source position in channel gradients
        int srcPos = idx;
        
        // Calculate destination position in interleaved input
        int dstPos = chunkIdx * inputStride + channelOffset + unitPos;
        
        // Direct assignment - no overlaps since each input position maps to one channel
        inputGradients[dstPos] = channelGradients[srcPos];
    }
}

extern "C"
__global__ void computeInputGradients(float* inputGradients, 
                                      float* outputGradients, 
                                      float* weights,
                                      int unitSize,
                                      int inputSize, 
                                      int outputSize, 
                                      int kernelWidth, 
                                      int batchSize) {
    // Each thread handles one input position
    int globalInputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalInputIdx >= inputSize * batchSize) 
        return;
    
    // Decode batch and position within batch
    int inputPos = globalInputIdx % inputSize;
    int batchIdx = globalInputIdx / inputSize;
    
    // Find range of output positions that use this input position
    int unitIndex = inputPos / unitSize;
    int firstOutputPos = max(0, unitIndex - (kernelWidth / unitSize) + 1);
    int lastOutputPos = min(outputSize - 1, unitIndex);
    
    // Accumulate gradient from all contributing output positions
    float totalGradient = 0.0f;
    
    for (int outputPos = firstOutputPos; outputPos <= lastOutputPos; outputPos++) {
        // Which weight connects this input to this output?
        int weightIdx = inputPos - (outputPos * unitSize);
        
        // Get gradient from this output position
        int outputIdx = batchIdx * outputSize + outputPos;
        float outputGrad = outputGradients[outputIdx];
        
        // Add contribution: weight * output_gradient
        totalGradient += weights[weightIdx] * outputGrad;
    }
    
    // Store final gradient for this input position
    inputGradients[globalInputIdx] = totalGradient;
}

/**
 * Adam weight update kernel
 */
extern "C"
__global__ void adamUpdateWeights(float* weights,
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
        
        // Update weight: w = w - lr * mHat / (sqrt(vHat) + epsilon)
        weights[idx] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
    }
}

/**
 * Adam bias update kernel
 */
extern "C"
__global__ void adamUpdateBias(float* bias,
                              float* gradient,
                              float* momentum,
                              float* velocity,
                              float learningRate,
                              float beta1,
                              float beta2,
                              float epsilon,
                              float beta1Power,
                              float beta2Power) {
    // Update momentum
    momentum[0] = beta1 * momentum[0] + (1.0f - beta1) * gradient[0];
    
    // Update velocity
    velocity[0] = beta2 * velocity[0] + (1.0f - beta2) * gradient[0] * gradient[0];
    
    // Bias correction
    float mHat = momentum[0] / (1.0f - beta1Power);
    float vHat = velocity[0] / (1.0f - beta2Power);
    
    // Update bias
    bias[0] -= learningRate * mHat / (sqrtf(vHat) + epsilon);
}