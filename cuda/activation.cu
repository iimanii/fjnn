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
 * Rectifier Linear Unit
 */
extern "C"
__global__ void ReLU(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size && v[row] < 0.0f)
        v[row] = 0.0f;
}
extern "C"
__global__ void ReLUPrime(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < size) {
        v[row] = v[row] > 0.0f ? 1.0f : 0.0f;
    }
}

/**
 * Leaky Rectifier Linear Unit
 */
extern "C"
__global__ void LeakyReLU(float* v, long size, float alpha) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size && v[row] < 0.0f)
        v[row] = v[row] * alpha;
}

/**
 * Gaussian Error Linear Unit (GELU)
 * Using the tanh approximation for GELU
 */
extern "C"
__global__ void GeLU(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < size) {
        float x = v[row];
        // GELU formula: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        v[row] = 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
    }
}

/**
 * Sigmoid
 */
extern "C"
__global__ void Sigmoid(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        v[row] = 1.0 / (1 + safe_exp(-v[row]));
}

extern "C"
__global__ void SigmoidPrime(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < size) {
        float sigmoid = 1.0f / (1.0f + safe_exp(-v[row]));
        v[row] = sigmoid * (1.0f - sigmoid);
    }
}

/**
 * Sin
 */
extern "C"
__global__ void Sin(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        v[row] = sin(v[row]);
}

/**
 * Tanh
 */
extern "C"
__global__ void Tanh(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        v[row] = tanh(v[row]);
}

extern "C"
__global__ void TanhPrime(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < size) {
        float tanhValue = tanh(v[row]);
        v[row] = 1.0f - tanhValue * tanhValue;
    }
}

/**
 * Step
 */
extern "C"
__global__ void Step(float* v, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        v[row] = v[row] >= 0 ? 1 : 0;
}


template<const int BLOCK_SIZE>
__device__ void SoftMax(float* v, long size) {
    __shared__ float temp[BLOCK_SIZE];
    
    int blockIndex = blockIdx.x * size;
    
    const int iterations = (size - 1) / BLOCK_SIZE + 1;
    
    float tmax = FLOAT_MIN;

    for(int i=0; i < iterations; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        
        if(index < size) {
            float value = v[blockIndex + index];
            tmax = max(tmax, value);
        }
    }
    
    /* get max over block */    
    float blockMax = reduceBlockMax<BLOCK_SIZE>(temp, tmax);
    
    /* calculate exp */
    float sum = 0;
    for(int i=0; i < iterations; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        
        if(index < size) {
            float value = safe_exp(v[blockIndex + index] - blockMax);
            sum += value;
            v[blockIndex + index] = value;
        }
    }
    
    /* get sum over block */
    float blockSum = reduceBlockSum<BLOCK_SIZE>(temp, sum);
    
    /* calculate final value and store */
    for(int i=0; i < iterations; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        
        if(index < size) {
            if(blockSum == 0) {
                v[blockIndex + index] = 1.0 / size;
            } else
                v[blockIndex + index] = v[blockIndex + index] / blockSum;
        }
    }
}

extern "C"
__global__ void SoftMax_32(float* v, long size) {
    SoftMax<32>(v, size);
}

extern "C"
__global__ void SoftMax_64(float* v, long size) {
    SoftMax<64>(v, size);
}

extern "C"
__global__ void SoftMax_128(float* v, long size) {
    SoftMax<128>(v, size);
}

extern "C"
__global__ void SoftMax_256(float* v, long size) {
    SoftMax<256>(v, size);
}

extern "C"
__global__ void SoftMax_512(float* v, long size) {
    SoftMax<512>(v, size);
}

extern "C"
__global__ void SoftMax_1(float* v, long size, float* sums) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row >= size) 
        return;
    
    cache[threadIdx.x] = safe_exp(v[row]);
    
    v[row] = cache[threadIdx.x];
    
    __syncthreads();
    
    sumBlock(cache, min((long)blockDim.x, size - blockDim.x * blockIdx.x));

    /* each block creates a sum */
    if(threadIdx.x == 0)
       sums[blockIdx.x] = cache[0];
}

extern "C"
__global__ void SoftMax_2(float* v, long size, float* sums, long sums_size) {
    __shared__ float cache[THREADS_PER_BLOCK];
    
    int loops = calcIterations(blockDim.x, sums_size);
    
    cache[threadIdx.x] = 0;
    
    /* 
     * Calculate total sum from "sums" array
     * should loop only once unless size > THREADS_PER_BLOCK^2
     */
    for(int i=0; i < loops; i++) {
        int index = i * blockDim.x + threadIdx.x;
        
        if(index < sums_size)
            cache[threadIdx.x] += sums[index];
    }
    
    __syncthreads();
        
    sumBlock(cache, min((long)blockDim.x, sums_size));

    __syncthreads();
        
    double sum = cache[0];
    
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        if(sum == 0) {
            v[row] = 1.0 / size;
        } else {
            v[row] /= sum;
        }
}

extern "C"
__global__ void empty_0() {

}

extern "C"
__global__ void empty_1(int i) {

}
