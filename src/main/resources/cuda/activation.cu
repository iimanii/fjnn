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
 * Gaussian Error Linear Unit (GELU)
 * Using the tanh approximation for GELU
 */
extern "C"
__global__ void GeLU(float* input, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < size) {
        float x = input[row];
        // GELU formula: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        output[row] = 0.5f * x * (1.0f + tanh(0.797885f * (x + 0.044715f * x * x * x)));
    }
}

extern "C"
__global__ void GeLUDerivative(float* preActivation, float* postActivation, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    // d/dx[GELU(x)] = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715x³))) + 0.5x * sech²(sqrt(2/π) * (x + 0.044715x³)) * sqrt(2/π) * (1 + 3*0.044715x²)
    if(row < size) {
        float x = preActivation[row];
        float inner = 0.797885f * (x + 0.044715f * x * x * x);
        float tanh_val = tanhf(inner);
        float sech2 = 1.0f - tanh_val * tanh_val;
        
        output[row] = 0.5f * (1.0f + tanh_val) + 
                      0.5f * x * sech2 * 0.797885f * (1.0f + 3.0f * 0.044715f * x * x);
    }
}

extern "C"
__global__ void GeLUGradient(float* preActivation, float* postActivation, float* gradient, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    // d/dx[GELU(x)] = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715x³))) + 0.5x * sech²(sqrt(2/π) * (x + 0.044715x³)) * sqrt(2/π) * (1 + 3*0.044715x²)
    if(row < size) {
        float x = preActivation[row];
        float inner = 0.797885f * (x + 0.044715f * x * x * x);
        float tanh_val = tanhf(inner);
        float sech2 = 1.0f - tanh_val * tanh_val;
        
        float derivative = 0.5f * (1.0f + tanh_val) + 
                           0.5f * x * sech2 * 0.797885f * (1.0f + 3.0f * 0.044715f * x * x);
        
        gradient[row] *= derivative;
    }
}

/**
 * Leaky Rectifier Linear Unit
 */
extern "C"
__global__ void LeakyReLU(float* input, float* output, long size, float alpha) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        output[row] = input[row] < 0.0f ? input[row] * alpha : input[row];
}
extern "C"
__global__ void LeakyReLUDerivative(float* preActivation, float* postActivation, float* output, long size, float alpha) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size) {
        output[row] = preActivation[row] < 0 ? alpha : 1.0f;
    }
}
extern "C"
__global__ void LeakyReLUGradient(float* preActivation, float* postActivation, float* gradient, long size, float alpha) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size) {
        gradient[row] *= preActivation[row] < 0 ? alpha : 1.0f;
    }
}

/**
 * Rectifier Linear Unit
 */
extern "C"
__global__ void ReLU(float* input, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        output[row] = input[row] < 0.0f ? 0.0f : input[row];
}

extern "C"
__global__ void ReLUDerivative(float* preActivation, float* postActivation, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < size) {
        output[row] = preActivation[row] > 0.0f ? 1.0f : 0.0f;
    }
}

extern "C"
__global__ void ReLUGradient(float* preActivation, float* postActivation, float* gradient, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size) {
        if(preActivation[row] <= 0)
            gradient[row] = 0;
        // if preActivation > 0, derivative is 1 so gradient stays the same
    }
}

/**
 * Sigmoid
 */
extern "C"
__global__ void Sigmoid(float* input, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        output[row] = 1.0f / (1.0f + safe_exp(-input[row]));
}

extern "C"
__global__ void SigmoidDerivative(float* preActivation, float* postActivation, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < size) {
        float post = postActivation[row];
        output[row] = post * (1.0f - post);
    }
}

extern "C"
__global__ void SigmoidGradient(float* preActivation, float* postActivation, float* gradient, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size) {
        float post = postActivation[row];
        gradient[row] *= post * (1.0f - post);
    }
}

extern "C"
__global__ void SigmoidBinaryCrossEntropyGradient(float* postActivation, float* truth, float* result, float alpha, float beta, long size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < size) {
        float weight = (truth[i] == 1.0f) ? alpha : beta;
        result[i] = weight * (postActivation[i] - truth[i]);
    }
}

/**
 * Sin
 */
extern "C"
__global__ void Sin(float* input, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        output[row] = sinf(input[row]);
}

extern "C"
__global__ void SinDerivative(float* preActivation, float* postActivation, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size) {
        output[row] = cosf(preActivation[row]);
    }
}

extern "C"
__global__ void SinGradient(float* preActivation, float* postActivation, float* gradient, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size) {
        gradient[row] *= cosf(preActivation[row]);
    }
}


/**
 * SoftMax
 */
template<const int BLOCK_SIZE>
__device__ void SoftMax(float* input, float* output, long size) {
    __shared__ float temp[BLOCK_SIZE];
    
    int blockIndex = blockIdx.x * size;
    
    const int iterations = (size - 1) / BLOCK_SIZE + 1;
    
    float tmax = FLOAT_MIN;

    for(int i=0; i < iterations; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        
        if(index < size) {
            float value = input[blockIndex + index];
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
            float value = safe_exp(input[blockIndex + index] - blockMax);
            sum += value;
            output[blockIndex + index] = value;
        }
    }
    
    /* get sum over block */
    float blockSum = reduceBlockSum<BLOCK_SIZE>(temp, sum);
    
    /* calculate final value and store */
    for(int i=0; i < iterations; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        
        if(index < size) {
            if(blockSum == 0) {
                output[blockIndex + index] = 1.0 / size;
            } else
                output[blockIndex + index] = output[blockIndex + index] / blockSum;
        }
    }
}

template<const int BLOCK_SIZE>
__device__ void SoftMaxGradient(float* preActivation, float* postActivation, float* gradient, long size) {
    __shared__ float temp[BLOCK_SIZE];
    
    int blockIndex = blockIdx.x * size;
    const int iterations = (size - 1) / BLOCK_SIZE + 1;
    
    // Calculate partial dot products
    float dot_sum = 0;
    for(int i = 0; i < iterations; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        if(index < size) {
            dot_sum += postActivation[blockIndex + index] * gradient[blockIndex + index];
        }
    }
    
    // Reduce to get total dot product
    float dot_product = reduceBlockSum<BLOCK_SIZE>(temp, dot_sum);
    
    // Calculate final gradients
    for(int i = 0; i < iterations; i++) {
        int index = threadIdx.x + i * BLOCK_SIZE;
        if(index < size) {
            gradient[blockIndex + index] = postActivation[blockIndex + index] * (gradient[blockIndex + index] - dot_product);
        }
    }
}

/* 
 * create a list of SoftMax_X and SoftMaxGradient_X
 */
#define SOFTMAX_KERNELS(SIZE) \
extern "C" \
__global__ void SoftMax_##SIZE(float* input, float* output, long size) { \
    SoftMax<SIZE>(input, output, size); \
} \
extern "C" \
__global__ void SoftMaxGradient_##SIZE(float* preActivation, float* postActivation, float* gradient, long size) { \
    SoftMaxGradient<SIZE>(preActivation, postActivation, gradient, size); \
}

SOFTMAX_KERNELS(32)
SOFTMAX_KERNELS(64)
SOFTMAX_KERNELS(128)
SOFTMAX_KERNELS(256)
SOFTMAX_KERNELS(512)

/**
 * Fused SoftMax Cross Entropy Gradient
 * For cross entropy loss with softmax activation, gradient simplifies to: y_i - t_i
 */
extern "C"
__global__ void SoftMaxCrossEntropyGradient(float* postActivation, float* truth, float* result, long size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(i < size) {
        result[i] = postActivation[i] - truth[i];
    }
}

/**
 * Step
 */
extern "C"
__global__ void Step(float* input, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        output[row] = input[row] >= 0 ? 1.0f : 0.0f;
}


/**
 * Swish (x * sigmoid)
 */
extern "C"
__global__ void Swish(float* input, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size) {
        float x = input[row];
        output[row] = x / (1.0f + safe_exp(-x));
    }
}

extern "C"
__global__ void SwishDerivative(float* preActivation, float* postActivation, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size) {
        float x = preActivation[row];
        float sigmoid = 1.0f / (1.0f + safe_exp(-x));
        output[row] = postActivation[row] + sigmoid * (1.0f - postActivation[row]);
    }
}

extern "C"
__global__ void SwishGradient(float* preActivation, float* postActivation, float* gradient, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size) {
        float x = preActivation[row];
        float sigmoid = 1.0f / (1.0f + safe_exp(-x));
        float derivative = postActivation[row] + sigmoid * (1.0f - postActivation[row]);
        gradient[row] *= derivative;
    }
}

/**
 * Tanh
 */
extern "C"
__global__ void Tanh(float* input, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size)
        output[row] = tanhf(input[row]);
}

extern "C"
__global__ void TanhDerivative(float* preActivation, float* postActivation, float* output, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size) {
        float tanh_val = postActivation[row];
        output[row] = 1.0f - tanh_val * tanh_val;
    }
}

extern "C"
__global__ void TanhGradient(float* preActivation, float* postActivation, float* gradient, long size) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(row < size) {
        float tanh_val = postActivation[row];
        gradient[row] *= (1.0f - tanh_val * tanh_val);
    }
}

/* legacy stuff */
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
