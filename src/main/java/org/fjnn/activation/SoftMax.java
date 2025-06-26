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
package org.fjnn.activation;

import java.nio.FloatBuffer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.util.intrinsic;

/**
 *
 * @author ahmed
 */
public class SoftMax extends Activation {
    
    @Override
    public float compute(float input) {
        throw new UnsupportedOperationException("SoftMax cannot be computed on a single value"); 
    }
    
    @Override
    public void compute(float[] input, float[] output, int inputDim, int batchSize) {
        for(int c=0; c < batchSize; c++) {
            int from = c * inputDim;
            int to = from + inputDim;
            
            float max = Float.NEGATIVE_INFINITY;
            for(int i=from; i < to; i++) {
                max = Math.max(input[i], max);
            }

            double sum = 0;
            for(int i=from; i < to; i++) {
                output[i] = SafeExp(input[i] - max);
                sum += output[i];
            }
            
            if(sum == 0)
                for(int i=from; i < to; i++)
                    output[i] = 1.0f / input.length;
            else
                for(int i=from; i < to; i++)
                    output[i] /= sum;
        }
    }

    @Override
    public float derivative(float preActivation, float postActivation) {
        throw new UnsupportedOperationException("SoftMax derivative cannot be computed on single values"); 
    }
    
    /*
     * https://mmuratarat.github.io/2019-01-27/derivation-of-softmax-function
     * https://www.youtube.com/watch?v=09c7bkxpv9I
     * Softmax derivative is: Si * (δij - Sj) where S is softmax output
     * δij = 1 if  i ==j otherwise 0
     */
    @Override
    public void derivative(float[] preActivation, float[] postActivation, float[] output, int inputDim, int batchSize) {
        throw new UnsupportedOperationException("SoftMax derivative requires full Jacobian matrix, not element-wise. Use gradient() method for backpropagation");
    }

    /*
     * gradient[i] = Si(gi - dot_product[Si.gi])
     */
    @Override
    public void gradient(float[] preActivation, float[] postActivation, float[] gradient, int inputDim, int batchSize) {
        for(int c = 0; c < batchSize; c++) {
            int from = c * inputDim;
            int to = from + inputDim;

            // Calculate dot product between output and gradient
            float dot_product = 0;
            for(int j = from; j < to; j++) {
                dot_product += postActivation[j] * gradient[j];
            }

            // Calculate final gradient: Si(gi - dot_product)
            for(int i = from; i < to; i++) {
                gradient[i] = postActivation[i] * (gradient[i] - dot_product);
            }
        }
    }
    
    
    @Override
    public void computeGPU(CUdeviceptr input, CUdeviceptr output, int inputDim, int batchSize, CUstream stream) {
        if(batchSize > Integer.MAX_VALUE)
            throw new RuntimeException("Batch size exceeds maximum for softmax compute: " + batchSize);
        
        CudaFunctions.activation.SoftMax(input, output, inputDim, (int)batchSize, stream);
    }
    
    @Override
    public void derivativeGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, int inputDim, int batchSize, CUstream stream) {
        throw new UnsupportedOperationException("SoftMax derivative requires full Jacobian matrix, not element-wise. Use gradient() method for backpropagation");
    }
    
    @Override
    public void gradientGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, int inputDim, int batchSize, CUstream stream) {
        CudaFunctions.activationGradient.SoftMaxGradient(preActivation, postActivation, gradient, inputDim, batchSize, stream);
    }
    
        /**
     * Compute gradient for fused softmax-cross-entropy
     * For cross entropy loss with softmax activation, the gradient simplifies to: y_i - t_i
     * where y_i is the softmax output and t_i is the target (one-hot encoded)
     */
    public void gradientCrossEntropy(float[] postActivation, float[] truth, float[] result, int outputDim, int batchSize) {
        int size = outputDim * batchSize;
        for (int i = 0; i < size; i++) {
            result[i] = postActivation[i] - truth[i];
        }
    }
    
    /**
     * GPU version of fused softmax-cross-entropy gradient
     */
    public void gradientCrossEntropyGPU(CUdeviceptr postActivation, CUdeviceptr truth, CUdeviceptr result, int outputDim, int batchSize, CUstream stream) {
        CudaFunctions.activationGradient.SoftMaxCrossEntropyGradient(postActivation, truth, result, outputDim * batchSize, stream);
    }
    
    @Override
    public void compute(FloatBuffer input, FloatBuffer output, int inputDim, int batchSize) {
        intrinsic.SoftMax(input, output, inputDim, batchSize);
    }
}
