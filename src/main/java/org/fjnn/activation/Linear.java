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

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.cuda.CudaUtil;

/**
 * Linear activation function: f(x) = x
 * The identity function that outputs the input unchanged.
 * 
 * @author ahmed
 */
public class Linear extends Activation {

    @Override
    public float compute(float input) {
        return input;
    }
    
    @Override
    public void compute(float[] input, float[] output, int inputDim, int batchSize) {
        /* Only copy if different arrays */
        if (input != output)
            System.arraycopy(input, 0, output, 0, inputDim * batchSize);
    }

    @Override
    public float derivative(float preActivation, float postActivation) {
        return 1.0f;
    }

    @Override
    public void derivative(float[] preActivation, float[] postActivation, float[] output, int inputDim, int batchSize) {
        for(int i = 0; i < inputDim * batchSize; i++) {
            output[i] = 1.0f;  // fill array with 1.0f as derivative is constant
        }
    }
    
    @Override
    public void gradient(float[] preActivation, float[] postActivation, float[] gradient, int inputDim, int batchSize) {
        // derivative is 1, so gradient remains unchanged
    }


    @Override
    public void computeGPU(CUdeviceptr input, CUdeviceptr output, int inputDim, int batchSize, CUstream stream) {
        /* Only copy if different pointers */
        if (!input.equals(output))
            JCudaDriver.cuMemcpyAsync(output, input, inputDim * batchSize * CudaUtil.FLOAT_SIZE, stream);
    }
    
    @Override
    public void derivativeGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, int inputDim, int batchSize, CUstream stream) {
        // fill array with 1.0f as derivative is constant
        JCudaDriver.cuMemsetD32Async(output, Float.floatToRawIntBits(1.0f), inputDim * batchSize, stream);
    }
    
    @Override
    public void gradientGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, int inputDim, int batchSize, CUstream stream) {
        // derivative is 1, so gradient remains unchanged
    }
}
