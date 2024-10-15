/*
 * The MIT License
 *
 * Copyright 2024 ahmed.
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
package org.fjnn.loss;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class MeanSquareError extends Loss {
    
    // Compute the Mean Squared Error
    @Override
    public float compute(float[] output, float[] expected) {
        float sum = 0;
        for (int i = 0; i < output.length; i++) {
            float diff = output[i] - expected[i];
            sum += diff * diff;
        }
        return sum / (2 * output.length);
    }

    // Compute the derivative of MSE with respect to the predicted values
    @Override
    public float[] derivative(float[] output, float[] expected) {
        float[] derivatives = new float[output.length];
        for (int i = 0; i < output.length; i++) {
            derivatives[i] = output[i] - expected[i];
        }
        return derivatives;
    }
    
    @Override
    public void derivativeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
        CudaFunctions.MeanSquareErrorPrime(output, expected, result, size, CudaUtil.PREFERRED_BLOCK_SIZE, stream);
    }
}

