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
public class Sigmoid extends Activation {

    @Override
    public float compute(float input) {
        return 1.0f / (1.0f + SafeExp(-input));
    }

    @Override
    public void compute(FloatBuffer input, int stride, int count) {
        intrinsic.Sigmoid(input, stride * count);
    }
    
    @Override
    public void compute(float[] input, float[] output, int stride, int count) {
        for(int i=0; i < stride * count; i++)
            output[i] = 1.0f / (1.0f + SafeExp(-input[i]));
    }
    
    @Override
    public void computeGPU(CUdeviceptr input, CUdeviceptr output, int stride, int count, CUstream stream) {
        CudaFunctions.activation.Sigmoid(input, output, stride * count, stream);
    }

    @Override
    public float derivative(float preActivation, float postActivation) {
        /* sigmoid(x) * (1 - sigmoid(x)) */
        return postActivation * (1.0f - postActivation);
    }
    
    @Override
    public void derivative(float[] preActivation, float[] postActivation, float[] output, int stride, int count) {
        /* sigmoid(x) * (1 - sigmoid(x)) */
        for (int i=0; i < stride * count; i++) {
            output[i] = postActivation[i] * (1.0f - postActivation[i]);
        }
    }
    
    @Override
    public void derivativeGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, int stride, int count, CUstream stream) {
        CudaFunctions.activationDerivative.SigmoidDerivative(preActivation, postActivation, output, stride * (long)count, stream);
    }
    
    @Override
    public void gradient(float[] preActivation, float[] postActivation, float[] gradient, int stride, int count) {
        for(int i = 0; i < stride * count; i++) {
            gradient[i] *= postActivation[i] * (1.0f - postActivation[i]);
        }
    }

    @Override
    public void gradientGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, int stride, int count, CUstream stream) {
        CudaFunctions.activationGradient.SigmoidGradient(preActivation, postActivation, gradient, stride * (long)count, stream);
    }

    public void gradientCrossEntropy(float[] postActivation, float[] truth, float[] result, int outputSize, int batchSize) {
        int size = outputSize * batchSize;
        for (int i = 0; i < size; i++) {
            result[i] = postActivation[i] - truth[i];
        }
    }
    
    public void gradientGPUCrossEntropy(CUdeviceptr postActivation, CUdeviceptr truth, CUdeviceptr result, int outputSize, int batchSize, CUstream stream) {
        CudaFunctions.activationGradient.SigmoidCrossEntropyGradient(postActivation, truth, result, outputSize * batchSize, stream);
    }

}
