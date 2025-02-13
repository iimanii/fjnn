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
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaModule;
import org.fjnn.cuda.CUdeviceptr2D;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.util.intrinsic;

/**
 *
 * @author ahmed
 */
public class ReLU extends Activation {

    @Override
    public float compute(float input) {
        return Math.max(0, input);
    }
    
    @Override
    public void compute(float[] input, float[] output, int stride, int count) {
        for(int i=0; i < stride * count; i++) {
            output[i] = input[i] < 0.0f ? 0.0f : input[i];

        }
    }

    @Override
    public void compute(FloatBuffer input, int stride, int count) {
        intrinsic.ReLU(input, stride * count);
    }
    
    @Override
    public void computeGPU(CUdeviceptr input, CUdeviceptr output, int stride, int count, CUstream stream) {
        CudaFunctions.activation.ReLU(input, output, stride * (long)count, stream);
    }

    @Override
    public float derivative(float preActivation, float postActivation) {
        return preActivation > 0 ? 1 : 0;
    }
    

    @Override
    public void derivative(float[] preActivation, float[] postActivation, float[] output, int stride, int count) {
        for(int i = 0; i < stride * count; i++) {
            output[i] = preActivation[i] > 0 ? 1.0f : 0.0f;
        }
    }

    @Override
    public void derivativeGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, int stride, int count, CUstream stream) {
        CudaFunctions.activationDerivative.ReLUDerivative(preActivation, postActivation, output, stride * (long)count, stream);
    }
    
    @Override
    public void gradient(float[] preActivation, float[] postActivation, float[] gradient, int stride, int count) {
        for(int i = 0; i < stride * count; i++) {
            if(preActivation[i] <= 0) {
                gradient[i] = 0;
            }
        }
    }

    @Override
    public void gradientGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, int stride, int count, CUstream stream) {
        CudaFunctions.activationGradient.ReLUGradient(preActivation, postActivation, gradient, stride * (long)count, stream);
    }
}
