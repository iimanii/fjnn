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
public class Sigmoid extends Activation {

    @Override
    public float compute(float input) {
        return (float) (1.0 / (1 + SafeExp(-input)));
    }

    @Override
    public void compute(float[] input, int stride, int count) {
        for(int i=0; i < input.length; i++)
            input[i] = (float) (1.0 / (1 + SafeExp(-input[i])));
    }

    @Override
    public float derivative(float input) {
        float sigmoid = (float) (1.0 / (1 + SafeExp(-input)));
        return sigmoid * (1 - sigmoid);
    }
    
    @Override
    public void computeGPU(CUdeviceptr ptr, long stride, long count, CUstream stream) {
        CudaFunctions.Sigmoid(ptr, stride * count, stream);
    }
    
    @Override
    public void derivative(float[] input, int from, int to) {
        /* sigmoid(x) * (1 - sigmoid(x)) */
        for (int i = from; i < to; i++) {
            float sigmoid = (float) (1.0 / (1 + SafeExp(-input[i])));
            input[i] = sigmoid * (1 - sigmoid);
        }
    }

    @Override
    public void compute(FloatBuffer input, int stride, int count) {
        intrinsic.Sigmoid(input, stride * count);
    }

    @Override
    public void derivativeGPU(CUdeviceptr ptr, long stride, long count, CUstream stream) {
        CudaFunctions.SigmoidPrime(ptr, stride * count, stream);
    }
}
