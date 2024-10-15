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
package org.fjnn.activation;

import java.nio.FloatBuffer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class GeLU extends Activation {

    @Override
    public float compute(float input) {
        // Using the tanh approximation of GELU
        return 0.5f * input * (1.0f + (float) Math.tanh(Math.sqrt(2 / Math.PI) * (input + 0.044715f * Math.pow(input, 3))));
    }

    @Override
    public void compute(float[] input, int stride, int count) {
        for (int i = 0; i < input.length; i++) {
            input[i] = 0.5f * input[i] * (1.0f + (float) Math.tanh(Math.sqrt(2 / Math.PI) * (input[i] + 0.044715f * Math.pow(input[i], 3))));
        }
    }

    @Override
    public void computeGPU(CUdeviceptr ptr, long stride, long count, CUstream stream) {
        CudaFunctions.GeLU(ptr, stride * count, stream);
    }

    @Override
    public void derivative(float[] input, int from, int to) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public void compute(FloatBuffer input, int stride, int count) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public float derivative(float input) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void derivativeGPU(CUdeviceptr ptr, long stride, long size, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}

