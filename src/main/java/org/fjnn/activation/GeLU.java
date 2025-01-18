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
    private static final float SQRT_2_PI = (float) Math.sqrt(2 / Math.PI);
    private static final float ALPHA = 0.044715f;
    
    @Override
    public float compute(float input) {
        // Using the tanh approximation of GELU
        return 0.5f * input * (1.0f + (float) Math.tanh(SQRT_2_PI * (input + ALPHA * Math.pow(input, 3))));
    }

    @Override
    public void compute(float[] input, int stride, int count) {
        for (int i = 0; i < input.length; i++) {
            input[i] = 0.5f * input[i] * (1.0f + (float) Math.tanh(SQRT_2_PI * (input[i] + ALPHA * Math.pow(input[i], 3))));
        }
    }

    @Override
    public void compute(FloatBuffer input, int stride, int count) {
        throw new UnsupportedOperationException("Not supported yet.");
    }
    
    @Override
    public void computeGPU(CUdeviceptr ptr, long stride, long count, CUstream stream) {
        CudaFunctions.activation.GeLU(ptr, stride * count, stream);
    }


    @Override
    public float derivative(float preActivation, float postActivation) {
        // d/dx[GELU(x)] = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715x³))) + 0.5x * sech²(sqrt(2/π) * (x + 0.044715x³)) * sqrt(2/π) * (1 + 3*0.044715x²)
        float x = preActivation;
        float inner = SQRT_2_PI * (x + ALPHA * x * x * x);
        float tanh = (float)Math.tanh(inner);
        float sech2 = 1.0f - tanh * tanh;

        return 0.5f * (1.0f + tanh) + 0.5f * x * sech2 * SQRT_2_PI * (1.0f + 3 * ALPHA * x * x);
    }

    @Override
    public void derivative(float[] preActivation, float[] postActivation, float[] output, int stride, int count) {
        // d/dx[GELU(x)] = 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715x³))) + 0.5x * sech²(sqrt(2/π) * (x + 0.044715x³)) * sqrt(2/π) * (1 + 3*0.044715x²)
        for (int i = 0; i < stride * count; i++) {
            float x = preActivation[i];
            float inner = SQRT_2_PI * (x + ALPHA * x * x * x);
            float tanh = (float)Math.tanh(inner);
            float sech2 = 1.0f - tanh * tanh;

            output[i] = 0.5f * (1.0f + tanh) + 0.5f * x * sech2 * SQRT_2_PI * (1.0f + 3 * ALPHA * x * x);
        }
    }

    @Override
    public void derivativeGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, long stride, long count, CUstream stream) {
        CudaFunctions.activationDerivative.GeLUDerivative(preActivation, postActivation, output, stride * count, stream);
    }
    
    @Override
    public void gradient(float[] preActivation, float[] postActivation, float[] gradient, int stride, int count) {
        for (int i = 0; i < stride * count; i++) {
            float x = preActivation[i];
            float inner = SQRT_2_PI * (x + ALPHA * x * x * x);
            float tanh = (float)Math.tanh(inner);
            float sech2 = 1.0f - tanh * tanh;

            float derivative = 0.5f * (1.0f + tanh) + 0.5f * x * sech2 * SQRT_2_PI * (1.0f + 3 * ALPHA * x * x);
            gradient[i] *= derivative;
        }
    }

    @Override
    public void gradientGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, long stride, long count, CUstream stream) {
        CudaFunctions.activationGradient.GeLUGradient(preActivation, postActivation, gradient, stride * count, stream);
    }
}

