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
import java.util.Map;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.cuda.CudaFunctions;

/**
 *
 * @author ahmed
 */
public class LeakyReLU extends Activation {
    public static final float DEFAULT_ALPHA = 0.3f;
    
    final float alpha;
    
    public LeakyReLU() {
        this.alpha = DEFAULT_ALPHA;
    }
    
    public LeakyReLU(float alpha) {
        this.alpha = alpha;
    }
    
    @Override
    public float compute(float input) {
        return input < 0 ? input * alpha : input;
    }
    
    @Override
    public void compute(float[] input, float[] output, int inputDim, int batchSize) {
        for(int i=0; i < inputDim * batchSize; i++)
            output[i] = input[i] < 0 ? input[i] * alpha : input[i];
    }

    @Override
    public float derivative(float preActivation, float postActivation) {
        return preActivation < 0 ? alpha : 1.0f;
    }

    @Override
    public void derivative(float[] preActivation, float[] postActivation, float[] output, int inputDim, int batchSize) {
        for(int i = 0; i < inputDim * batchSize; i++) {
            output[i] = preActivation[i] < 0 ? alpha : 1.0f;
        }
    }
    
    @Override
    public void gradient(float[] preActivation, float[] postActivation, float[] gradient, int inputDim, int batchSize) {
        for(int i = 0; i < inputDim * batchSize; i++) {
            gradient[i] *= preActivation[i] < 0 ? alpha : 1.0f;
        }
    }

    
    @Override
    public void computeGPU(CUdeviceptr input, CUdeviceptr output, int inputDim, int batchSize, CUstream stream) {
        CudaFunctions.activation.LeakyReLU(input, output, inputDim * batchSize, alpha, stream);
    }

    @Override
    public void derivativeGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, int inputDim, int batchSize, CUstream stream) {
        CudaFunctions.activationDerivative.LeakyReLUDerivative(preActivation, postActivation, output, inputDim * batchSize, alpha, stream);
    }
    
    @Override
    public void gradientGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, int inputDim, int batchSize, CUstream stream) {
        CudaFunctions.activationGradient.LeakyReLUGradient(preActivation, postActivation, gradient, inputDim * batchSize, alpha, stream);
    }
    
    
    @Override
    public Map serialize() {
        Map obj = super.serialize();
        obj.put("alpha", alpha);
        
        return obj;
    }
    
    public static LeakyReLU deserialize(Map serialized) {
        float alpha = ((Number)serialized.get("alpha")).floatValue();
        
        return new LeakyReLU(alpha);
    }
}
