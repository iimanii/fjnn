/*
 * The MIT License
 *
 * Copyright 2025 ahmed.
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

import java.util.HashMap;
import java.util.Map;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.activation.Activation;
import org.fjnn.activation.SoftMax;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;

/**
 * Cross Entropy Loss for multi-class classification
 * L = -sum(t_i * log(y_i)) where t is one-hot encoded target and y is softmax output
 * 
 * For fused softmax-cross-entropy:
 * dL/dz_i = y_i - t_i (where z is pre-softmax activation)
 * 
 * @author ahmed
 */
public class CrossEntropy extends Loss {
    final static float eps = 1e-7f;

    @Override
    public float compute(float[] output, float[] expected) {
        if (output.length != expected.length)
            throw new RuntimeException("Output and expected arrays must have the same length");
        
        float loss = 0;
        for (int i = 0; i < output.length; i++) {
            float clipped = Math.max(eps, Math.min(1-eps, output[i])); 
            if (expected[i] > 0) {  // Only compute for non-zero targets
                loss -= expected[i] * Math.log(clipped);
            }
        }
        
        return loss / output.length;
    }

    @Override
    public void computeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
        // Create temporary buffer for per-element results
        CUdeviceptr tempBuffer = CudaUtil.createFloatAsync(size, stream);
        
        // Compute per-element cross entropy loss
        CudaFunctions.loss.CrossEntropy(output, expected, tempBuffer, size, stream);
        
        // Reduce sum and average
        CudaFunctions.vector.reduceSum(result, tempBuffer, 1, (int)size, stream);
        CudaFunctions.vector.scale(result, 1.0f / size, 1, stream);
        
        // Cleanup
        CudaUtil.freeAsync(tempBuffer, stream);
    }
    
    @Override
    public float[] derivative(float[] output, float[] expected) {
        float[] derivatives = new float[output.length];
        
        for (int i = 0; i < output.length; i++) {
            float clipped = Math.max(eps, Math.min(1-eps, output[i]));
            // Derivative of cross entropy w.r.t output: -t_i / y_i
            derivatives[i] = -expected[i] / clipped;
        }
        return derivatives;
    }

    @Override
    public void derivativeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
        CudaFunctions.loss.CrossEntropyDerivative(output, expected, result, size, stream);
    }
    
    @Override
    public String name() {
        return "CrossEntropy";
    }
    
    @Override
    public boolean canFuseWith(Activation activation) {
        return activation instanceof SoftMax;
    }
    
    @Override
    public void fusedGradient(float[] postActivation, float[] expected, float[] result, Activation activation, int outputDim, int batchSize) {
        if (activation instanceof SoftMax) {
            SoftMax softmax = (SoftMax) activation;
            softmax.gradientCrossEntropy(postActivation, expected, result, outputDim, batchSize);
        } else {
            throw new IllegalArgumentException("Can only fuse with SoftMax");
        }
    }
    
    @Override
    public void fusedGradientGPU(CUdeviceptr postActivation, CUdeviceptr expected, CUdeviceptr result, Activation activation, int outputDim, int batchSize, CUstream stream) {
        if (activation instanceof SoftMax) {
            SoftMax softmax = (SoftMax) activation;
            softmax.gradientCrossEntropyGPU(postActivation, expected, result, outputDim, batchSize, stream);
        } else {
            throw new IllegalArgumentException("Can only fuse with SoftMax");
        }
    }
}