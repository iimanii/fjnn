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
import org.fjnn.activation.Sigmoid;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 * 
 * L = -[t * log(y) + (1-t) * log(1-y)]
 * 
 * dL/dy = -t/y + (1-t)/(1-y)
 *       = (-t+y)/(y(1-y))
 */
public class BinaryCrossEntropy extends Loss {
    final static float eps = 1e-7f;

    public final float alpha;
    public final float beta;

    public BinaryCrossEntropy() {
        this(1.0f, 1.0f);
    }
    
    public BinaryCrossEntropy(float alpha, float beta) {
        this.alpha = alpha;
        this.beta = beta;
    }
    
    
    @Override
    public float compute(float[] output, float[] expected) {
        if (output.length != expected.length)
            throw new RuntimeException();
        
        float loss = 0;
        for (int i = 0; i < output.length; i++) {
            float clipped = Math.max(eps, Math.min(1-eps, output[i])); 
            float weight = (expected[i] == 1.0f) ? alpha : beta;

            loss += weight * (-expected[i] * Math.log(clipped) - (1 - expected[i]) * Math.log(1 - clipped));
        }
        
        return loss / output.length;
    }

    @Override
    public void computeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
        // Create temporary buffer for per-element results
        CUdeviceptr tempBuffer = CudaUtil.createFloatAsync(size, stream);
        
        // Compute per-element cross entropy loss
        CudaFunctions.loss.BinaryCrossEntropy(output, expected, tempBuffer, alpha, beta, size, stream);
        
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
            
            float weight = (expected[i] == 1.0f) ? alpha : beta;
            derivatives[i] = weight * (clipped - expected[i]) / (clipped * (1 - clipped));
        }
        return derivatives;
    }

    @Override
    public void derivativeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
        CudaFunctions.loss.BinaryCrossEntropyDerivative(output, expected, result, alpha, beta, size, stream);
    }
    
    @Override
    public Map serialize() {
        Map result = new HashMap();
        result.put("type", "BinaryCrossEntropy");
        result.put("alpha", alpha);
        result.put("beta", beta);
        return result;
    }
    
    public static BinaryCrossEntropy deserialize(Map serialized) {
        float alpha = (Float)serialized.get("alpha");
        float beta = (Float)serialized.get("beta");
        return new BinaryCrossEntropy(alpha, beta);
    }

    @Override
    public String name() {
        return String.format("BCE_%.1f_%.1f", alpha, beta);
    }
    
    @Override
    public boolean canFuseWith(Activation activation) {
        return activation instanceof Sigmoid;
    }
    
    @Override
    public void fusedGradient(float[] postActivation, float[] expected, float[] result, Activation activation, int outputDim, int batchSize) {
        if (activation instanceof Sigmoid) {
            Sigmoid sigmoid = (Sigmoid) activation;
            sigmoid.gradientCrossEntropy(postActivation, expected, result, alpha, beta, outputDim, batchSize);
        } else
            throw new IllegalArgumentException("Can only fuse with Sigmoid");

    }
    
    @Override
    public void fusedGradientGPU(CUdeviceptr postActivation, CUdeviceptr expected, CUdeviceptr result, Activation activation, int outputDim, int batchSize, CUstream stream) {
        if (activation instanceof Sigmoid) {
            Sigmoid sigmoid = (Sigmoid) activation;
            sigmoid.gradientGPUCrossEntropy(postActivation, expected, result, alpha, beta, outputDim, batchSize, stream);
        } else {
            throw new IllegalArgumentException("Can only fuse with Sigmoid");
        }
    }
}
