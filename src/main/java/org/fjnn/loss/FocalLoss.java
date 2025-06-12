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

import java.util.Map;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 * 
 * Incomplete class .. needs testing
 */
public class FocalLoss extends Loss {
    public final float gamma;           // focusing parameter
    final static float eps = 1e-7f;
    
    public FocalLoss(float gamma) {
        this.gamma = gamma;
    }
    
    @Override
    public float compute(float[] output, float[] expected) {
        if (output.length != expected.length)
            throw new RuntimeException();
        
        float loss = 0;
        for (int i = 0; i < output.length; i++) {
            float clipped = Math.max(eps, Math.min(1-eps, output[i]));
            
            float pt = expected[i] == 1 ? clipped : (1 - clipped);
            float modulating_factor = (float)Math.pow(1 - pt, gamma);
            loss += -modulating_factor * (expected[i] * Math.log(clipped) + (1 - expected[i]) * Math.log(1 - clipped));
        }
        
        return loss / output.length;
    }

    @Override
    public float[] derivative(float[] output, float[] expected) {
        float[] derivatives = new float[output.length];
        
        for (int i = 0; i < output.length; i++) {
            float clipped = Math.max(eps, Math.min(1-eps, output[i]));
            float pt = expected[i] == 1 ? clipped : (1 - clipped);
            float modulating_factor = (float)Math.pow(1 - pt, gamma);
            
            // Combining BCE derivative with focal loss terms
            float bce_derivative = (clipped - expected[i]) / (clipped * (1 - clipped));
            float focal_term = modulating_factor * (gamma * (float)Math.log(pt) * (expected[i] == 1 ? -1 : 1) + 1);
            
            derivatives[i] = bce_derivative * focal_term;
        }
        return derivatives;
    }
    
    @Override
    public void computeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
        // Create temporary buffer for per-element results
        CUdeviceptr tempBuffer = CudaUtil.createFloatAsync(size, stream);

        // Compute per-element focal loss
        CudaFunctions.loss.FocalLoss(output, expected, tempBuffer, gamma, size, stream);

        // Reduce sum and average
        CudaFunctions.vector.reduceSum(result, tempBuffer, 1, (int)size, stream);
        CudaFunctions.vector.scale(result, 1.0f / size, 1, stream);

        // Cleanup
        CudaUtil.freeAsync(tempBuffer, stream);
    }
    
    @Override
    public void derivativeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
        CudaFunctions.loss.FocalLossDerivative(output, expected, result, gamma, size, stream);
    }
    
    @Override
    public Map serialize() {
        Map result = super.serialize();
        result.put("gamma", gamma);
        return result;
    }
    
    public static FocalLoss deserialize(Map serialized) {
        float gamma = (Float)serialized.get("gamma");
        return new FocalLoss(gamma);
    }

    @Override
    public String name() {
        return "Focal";
    }
}
