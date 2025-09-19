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

import java.util.HashMap;
import java.util.Map;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaModule;
import org.fjnn.cuda.CudaUtil;

/**
 * Bird Loss Function: L(x) = α * ln(βx² + 1)
 * 
 * This loss function maintains relatively high values even when predictions are
 * close to targets, unlike MSE which drops quickly. For example, an error of 0.01
 * might yield MSE ≈ 0.0001, but Bird Loss ≈ 0.05 (with typical parameters).
 * This forces the network to continue learning as the loss considers "close" 
 * predictions still inadequate until very near the target.
 * 
 * @author ahmed
 */
public class BirdLoss extends Loss {
    
    public float alpha;  // Scaling factor (typically < 1)
    public float beta;   // Error sensitivity factor (typically large, e.g., 100-10000)
    
    /**
     * Create a falcon loss with default parameters
     */
    public BirdLoss() {
        this(0.25f, 1000.0f);
    }
    
    /**
     * Create a falcon loss with specified parameters
     * @param alpha Scaling factor (typically < 1)
     * @param beta Controls loss magnitude for small errors (typically 100-1000)
     */
    public BirdLoss(float alpha, float beta) {
        if (alpha <= 0)
            throw new IllegalArgumentException("Parameter alpha must be positive");

        if (beta <= 0)
            throw new IllegalArgumentException("Parameter beta must be positive");

        this.alpha = alpha;
        this.beta = beta;
    }
    
    @Override
    public float compute(float[] output, float[] expected) {
        if (output.length != expected.length)
            throw new IllegalArgumentException("Output and expected arrays must have same length");
        
        float totalLoss = 0.0f;
        for (int i = 0; i < output.length; i++) {
            float diff = output[i] - expected[i];
            totalLoss += alpha * (float)Math.log(beta * diff * diff + 1);
        }
        
        return totalLoss / output.length;
    }
    
    @Override
    public void computeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
                // Create temporary buffer for per-element results
        CUdeviceptr tempBuffer = CudaUtil.createFloatAsync(size, stream);
        
        // Compute per-element cross entropy loss
        CudaFunctions.loss.BirdLoss(output, expected, tempBuffer, alpha, beta, size, stream);
        
        // Reduce sum and average
        CudaFunctions.vector.reduceSum(result, tempBuffer, 1, (int)size, stream);
        CudaFunctions.vector.scale(result, 1.0f / size, 1, stream);
        
        // Cleanup
        CudaUtil.freeAsync(tempBuffer, stream);
    }
    
    @Override
    public float[] derivative(float[] output, float[] expected) {
        if (output.length != expected.length) {
            throw new IllegalArgumentException("Output and expected arrays must have same length");
        }
        
        float[] gradient = new float[output.length];
        float scale = 1.0f / output.length;  // Include averaging in gradient
        
        for (int i = 0; i < output.length; i++) {
            float diff = output[i] - expected[i];
            // Derivative: d/dx[α*ln(βx² + 1)] = 2αβx / (βx² + 1)
            gradient[i] = scale * 2 * alpha * beta * diff / (beta * diff * diff + 1);
        }
        
        return gradient;
    }
    
    @Override
    public void derivativeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
        CudaFunctions.loss.BirdLossDerivative(output, expected, result, alpha, beta, size, stream);
    }
    
    @Override
    public String name() {
        return String.format("BirdLoss_%.2f_%.0f", alpha, beta);
    }
    
    @Override
    public Map serialize() {
        HashMap result = new HashMap();
        result.put("type", getClass().getSimpleName());
        result.put("alpha", alpha);
        result.put("beta", beta);
        return result;
    }
    
    public static BirdLoss deserialize(Map serialized) {
        float alpha = ((Number) serialized.get("alpha")).floatValue();
        float beta = ((Number) serialized.get("beta")).floatValue();
        return new BirdLoss(alpha, beta);
    }
}