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

import java.util.Map;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.trainer.backpropagate.Dataset;

/**
 *
 * @author ahmed
 */
public class WeightedMeanSquareError extends Loss {
    float[] weights;
    CUdeviceptr weightsGPU;
    final WeightingStrategy strategy;
    
    Dataset set;
    
    public enum WeightingStrategy {
        INVERSE_FREQUENCY,    // 1 / frequency
        INVERSE_SQRT,         // 1 / sqrt(frequency)  
        BOUNDARY_FOCUS,       // Custom for Mandelbrot boundary
        CUSTOM                // User-provided weights
    }

    public WeightedMeanSquareError(float[] weights, CUdeviceptr weightsGPU) {
        this.weights = weights;
        this.weightsGPU = weightsGPU;
        this.strategy = WeightingStrategy.CUSTOM;
    }
    
    // Compute the Mean Squared Error
    @Override
    public float compute(float[] output, float[] expected) {
        if(output.length != expected.length || output.length != weights.length)
            throw new RuntimeException();
        
        float sum = 0;
        for (int i = 0; i < output.length; i++) {
            float diff = output[i] - expected[i];
            sum += weights[i] * diff * diff;
        }
        return sum / output.length;
    }

    @Override
    public void computeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
        // Create temporary buffer for per-element results
        CUdeviceptr tempBuffer = CudaUtil.createFloatAsync(size, stream);
        
        // Compute per-element weighted squared differences
        CudaFunctions.loss.WeightedMeanSquareError(output, expected, weightsGPU, tempBuffer, size, stream);
        
        // Reduce sum and average
        CudaFunctions.vector.reduceSum(result, tempBuffer, 1, (int)size, stream);
        CudaFunctions.vector.scale(result, 1.0f / size, 1, stream);
        
        // Cleanup
        CudaUtil.freeAsync(tempBuffer, stream);
    }
    
    // Compute the derivative of MSE with respect to the predicted values
    @Override
    public float[] derivative(float[] output, float[] expected) {
        if(output.length != expected.length || output.length != weights.length)
            throw new RuntimeException();
        
        float[] derivatives = new float[output.length];        
        float multiplier = 2.0f / output.length;

        for (int i = 0; i < output.length; i++) {
            derivatives[i] = multiplier * weights[i] * (output[i] - expected[i]);
        }
        return derivatives;
    }
    
    @Override
    public void derivativeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
        CudaFunctions.loss.WeightedMeanSquareErrorDerivative(output, expected, weightsGPU, result, size, stream);
    }
    
    @Override
    public Map serialize() {
        Map result = super.serialize();
        result.put("weights", weights);  // Store the weights array
        return result;
    }
    
    @Override
    public String name() {
        return "WMSE";
    }
}

