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
package org.fjnn.convolution;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.jcublas.cublasHandle;
import org.fjnn.convolution.output.ConvolutionUnitForwardOutput;
import org.fjnn.convolution.output.ConvolutionUnitForwardOutputGPU;

/**
 * Common interface for convolution layers
 * @author ahmed
 */

public interface ConvolutionUnit {
    /**
     * CPU forward pass
     */
    ConvolutionUnitForwardOutput feedForward(float[] input, int batchSize);
    
    /**
     * GPU forward pass  
     */
    ConvolutionUnitForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchSize, CUstream stream, cublasHandle handle);
    
    /**
     * Compute output size for given input
     */
    int computeOutputSize(int inputSize);
    
    /**
     * Prepare GPU resources
     */
    void prepareGPU(CUstream stream);
    
    /**
     * Free GPU resources
     */
    void freeGPU(CUstream stream);
    
    /**
     * Check if GPU ready
     */
    boolean gpuReady();
}
