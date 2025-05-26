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
package org.fjnn.convolution.output;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.cuda.CudaUtil;

/**
 * GPU version of convolution unit backpropagation output
 * @author ahmed
 */
public class ConvolutionUnitBackpropagateOutputGPU {
    public final CUdeviceptr inputGradients;
    public final int inputSize;
    public final int batchSize;
    
    // Weight/bias gradients
    public final CUdeviceptr[] weightGradients;  // [kernel_index]
    public final CUdeviceptr[] biasGradients;    // [kernel_index]
    
    // Single kernel constructor
    public ConvolutionUnitBackpropagateOutputGPU(CUdeviceptr inputGradients, int inputSize, int batchSize,
                                                 CUdeviceptr weightGradients, CUdeviceptr biasGradient) {
        this.inputGradients = inputGradients;
        this.inputSize = inputSize;
        this.batchSize = batchSize;
        this.weightGradients = new CUdeviceptr[]{weightGradients};
        this.biasGradients = new CUdeviceptr[]{biasGradient};
    }
    
    // Multi-kernel constructor  
    public ConvolutionUnitBackpropagateOutputGPU(CUdeviceptr inputGradients, int inputSize, int batchSize,
                                                 CUdeviceptr[] weightGradients, CUdeviceptr[] biasGradients) {
        this.inputGradients = inputGradients;
        this.inputSize = inputSize;
        this.batchSize = batchSize;
        this.weightGradients = weightGradients;
        this.biasGradients = biasGradients;
    }
    
    public void free() {
        if (inputGradients != null)
            CudaUtil.free(inputGradients);
        
        if (weightGradients != null)
            for (CUdeviceptr ptr : weightGradients) {
                if (ptr != null) CudaUtil.free(ptr);
            }
        
        if (biasGradients != null)
            for (CUdeviceptr ptr : biasGradients) {
                if (ptr != null) CudaUtil.free(ptr);
            }
    }
    
    public void freeAsync(CUstream stream) {
        if (inputGradients != null)
            CudaUtil.freeAsync(inputGradients, stream);
        
        if (weightGradients != null)
            for (CUdeviceptr ptr : weightGradients) {
                if (ptr != null) CudaUtil.freeAsync(ptr, stream);
            }
        
        if (biasGradients != null)
            for (CUdeviceptr ptr : biasGradients) {
                if (ptr != null) CudaUtil.freeAsync(ptr, stream);
            }
    }
}