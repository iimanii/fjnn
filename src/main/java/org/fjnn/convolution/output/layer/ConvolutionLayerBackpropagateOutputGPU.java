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
package org.fjnn.convolution.output.layer;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.convolution.output.unit.ConvolutionUnitBackpropagateOutputGPU;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class ConvolutionLayerBackpropagateOutputGPU {
    public final CUdeviceptr inputGradients;                            // Aggregated input gradients on GPU  
    public final ConvolutionUnitBackpropagateOutputGPU[] unitBackprops; // Individual unit outputs
    public final int inputSize;
    public final int batchSize;
    
    public ConvolutionLayerBackpropagateOutputGPU(CUdeviceptr inputGradients, 
                                                  ConvolutionUnitBackpropagateOutputGPU[] unitBackprops, 
                                                  int inputSize, 
                                                  int batchSize) {
        this.inputGradients = inputGradients;
        this.unitBackprops = unitBackprops;
        this.inputSize = inputSize;
        this.batchSize = batchSize;
    }
    
    public void free() {
        if (inputGradients != null)
            CudaUtil.free(inputGradients);
        
        if (unitBackprops != null) {
            for (ConvolutionUnitBackpropagateOutputGPU unitBackprop : unitBackprops) {
                if (unitBackprop != null)
                    unitBackprop.free();
            }
        }
    }
    
    public void freeAsync(CUstream stream) {
        if (inputGradients != null)
            CudaUtil.freeAsync(inputGradients, stream);
        
        if (unitBackprops != null) {
            for (ConvolutionUnitBackpropagateOutputGPU unitBackprop : unitBackprops) {
                if (unitBackprop != null)
                    unitBackprop.freeAsync(stream);
            }
        }
    }
}
