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
import org.fjnn.convolution.output.unit.ConvolutionUnitForwardOutputGPU;

/**
 *
 * @author ahmed
 */
public class ConvolutionLayerForwardOutputGPU {
    public final ConvolutionUnitForwardOutputGPU[] unitOutputs;
    public final CUdeviceptr input;
    public final int batchSize;

    public ConvolutionLayerForwardOutputGPU(ConvolutionUnitForwardOutputGPU[] unitOutputs, CUdeviceptr input, int batchSize) {
        this.unitOutputs = unitOutputs;
        this.input = input;
        this.batchSize = batchSize;
    }
    
    public void free() {
        if (unitOutputs != null) {
            for (ConvolutionUnitForwardOutputGPU unitOutput : unitOutputs) {
                if (unitOutput != null) {
                    unitOutput.free();
                }
            }
        }
        // Note: input CUdeviceptr is not freed as it's typically owned by the caller
    }

    public void freeAsync(CUstream stream) {
        if (unitOutputs != null) {
            for (ConvolutionUnitForwardOutputGPU unitOutput : unitOutputs) {
                if (unitOutput != null) {
                    unitOutput.freeAsync(stream);
                }
            }
        }
        // Note: input CUdeviceptr is not freed as it's typically owned by the caller
    }    
}
