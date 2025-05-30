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
package org.fjnn.convolution.output.unit;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class ConvolutionUnitForwardOutputGPU {
    public final CUdeviceptr output;    
    public final int unitCount;
    public final int outputOrigin;      /* the input position that maps to this unit's output[0] */
    
    public final int outputSize;
    public final int batchSize;
    
    // For backpropagation
    public final CUdeviceptr input;
    public final CUdeviceptr[] channelInputs;      // [kernel_index][channel_input]
    public final CUdeviceptr[] im2colMatrix;       // im2col transformation result
    public final CUdeviceptr[] kernelOutputs;      // [kernel_index][raw_output]
    public final CUdeviceptr[] sigmoidOutputs;     // [kernel_index][sigmoid_output] or null for single kernel
    
    // Single kernel constructor (for Kernel class)
    public ConvolutionUnitForwardOutputGPU(CUdeviceptr output, int outputSize, int batchSize, 
                                           CUdeviceptr forwardPassInputs, CUdeviceptr im2colMatrix, int unitCount) {
        this.unitCount = unitCount;
        this.outputOrigin = (unitCount - 1) / 2;
        
        this.output = output;
        this.outputSize = outputSize;
        this.batchSize = batchSize;
        
        this.input = forwardPassInputs;
        this.channelInputs = null;                 // no channels for single kernel
        this.im2colMatrix = new CUdeviceptr[]{im2colMatrix};
        this.kernelOutputs = null;
        this.sigmoidOutputs = null;
    }
    
    // Multi-kernel constructor (for KernelGroup class)
    public ConvolutionUnitForwardOutputGPU(CUdeviceptr output, int outputSize, int batchSize,
                                           CUdeviceptr forwardPassInputs, int unitCount,
                                           CUdeviceptr[] kernelOutputs, CUdeviceptr[] sigmoidOutputs,
                                           CUdeviceptr[] channelInputs, CUdeviceptr[] im2colMatrix) {
        this.unitCount = unitCount;
        this.outputOrigin = (unitCount - 1) / 2;
        
        this.output = output;
        this.outputSize = outputSize;
        this.batchSize = batchSize;
        
        this.input = forwardPassInputs;
        this.channelInputs = channelInputs;
        this.im2colMatrix = im2colMatrix;
        this.kernelOutputs = kernelOutputs;
        this.sigmoidOutputs = sigmoidOutputs;
    }
    
    public void free() {
        if (output != null)
            CudaUtil.free(output);

        if (kernelOutputs != null)
            for (CUdeviceptr ptr : kernelOutputs) {
                 CudaUtil.free(ptr);
            }

        if (sigmoidOutputs != null)
            for (CUdeviceptr ptr : sigmoidOutputs) {
                 CudaUtil.free(ptr);
            }

        if (channelInputs != null)
            for (CUdeviceptr ptr : channelInputs) {
                 CudaUtil.free(ptr);
            }

        if (im2colMatrix != null)
            for (CUdeviceptr ptr : im2colMatrix) {
                CudaUtil.free(ptr);
            }
    }

    public void freeAsync(CUstream stream) {
        if (output != null)
            CudaUtil.freeAsync(output, stream);

        if (kernelOutputs != null)
            for (CUdeviceptr ptr : kernelOutputs) {
                 CudaUtil.freeAsync(ptr, stream);
            }

        if (sigmoidOutputs != null)
            for (CUdeviceptr ptr : sigmoidOutputs) {
                 CudaUtil.freeAsync(ptr, stream);
            }

        if (channelInputs != null)
            for (CUdeviceptr ptr : channelInputs) {
                 CudaUtil.freeAsync(ptr, stream);
            }

        if (im2colMatrix != null)
            for (CUdeviceptr ptr : im2colMatrix) {
                 CudaUtil.freeAsync(ptr, stream);
            }
    }
}
