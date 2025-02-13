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
package org.fjnn.transformer.output;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class AttentionWeightsOutputGPU extends FeedForwardOutputGPU {
    public final CUdeviceptr input;         // original input values
    public final CUdeviceptr output;        // final output after weight transformation
    
    public AttentionWeightsOutputGPU(CUdeviceptr input, int inputDim, int batchSize, CUstream stream) {
        super(inputDim, batchSize);
        
        this.input = input;
        this.output = CudaUtil.createFloatAsync(outputDim * batchSize, stream);
    }

    @Override
    public CUdeviceptr output() {
        return output;
    }

    @Override
    public void free() {
        // Don't free input as it was passed in
        CudaUtil.free(output);
    }

    @Override
    public void freeAsync(CUstream stream) {
        // Don't free input as it was passed in
        CudaUtil.freeAsync(output, stream);
    }
}
