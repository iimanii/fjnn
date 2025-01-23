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
package org.fjnn.normalizer.outputs;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class LayerNormalizerForwardOutputGPU extends FeedForwardOutputGPU {
    // Fields for layer normalization
    public CUdeviceptr preNormalization;        // input values
    public CUdeviceptr normalized;              // x̂ values (after normalization, before γ and β)    
    public CUdeviceptr postNormalization;       // final output (after γ and β)
//    public CUdeviceptr means;                   // means per layer
    public CUdeviceptr stds;                    // variances per layer

    public LayerNormalizerForwardOutputGPU(CUdeviceptr preNormalization, int outputDim, int batchSize, CUstream stream) {
        super(outputDim, batchSize);
        
        this.preNormalization = preNormalization;
        this.normalized = CudaUtil.createFloatAsync(outputDim * batchSize, stream);
        this.postNormalization = CudaUtil.createFloatAsync(outputDim * batchSize, stream);
//        this.means = CudaUtil.createFloatAsync(batchSize, stream);
        this.stds = CudaUtil.createFloatAsync(batchSize, stream);
    }

    @Override
    public CUdeviceptr output() {
        return postNormalization;
    }

    @Override
    public void free() {
        /* we do not free the preNormalization as its by reference */
        CudaUtil.free(normalized);
        CudaUtil.free(postNormalization);
//        CudaUtil.free(means);
        CudaUtil.free(stds);
        
        preNormalization = null;
        postNormalization = null;
//        means = null;
        stds = null;
    }

    @Override
    public void freeAsync(CUstream stream) {
        /* we do not free the preNormalization as its by reference */
        CudaUtil.freeAsync(normalized, stream);
        CudaUtil.freeAsync(postNormalization, stream);
//        CudaUtil.freeAsync(means, stream);
        CudaUtil.freeAsync(stds, stream);
        
        preNormalization = null;
        postNormalization = null;
//        means = null;
        stds = null;
    }
}
