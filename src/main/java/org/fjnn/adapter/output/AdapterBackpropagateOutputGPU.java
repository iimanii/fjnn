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
package org.fjnn.adapter.output;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class AdapterBackpropagateOutputGPU extends BackpropagateOutputGPU {  
    private final CUdeviceptr deltaLoss;
    private final boolean byReference;  /* whether or not the output ptr is just a reference to another pointer */

    public AdapterBackpropagateOutputGPU(int deltaLossDim, int batchSize, CUdeviceptr deltaLoss, boolean byReference) {
        super(deltaLossDim, batchSize);
        
        this.deltaLoss = deltaLoss;
        this.byReference = byReference;
    }

    @Override
    public CUdeviceptr deltaLoss() {
        return deltaLoss;
    }    

    @Override
    public void free() {
        if(!byReference)
            CudaUtil.free(deltaLoss);
    }

    @Override
    public void freeAsync(CUstream stream) {
        if(!byReference)
            CudaUtil.freeAsync(deltaLoss, stream);
    }
}
