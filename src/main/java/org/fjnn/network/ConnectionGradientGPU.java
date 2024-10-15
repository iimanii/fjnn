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
package org.fjnn.network;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class ConnectionGradientGPU {
    public CUdeviceptr weightGradients;  // Gradient for each weight in the connection
    public CUdeviceptr biasGradients;    // Gradient for each bias in the connection

    public ConnectionGradientGPU(int neurons, int links, CUstream stream) {
        weightGradients = CudaUtil.createFloatAsync(neurons * links, stream);
//        JCudaDriver.cuMemsetD32Async(weightGradients, 0, neurons * links, stream);
        
        biasGradients = CudaUtil.createFloatAsync(links, stream);        
//        JCudaDriver.cuMemsetD32Async(biasGradients, 0, links, stream);
    }
    
    public void free(CUstream stream) {
        CudaUtil.freeAsync(weightGradients, stream);
        CudaUtil.freeAsync(biasGradients, stream);
    }
}
