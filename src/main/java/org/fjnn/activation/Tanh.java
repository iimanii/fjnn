/*
 * The MIT License
 *
 * Copyright 2018 Ahmed Tarek.
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
package org.fjnn.activation;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaEngine.CUdeviceptr2D;
import org.fjnn.cuda.CudaModule;
import org.fjnn.cuda.CudaThread;

/**
 *
 * @author ahmed
 */
public class Tanh extends Activation {

    @Override
    public float compute(float input) {
        return (float)Math.tanh(input);
    }
    
    @Override
    public void compute(float[] input) {
        for(int i=0; i < input.length; i++)
            input[i] = (float)Math.tanh(input[i]);
    }

    @Override
    public void computeGPU(CUdeviceptr ptr, int size, CUstream stream) {
        int device = CudaThread.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "Tanh", device);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr),
            Pointer.to(new long[]{size})
        );
        
        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(device), size);
        int gridSizeX = (size - 1) / blockSizeX + 1;
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, 1, 1,       // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, stream,             // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
    }

    @Override
    public void computeMultiGPU(CUdeviceptr2D ptr, int width, int height, CUstream stream) {
        int device = CudaThread.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "multi_Tanh", device);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr.ptr),
            Pointer.to(new long[]{width}),
            Pointer.to(new long[]{ptr.pitch})
        );
        
        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(device), width);
        int gridSizeX = (width - 1) / blockSizeX + 1;
        int gridSizeY = height;
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, gridSizeY, 1,   // Grid dimension
            blockSizeX, 1, 1,          // Block dimension
            0, stream,                 // Shared memory size and stream
            kernelParameters, null     // Kernel- and extra parameters
        );
    }

    @Override
    public void computeConditional(float[] input, boolean[] compute) {
        for(int i=0; i < input.length; i++)
            if(compute[i])
                input[i] = (float)Math.tanh(input[i]);
    }

    @Override
    public void computeGPUConditional(CUdeviceptr ptr, CUdeviceptr compute, int size, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void computeMultiGPUConditional(CUdeviceptr2D ptr, CUdeviceptr compute, int width, int height, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
