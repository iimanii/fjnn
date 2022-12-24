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
import org.fjnn.cuda.CudaModule;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.cuda.CUdeviceptr2D;

/**
 *
 * @author ahmed
 */
public class SoftMax extends Activation {
    
    @Override
    public float compute(float input) {
        return 1.0f;
    }

    @Override
    public void compute(float[] input, int from, int to) {
        double sum = 0;
        float max = Float.NEGATIVE_INFINITY;
        
        for(int i=from; i < to; i++) {
            max = Math.max(input[i], max);
        }
        
        for(int i=from; i < to; i++) {
            input[i] = (float) Math.exp(input[i] - max);
            sum += input[i];
        }
        
        if(sum == 0)
            for(int i=from; i < to; i++)
                input[i] = 1.0f / input.length;
        else
            for(int i=from; i < to; i++)
                input[i] /= sum;
    }

    @Override
    public void computeGPU(CUdeviceptr ptr, int size, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(device), size);
        int gridSizeX = (size - 1) / blockSizeX + 1;
        
        /* Create temp array to put sums */
        int sums_size = gridSizeX;
        CUdeviceptr sums = CudaUtil.create(sums_size);//CudaEngine.getSharedResource(sums_size, device);//new CUdeviceptr();
//        JCudaDriver.cuMemAlloc(sums, sums_size * (long)Sizeof.FLOAT);
        
        if(true)
            throw new RuntimeException("bad implementation reimplement to subtract MAX");
        
        /* Phase 1 */
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "SoftMax_1", device);
                    
        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr),
            Pointer.to(new long[]{size}),
            Pointer.to(sums)
        );
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, 1, 1,       // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, stream,             // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        
        /* Phase 2 */
        function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "SoftMax_2", device);
                            
        kernelParameters = Pointer.to(
            Pointer.to(ptr),
            Pointer.to(new long[]{size}),
            Pointer.to(sums),
            Pointer.to(new long[]{sums_size})
        );
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, 1, 1,       // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, stream,             // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        
//        CudaEngine.freeSharedResource(sums, device);
        JCudaDriver.cuMemFree(sums);
    }

    @Override
    public void computeMultiGPU(CUdeviceptr2D ptr, int width, int height, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(device), width);
        int gridSizeX = (width - 1) / blockSizeX + 1;
        int gridSizeY = height;
        
        /* Create temp array to put sums */
        CUdeviceptr2D sums = CUdeviceptr2D.createPitch(width, height);
        
        /* Phase 1 */
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "multi_SoftMax_1", device);
                    
        //float* v, long size, size_t pitch_v, float* sums, size_t pitch_s
        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr.ptr),
            Pointer.to(new long[]{width}),
            Pointer.to(new long[]{ptr.pitch}),
            Pointer.to(sums.ptr),
            Pointer.to(new long[]{sums.pitch})
        );
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, gridSizeY, 1,    // Grid dimension
            blockSizeX, 1, 1,           // Block dimension
            0, stream,                  // Shared memory size and stream
            kernelParameters, null      // Kernel- and extra parameters
        );
        
        /* Phase 2 */
        function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "multi_SoftMax_2", device);
                            
        //float* v, long size, size_t pitch_v, float* sums, long sums_size, size_t pitch_s
        kernelParameters = Pointer.to(
            Pointer.to(ptr.ptr),
            Pointer.to(new long[]{width}),
            Pointer.to(new long[]{ptr.pitch}),            
            Pointer.to(sums.ptr),
            Pointer.to(new long[]{gridSizeX}),
            Pointer.to(new long[]{sums.pitch})
        );
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, gridSizeY, 1,  // Grid dimension
            blockSizeX, 1, 1,         // Block dimension
            0, stream,                // Shared memory size and stream
            kernelParameters, null    // Kernel- and extra parameters
        );
        
        JCudaDriver.cuMemFree(sums.ptr);
    }

    @Override
    public void computeConditional(float[] input, boolean[] compute) {
        double sum = 0;
        
        for(int i=0; i < input.length; i++) {
            if(compute[i]) {
                input[i] = SafeExp(input[i]);
                sum += input[i];
            }
        }
        
        if(sum == 0)
            for(int i=0; i < input.length; i++) {
                if(compute[i])
                    input[i] = 1.0f / input.length;
            }
        else
            for(int i=0; i < input.length; i++) {
                if(compute[i])
                    input[i] /= sum;
            }
    }

    @Override
    public void computeGPUConditional(CUdeviceptr ptr, CUdeviceptr compute, int size, CUstream stream, int count) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void computeMultiGPUConditional(CUdeviceptr2D ptr, CUdeviceptr compute, int width, int height, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    /*
     * https://mmuratarat.github.io/2019-01-27/derivation-of-softmax-function
     * https://www.youtube.com/watch?v=09c7bkxpv9I
     */
    @Override
    public void derivative(float[] input, int from, int to) {

    }
}
