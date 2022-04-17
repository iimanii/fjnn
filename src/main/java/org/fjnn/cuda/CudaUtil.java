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
package org.fjnn.cuda;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import run.timer;

/**
 *
 * @author ahmed
 */
public class CudaUtil {
    
    private static final long FLOAT_SIZE = Sizeof.FLOAT;
    
    public static CUdeviceptr create(long size) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, size * FLOAT_SIZE);
        
        return ptr;
    }

    public static CUdeviceptr createBytes(long size) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, size);
        
        return ptr;
    }
    
    public static CUdeviceptr toGPU(float[] array) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, array.length * FLOAT_SIZE);
        JCudaDriver.cuMemcpyHtoD(ptr, Pointer.to(array), array.length * FLOAT_SIZE);
        
        return ptr;
    }
        
    public static CUdeviceptr toGPU(float[] array, CUstream stream) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, array.length * FLOAT_SIZE);
        JCudaDriver.cuMemcpyHtoDAsync(ptr, Pointer.to(array), array.length * FLOAT_SIZE, stream);
        
        return ptr;
    }
    
    public static float[] fromGPU(CUdeviceptr src, int size) {
        float[] array = new float[size];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(array), src, size * FLOAT_SIZE);
        
        return array;
    }
    
    public static float[] fromGPU(CUdeviceptr src, int size, CUstream stream) {
        float[] array = new float[size];
        JCudaDriver.cuMemcpyDtoHAsync(Pointer.to(array), src, size * FLOAT_SIZE, stream);
        
        return array;
    }
        
    public static long length(CUdeviceptr ptr) {
        long[] size = new long[1];
        JCudaDriver.cuMemGetAddressRange(null, size, ptr);
        
        return size[0];
    }

    /* for pinned memory */
    public static Pointer createPinned(long size) {
        Pointer ptr = new Pointer();
        JCudaDriver.cuMemAllocHost(ptr, size);
        
        return ptr;
    }
    
    public static CUdeviceptr toGPU(Pointer pinned, long size, CUstream stream) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, size * FLOAT_SIZE);
        JCudaDriver.cuMemcpyHtoDAsync(ptr, pinned, size * FLOAT_SIZE, stream);
        
        return ptr;
    }

    public static float[] fromGPU(Pointer pinned, CUdeviceptr src, int size, CUstream stream) {
        float[] array = new float[size];
        JCudaDriver.cuMemcpyDtoHAsync(pinned, src, size * FLOAT_SIZE, stream);
        
        /* must sync here to actually get the data */
        JCudaDriver.cuStreamSynchronize(stream);
        
        pinned.getByteBuffer().asFloatBuffer().get(array);
        
        return array;
    }
    
    /* utility functions */
    public static float sum_abs_differenceGPU(CUdeviceptr array1, CUdeviceptr array2, int size, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        int threadsPerBlock = CudaEngine.getMaxThreadsPerBlock(device);        
        CUfunction matrixMulVector = CudaEngine.getKernel(CudaModule.MODULE_ACCUMULATE, "sum_abs_difference", device);
        
        int blockSizeX = Math.min(threadsPerBlock, size);
        int gridSizeX  = (size-1) / (blockSizeX) + 1;
        gridSizeX = (int) Math.max(1, Math.ceil(Math.sqrt(gridSizeX)));

        CUdeviceptr result = CudaUtil.create(gridSizeX);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(array1),
            Pointer.to(array2),
            Pointer.to(new long[]{size}),
            Pointer.to(result)
        );

        JCudaDriver.cuLaunchKernel(matrixMulVector,
            gridSizeX, 1, 1,        // Grid dimension
            blockSizeX, 1, 1,       // Block dimension
            0, stream,                // Shared memory size and stream
            kernelParameters, null  // Kernel- and extra parameters
        );
        
        float sum = sumGPU(result, gridSizeX, stream);
        JCudaDriver.cuMemFree(result);
        
        return sum;
    }
    
    public static float sumGPU(CUdeviceptr array, int size, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        int threadsPerBlock = CudaEngine.getMaxThreadsPerBlock(device);        
        CUfunction matrixMulVector = CudaEngine.getKernel(CudaModule.MODULE_ACCUMULATE, "accumulate_vector", device);
        
        int blockSizeX = Math.min(threadsPerBlock, size);
        int gridSizeX  = (size-1) / (blockSizeX * 2) + 1;
        gridSizeX = (int) Math.max(1, Math.ceil(Math.sqrt(gridSizeX)));

        CUdeviceptr result = CudaUtil.create(gridSizeX);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(array),
            Pointer.to(new long[]{size}),
            Pointer.to(result)
        );

        JCudaDriver.cuLaunchKernel(matrixMulVector,
            gridSizeX, 1, 1,        // Grid dimension
            blockSizeX, 1, 1,       // Block dimension
            0, stream,                // Shared memory size and stream
            kernelParameters, null  // Kernel- and extra parameters
        );
        
        float sum;
        
        if(gridSizeX > 1)
            sum = sumGPU(result, gridSizeX, stream);
        else {
            sum = CudaUtil.fromGPU(result, 1, stream)[0];
        }
        
        JCudaDriver.cuMemFree(result);
        
        return sum;
    }
    
    
    public static void print(CUdeviceptr ptr, int length, CUstream stream) {
        float[] temp = fromGPU(ptr, length, null);
        
        for(float t : temp)
            System.out.print(t + " ");
        System.out.println();
    }
        
    public static void printMemUsage(boolean cpu) {
        if(cpu) {
            /* Total number of processors or cores available to the JVM */
            System.out.println("Available processors (cores): "
                    + Runtime.getRuntime().availableProcessors());

            /* Total amount of free memory available to the JVM */
            System.out.println("Free memory (bytes): "
                    + Runtime.getRuntime().freeMemory());

            /* This will return Long.MAX_VALUE if there is no preset limit */
            long maxMemory = Runtime.getRuntime().maxMemory();
            /* Maximum amount of memory the JVM will attempt to use */
            System.out.println("Maximum memory (bytes): "
                    + (maxMemory == Long.MAX_VALUE ? "no limit" : maxMemory));

            /* Total memory currently in use by the JVM */
            System.out.println("Total memory (bytes): "
                    + Runtime.getRuntime().totalMemory());
        }

        long[] free = new long[1];
        long[] total = new long[1];
        
        for(int i=0; i < CudaEngine.getDeviceCount(); i++) {
            CudaEngine.prepareThread(i);
            JCudaDriver.cuMemGetInfo(free, total);
            System.out.println("Device " + i + ": " + free[0] + " " + total[0] + " " + (1.0*free[0])/total[0]);
            CudaEngine.finalizeThread();
        }
    }    
}
