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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.util.util;
import run.timer;

/**
 *
 * @author ahmed
 */
public class CudaUtil {
    
    public static final long FLOAT_SIZE = Sizeof.FLOAT;
    public static int DEFAULT_MEM_ALIGN = 1024;
    public static int PREFERRED_BLOCK_SIZE = 128;
    
    public static CUdeviceptr createFloat(long size) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, size * FLOAT_SIZE);
        
        return ptr;
    }

    public static CUdeviceptr createByte(long size) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, size);
        
        return ptr;
    }
    
    public static CUdeviceptr createFloat(long size, int device) {
        CudaEngine.prepareThread(device);
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, size * FLOAT_SIZE);
        CudaEngine.finalizeThread();
        
        return ptr;
    }

    public static CUdeviceptr createByte(long size, int device) {
        CudaEngine.prepareThread(device);
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, size);
        CudaEngine.finalizeThread();
        
        return ptr;
    }

    public static void free(CUdeviceptr ptr) {
        JCudaDriver.cuMemFree(ptr);
    }
    
    public static CUdeviceptr toGPU(float[] array) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, array.length * FLOAT_SIZE);
        JCudaDriver.cuMemcpyHtoD(ptr, Pointer.to(array), array.length * FLOAT_SIZE);
        
        return ptr;
    }
        
    public static CUdeviceptr toGPU(float[] input, int device) {
        CudaEngine.prepareThread(device);
        CUdeviceptr ptr = toGPU(input);
        CudaEngine.finalizeThread();
        
        return ptr;
    }
    
    public static CUdeviceptr toGPU(float[][] input, int device) {
        CudaEngine.prepareThread(device);
        CUdeviceptr ptr = CudaUtil.toGPU(util.to1D(input, input[0].length));
        CudaEngine.finalizeThread();
        
        return ptr;
    }
    
    public static CUdeviceptr toGPU(float[] array, CUstream stream) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, array.length * FLOAT_SIZE);
        JCudaDriver.cuMemcpyHtoDAsync(ptr, Pointer.to(array), array.length * FLOAT_SIZE, stream);
        
        return ptr;
    }
    
    public static float[] fromGPUFloat(CUdeviceptr src, int size) {
        float[] array = new float[size];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(array), src, size * FLOAT_SIZE);
        
        return array;
    }
    
    public static float[] fromGPUFloat(CUdeviceptr src, int size, CUstream stream) {
        float[] array = new float[size];
        JCudaDriver.cuMemcpyDtoHAsync(Pointer.to(array), src, size * FLOAT_SIZE, stream);
        
        return array;
    }
        
    public static FloatBuffer toInputBuffer(float[] input) {
        FloatBuffer result = ByteBuffer.allocateDirect(input.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        result.put(input);
        
        return result;
    }
    
    public static FloatBuffer toInputBuffer(float[][] input) {
        /* all inputs must be equal in size */
        int size = input.length * input[0].length;
        
        FloatBuffer result = ByteBuffer.allocateDirect(size * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        for(float[] i : input)
            result.put(i);
        
        return result;
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
    
    public static void print(CUdeviceptr ptr, int length, CUstream stream) {
        float[] temp = CudaUtil.fromGPUFloat(ptr, length, null);
        
        for(float t : temp)
            System.out.print(t + " ");
        System.out.println();
    }
        
    public static void printMemUsage(boolean cpu, int deviceId) {
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
        
        CudaEngine.prepareThread(deviceId);
        JCudaDriver.cuMemGetInfo(free, total);
        System.out.printf("Device %d: %.4f %.4f %.4f\n", deviceId, free[0] / 1e6f, total[0] / 1e6f, (1.0*free[0])/total[0]);
        CudaEngine.finalizeThread();
    }    

    public static long alignLength(long length, int alignment) {
        return ((length + (alignment - 1)) & ~(alignment - 1));
    }
}
