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
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 */
public class CudaUtil {
    
    public static final long FLOAT_SIZE = Sizeof.FLOAT;
    public static int DEFAULT_MEM_ALIGN = 256;
    public static int DEFAULT_MEM_ALIGN_FLOAT = 256 / Sizeof.FLOAT;
    public static int PREFERRED_BLOCK_SIZE = 128;
    
    public static Map<Integer, CudaMempool2> mempool = new HashMap<>();
    
    public static synchronized void initMemPool(long maxSize, int deviceId) {
        if (mempool.containsKey(deviceId))
            throw new IllegalStateException("Memory pool for device " + deviceId + " already exists.");
        
        CudaMempool2 pool = new CudaMempool2(maxSize, deviceId);
        mempool.put(deviceId, pool);
    }
    
    public static synchronized void destroyMemPool(int deviceId) {
        CudaMempool2 pool = mempool.get(deviceId);
        
        if (pool != null) {
            pool.destroy();
            mempool.remove(deviceId);
        } else
            throw new IllegalStateException("Memory pool for device " + deviceId + " doesn't exists.");
    }
    
    public static CUdeviceptr getMemPoolAsync(long size, CUstream stream) {
        return getMemPoolAsync(size, CudaEngine.getThreadDeviceId(), stream);
    }
    
    public static CUdeviceptr getMemPoolAsync(long size, int deviceId, CUstream stream) {
        CudaMempool2 pool = mempool.get(deviceId);
        
        if (pool == null)
            throw new IllegalStateException("Memory pool for device " + deviceId + " not initialized.");
        
        return pool.get(size, stream);
    }
    
    public static CUdeviceptr getMemPoolFloatAsync(long size, CUstream stream) {
        return getMemPoolFloatAsync(size, CudaEngine.getThreadDeviceId(), stream);
    }
    
    public static CUdeviceptr getMemPoolFloatAsync(long size, int deviceId, CUstream stream) {
        CudaMempool2 pool = mempool.get(deviceId);
        
        if (pool == null)
            throw new IllegalStateException("Memory pool for device " + deviceId + " not initialized.");

        return pool.getFloat(size, stream);
    }
    
    public static CUdeviceptr copyMemPoolAsync(CUdeviceptr src, long size, CUstream stream) {
        CUdeviceptr dest = getMemPoolAsync(size, stream);
        
        JCudaDriver.cuMemcpyAsync(dest, src, size, stream);
        
        return dest;
    }
    
    public static CUdeviceptr copyMemPoolFloatAsync(CUdeviceptr src, long size, CUstream stream) {
        return copyAsync(src, size * FLOAT_SIZE, stream);
    }

    static AtomicInteger netMemoryAllocCount = new AtomicInteger();
    
    public static int getNetMemoryAllocCounter() {
        return netMemoryAllocCount.get();
    }
    
    public static CUdeviceptr create(long size) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, size);

        netMemoryAllocCount.incrementAndGet();
        
        return ptr;
    }
    
    public static CUdeviceptr createAsync(long size, CUstream stream) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAllocAsync(ptr, size, stream);
        
        netMemoryAllocCount.incrementAndGet();
        
        return ptr;
    }
        
    public static CUdeviceptr createFloat(long size) {
        return create(size * FLOAT_SIZE);
    }
    
    public static CUdeviceptr createFloatAsync(long size, CUstream stream) {
        return createAsync(size * FLOAT_SIZE, stream);
    }
    
    
    public static void free(CUdeviceptr ptr) {
        JCudaDriver.cuMemFree(ptr);
        
        netMemoryAllocCount.decrementAndGet();
    }
    
    public static void freeAsync(CUdeviceptr ptr, CUstream stream) {
        JCudaDriver.cuMemFreeAsync(ptr, stream);
        
        netMemoryAllocCount.decrementAndGet();
    }
    
    
    public static CUdeviceptr copy(CUdeviceptr src, long size) {
        CUdeviceptr dest = create(size);
        
        JCudaDriver.cuMemcpy(dest, src, size);
        
        return dest;
    }
    
    public static CUdeviceptr copyAsync(CUdeviceptr src, long size, CUstream stream) {
        CUdeviceptr dest = createAsync(size, stream);
        
        JCudaDriver.cuMemcpyAsync(dest, src, size, stream);
        
        return dest;
    }
    
    public static CUdeviceptr copyFloat(CUdeviceptr src, long size) {
        return copy(src, size * FLOAT_SIZE);
    }
    
    public static CUdeviceptr copyFloatAsync(CUdeviceptr src, long size, CUstream stream) {
        return copyAsync(src, size * FLOAT_SIZE, stream);
    }

    /* moving memory */
    public static CUdeviceptr toGPU(float[] array) {
        CUdeviceptr ptr = create(array.length * FLOAT_SIZE);
        
        toGPU(array, ptr);
        
        return ptr;
    }
    
    public static void toGPU(float[] array, CUdeviceptr ptr) {
        JCudaDriver.cuMemcpyHtoD(ptr, Pointer.to(array), array.length * FLOAT_SIZE);
    }
    
    public static CUdeviceptr toGPUAsync(float[] array, CUstream stream) {
        CUdeviceptr ptr = createAsync(array.length * FLOAT_SIZE, stream);
        
        toGPUAsync(array, ptr, stream);
        
        return ptr;
    }
    
    public static void toGPUAsync(float[] array, CUdeviceptr ptr, CUstream stream) {
        JCudaDriver.cuMemcpyHtoDAsync(ptr, Pointer.to(array), array.length * FLOAT_SIZE, stream);
    }
    
    public static byte[] fromGPU(CUdeviceptr src, int size) {
        byte[] array = new byte[size];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(array), src, size);
        
        return array;
    }
    
    public static byte[] fromGPUAsync(CUdeviceptr src, int size, CUstream stream) {
        byte[] array = new byte[size];
        JCudaDriver.cuMemcpyDtoHAsync(Pointer.to(array), src, size, stream);
        
        return array;
    }
    
    public static float[] fromGPUFloat(CUdeviceptr src, int size) {
        float[] array = new float[size];
        JCudaDriver.cuMemcpyDtoH(Pointer.to(array), src, size * FLOAT_SIZE);
        
        return array;
    }
    
    public static float[] fromGPUFloatAsync(CUdeviceptr src, int size, CUstream stream) {
        float[] array = new float[size];
        JCudaDriver.cuMemcpyDtoHAsync(Pointer.to(array), src, size * FLOAT_SIZE, stream);
        
        return array;
    }
    
        
    /* streams */
    public static CUstream createStream() {
        CUstream stream = new CUstream();
        JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_NON_BLOCKING);
        return stream;
    }
    
    public static void freeStream(CUstream stream) {
        JCudaDriver.cuStreamDestroy(stream);
    }
    
    /* float buffers */
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
    
    /* other util functions */
    public static void print(CUdeviceptr ptr, int length) {
        float[] temp = CudaUtil.fromGPUFloat(ptr, length);
        
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
