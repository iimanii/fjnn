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

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUmemorytype;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.cuda.CudaEngine.CUdeviceptr2D;

/**
 *
 * @author ahmed
 */
public class CudaUtil {
    
    public static final String CUDA_COMPILER = "nvcc";
    
    /**
     * Compiles CU files, throws runtime exception if anything went wrong
     */ 
    static void compileCU(String cuPath, String ptxPath) {
        File cuFile = new File(cuPath);
        
        if(!cuFile.exists())
            throw new RuntimeException("Cuda file not found: " + cuPath);
        
        String arch = System.getProperty("sun.arch.data.model");
        
        String command = String.format("%s -ptx %s -o %s -m %s -lineinfo",
                                        CUDA_COMPILER, cuPath, ptxPath, arch);

        System.out.println("Compiling: \n" + command);

        try {
            Process process = Runtime.getRuntime().exec(command);
            
            int result = process.waitFor();
            
            if(result != 0) {
                String output = readStream(process.getInputStream());
                String error  = readStream(process.getErrorStream());
                
                System.out.println("nvcc exit: " + result);
                System.out.println("outputMessage:\n" + output);
                System.out.println("errorMessage:\n" + error);
                
                throw new RuntimeException("Unable to create .ptx file: "+ error);
            }
        } catch(IOException | InterruptedException ex) {
            throw new RuntimeException(ex);
        }
    }
    
    static String readStream(InputStream stream) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        byte buffer[] = new byte[4096];
        
        int len;
        
        while((len = stream.read(buffer)) != -1)
            output.write(buffer, 0, len);

        return output.toString("UTF-8");
    }

    public static CUdeviceptr toGPU(float[] array) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, array.length * (long)Sizeof.FLOAT);

        JCudaDriver.cuMemcpyHtoD(ptr, Pointer.to(array), array.length * (long)Sizeof.FLOAT);
        
        return ptr;
    }
    
    public static CUdeviceptr toGPU(float[] array, CUstream stream) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, array.length * (long)Sizeof.FLOAT);
        JCudaDriver.cuMemcpyHtoDAsync(ptr, Pointer.to(array), array.length * (long)Sizeof.FLOAT, stream);
        
        return ptr;
    }
    public static CUdeviceptr toGPU(Pointer array, long size, CUstream stream) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(ptr, size * (long)Sizeof.FLOAT);
        JCudaDriver.cuMemcpyHtoDAsync(ptr, array, size * (long)Sizeof.FLOAT, stream);
        
        return ptr;
    }
    
    public static CUdeviceptr2D createPitch(int width, int height) {
        CUdeviceptr ptr = new CUdeviceptr();
        long[] pitch = new long[1];
        
        JCudaDriver.cuMemAllocPitch(ptr, pitch, width * (long)Sizeof.FLOAT, height, Sizeof.FLOAT);

        return new CUdeviceptr2D(ptr, pitch[0] / Sizeof.FLOAT);
    }
    
    public static float[] fromGPU(CUdeviceptr ptr, int size, CUstream stream) {
        float[] array = new float[size];
        JCudaDriver.cuMemcpyDtoHAsync(Pointer.to(array), ptr, size * (long)Sizeof.FLOAT, stream);
        
        return array;
    }
    
    public static long length(CUdeviceptr ptr) {
        long[] size = new long[1];
        JCudaDriver.cuMemGetAddressRange(null, size, ptr);
        
        return size[0];
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
        CudaThread.prepareThread();
        JCudaDriver.cuMemGetInfo(free, total);
        CudaThread.finalizeThread();

        System.out.println(Thread.currentThread().getId() + ": waiting -> " + free[0] + " " + total[0]);

    }

    public static float[][] fromGPU(CUdeviceptr2D ptr, int width, int height, CUstream mainstream) {
        float[] temp = new float[height * width];
        
        CUDA_MEMCPY2D memcpy = new CUDA_MEMCPY2D();
        memcpy.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
        memcpy.srcDevice = ptr.ptr;
        memcpy.srcPitch = ptr.pitch * (long) Sizeof.FLOAT;

        memcpy.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        memcpy.dstHost = Pointer.to(temp);
        memcpy.dstPitch = width * (long) Sizeof.FLOAT;

        memcpy.WidthInBytes = width * (long) Sizeof.FLOAT;
        memcpy.Height = height;

        memcpy.srcXInBytes = memcpy.srcY = memcpy.dstXInBytes = memcpy.dstY = 0;

        JCudaDriver.cuMemcpy2DAsync(memcpy, mainstream);
        
        JCudaDriver.cuStreamSynchronize(mainstream);
        
        float[][] result = new float[height][width];
        
        for(int i=0; i < height; i++)
            System.arraycopy(temp, i * width, result[i], 0, width);
        
        return result;
    }
}
