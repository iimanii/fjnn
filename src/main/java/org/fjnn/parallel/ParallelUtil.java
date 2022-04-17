/*
 * The MIT License
 *
 * Copyright 2020 ahmed.
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
package org.fjnn.parallel;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUmemorytype;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;

/**
 *
 * @author ahmed
 */
public class ParallelUtil {
        
    public static class CUdeviceptr2D {
        public CUdeviceptr ptr;
        public long pitch;

        public CUdeviceptr2D(CUdeviceptr ptr, long pitch_ptr) {
            this.ptr = ptr;
            this.pitch = pitch_ptr;
        }
    }
    
    public static CUdeviceptr2D createPitch(int width, int height) {
        CUdeviceptr ptr = new CUdeviceptr();
        long[] pitch = new long[1];
        
        JCudaDriver.cuMemAllocPitch(ptr, pitch, width * (long)Sizeof.FLOAT, height, Sizeof.FLOAT);

        return new CUdeviceptr2D(ptr, pitch[0] / Sizeof.FLOAT);
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
