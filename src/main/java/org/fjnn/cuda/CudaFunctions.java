/*
 * The MIT License
 *
 * Copyright 2023 ahmed.
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
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;

/**
 *
 * @author ahmed
 */
public class CudaFunctions {
    
    public static void addStride(CUdeviceptr a, CUdeviceptr b, int stride, int count, CUstream stream) {
        addStride(a, b, stride, count, CudaUtil.PREFERRED_BLOCK_SIZE, stream);
    }
    
    public static void addStride(CUdeviceptr a, CUdeviceptr b, int stride, int count, int threadsPerBlock, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACCUMULATE, "add_stride", device);
        
        int blockSizeX = Math.min(threadsPerBlock, stride);
        int blockSizeY = threadsPerBlock / blockSizeX;
        int gridSizeX = (stride - 1) / threadsPerBlock + 1;
        
        int blocksCount = (count - 1) / blockSizeY + 1;
        int gridSizeY = Math.min(CudaEngine.getMaxGridSize(device)[1], blocksCount);
        int gridSizeZ = (blocksCount - 1) / gridSizeY + 1;
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(a),
            Pointer.to(b),
            Pointer.to(new int[]{stride}),
            Pointer.to(new long[]{(long)count * stride})
        );
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, gridSizeY, gridSizeZ,    // Grid dimension
            blockSizeX, blockSizeY, 1,          // Block dimension
            0, stream,                          // Shared memory size and stream
            kernelParameters, null              // Kernel- and extra parameters
        );
    }
    
    public static void addStrideVectorizedLoop(CUdeviceptr a, CUdeviceptr b, int stride, int count, int threadsPerBlock, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACCUMULATE, "add_stride_vectorized_loop", device);
        
        int effectiveStride = Math.max(stride / 4, stride % 4);
        
        int blockSizeX = Math.max(32, Math.min(threadsPerBlock, effectiveStride));
        int blockSizeY = Math.min(threadsPerBlock / blockSizeX, count);
        int gridSizeX = (effectiveStride - 1) / threadsPerBlock + 1;
        
        int iterations = 4;//(int) Math.sqrt(count);
        
        int blocksCount = (count - 1) / (iterations * blockSizeY) + 1;
        int gridSizeY = Math.min(CudaEngine.getMaxGridSize(device)[1], blocksCount);
        int gridSizeZ = (blocksCount - 1) / gridSizeY + 1;
        
//        System.out.printf("%d %d %d %d %d %d %d %d\n", threadsPerBlock, blockSizeX, blockSizeY, gridSizeX, iterations, blocksCount, gridSizeY, gridSizeZ);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(a),
            Pointer.to(b),
            Pointer.to(new int[]{stride}),
            Pointer.to(new int[]{count}),
            Pointer.to(new int[]{iterations})
        );
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, gridSizeY, gridSizeZ,    // Grid dimension
            blockSizeX, blockSizeY, 1,          // Block dimension
            0, stream,                          // Shared memory size and stream
            kernelParameters, null              // Kernel- and extra parameters
        );
    }
    
    public static void addStrideVectorized(CUdeviceptr a, CUdeviceptr b, int stride, int count, int threadsPerBlock, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACCUMULATE, "add_stride_vectorized", device);
        
        int effectiveStride = Math.max(stride / 4, stride % 4);
        
        int blockSizeX = Math.min(threadsPerBlock, effectiveStride);
        int blockSizeY = Math.min(threadsPerBlock / blockSizeX, count);
        int gridSizeX = (effectiveStride - 1) / threadsPerBlock + 1;
        
        int blocksCount = (count - 1) / (blockSizeY) + 1;
        int gridSizeY = Math.min(CudaEngine.getMaxGridSize(device)[1], blocksCount);
        int gridSizeZ = (blocksCount - 1) / gridSizeY + 1;
        
//        System.out.printf("%d %d %d %d %d %d %d\n", blockSizeX, blockSizeY, gridSizeX, iterations, blocksCount, gridSizeY, gridSizeZ);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(a),
            Pointer.to(b),
            Pointer.to(new int[]{stride}),
            Pointer.to(new int[]{count})
        );
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, gridSizeY, gridSizeZ,    // Grid dimension
            blockSizeX, blockSizeY, 1,          // Block dimension
            0, stream,                          // Shared memory size and stream
            kernelParameters, null              // Kernel- and extra parameters
        );
    }
    
    public static void addStrideLoop(CUdeviceptr a, CUdeviceptr b, int stride, int count, int threadsPerBlock, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACCUMULATE, "add_stride_loop", device);
        
        int blockSizeX = Math.min(threadsPerBlock, stride);
        int blockSizeY = threadsPerBlock / blockSizeX;
        
        int blocksCount = (count - 1) / blockSizeY + 1;
        int gridSizeX = Math.min(CudaEngine.getMaxGridSize(device)[0], blocksCount);
        int gridSizeY = (blocksCount - 1) / gridSizeX + 1;
        
        int iterations = (stride - 1) / blockSizeX + 1;
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(a),
            Pointer.to(b),
            Pointer.to(new int[]{stride}),
            Pointer.to(new int[]{blockSizeY}),
            Pointer.to(new int[]{iterations}),
            Pointer.to(new long[]{(long)count * stride})
        );
        
//        System.out.printf("%d %d %d %d\n", blockSizeX, blockSizeY, gridSizeX, gridSizeY);
        
//        int gridSizeY = (count - 1) / stridesPerBlock + 1;
//        int gridSizeY = Math.min(CudaEngine.getMaxGridSize(device)[1], count);
//        int gridSizeZ = (count - 1) / gridSizeY + 1;
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, gridSizeY, 1,            // Grid dimension
            blockSizeX, blockSizeY, 1,          // Block dimension
            0, stream,                          // Shared memory size and stream
            kernelParameters, null              // Kernel- and extra parameters
        );
    }
    
    public static void addStrideOld(CUdeviceptr a, CUdeviceptr b, int stride, int count, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACCUMULATE, "add_stride_old", device);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(a),
            Pointer.to(b),
            Pointer.to(new long[]{stride}),
            Pointer.to(new long[]{stride * count})
        );
        
        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(device), stride);
        int gridSizeX = (stride - 1) / blockSizeX + 1;
        int gridSizeY = Math.min(CudaEngine.getMaxGridSize(device)[1], count);
        int gridSizeZ = (count - 1) / gridSizeY + 1;
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, gridSizeY, gridSizeZ,    // Grid dimension
            blockSizeX, 1, 1,                   // Block dimension
            0, stream,                          // Shared memory size and stream
            kernelParameters, null              // Kernel- and extra parameters
        );
    }
    
    public static void ReLU(CUdeviceptr ptr, int size, int threadsPerBlock, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();

        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "ReLU", device);

        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr),
            Pointer.to(new long[]{size})
        );
        
        int blockSizeX = Math.min(threadsPerBlock, size);
        int gridSizeX = (size - 1) / blockSizeX + 1;

        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, 1, 1,       // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, stream,             // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
    }
    
    public static void LeakyReLU(CUdeviceptr ptr, int size, float alpha, int threadsPerBlock, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();

        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "LeakyReLU", device);

        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr),
            Pointer.to(new long[]{size}),
            Pointer.to(new float[]{alpha})
        );
        
        int blockSizeX = Math.min(threadsPerBlock, size);
        int gridSizeX = (size - 1) / blockSizeX + 1;

        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, 1, 1,       // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, stream,             // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
    }
    
    public static void Sigmoid(CUdeviceptr ptr, int size, int threadsPerBlock, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "Sigmoid", device);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr),
            Pointer.to(new long[]{size})
        );
        
        int blockSizeX = Math.min(threadsPerBlock, size);
        int gridSizeX = (size - 1) / blockSizeX + 1;
        
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, 1, 1,       // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, stream,             // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
    }
    
    public static void crossoverMutate(CUdeviceptr a, CUdeviceptr b, CUdeviceptr r, int size, float min, float max, double mutation,
                                       CUdeviceptr rng_crossover, CUdeviceptr rng_mutate, CUdeviceptr rng_pool, 
                                       int threadsPerBlock, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_GENETIC, "cross_over_mutate", device);
        
        int blockSizeX = Math.min(threadsPerBlock, size);
        int gridSizeX = (size - 1) / blockSizeX + 1;

        Pointer kernelParameters = Pointer.to(
            Pointer.to(a),
            Pointer.to(b),
            Pointer.to(r),
            Pointer.to(new long[]{size}),
            Pointer.to(new float[]{min}),
            Pointer.to(new float[]{max}),
            Pointer.to(new double[]{mutation}),
            Pointer.to(rng_crossover),
            Pointer.to(rng_mutate),
            Pointer.to(rng_pool)
        );        
                
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, 1, 1,       // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, stream,             // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
    }

    /* utility functions */
    public static float sum_abs_differenceGPU(CUdeviceptr array1, CUdeviceptr array2, int size, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        int threadsPerBlock = CudaEngine.getMaxThreadsPerBlock(device);
        CUfunction matrixMulVector = CudaEngine.getKernel(CudaModule.MODULE_ACCUMULATE, "sum_abs_difference", device);
        int blockSizeX = Math.min(threadsPerBlock, size);
        int gridSizeX = (size - 1) / (blockSizeX) + 1;
        gridSizeX = (int) Math.max(1, Math.ceil(Math.sqrt(gridSizeX)));
        CUdeviceptr result = CudaUtil.createFloat(gridSizeX);
        Pointer kernelParameters = Pointer.to(Pointer.to(array1), Pointer.to(array2), Pointer.to(new long[]{size}), Pointer.to(result));
        JCudaDriver.cuLaunchKernel(matrixMulVector, gridSizeX, 1, 1, blockSizeX, 1, 1, // Block dimension
        0, stream, kernelParameters, null // Kernel- and extra parameters
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
        int gridSizeX = (size - 1) / (blockSizeX * 2) + 1;
        gridSizeX = (int) Math.max(1, Math.ceil(Math.sqrt(gridSizeX)));
        CUdeviceptr result = CudaUtil.createFloat(gridSizeX);
        Pointer kernelParameters = Pointer.to(Pointer.to(array), Pointer.to(new long[]{size}), Pointer.to(result));
        JCudaDriver.cuLaunchKernel(matrixMulVector, gridSizeX, 1, 1, blockSizeX, 1, 1, // Block dimension
        0, stream, kernelParameters, null // Kernel- and extra parameters
        );
        float sum;
        if (gridSizeX > 1) {
            sum = sumGPU(result, gridSizeX, stream);
        } else {
            sum = CudaUtil.fromGPUFloat(result, 1, stream)[0];
        }
        JCudaDriver.cuMemFree(result);
        return sum;
    }
}
