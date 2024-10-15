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
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 */
public class CudaFunctions {
    
    public static class vector {
        public static void copyStride(CUdeviceptr src, CUdeviceptr dst, int stride, int spread, int count, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACCUMULATE, "copy_stride", device);

            int blockSizeX = Math.min(threadsPerBlock, stride);
            int blockSizeY = threadsPerBlock / blockSizeX;
            int gridSizeY  = (stride - 1) / threadsPerBlock + 1;

            int gridSizeX = (count - 1) / blockSizeY + 1;

            Pointer kernelParameters = Pointer.to(
                Pointer.to(src),
                Pointer.to(dst),
                Pointer.to(new int[]{stride}),
                Pointer.to(new int[]{spread}),
                Pointer.to(new int[]{count})
            );

            JCudaDriver.cuLaunchKernel(function,
                gridSizeX, gridSizeY, 1,            // Grid dimension
                blockSizeX, blockSizeY, 1,          // Block dimension
                0, stream,                          // Shared memory size and stream
                kernelParameters, null              // Kernel- and extra parameters
            );
        }

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
        
        public static void scale(CUdeviceptr a, float factor, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_VECTOR, "scale", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(new float[]{factor}),
                Pointer.to(new long[]{size})
            );

            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, size);
            long gridSizeX = (size - 1) / blockSizeX + 1;

            if(gridSizeX > Integer.MAX_VALUE)
                throw new RuntimeException();

            JCudaDriver.cuLaunchKernel(function,
            (int)gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, stream,             // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
            );
        }
    
        public static void multiply(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_VECTOR, "multiply", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(c),
                Pointer.to(new long[]{size})
            );

            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, size);
            long gridSizeX = (size - 1) / blockSizeX + 1;

            if(gridSizeX > Integer.MAX_VALUE)
                throw new RuntimeException();

            JCudaDriver.cuLaunchKernel(function,
            (int)gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, stream,             // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
            );
        }
        
        public static void reduceStride(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, long sizeA, long sizeB, CUstream stream) {
            if(sizeA < sizeB)
                throw new RuntimeException("Size A must be larger or equal to Size B");
            
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_VECTOR, "reduce_stride", device);
            
            Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(c),
                Pointer.to(new long[]{sizeA}),
                Pointer.to(new long[]{sizeB})
            );
                        
            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, sizeA);
            long gridSizeX = (sizeA - 1) / blockSizeX + 1;

            if(gridSizeX > Integer.MAX_VALUE)
                throw new RuntimeException();
            
            // Launch the reduceStride kernel
            JCudaDriver.cuLaunchKernel(function,
            (int)gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, stream,             // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
            );
        }

        public static void reduceStrideInPlace(CUdeviceptr a, CUdeviceptr b, long sizeB, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_VECTOR, "reduce_stride_in_place", device);
            
            Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(new long[]{sizeB})
            );
                        
            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, sizeB);
            long gridSizeX = (sizeB - 1) / blockSizeX + 1;

            if(gridSizeX > Integer.MAX_VALUE)
                throw new RuntimeException();
            
            // Launch the reduceStride kernel
            JCudaDriver.cuLaunchKernel(function,
            (int)gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, stream,             // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
            );
        }
        
        public static void reduceSum(CUdeviceptr dest, CUdeviceptr src, long stride, int count, CUstream stream) {
            if(count == 1) {                
                JCudaDriver.cuMemcpyAsync(dest, src, stride * CudaUtil.FLOAT_SIZE, stream);
                return;
            } else if(count == 2) {
                reduceStride(src, src.withByteOffset(stride * CudaUtil.FLOAT_SIZE), dest, stride, stride, stream);
                return;
            }
            
            /* create temp memory to reduce */
            int curr = (count + 1) / 2;
            CUdeviceptr temp = CudaUtil.createFloatAsync(stride * curr, stream);
            
            // Initial reduction: src -> temp
            reduceStride(src, src.withByteOffset(stride * curr * CudaUtil.FLOAT_SIZE), temp, stride * curr, stride * (count - curr), stream);
            
            // Subsequent reductions: temp -> temp
            while(curr > 2) {
                int half = (curr + 1) / 2;
                reduceStrideInPlace(temp, temp.withByteOffset(stride * half * CudaUtil.FLOAT_SIZE), stride * (curr - half), stream);                
                curr = half;
            }
            
            reduceStride(temp, temp.withByteOffset(stride * CudaUtil.FLOAT_SIZE), dest, stride, stride, stream);
            
            CudaUtil.freeAsync(temp, stream);
        }
    }
    
    /* Activation functions */
    public static void Activation(String name, CUdeviceptr ptr, long size, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, name, device);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr),
            Pointer.to(new long[]{size})
        );
        
        int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, size);
        long gridSizeX = (size - 1) / blockSizeX + 1;
        
        if(gridSizeX > Integer.MAX_VALUE)
            throw new RuntimeException();
                    
        JCudaDriver.cuLaunchKernel(function,
        (int)gridSizeX, 1, 1,      // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, stream,             // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
    }
    
    public static void ReLU(CUdeviceptr ptr, long size, CUstream stream) {
        Activation("ReLU", ptr, size, stream);
    }
    
    public static void LeakyReLU(CUdeviceptr ptr, long size, float alpha, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();

        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "LeakyReLU", device);

        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr),
            Pointer.to(new long[]{size}),
            Pointer.to(new float[]{alpha})
        );
        

        int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, size);
        long gridSizeX = (size - 1) / blockSizeX + 1;
        
        if(gridSizeX > Integer.MAX_VALUE)
            throw new RuntimeException();

        JCudaDriver.cuLaunchKernel(function,
        (int)gridSizeX, 1, 1,      // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, stream,             // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
    }
    
    public static void Sigmoid(CUdeviceptr ptr, long size, CUstream stream) {
        Activation("Sigmoid", ptr, size, stream);
    }
    
    public static void Sin(CUdeviceptr ptr, long size, CUstream stream) {
        Activation("Sin", ptr, size, stream);
    }
    
    public static void SoftMax(CUdeviceptr ptr, long stride, long count, CUstream stream) {
        if(count > Integer.MAX_VALUE || stride > Integer.MAX_VALUE)
            throw new RuntimeException();
        
        int device = CudaEngine.getThreadDeviceId();
        
        int blockSize;
        
        long temp = stride / 1;
        
        if(temp <= 32)
            blockSize = 32;
        else if(temp <= 64)
            blockSize = 64;
        else if(temp <= 128)
            blockSize = 128;
        else if(temp <= 256)
            blockSize = 256;
        else
            blockSize = 512;
        
        /* Compute Matrix Multiplication */
        CUfunction fn = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, true, String.format("SoftMax_%d", blockSize), device);
        
        int blockSizeX = blockSize;
        int gridSizeX = (int)count;
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(ptr),
            Pointer.to(new long[]{stride})
        );

        JCudaDriver.cuLaunchKernel(fn,
            gridSizeX, 1, 1,            // Grid dimension
            blockSizeX, 1, 1,           // Block dimension
            0, stream,                  // Shared memory size and stream
            kernelParameters, null      // Kernel- and extra parameters
        );
    }
    
    public static void Step(CUdeviceptr ptr, long size, CUstream stream) {
        Activation("Step", ptr, size, stream);
    }
    
    public static void Tanh(CUdeviceptr ptr, long size, CUstream stream) {
        Activation("Tanh", ptr, size, stream);
    }
    
    public static void GeLU(CUdeviceptr ptr, long size, CUstream stream) {
        Activation("GeLU", ptr, size, stream);
    }
    
    /* Activation derivative functions */    
    public static void ReLUPrime(CUdeviceptr ptr, long size, CUstream stream) {
        Activation("ReLUPrime", ptr, size, stream);
    }
    
    public static void TanhPrime(CUdeviceptr ptr, long size, CUstream stream) {
        Activation("TanhPrime", ptr, size, stream);
    }
    
    public static void SigmoidPrime(CUdeviceptr ptr, long size, CUstream stream) {
        Activation("SigmoidPrime", ptr, size, stream);
    }
    
    /* Loss Functions */
    public static void MeanSquareErrorPrime(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, int threadsPerBlock, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_LOSS, "MeanSquareErrorPrime", device);
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(output),
            Pointer.to(expected),
            Pointer.to(result),
            Pointer.to(new long[]{size})
        );
        
        int blockSizeX = (int)Math.min(threadsPerBlock, size);
        long gridSizeX = (size - 1) / blockSizeX + 1;
        
        if(gridSizeX > Integer.MAX_VALUE)
            throw new RuntimeException();
        
        JCudaDriver.cuLaunchKernel(function,
        (int)gridSizeX, 1, 1,      // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, stream,             // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
    }
    
    
    /* utility functions */
    public static void MatrixMultiply(CUdeviceptr a, CUdeviceptr b, CUdeviceptr r, int rows_a, int cols_a, int cols_b, float alpha, CUstream stream) {
        boolean unaligned = cols_a % 4 != 0 || cols_b % 4 != 0;
        
        int[] dims = getDimentions(rows_a, cols_a, cols_b, unaligned);
        
        String name = String.format("matrix_mul_matrix_%dx%dx%d%s", dims[0], dims[1], dims[2], unaligned ? "_unaligned" : "");
        String module = unaligned ? CudaModule.MODULE_MATRIX_UNALIGNED : CudaModule.MODULE_MATRIX;
        
        int deviceId = CudaEngine.getThreadDeviceId();
        
        System.out.println(name);
        
        /* Compute Matrix Multiplication */
        CUfunction function = CudaEngine.getKernel(module, name, deviceId);
        
        int blockSizeX = getBlockSize(name);
        int blockSizeY = 1;
        int gridSizeX = (cols_b - 1) / dims[2] + 1;
        int gridSizeY = (rows_a - 1) / dims[0] + 1;

        Pointer kernelParameters = Pointer.to(
            Pointer.to(a),
            Pointer.to(b),
            Pointer.to(r),
            Pointer.to(new int[]{rows_a}),
            Pointer.to(new int[]{cols_a}),
            Pointer.to(new int[]{cols_b}),
            Pointer.to(new float[]{alpha})
        );

        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, gridSizeY, 1,            // Grid dimension
            blockSizeX, blockSizeY, 1,          // Block dimension
            0, stream,                          // Shared memory size and stream
            kernelParameters, null              // Kernel- and extra parameters
        );
    }
    
    public static void CrossoverMutate(CUdeviceptr a, CUdeviceptr b, CUdeviceptr r, long size, float min, float max, double mutation,
                                       CUdeviceptr rng_crossover, CUdeviceptr rng_mutate, CUdeviceptr rng_pool, 
                                       int threadsPerBlock, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_GENETIC, "cross_over_mutate", device);
        
        int blockSizeX = (int) Math.min(threadsPerBlock, size);
        int gridSizeX = (int) ((size - 1) / blockSizeX + 1);

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

    public static void ClipWeights(CUdeviceptr weights, long size, float min, float max, CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_GENETIC, "clip_weights", device);
        
        int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, size);
        int gridSizeX = (int) ((size - 1) / blockSizeX + 1);

        Pointer kernelParameters = Pointer.to(
            Pointer.to(weights),
            Pointer.to(new long[]{size}),
            Pointer.to(new float[]{min}),
            Pointer.to(new float[]{max})
        );        
                
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, 1, 1,       // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, stream,             // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
    }
    
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
            sum = CudaUtil.fromGPUFloatAsync(result, 1, stream)[0];
        }
        JCudaDriver.cuMemFree(result);
        return sum;
    }

    public static int[] getDimentions(int rows_a, int cols_a, int cols_b, boolean unaligned) {
        int b_cols = util.log2(cols_b);
        b_cols = util.clip(b_cols, 3, 7);
        b_cols = (int) Math.pow(2, b_cols);
        
        int a_cols = util.log2(cols_a);
        
        if(unaligned) {
            switch(b_cols) {
                case 8:
                    a_cols = util.clip(a_cols, 3, 7);
                    break;
                case 16:
                    a_cols = util.clip(a_cols, 3, 7);
                    break;
                case 32:
                    a_cols = util.clip(a_cols, 4, 7);
                    break;
                case 64:
                    a_cols = util.clip(a_cols, 5, 7);
                    break;
                case 128:
                    a_cols = util.clip(a_cols, 5, 5);
                    break;
            }
        } else {
            switch(b_cols) {
                case 8:
                    a_cols = util.clip(a_cols, 4, 7);
                    break;
                case 16:
                    a_cols = util.clip(a_cols, 3, 7);
                    break;
                case 32:
                    a_cols = util.clip(a_cols, 3, 7);
                    break;
                case 64:
                    a_cols = util.clip(a_cols, 3, 7);
                    break;
                case 128:
                    a_cols = util.clip(a_cols, 3, 6);
                    break;
            }            
        }
        
        a_cols = (int) Math.pow(2, a_cols);
        return new int[]{select(a_cols, b_cols, unaligned), a_cols, b_cols};
    }
    
    private static int select(int cols_a, int cols_b, boolean unaligned) {
        String hash = String.format("%dx%d", cols_a, cols_b);
        
        if(unaligned) {
            switch(hash) {
                case "128x16":
                    return 16;
                case "128x32":
                    return 8;
                case "128x64":
                    return 8;
                case "128x8":
                    return 16;
                case "16x16":
                    return 8;
                case "16x32":
                    return 8;
                case "16x8":
                    return 16;
                case "32x128":
                    return 8;
                case "32x16":
                    return 8;
                case "32x32":
                    return 8;
                case "32x64":
                    return 8;
                case "32x8":
                    return 16;
                case "64x16":
                    return 8;
                case "64x32":
                    return 8;
                case "64x64":
                    return 8;
                case "64x8":
                    return 16;
                case "8x16":
                    return 8;
                case "8x8":
                    return 16;
            }
        } else {
            switch(hash) {
                case "128x16":
                    return 16;
                case "128x32":
                    return 16;
                case "128x64":
                    return 32;
                case "128x8":
                    return 32;
                case "16x128":
                    return 64;
                case "16x16":
                    return 16;
                case "16x32":
                    return 64;
                case "16x64":
                    return 128;
                case "16x8":
                    return 16;
                case "32x128":
                    return 64;
                case "32x16":
                    return 16;
                case "32x32":
                    return 8;
                case "32x64":
                    return 128;
                case "32x8":
                    return 32;
                case "64x128":
                    return 32;
                case "64x16":
                    return 16;
                case "64x32":
                    return 8;
                case "64x64":
                    return 64;
                case "64x8":
                    return 32;
                case "8x128":
                    return 64;
                case "8x16":
                    return 16;
                case "8x32":
                    return 32;
                case "8x64":
                    return 64;
            }
        }
        
        return 0;
    }
    
    private static int getBlockSize(String name) {
        switch(name) {
            case "matrix_mul_matrix_128x16x16":
            case "matrix_mul_matrix_64x16x16":
            case "matrix_mul_matrix_32x16x16":
            case "matrix_mul_matrix_16x16x16":
                return 64;
            case "matrix_mul_matrix_16x16x16_unaligned":
                return 256;
        }
        
        return 128;
    }
}
