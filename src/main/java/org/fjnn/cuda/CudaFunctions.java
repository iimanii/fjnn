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
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.convolution.output.ConvolutionForwardOutputGPU;
import org.fjnn.convolution.output.layer.ConvolutionLayerForwardOutputGPU;
import org.fjnn.convolution.output.unit.ConvolutionUnitForwardOutputGPU;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 */
public class CudaFunctions {
    
    public static class vector {
        /* a[i] *= factor */
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
        
        /* c[i] = a[i] + b[i] */
        public static void add(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_VECTOR, "add", device);

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
        
        /* c[i] = a[i] + alpha * b[i] */
        public static void add(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, float alpha, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_VECTOR, "add_multiply", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(c),
                Pointer.to(new float[]{alpha}),
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
        
        /* c[i] = a[i] * b[i] */
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
        
        /*
         * stride_base = stride_id * stride_len -> stride_id in [0, batchSize)
         * a[i + stride_base] += b[i]
         */
        public static void addStride(CUdeviceptr a, CUdeviceptr b, int stride, int batchSize, CUstream stream) {
            addStride(a, b, a, stride, batchSize, stream);
        }
        
        /*
         * stride_base = stride_id * stride_len -> stride_id in [0, batchSize) 
         * c[x + stride_base] = a[x + stride_base] + b[x]
         */
        public static void addStride(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, long stride, int batchSize, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_VECTOR, "add_stride", device);

            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;
            
            int blockSizeX = (int) Math.min(threadsPerBlock, stride);
            int blockSizeY = threadsPerBlock / blockSizeX;
            int gridSizeX = (int)((stride - 1) / threadsPerBlock + 1);

            int blocksCount = (int)((batchSize - 1) / blockSizeY + 1);
            int gridSizeY = Math.min(CudaEngine.getMaxGridSize(device)[1], blocksCount);
            int gridSizeZ = (blocksCount - 1) / gridSizeY + 1;

            Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(c),
                Pointer.to(new long[]{stride}),
                Pointer.to(new long[]{(long)batchSize * stride})
            );

            JCudaDriver.cuLaunchKernel(function,
                gridSizeX, gridSizeY, gridSizeZ,    // Grid dimension
                blockSizeX, blockSizeY, 1,          // Block dimension
                0, stream,                          // Shared memory size and stream
                kernelParameters, null              // Kernel- and extra parameters
            );
        }
        
        /*
         * stride_base = stride_id * stride_len -> stride_id in [0, count)
         * c[i + stride_base] = alpha * a[i + stride_base] * b[i]
         */
        public static void multiplyStride(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, float alpha, long stride, long count, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_VECTOR, "multiply_stride", device);
            
            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;
            
            int blockSizeX = (int) Math.min(threadsPerBlock, stride);
            int blockSizeY = threadsPerBlock / blockSizeX;
            int gridSizeX = (int)((stride - 1) / threadsPerBlock + 1);
            int blocksCount = (int)((count - 1) / blockSizeY + 1);
            
            int gridSizeY = Math.min(CudaEngine.getMaxGridSize(device)[1], blocksCount);
            int gridSizeZ = (blocksCount - 1) / gridSizeY + 1;

            Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(c),
                Pointer.to(new float[]{alpha}),
                Pointer.to(new long[]{stride}),
                Pointer.to(new long[]{count * stride})
            );

            JCudaDriver.cuLaunchKernel(function,
                gridSizeX, gridSizeY, gridSizeZ,
                blockSizeX, blockSizeY, 1,
                0, stream,
                kernelParameters, null
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
        
        
        public static void reduceStride2(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, long sizeA, long sizeB, CUstream stream) {
            if(sizeA < sizeB)
                throw new RuntimeException("Size A must be larger or equal to Size B");
            
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_VECTOR, "optimized_reduce_stride", device);
            
                        
            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, sizeA);
            long gridSizeX = (sizeA - 1) / blockSizeX + 1;
            long iterationsPerBlock = 1;
            
            if(gridSizeX > 4096) {
                gridSizeX = 4096;
                iterationsPerBlock = (sizeA - 1) / (blockSizeX * gridSizeX) + 1;
                
//                System.out.println("new grid size: " + gridSizeX + " " + iterationsPerBlock);
            }
                    
            if(gridSizeX > Integer.MAX_VALUE)
                throw new RuntimeException();
            
            Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(c),
                Pointer.to(new long[]{sizeA}),
                Pointer.to(new long[]{sizeB}),
                Pointer.to(new long[]{iterationsPerBlock})
            );
            
            // Launch the reduceStride kernel
            JCudaDriver.cuLaunchKernel(function,
            (int)gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, stream,             // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
            );
        }

        public static void testReduceStride(CUdeviceptr dest, CUdeviceptr src, long stride, int count, CUstream stream) {
            if(count == 1) {                
                JCudaDriver.cuMemcpyAsync(dest, src, stride * CudaUtil.FLOAT_SIZE, stream);
                return;
            } else if(count == 2) {
                reduceStride2(src, src.withByteOffset(stride * CudaUtil.FLOAT_SIZE), dest, stride, stride, stream);
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
                reduceStride(temp, temp.withByteOffset(stride * half * CudaUtil.FLOAT_SIZE), temp, stride * half, stride * (curr - half), stream);
//                reduceStrideInPlace(temp, temp.withByteOffset(stride * half * CudaUtil.FLOAT_SIZE), stride * (curr - half), stream);      
                curr = half;
            }
            
            reduceStride(temp, temp.withByteOffset(stride * CudaUtil.FLOAT_SIZE), dest, stride, stride, stream);
            
            CudaUtil.freeAsync(temp, stream);
        }

        public static void threshold(CUdeviceptr mask, float rate, int size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_VECTOR, "threshold", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(mask),
                Pointer.to(new float[]{rate}),
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

        public static void addScalar(CUdeviceptr array, float scalar, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_VECTOR, "add_scalar", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(array),
                Pointer.to(new float[]{scalar}),
                Pointer.to(new long[]{size})
            );

            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, size);
            long gridSizeX = (size - 1) / blockSizeX + 1;

            if(gridSizeX > Integer.MAX_VALUE)
                throw new RuntimeException();

            JCudaDriver.cuLaunchKernel(function,
                (int)gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,          // Block dimension
                0, stream,                 // Shared memory size and stream
                kernelParameters, null     // Kernel- and extra parameters
            );
        }
    }
    
    public static class convolution {
        /**
         * Transforms input data for 1D convolution using im2col on GPU.
         * 
         * @param input The input data on GPU
         * @param inputDim The dimension of each input (sequence length)
         * @param kernelWidth
         * @param unitSize
         * @param outputDim
         * @param batchSize Number of batches in the input
         * @param stream CUDA stream
         * @return The transformed im2col matrix on GPU
         */
        public static CUdeviceptr im2col(CUdeviceptr input, int inputDim, int kernelWidth, int unitSize, int outputDim, int batchSize, CUstream stream) {
            int totalSize = batchSize * outputDim * kernelWidth;

            // Allocate output memory on GPU
            CUdeviceptr im2colOutput = CudaUtil.createFloatAsync(totalSize, stream);

            // Launch custom CUDA kernel to perform im2col operation
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_CONVOLUTION, "im2col", device);

            // Set up kernel parameters
            Pointer kernelParams = Pointer.to(
                Pointer.to(input),
                Pointer.to(im2colOutput),
                Pointer.to(new int[]{inputDim}),
                Pointer.to(new int[]{outputDim}),
                Pointer.to(new int[]{kernelWidth}),
                Pointer.to(new int[]{unitSize}),
                Pointer.to(new int[]{batchSize})
            );

            // Each thread processes one element in the output matrix
            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;
            int blocksPerGrid = (totalSize + threadsPerBlock - 1) / threadsPerBlock;

            JCudaDriver.cuLaunchKernel(function,
                                     blocksPerGrid, 1, 1,         // Grid dimensions
                                     threadsPerBlock, 1, 1,       // Block dimensions  
                                     0, stream,                   // Shared memory and stream
                                     kernelParams, null);         // Parameters

            return im2colOutput;
        }
        
        /**
        * Extracts channel data from interleaved input for KernelGroup processing
         * @param input
         * @param output
         * @param inputStride
         * @param unitSize
         * @param channelOffset
         * @param numChunks
         * @param batchSize
         * @param stream
        */
        public static void extractChannel(CUdeviceptr input, CUdeviceptr output, 
                                          int inputStride, int unitSize, int channelOffset, 
                                          int totalChunks, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_CONVOLUTION, "extractChannel", device);

            int totalElements = unitSize * totalChunks;
            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, totalElements);
            int gridSizeX = (totalElements - 1) / blockSizeX + 1;

            Pointer kernelParameters = Pointer.to(
                Pointer.to(input),
                Pointer.to(output),
                Pointer.to(new int[]{inputStride}),
                Pointer.to(new int[]{unitSize}),
                Pointer.to(new int[]{channelOffset}),
                Pointer.to(new int[]{totalChunks})
            );

            JCudaDriver.cuLaunchKernel(function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,     // Block dimension
                0, stream,            // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
            );
        }
        
        /**
         * Distributes gradients from a channel back to interleaved input positions
         */
        public static void distributeChannelGradients(CUdeviceptr channelGradients, 
                                                     CUdeviceptr inputGradients,
                                                     int inputStride, 
                                                     int unitSize, 
                                                     int channelOffset, 
                                                     int totalChunks, 
                                                     CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_CONVOLUTION, 
                                                     "distributeChannelGradients", device);

            int totalElements = unitSize * totalChunks;
            int threadsPerBlock = 256;
            int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

            Pointer kernelParams = Pointer.to(
                Pointer.to(channelGradients),
                Pointer.to(inputGradients),
                Pointer.to(new int[]{inputStride}),
                Pointer.to(new int[]{unitSize}),
                Pointer.to(new int[]{channelOffset}),
                Pointer.to(new int[]{totalChunks})
            );

            JCudaDriver.cuLaunchKernel(function,
                                     blocksPerGrid, 1, 1,
                                     threadsPerBlock, 1, 1,
                                     0, stream,
                                     kernelParams, null);
        }
        
        /**
         * Distribute output gradients to input gradients (col2im operation)
         */
        public static void computeInputGradients(CUdeviceptr inputGradients, 
                                                 CUdeviceptr outputGradients, 
                                                 CUdeviceptr weights,
                                                 int unitSize,
                                                 int inputSize, 
                                                 int outputSize, 
                                                 int kernelWidth, 
                                                 int batchSize, 
                                                 CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_CONVOLUTION, 
                                                     "computeInputGradients", device);

            int totalElements = inputSize * batchSize;
            int threadsPerBlock = 256;
            int blocksPerGrid = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

            Pointer kernelParams = Pointer.to(
                Pointer.to(inputGradients),
                Pointer.to(outputGradients),
                Pointer.to(weights),
                Pointer.to(new int[]{unitSize}),
                Pointer.to(new int[]{inputSize}),
                Pointer.to(new int[]{outputSize}),
                Pointer.to(new int[]{kernelWidth}),
                Pointer.to(new int[]{batchSize})
            );

            JCudaDriver.cuLaunchKernel(function,
                                     blocksPerGrid, 1, 1,
                                     threadsPerBlock, 1, 1,
                                     0, stream,
                                     kernelParams, null);
        }
        
        public static void adamUpdateWeights(CUdeviceptr weights,
                                           CUdeviceptr gradients,
                                           CUdeviceptr momentum,
                                           CUdeviceptr velocity,
                                           float learningRate,
                                           float beta1,
                                           float beta2,
                                           float epsilon,
                                           float beta1Power,
                                           float beta2Power,
                                           long size,
                                           CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_CONVOLUTION, "adamUpdateWeights", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(weights),
                Pointer.to(gradients),
                Pointer.to(momentum),
                Pointer.to(velocity),
                Pointer.to(new float[]{learningRate}),
                Pointer.to(new float[]{beta1}),
                Pointer.to(new float[]{beta2}),
                Pointer.to(new float[]{epsilon}),
                Pointer.to(new float[]{beta1Power}),
                Pointer.to(new float[]{beta2Power}),
                Pointer.to(new long[]{size})
            );

            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, size);
            long gridSizeX = (size - 1) / blockSizeX + 1;

            if(gridSizeX > Integer.MAX_VALUE)
                throw new RuntimeException();

            JCudaDriver.cuLaunchKernel(function,
                (int)gridSizeX, 1, 1,
                blockSizeX, 1, 1,
                0, stream,
                kernelParameters, null
            );
        }

        public static void adamUpdateBias(CUdeviceptr bias,
                                        CUdeviceptr gradient,
                                        CUdeviceptr momentum,
                                        CUdeviceptr velocity,
                                        float learningRate,
                                        float beta1,
                                        float beta2,
                                        float epsilon,
                                        float beta1Power,
                                        float beta2Power,
                                        CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_CONVOLUTION, "adamUpdateBias", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(bias),
                Pointer.to(gradient),
                Pointer.to(momentum),
                Pointer.to(velocity),
                Pointer.to(new float[]{learningRate}),
                Pointer.to(new float[]{beta1}),
                Pointer.to(new float[]{beta2}),
                Pointer.to(new float[]{epsilon}),
                Pointer.to(new float[]{beta1Power}),
                Pointer.to(new float[]{beta2Power})
            );

            JCudaDriver.cuLaunchKernel(function,
                1, 1, 1,
                1, 1, 1,
                0, stream,
                kernelParameters, null
            );
        }
        
        /**
        * Copy one unit's output to the positional concatenation result with strided updates
        * @param unitOutput      GPU pointer to this unit's output data [unitOutputSize * batchSize]
        * @param result          GPU pointer to the final reshaped output buffer [refOutputSize * numUnits * batchSize]
        * @param unitIdx         Index of this unit (0 to numUnits-1), determines stride offset in result
        * @param numUnits        Total number of convolution units being combined (stride size)
        * @param refFirstInput
        * @param unitFirstInput
        * @param unitOutputSize
        * @param refOutputSize   Number of output values the reference unit produces (limits copy range)
        * @param batchSize       Number of batches being processed
        * @param stream          CUDA stream for asynchronous execution
        */
        public static void copyUnitToPositionalOutput(CUdeviceptr unitOutput,
                                             CUdeviceptr result,
                                             int unitIdx,
                                             int numUnits,
                                             int refFirstInput,
                                             int unitFirstInput,
                                             int unitOutputSize,
                                             int refOutputSize,
                                             int batchSize,
                                             CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_CONVOLUTION, 
                                                      "copyUnitToPositionalOutput", device);

            int totalElements = refOutputSize * batchSize;
            int blockSize = CudaUtil.PREFERRED_BLOCK_SIZE;
            int gridSize = (totalElements + blockSize - 1) / blockSize;

            Pointer kernelParams = Pointer.to(
                Pointer.to(unitOutput),
                Pointer.to(result),
                Pointer.to(new int[]{unitIdx}),
                Pointer.to(new int[]{numUnits}),
                Pointer.to(new int[]{refFirstInput}),
                Pointer.to(new int[]{unitFirstInput}),
                Pointer.to(new int[]{unitOutputSize}),
                Pointer.to(new int[]{refOutputSize}),
                Pointer.to(new int[]{batchSize})
            );

            JCudaDriver.cuLaunchKernel(function, 
                    gridSize, 1, 1, 
                    blockSize, 1, 1, 
                    0, stream, 
                    kernelParams, null);
        }
        
        public static void copyUnitStrided(CUdeviceptr src, int srcDim, int srcStart, 
                                           CUdeviceptr dst, int dstDim, int dstStart,
                                           int stride, int count, int batchSize,
                                           CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_CONVOLUTION, 
                                                      "copyUnitStrided", device);

            int totalElements = count * batchSize;
            int blockSize = CudaUtil.PREFERRED_BLOCK_SIZE;
            int gridSize = (totalElements + blockSize - 1) / blockSize;

            Pointer kernelParams = Pointer.to(
                Pointer.to(src),
                Pointer.to(new int[]{srcDim}),
                Pointer.to(new int[]{srcStart}),
                Pointer.to(dst),
                Pointer.to(new int[]{dstDim}),
                Pointer.to(new int[]{dstStart}),
                Pointer.to(new int[]{stride}),
                Pointer.to(new int[]{count}),
                Pointer.to(new int[]{batchSize})
            );

            JCudaDriver.cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1,
                                     0, stream, kernelParams, null);
        }
        
        public static void copyUnitStridedReverse(CUdeviceptr src, CUdeviceptr dst,
                                                 int srcStart, int dstStart,
                                                 int srcSize, int stride, int dstSize,
                                                 int count, int batchSize,
                                                 CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_CONVOLUTION, 
                                                      "copyUnitStridedReverse", device);

            int totalElements = count * batchSize;
            int blockSize = CudaUtil.PREFERRED_BLOCK_SIZE;
            int gridSize = (totalElements + blockSize - 1) / blockSize;

            Pointer kernelParams = Pointer.to(
                Pointer.to(src),
                Pointer.to(dst),
                Pointer.to(new int[]{srcStart}),
                Pointer.to(new int[]{dstStart}),
                Pointer.to(new int[]{srcSize}),
                Pointer.to(new int[]{stride}),
                Pointer.to(new int[]{dstSize}),
                Pointer.to(new int[]{count}),
                Pointer.to(new int[]{batchSize})
            );

            JCudaDriver.cuLaunchKernel(function, gridSize, 1, 1, blockSize, 1, 1,
                                     0, stream, kernelParams, null);
        }
    }
    
    /* Activation functions */
    public static class activation {
        private static void Activation(String name, CUdeviceptr input, CUdeviceptr output, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, name, device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(input),
                Pointer.to(output),
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

        public static void GeLU(CUdeviceptr input, CUdeviceptr output, long size, CUstream stream) {
            Activation("GeLU", input, output, size, stream);
        }
        
        public static void LeakyReLU(CUdeviceptr input, CUdeviceptr output, long size, float alpha, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "LeakyReLU", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(input),
                Pointer.to(output),
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

        public static void ReLU(CUdeviceptr input, CUdeviceptr output, long size, CUstream stream) {
            Activation("ReLU", input, output, size, stream);
        }

        public static void Sigmoid(CUdeviceptr input, CUdeviceptr output, long size, CUstream stream) {
            Activation("Sigmoid", input, output, size, stream);
        }

        public static void Sin(CUdeviceptr input, CUdeviceptr output, long size, CUstream stream) {
            Activation("Sin", input, output, size, stream);
        }

        public static void SoftMax(CUdeviceptr input, CUdeviceptr output, int inputDim, int batchSize, CUstream stream) {
            if(inputDim > Integer.MAX_VALUE || batchSize > Integer.MAX_VALUE)
                throw new RuntimeException();

            int device = CudaEngine.getThreadDeviceId();
            int blockSize;
            
            if(inputDim <= 32)
                blockSize = 32;
            else if(inputDim <= 64)
                blockSize = 64;
            else if(inputDim <= 128)
                blockSize = 128;
            else if(inputDim <= 256)
                blockSize = 256;
            else
                blockSize = 512;

            /* Compute Matrix Multiplication */
            CUfunction fn = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, true, String.format("SoftMax_%d", blockSize), device);

            int blockSizeX = blockSize;
            int gridSizeX = batchSize;

            Pointer kernelParameters = Pointer.to(
                Pointer.to(input),
                Pointer.to(output),
                Pointer.to(new long[]{inputDim})
            );

            JCudaDriver.cuLaunchKernel(fn,
                gridSizeX, 1, 1,            // Grid dimension
                blockSizeX, 1, 1,           // Block dimension
                0, stream,                  // Shared memory size and stream
                kernelParameters, null      // Kernel- and extra parameters
            );
        }

        public static void Step(CUdeviceptr input, CUdeviceptr output, long size, CUstream stream) {
            Activation("Step", input, output, size, stream);
        }

        public static void Swish(CUdeviceptr input, CUdeviceptr output, long size, CUstream stream) {
            Activation("Swish", input, output, size, stream);
        }

        public static void Tanh(CUdeviceptr input, CUdeviceptr output, long size, CUstream stream) {
            Activation("Tanh", input, output, size, stream);
        }
    }
    
    /* Activation derivative functions */
    public static class activationDerivative {
        private static void ActivationDerivative(String name, CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, name, device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(preActivation),
                Pointer.to(postActivation),
                Pointer.to(output),
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
        
        public static void GeLUDerivative(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, long size, CUstream stream) {
            ActivationDerivative("GeLUDerivative", preActivation, postActivation, output, size, stream);
        }
        
        public static void LeakyReLUDerivative(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, long size, float alpha, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "LeakyReLUDerivative", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(preActivation),
                Pointer.to(postActivation),
                Pointer.to(output),
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
        
        public static void LinearDerivative(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, long size, CUstream stream) {
            JCudaDriver.cuMemsetD32Async(output, Float.floatToRawIntBits(1.0f), size, stream);
        }
        
        public static void ReLUDerivative(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, long size, CUstream stream) {
            ActivationDerivative("ReLUDerivative", preActivation, postActivation, output, size, stream);
        }

        public static void SinDerivative(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, long size, CUstream stream) {
            ActivationDerivative("SinDerivative", preActivation, postActivation, output, size, stream);
        }
        
        public static void SoftMaxDerivative(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, long stride, long count, CUstream stream) {
            if(count > Integer.MAX_VALUE || stride > Integer.MAX_VALUE)
                throw new RuntimeException();

            int device = CudaEngine.getThreadDeviceId();
            int blockSizeX;
            
            if(stride <= 32)
                blockSizeX = 32;
            else if(stride <= 64)
                blockSizeX = 64;
            else if(stride <= 128)
                blockSizeX = 128;
            else if(stride <= 256)
                blockSizeX = 256;
            else
                blockSizeX = 512;

            /* Compute Matrix Multiplication */
            CUfunction fn = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, true, String.format("SoftMaxDerivative", blockSizeX), device);
            int gridSizeX = (int)count;

            Pointer kernelParameters = Pointer.to(
                Pointer.to(preActivation),
                Pointer.to(postActivation),
                Pointer.to(output),
                Pointer.to(new long[]{stride})
            );

            JCudaDriver.cuLaunchKernel(fn,
                gridSizeX, 1, 1,            // Grid dimension
                blockSizeX, 1, 1,           // Block dimension
                0, stream,                  // Shared memory size and stream
                kernelParameters, null      // Kernel- and extra parameters
            );
        }
        
        public static void SigmoidDerivative(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, long size, CUstream stream) {
            ActivationDerivative("SigmoidDerivative", preActivation, postActivation, output, size, stream);
        }

        public static void SwishDerivative(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, long size, CUstream stream) {
            ActivationDerivative("SwishDerivative", preActivation, postActivation, output, size, stream);
        }

        public static void TanhDerivative(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, long size, CUstream stream) {
            ActivationDerivative("TanhDerivative", preActivation, postActivation, output, size, stream);
        }
    }
    
    public static class activationGradient {
        private static void ActivationGradient(String name, CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, name, device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(preActivation),
                Pointer.to(postActivation),
                Pointer.to(gradient),
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
        
        public static void GeLUGradient(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, long size, CUstream stream) {
            ActivationGradient("GeLUGradient", preActivation, postActivation, gradient, size, stream);
        }
        
        public static void LeakyReLUGradient(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, long size, float alpha, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "LeakyReLUGradient", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(preActivation),
                Pointer.to(postActivation),
                Pointer.to(gradient),
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
                
        public static void ReLUGradient(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, long size, CUstream stream) {
            ActivationGradient("ReLUGradient", preActivation, postActivation, gradient, size, stream);
        }

        public static void SigmoidGradient(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, long size, CUstream stream) {
            ActivationGradient("SigmoidGradient", preActivation, postActivation, gradient, size, stream);
        }
        
        public static void SigmoidBinaryCrossEntropyGradient(CUdeviceptr postActivation, 
                                                       CUdeviceptr truth, 
                                                       CUdeviceptr gradient, 
                                                       float alpha, float beta,
                                                       long size, 
                                                       CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "SigmoidBinaryCrossEntropyGradient", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(postActivation),
                Pointer.to(truth),
                Pointer.to(gradient),
                Pointer.to(new float[]{alpha}),
                Pointer.to(new float[]{beta}),                
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
        
        public static void SinGradient(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, long size, CUstream stream) {
            ActivationGradient("SinGradient", preActivation, postActivation, gradient, size, stream);
        }
        
        public static void SoftMaxGradient(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, long stride, long count, CUstream stream) {
            if(count > Integer.MAX_VALUE || stride > Integer.MAX_VALUE)
                throw new RuntimeException();

            int device = CudaEngine.getThreadDeviceId();
            int blockSize;
            
            if(stride <= 32)
                blockSize = 32;
            else if(stride <= 64)
                blockSize = 64;
            else if(stride <= 128)
                blockSize = 128;
            else if(stride <= 256)
                blockSize = 256;
            else
                blockSize = 512;

            /* Compute Matrix Multiplication */
            CUfunction fn = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, true, String.format("SoftMaxGradient_%d", blockSize), device);

            int blockSizeX = blockSize;
            int gridSizeX = (int)count;

            Pointer kernelParameters = Pointer.to(
                Pointer.to(preActivation),
                Pointer.to(postActivation),
                Pointer.to(gradient),
                Pointer.to(new long[]{stride})
            );

            JCudaDriver.cuLaunchKernel(fn,
                gridSizeX, 1, 1,            // Grid dimension
                blockSizeX, 1, 1,           // Block dimension
                0, stream,                  // Shared memory size and stream
                kernelParameters, null      // Kernel- and extra parameters
            );
        }
        
        public static void SoftMaxCrossEntropyGradient(CUdeviceptr postActivation, CUdeviceptr truth, CUdeviceptr gradient, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_ACTIVATION, "SoftMaxCrossEntropyGradient", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(postActivation),
                Pointer.to(truth),
                Pointer.to(gradient),
                Pointer.to(new long[]{size})
            );

            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, size);
            long gridSizeX = (size - 1) / blockSizeX + 1;

            if(gridSizeX > Integer.MAX_VALUE)
                throw new RuntimeException();

            JCudaDriver.cuLaunchKernel(function,
                (int)gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,          // Block dimension
                0, stream,                 // Shared memory size and stream
                kernelParameters, null     // Kernel- and extra parameters
            );          
        }

        public static void SwishGradient(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, long size, CUstream stream) {
            ActivationGradient("SwishGradient", preActivation, postActivation, gradient, size, stream);
        }
        
        public static void TanhGradient(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, long size, CUstream stream) {
            ActivationGradient("TanhGradient", preActivation, postActivation, gradient, size, stream);
        }
    }
    
    public static class normalization {
        public static void LayerNormalizer(CUdeviceptr preNormalization, 
                                           CUdeviceptr normalized,
                                           CUdeviceptr postNormalization, 
                                           CUdeviceptr stds,
                                           CUdeviceptr gamma, CUdeviceptr beta,
                                           long stride, long count, CUstream stream) {
            if(count > Integer.MAX_VALUE || stride > Integer.MAX_VALUE)
                throw new RuntimeException();

            int device = CudaEngine.getThreadDeviceId();
            int blockSize;
            
            if(stride <= 32)
                blockSize = 32;
            else if(stride <= 64)
                blockSize = 64;
            else if(stride <= 128)
                blockSize = 128;
            else if(stride <= 256)
                blockSize = 256;
            else
                blockSize = 512;

            /* Compute Matrix Multiplication */
            CUfunction fn = CudaEngine.getKernel(CudaModule.MODULE_NORMALIZER, true, String.format("LayerNormalizer_%d", blockSize), device);

            int blockSizeX = blockSize;
            int gridSizeX = (int)count;

            Pointer kernelParameters = Pointer.to(
                Pointer.to(preNormalization),
                Pointer.to(normalized),
                Pointer.to(postNormalization),
                Pointer.to(stds),
                Pointer.to(gamma),
                Pointer.to(beta),
                Pointer.to(new long[]{stride})
            );

            JCudaDriver.cuLaunchKernel(fn,
                gridSizeX, 1, 1,            // Grid dimension
                blockSizeX, 1, 1,           // Block dimension
                0, stream,                  // Shared memory size and stream
                kernelParameters, null      // Kernel- and extra parameters
            );
        }

        public static void LayerNormalizerBackpropagate(CUdeviceptr normalized,
                                                        CUdeviceptr stds, 
                                                        CUdeviceptr gamma, 
                                                        CUdeviceptr gradients, 
                                                        CUdeviceptr deltaLoss, 
                                                        int stride, int count, CUstream stream) {
            if(count > Integer.MAX_VALUE || stride > Integer.MAX_VALUE)
                throw new RuntimeException();

            int device = CudaEngine.getThreadDeviceId();
            int blockSize;
            
            if(stride <= 32)
                blockSize = 32;
            else if(stride <= 64)
                blockSize = 64;
            else if(stride <= 128)
                blockSize = 128;
            else if(stride <= 256)
                blockSize = 256;
            else
                blockSize = 512;

            /* Compute Matrix Multiplication */
            CUfunction fn = CudaEngine.getKernel(CudaModule.MODULE_NORMALIZER, true, String.format("LayerNormalizerDerivative_%d", blockSize), device);

            int blockSizeX = blockSize;
            int gridSizeX = (int)count;

            Pointer kernelParameters = Pointer.to(
                Pointer.to(normalized),
                Pointer.to(stds),
                Pointer.to(gamma),
                Pointer.to(gradients),
                Pointer.to(deltaLoss),
                Pointer.to(new long[]{stride})
            );

            JCudaDriver.cuLaunchKernel(fn,
                gridSizeX, 1, 1,            // Grid dimension
                blockSizeX, 1, 1,           // Block dimension
                0, stream,                  // Shared memory size and stream
                kernelParameters, null      // Kernel- and extra parameters
            );
        }
        
    }
    
    public static class connection {
        /* Weights */
        public static void updateWeightsWithDecay(CUdeviceptr weights, 
                                                  CUdeviceptr gradients, 
                                                  float learningRate, 
                                                  float weightDecay, 
                                                  long size, 
                                                  CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_CONNECTION, "updateWeightsWithDecay", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(weights),
                Pointer.to(gradients),
                Pointer.to(new float[]{learningRate}),
                Pointer.to(new float[]{weightDecay}),
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
        
        public static void adamUpdateConnectionWeights(CUdeviceptr weights,
                                                       CUdeviceptr gradients,
                                                       CUdeviceptr momentum,
                                                       CUdeviceptr velocity,
                                                       float learningRate,
                                                       float beta1,
                                                       float beta2,
                                                       float epsilon,
                                                       float beta1Power,
                                                       float beta2Power,
                                                       float weightDecay,
                                                       long size,
                                                       CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_CONNECTION, "adamUpdateConnectionWeights", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(weights),
                Pointer.to(gradients),
                Pointer.to(momentum),
                Pointer.to(velocity),
                Pointer.to(new float[]{learningRate}),
                Pointer.to(new float[]{beta1}),
                Pointer.to(new float[]{beta2}),
                Pointer.to(new float[]{epsilon}),
                Pointer.to(new float[]{beta1Power}),
                Pointer.to(new float[]{beta2Power}),
                Pointer.to(new float[]{weightDecay}),
                Pointer.to(new long[]{size})
            );

            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, size);
            long gridSizeX = (size - 1) / blockSizeX + 1;

            if(gridSizeX > Integer.MAX_VALUE)
                throw new RuntimeException();

            JCudaDriver.cuLaunchKernel(function,
                (int)gridSizeX, 1, 1,
                blockSizeX, 1, 1,
                0, stream,
                kernelParameters, null
            );
        }

        public static void adamUpdateConnectionBiases(CUdeviceptr biases,
                                                      CUdeviceptr gradients,
                                                      CUdeviceptr momentum,
                                                      CUdeviceptr velocity,
                                                      float learningRate,
                                                      float beta1,
                                                      float beta2,
                                                      float epsilon,
                                                      float beta1Power,
                                                      float beta2Power,
                                                      long size,
                                                      CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_CONNECTION, "adamUpdateConnectionBiases", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(biases),
                Pointer.to(gradients),
                Pointer.to(momentum),
                Pointer.to(velocity),
                Pointer.to(new float[]{learningRate}),
                Pointer.to(new float[]{beta1}),
                Pointer.to(new float[]{beta2}),
                Pointer.to(new float[]{epsilon}),
                Pointer.to(new float[]{beta1Power}),
                Pointer.to(new float[]{beta2Power}),
                Pointer.to(new long[]{size})
            );

            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, size);
            long gridSizeX = (size - 1) / blockSizeX + 1;

            if(gridSizeX > Integer.MAX_VALUE)
                throw new RuntimeException();

            JCudaDriver.cuLaunchKernel(function,
                (int)gridSizeX, 1, 1,
                blockSizeX, 1, 1,
                0, stream,
                kernelParameters, null
            );
        }
    }
    
    
    /* Loss Functions */
    public static class loss {
        public static void MeanSquareError(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_LOSS, "MeanSquareError", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(output),
                Pointer.to(expected),
                Pointer.to(result),
                Pointer.to(new long[]{size})
            );

            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;

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
        
        public static void MeanSquareErrorDerivative(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_LOSS, "MeanSquareErrorDerivative", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(output),
                Pointer.to(expected),
                Pointer.to(result),
                Pointer.to(new long[]{size})
            );

            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;

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
        
        public static void BinaryCrossEntropy(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, float alpha, float beta, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_LOSS, "BinaryCrossEntropy", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(output),
                Pointer.to(expected),
                Pointer.to(result),
                Pointer.to(new float[]{alpha}),
                Pointer.to(new float[]{beta}),
                Pointer.to(new long[]{size})
            );

            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;

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

        public static void BinaryCrossEntropyDerivative(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, float alpha, float beta, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_LOSS, "BinaryCrossEntropyDerivative", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(output),
                Pointer.to(expected),
                Pointer.to(result),
                Pointer.to(new float[]{alpha}),
                Pointer.to(new float[]{beta}),
                Pointer.to(new long[]{size})
            );

            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;

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

        public static void WeightedMeanSquareError(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr weights, CUdeviceptr result, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_LOSS, "WeightedMeanSquareError", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(output),
                Pointer.to(expected),
                Pointer.to(weights),
                Pointer.to(result),
                Pointer.to(new long[]{size})
            );

            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;

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

        public static void WeightedMeanSquareErrorDerivative(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr weights, CUdeviceptr result, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_LOSS, "WeightedMeanSquareErrorDerivative", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(output),
                Pointer.to(expected),
                Pointer.to(weights),
                Pointer.to(result),
                Pointer.to(new long[]{size})
            );

            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;

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

        public static void FocalLoss(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, float gamma, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_LOSS, "FocalLoss", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(output),
                Pointer.to(expected),
                Pointer.to(result),
                Pointer.to(new float[]{gamma}),
                Pointer.to(new long[]{size})
            );

            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;

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

        public static void FocalLossDerivative(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, float gamma, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_LOSS, "FocalLossDerivative", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(output),
                Pointer.to(expected),
                Pointer.to(result),
                Pointer.to(new float[]{gamma}),
                Pointer.to(new long[]{size})
            );

            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;

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
        
        public static void CrossEntropy(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_LOSS, "CrossEntropy", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(output),
                Pointer.to(expected),
                Pointer.to(result),
                Pointer.to(new long[]{size})
            );

            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, size);
            long gridSizeX = (size - 1) / blockSizeX + 1;

            if(gridSizeX > Integer.MAX_VALUE)
                throw new RuntimeException();

            JCudaDriver.cuLaunchKernel(function,
                (int)gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,          // Block dimension
                0, stream,                 // Shared memory size and stream
                kernelParameters, null     // Kernel- and extra parameters
            );
        }
        
        public static void CrossEntropyDerivative(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();

            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_LOSS, "CrossEntropyDerivative", device);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(output),
                Pointer.to(expected),
                Pointer.to(result),
                Pointer.to(new long[]{size})
            );

            int blockSizeX = (int) Math.min(CudaUtil.PREFERRED_BLOCK_SIZE, size);
            long gridSizeX = (size - 1) / blockSizeX + 1;

            if(gridSizeX > Integer.MAX_VALUE)
                throw new RuntimeException();

            JCudaDriver.cuLaunchKernel(function,
                (int)gridSizeX, 1, 1,      // Grid dimension
                blockSizeX, 1, 1,          // Block dimension
                0, stream,                 // Shared memory size and stream
                kernelParameters, null     // Kernel- and extra parameters
            );
        }
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
    
    public static class playground {
        /**
         * Simple 1D version: c[x + stride * i] = a[x + stride * i] + b[x]
         * Uses 1D grid, no shared memory.
         */
        public static void addStride1DSimple(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, long stride, long count, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_TEST, "add_stride_1d_simple", device);

            long totalElements = stride * count;
            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;
            long gridSize = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

            if (gridSize > Integer.MAX_VALUE) {
                throw new RuntimeException("Grid size too large: " + gridSize);
            }

            Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(c),
                Pointer.to(new long[]{stride}),
                Pointer.to(new long[]{totalElements})
            );

            JCudaDriver.cuLaunchKernel(function,
                (int)gridSize, 1, 1,         // Grid dimension
                threadsPerBlock, 1, 1,       // Block dimension
                0, stream,                   // Shared memory size and stream
                kernelParameters, null       // Kernel parameters
            );
        }

        /**
         * Shared memory 1D version: c[x + stride * i] = a[x + stride * i] + b[x]
         * Uses 1D grid with shared memory for array b.
         */
        public static void addStride1DShared(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, long stride, long count, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_TEST, "add_stride_1d_shared", device);

            long totalElements = stride * count;
            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;
            long gridSize = (totalElements + threadsPerBlock - 1) / threadsPerBlock;

            if (gridSize > Integer.MAX_VALUE) {
                throw new RuntimeException("Grid size too large: " + gridSize);
            }

            int sharedMemSize = (int)(stride * CudaUtil.FLOAT_SIZE);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(c),
                Pointer.to(new long[]{stride}),
                Pointer.to(new long[]{totalElements})
            );

            JCudaDriver.cuLaunchKernel(function,
                (int)gridSize, 1, 1,         // Grid dimension
                threadsPerBlock, 1, 1,       // Block dimension
                sharedMemSize, stream,       // Shared memory size and stream
                kernelParameters, null       // Kernel parameters
            );
        }

        /**
         * Simple 2D version: c[x + stride * i] = a[x + stride * i] + b[x]
         * Uses 2D grid similar to original implementation, no shared memory.
         */
        public static void addStride2DSimple(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, long stride, long count, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_TEST, "add_stride_2d_simple", device);

            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;

            int blockSizeX = Math.min(threadsPerBlock, (int)stride);
            int blockSizeY = threadsPerBlock / blockSizeX;
            int gridSizeX = ((int)stride - 1) / blockSizeX + 1;

            int blocksCount = ((int)count - 1) / blockSizeY + 1;
            int gridSizeY = Math.min(CudaEngine.getMaxGridSize(device)[1], blocksCount);
            int gridSizeZ = (blocksCount - 1) / gridSizeY + 1;

            Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(c),
                Pointer.to(new long[]{stride}),
                Pointer.to(new long[]{count * stride})
            );

            JCudaDriver.cuLaunchKernel(function,
                gridSizeX, gridSizeY, gridSizeZ,    // Grid dimension
                blockSizeX, blockSizeY, 1,          // Block dimension
                0, stream,                          // Shared memory size and stream
                kernelParameters, null              // Kernel parameters
            );
        }

        /**
         * Shared memory 2D version: c[x + stride * i] = a[x + stride * i] + b[x]
         * Uses 2D grid with shared memory for array b.
         */
        public static void addStride2DShared(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, long stride, long count, CUstream stream) {
            int device = CudaEngine.getThreadDeviceId();
            CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_TEST, "add_stride_2d_shared", device);

            int threadsPerBlock = CudaUtil.PREFERRED_BLOCK_SIZE;

            int blockSizeX = Math.min(threadsPerBlock, (int)stride);
            int blockSizeY = threadsPerBlock / blockSizeX;
            int gridSizeX = ((int)stride - 1) / blockSizeX + 1;

            int blocksCount = ((int)count - 1) / blockSizeY + 1;
            int gridSizeY = Math.min(CudaEngine.getMaxGridSize(device)[1], blocksCount);
            int gridSizeZ = (blocksCount - 1) / gridSizeY + 1;

            int sharedMemSize = (int)(stride * CudaUtil.FLOAT_SIZE);

            Pointer kernelParameters = Pointer.to(
                Pointer.to(a),
                Pointer.to(b),
                Pointer.to(c),
                Pointer.to(new long[]{stride}),
                Pointer.to(new long[]{count})
            );

            JCudaDriver.cuLaunchKernel(function,
                gridSizeX, gridSizeY, gridSizeZ,    // Grid dimension
                blockSizeX, blockSizeY, 1,          // Block dimension
                sharedMemSize, stream,              // Shared memory size and stream
                kernelParameters, null              // Kernel parameters
            );
        }
    }

}
