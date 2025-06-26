/*
 * The MIT License
 *
 * Copyright 2025 ahmed.
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
package junit.unit.convolution;

import junit2.convolution.*;
import static org.junit.Assert.*;
import org.junit.BeforeClass;
import org.junit.Before;
import org.junit.After;
import org.junit.Test;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.convolution.Kernel;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class Im2ColTest extends ConvolutionBaseTest {

    @Test
    public void testIm2ColMatch() {
        // Setup parameters
        int inputDim = 7;
        int kernelWidth = 3;
        int batchSize = 2;
        int outputDim = inputDim - kernelWidth + 1; // 5

        // Create input data with a pattern
        float[] input = new float[inputDim * batchSize];
        for (int i = 0; i < input.length; i++) {
            input[i] = i + 1; // 1-indexed for easier debugging
        }

        // Get CPU im2col result
        Kernel kernel = new Kernel(kernelWidth);
        float[] cpuIm2col = kernel.im2col(input, inputDim, batchSize);

        // Setup GPU
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);

        // Run GPU im2col
        CUdeviceptr gpuIm2col = CudaFunctions.convolution.im2col(
            inputGPU, inputDim, kernelWidth, 1, outputDim, batchSize, stream);

        // Get results back
        JCudaDriver.cuStreamSynchronize(stream);
        float[] gpuIm2colData = CudaUtil.fromGPUFloat(gpuIm2col, outputDim * kernelWidth * batchSize);

        // Verify results
        assertArrayEquals("CPU and GPU im2col results don't match", 
                         cpuIm2col, gpuIm2colData, 1e-5f);

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeAsync(gpuIm2col, stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testIm2ColContent() {
        // Simple test to verify im2col content is correct
        int inputDim = 5;
        int kernelWidth = 3;
        int batchSize = 1;
        int outputDim = inputDim - kernelWidth + 1; // 3

        // Input: [1, 2, 3, 4, 5]
        float[] input = {1, 2, 3, 4, 5};

        // Expected im2col:
        // Row 0: [1, 2, 3] - kernel at position 0
        // Row 1: [2, 3, 4] - kernel at position 1
        // Row 2: [3, 4, 5] - kernel at position 2
        float[] expected = {
            1, 2, 3,
            2, 3, 4,
            3, 4, 5
        };

        Kernel kernel = new Kernel(kernelWidth);
        float[] im2colResult = kernel.im2col(input, inputDim, batchSize);

        assertArrayEquals("im2col transformation incorrect", 
                         expected, im2colResult, 1e-5f);
    }

    @Test
    public void testMultiBatchIm2Col() {
        // Test with multiple batches
        int inputDim = 4;
        int kernelWidth = 2;
        int batchSize = 2;
        int outputDim = inputDim - kernelWidth + 1; // 3

        // Input - 2 batches: [1,2,3,4], [5,6,7,8]
        float[] input = {1, 2, 3, 4, 5, 6, 7, 8};

        // Expected im2col:
        // Batch 1:
        // Row 0: [1, 2] - kernel at position 0
        // Row 1: [2, 3] - kernel at position 1
        // Row 2: [3, 4] - kernel at position 2
        // Batch 2:
        // Row 3: [5, 6] - kernel at position 0
        // Row 4: [6, 7] - kernel at position 1
        // Row 5: [7, 8] - kernel at position 2
        float[] expected = {
            1, 2,
            2, 3,
            3, 4,
            5, 6,
            6, 7,
            7, 8
        };

        Kernel kernel = new Kernel(kernelWidth);
        float[] im2colResult = kernel.im2col(input, inputDim, batchSize);

        assertArrayEquals("Multi-batch im2col incorrect", 
                         expected, im2colResult, 1e-5f);

        // Test GPU implementation also
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);

        CUdeviceptr gpuIm2col = CudaFunctions.convolution.im2col(
            inputGPU, inputDim, kernelWidth, 1, outputDim, batchSize, stream);

        JCudaDriver.cuStreamSynchronize(stream);
        float[] gpuResult = CudaUtil.fromGPUFloat(gpuIm2col, outputDim * kernelWidth * batchSize);

        assertArrayEquals("GPU multi-batch im2col incorrect", 
                         expected, gpuResult, 1e-5f);

        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeAsync(gpuIm2col, stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testLargeMultiBatchIm2ColCorrectness() {
        // Larger test parameters
        int inputDim = 256;
        int kernelWidth = 16;
        int batchSize = 64;
        int outputDim = inputDim - kernelWidth + 1;

        // Generate deterministic input data
        float[] input = new float[inputDim * batchSize];
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < inputDim; i++) {
                // Use a deterministic pattern for easier debugging
                input[b * inputDim + i] = (b * 0.1f) + (i * 0.01f);
            }
        }

        // Compute CPU result
        Kernel kernel = new Kernel(kernelWidth);
        float[] cpuResult = kernel.im2col(input, inputDim, batchSize);

        // Compute GPU result
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        CUdeviceptr gpuIm2col = CudaFunctions.convolution.im2col(inputGPU, inputDim, kernelWidth, 1, outputDim, batchSize, stream);
        JCudaDriver.cuStreamSynchronize(stream);
        float[] gpuResult = CudaUtil.fromGPUFloat(gpuIm2col, outputDim * kernelWidth * batchSize);

        // Verify results match
        assertArrayEquals("CPU and GPU im2col results don't match for large dataset", 
                         cpuResult, gpuResult, 1e-5f);

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeAsync(gpuIm2col, stream);
        CudaUtil.freeStream(stream);

        System.out.println("Large im2col test passed with " + batchSize + 
                         " batches of size " + inputDim + " and kernel width " + kernelWidth);
    }
    
    @Test
    public void testIm2ColWithUnits() {
        // Test unitSize=2, unitCount=2 (width=4)
        Kernel kernel = new Kernel(2, 2);

        // Input: 5 units of size 2 = [1,2] [3,4] [5,6] [7,8] [9,10]
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

        float[] im2colResult = kernel.im2col(input, 10, 1);

        // Expected: 4 output positions, each with 4 consecutive values
        // Position 0: [1,2,3,4]
        // Position 1: [3,4,5,6] 
        // Position 2: [5,6,7,8]
        // Position 3: [7,8,9,10]
        float[] expected = {
            1, 2, 3, 4,
            3, 4, 5, 6,
            5, 6, 7, 8,  
            7, 8, 9, 10
        };

        assertArrayEquals("im2col with units incorrect", expected, im2colResult, 1e-5f);
    }

    @Test
    public void testIm2ColWithUnitsMultiBatch() {
        // Test unitSize=2, unitCount=2 (width=4) with 2 batches
        Kernel kernel = new Kernel(2, 2);

        // Input: 2 batches, each with 4 units of size 2
        // Batch 1: [1,2] [3,4] [5,6] [7,8]
        // Batch 2: [9,10] [11,12] [13,14] [15,16]
        float[] input = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f
        };

        float[] im2colResult = kernel.im2col(input, 8, 2);

        // Expected: 3 positions per batch, 4 values each
        // Batch 1: [1,2,3,4], [3,4,5,6], [5,6,7,8]
        // Batch 2: [9,10,11,12], [11,12,13,14], [13,14,15,16]
        float[] expected = {
            1, 2, 3, 4,
            3, 4, 5, 6,
            5, 6, 7, 8,
            9, 10, 11, 12,
            11, 12, 13, 14,
            13, 14, 15, 16
        };

        assertArrayEquals("Multi-batch im2col with units incorrect", expected, im2colResult, 1e-5f);
    }
    
    @Test
    public void testIm2ColCPUvsGPU() {
        // Test unitSize=2, unitCount=3 (width=6)
        Kernel kernel = new Kernel(2, 3);

        // Input: 2 batches, each with 5 units of size 2
        float[] input = new float[20]; // 2 batches × 5 units × 2 values
        for (int i = 0; i < input.length; i++) {
            input[i] = i + 1;
        }

        // CPU im2col
        float[] cpuResult = kernel.im2col(input, 10, 2);

        // GPU im2col
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        CUdeviceptr gpuIm2col = kernel.im2colGPU(inputGPU, 10, 2, stream);

        JCudaDriver.cuStreamSynchronize(stream);
        float[] gpuResult = CudaUtil.fromGPUFloat(gpuIm2col, cpuResult.length);

        assertArrayEquals("CPU and GPU im2col results don't match", 
                         cpuResult, gpuResult, 1e-5f);

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeAsync(gpuIm2col, stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testExtractChannelGPU() {
        CUstream stream = CudaUtil.createStream();

        // Test configuration: 3 kernels with different unit sizes
        int[] unitSizes = {2, 1, 3};  // kernel1=2, kernel2=1, kernel3=3
        int inputStride = 6;  // total stride per chunk
        int numChunks = 3;
        int batchSize = 2;

        // Create interleaved input: 2 batches * 3 chunks * 6 values = 36 values
        float[] input = {
            // Batch 1
            // chunk 0: [k1: 1,2] [k2: 10] [k3: 20,21,22]
            1.0f, 2.0f, 10.0f, 20.0f, 21.0f, 22.0f,
            // chunk 1: [k1: 3,4] [k2: 11] [k3: 23,24,25] 
            3.0f, 4.0f, 11.0f, 23.0f, 24.0f, 25.0f,
            // chunk 2: [k1: 5,6] [k2: 12] [k3: 26,27,28]
            5.0f, 6.0f, 12.0f, 26.0f, 27.0f, 28.0f,

            // Batch 2  
            // chunk 0: [k1: 7,8] [k2: 13] [k3: 29,30,31]
            7.0f, 8.0f, 13.0f, 29.0f, 30.0f, 31.0f,
            // chunk 1: [k1: 9,10] [k2: 14] [k3: 32,33,34]
            9.0f, 10.0f, 14.0f, 32.0f, 33.0f, 34.0f,
            // chunk 2: [k1: 11,12] [k2: 15] [k3: 35,36,37]
            11.0f, 12.0f, 15.0f, 35.0f, 36.0f, 37.0f
        };

        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);

        // Test each channel extraction
        for (int k = 0; k < unitSizes.length; k++) {
            int unitSize = unitSizes[k];
            int channelSize = unitSize * numChunks * batchSize;

            // Calculate channel offset
            int channelOffset = 0;
            for (int i = 0; i < k; i++) {
                channelOffset += unitSizes[i];
            }

            // Extract channel
            CUdeviceptr channelOutput = CudaUtil.createFloatAsync(channelSize, stream);
            CudaFunctions.convolution.extractChannel(
                inputGPU, channelOutput, 
                inputStride, unitSize, channelOffset, 
                numChunks * batchSize, stream
            );

            // Get result back to CPU
            float[] result = CudaUtil.fromGPUFloatAsync(channelOutput, channelSize, stream);
            JCudaDriver.cuStreamSynchronize(stream);

            // Verify extracted channel data
            if (k == 0) { // Kernel 1: unitSize=2, channelOffset=0
                float[] expected = {
                    // Batch 1: [1,2], [3,4], [5,6]
                    1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                    // Batch 2: [7,8], [9,10], [11,12] 
                    7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f
                };
                assertArrayEquals("Kernel 1 channel extraction failed", expected, result, 1e-6f);

            } else if (k == 1) { // Kernel 2: unitSize=1, channelOffset=2
                float[] expected = {
                    // Batch 1: [10], [11], [12]
                    10.0f, 11.0f, 12.0f,
                    // Batch 2: [13], [14], [15]
                    13.0f, 14.0f, 15.0f
                };
                assertArrayEquals("Kernel 2 channel extraction failed", expected, result, 1e-6f);

            } else if (k == 2) { // Kernel 3: unitSize=3, channelOffset=3
                float[] expected = {
                    // Batch 1: [20,21,22], [23,24,25], [26,27,28]
                    20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f,
                    // Batch 2: [29,30,31], [32,33,34], [35,36,37]
                    29.0f, 30.0f, 31.0f, 32.0f, 33.0f, 34.0f, 35.0f, 36.0f, 37.0f
                };
                assertArrayEquals("Kernel 3 channel extraction failed", expected, result, 1e-6f);
            }

            CudaUtil.freeAsync(channelOutput, stream);
        }

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testDistributeChannelGradientsGPU() {
        CUstream stream = CudaUtil.createStream();

        // Test configuration: 3 kernels with different unit sizes
        int[] unitSizes = {2, 1, 3};  // kernel1=2, kernel2=1, kernel3=3
        int inputStride = 6;  // total stride per chunk
        int numChunks = 3;
        int batchSize = 2;
        int totalInputSize = inputStride * numChunks * batchSize;

        // Create channel gradients for each kernel
        float[][] channelGradients = {
            // Kernel 1: unitSize=2, 6 chunks total (3 per batch)
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
            // Kernel 2: unitSize=1, 6 chunks total
            {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f},
            // Kernel 3: unitSize=3, 6 chunks total
            {100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f, 108.0f, 
             109.0f, 110.0f, 111.0f, 112.0f, 113.0f, 114.0f, 115.0f, 116.0f, 117.0f}
        };

        // Expected interleaved result
        float[] expectedOutput = {
            // Batch 1
            // chunk 0: [k1: 1,2] [k2: 0.1] [k3: 100,101,102]
            1.0f, 2.0f, 0.1f, 100.0f, 101.0f, 102.0f,
            // chunk 1: [k1: 3,4] [k2: 0.2] [k3: 103,104,105]
            3.0f, 4.0f, 0.2f, 103.0f, 104.0f, 105.0f,
            // chunk 2: [k1: 5,6] [k2: 0.3] [k3: 106,107,108]
            5.0f, 6.0f, 0.3f, 106.0f, 107.0f, 108.0f,

            // Batch 2
            // chunk 0: [k1: 7,8] [k2: 0.4] [k3: 109,110,111]
            7.0f, 8.0f, 0.4f, 109.0f, 110.0f, 111.0f,
            // chunk 1: [k1: 9,10] [k2: 0.5] [k3: 112,113,114]
            9.0f, 10.0f, 0.5f, 112.0f, 113.0f, 114.0f,
            // chunk 2: [k1: 11,12] [k2: 0.6] [k3: 115,116,117]
            11.0f, 12.0f, 0.6f, 115.0f, 116.0f, 117.0f
        };

        // Create output buffer (initialized to zero)
        CUdeviceptr outputGradients = CudaUtil.createFloatAsync(totalInputSize, stream);
        JCudaDriver.cuMemsetD32Async(outputGradients, 0, totalInputSize, stream);

        // Distribute gradients from each channel
        for (int k = 0; k < unitSizes.length; k++) {
            int unitSize = unitSizes[k];
            int channelSize = unitSize * numChunks * batchSize;

            // Calculate channel offset
            int channelOffset = 0;
            for (int i = 0; i < k; i++) {
                channelOffset += unitSizes[i];
            }

            // Upload channel gradients to GPU
            CUdeviceptr channelGradientsGPU = CudaUtil.toGPUAsync(channelGradients[k], stream);

            // Distribute channel gradients back to interleaved positions
            CudaFunctions.convolution.distributeChannelGradients(
                channelGradientsGPU, outputGradients,
                inputStride, unitSize, channelOffset,
                numChunks * batchSize, stream
            );

            CudaUtil.freeAsync(channelGradientsGPU, stream);
        }

        // Get result back to CPU
        float[] result = CudaUtil.fromGPUFloatAsync(outputGradients, totalInputSize, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        // Verify the distributed gradients match expected interleaved format
        assertArrayEquals("Channel gradient distribution failed", expectedOutput, result, 1e-6f);

        // Cleanup
        CudaUtil.freeAsync(outputGradients, stream);
        CudaUtil.freeStream(stream);
    }    
}

