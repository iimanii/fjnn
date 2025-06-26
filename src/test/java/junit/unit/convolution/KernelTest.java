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

import java.util.Arrays;
import static org.junit.Assert.*;
import org.junit.Test;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.cublasHandle;
import org.fjnn.convolution.Kernel;
import org.fjnn.convolution.output.unit.ConvolutionUnitBackpropagateOutput;
import org.fjnn.convolution.output.unit.ConvolutionUnitBackpropagateOutputGPU;
import org.fjnn.convolution.output.unit.ConvolutionUnitForwardOutput;
import org.fjnn.convolution.output.unit.ConvolutionUnitForwardOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaUtil;
import org.junit.Ignore;

/**
 *
 * @author ahmed
 */
public class KernelTest extends ConvolutionBaseTest {
    
    @Test
    public void testKernelForward() {
        // Create kernel with width 3
        Kernel kernel = new Kernel(3);
        
        // Set specific weights and bias for predictable output
        kernel.setWeights(new float[]{0.5f, 1.0f, 0.5f});
        kernel.setBias(1.0f);
        
        // Input: [1, 2, 3, 4, 5]
        float[] input = {1, 2, 3, 4, 5};
        
        // Expected output: [1 + (1*0.5 + 2*1.0 + 3*0.5), 1 + (2*0.5 + 3*1.0 + 4*0.5), 1 + (3*0.5 + 4*1.0 + 5*0.5)]
        // = [5.0, 7.0, 9.0]
        float[] expected = {5.0f, 7.0f, 9.0f};
        
        // Perform forward pass
        ConvolutionUnitForwardOutput output = kernel.feedForward(input, 1);
        
        // Check results
        assertArrayEquals("Forward pass output incorrect", expected, output.output, 1e-5f);
    }
    
    @Test
    public void testKernelForwardWithUnits() {
        // Test unitSize=3, unitCount=2 (width=6)
        Kernel kernel = new Kernel(3, 2);
        kernel.setWeights(new float[]{0.5f, 1.0f, 1.5f, 2.0f, 0.5f, 1.2f});
        kernel.setBias(0.0f);

        // Input: 5 units of size 3 each = [1,2,3] [4,5,6] [7,8,9] [10,11,12] [13,14,15]
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};

        ConvolutionUnitForwardOutput output = kernel.feedForward(input, 1);

        // Manual calculation (sliding window of 6 values by unitSize=3):
        // Position 0: [1,2,3,4,5,6] → 1*0.5 + 2*1.0 + 3*1.5 + 4*2.0 + 5*0.5 + 6*1.2 = 0.5+2.0+4.5+8.0+2.5+7.2 = 24.7
        // Position 1: [4,5,6,7,8,9] → 4*0.5 + 5*1.0 + 6*1.5 + 7*2.0 + 8*0.5 + 9*1.2 = 2.0+5.0+9.0+14.0+4.0+10.8 = 44.8
        // Position 2: [7,8,9,10,11,12] → 7*0.5 + 8*1.0 + 9*1.5 + 10*2.0 + 11*0.5 + 12*1.2 = 3.5+8.0+13.5+20.0+5.5+14.4 = 64.9
        // Position 3: [10,11,12,13,14,15] → 10*0.5 + 11*1.0 + 12*1.5 + 13*2.0 + 14*0.5 + 15*1.2 = 5.0+11.0+18.0+26.0+7.0+18.0 = 85.0
        float[] expected = {24.7f, 44.8f, 64.9f, 85.0f};

        assertEquals(4, output.outputSize);
        assertArrayEquals(expected, output.output, 1e-5f);
    }
    
    @Test
    public void testKernelWithMultiBatches() {
        // Create kernel with width 2
        Kernel kernel = new Kernel(2);
        
        // Set weights and bias
        kernel.setWeights(new float[]{1.0f, 1.0f});
        kernel.setBias(0.5f);
        
        // Input: batch 1: [1, 2, 3], batch 2: [4, 5, 6]
        float[] input = {1, 2, 3, 4, 5, 6};
        
        // Expected: batch 1: [3.5, 5.5], batch 2: [9.5, 11.5]
        float[] expected = {3.5f, 5.5f, 9.5f, 11.5f};
        
        // Output size = (3 - 2 + 1) * 2 batches = 4
        // Perform forward pass with 2 batches
        ConvolutionUnitForwardOutput output = kernel.feedForward(input, 2);
        
        // Check results
        assertArrayEquals("Multi-batch output incorrect", expected, output.output, 1e-5f);
    }
    
    @Test
    public void testKernelWithUnitsMultiBatches() {
        // Test unitSize=3, unitCount=2 (width=6)
        Kernel kernel = new Kernel(3, 2);
        kernel.setWeights(new float[]{0.5f, 1.0f, 1.5f, 2.0f, 0.5f, 1.2f});
        kernel.setBias(0.0f);

        // Input: 2 batches, each with 4 units of size 3
        // Batch 1: [1,2,3] [4,5,6] [7,8,9] [10,11,12]
        // Batch 2: [13,14,15] [16,17,18] [19,20,21] [22,23,24]
        float[] input = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,      // Batch 1
            13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f // Batch 2
        };

        ConvolutionUnitForwardOutput output = kernel.feedForward(input, 2);

        // Expected outputs:
        // Batch 1: [24.7, 44.8, 64.9] (3 outputs from 4 units)
        // Batch 2: Position 0: [13,14,15,16,17,18] → 13*0.5 + 14*1.0 + 15*1.5 + 16*2.0 + 17*0.5 + 18*1.2 = 6.5+14.0+22.5+32.0+8.5+21.6 = 105.1
        //         Position 1: [16,17,18,19,20,21] → 16*0.5 + 17*1.0 + 18*1.5 + 19*2.0 + 20*0.5 + 21*1.2 = 8.0+17.0+27.0+38.0+10.0+25.2 = 125.2
        //         Position 2: [19,20,21,22,23,24] → 19*0.5 + 20*1.0 + 21*1.5 + 22*2.0 + 23*0.5 + 24*1.2 = 9.5+20.0+31.5+44.0+11.5+28.8 = 145.3
        float[] expected = {24.7f, 44.8f, 64.9f, 105.1f, 125.2f, 145.3f};

        assertEquals(3, output.outputSize);
        assertEquals(2, output.batchSize);
        assertArrayEquals(expected, output.output, 1e-5f);
    }

    
    @Test
    public void testKernelForwardGPU() {
        // Create kernel with width 3
        Kernel kernel = new Kernel(3);
        
        // Set weights and bias
        kernel.setWeights(new float[]{0.5f, 1.0f, 0.5f});
        kernel.setBias(1.0f);
        
        CUstream stream = CudaUtil.createStream();
        
        // Prepare GPU
        kernel.prepareGPU(stream);
        
        float[] input = {1, 2, 3, 4, 5, 6, 7};
        
        // Generate CPU output to compare with GPU results
        float[] expected = kernel.feedForward(input, 1).output; // outputSize = 7-3+1 = 5
        
        // Create stream
        cublasHandle handle = CudaEngine.getCublasHandle(0);
        
        // Move input to GPU
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);

        // Perform GPU forward pass
        ConvolutionUnitForwardOutputGPU gpuOutput = kernel.feedForwardGPU(inputGPU, 1, stream, handle);
        
        // Get results back to CPU
        float[] output = CudaUtil.fromGPUFloatAsync(gpuOutput.output, expected.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);
        
        // Check results
        assertArrayEquals("GPU forward pass output incorrect", expected, output, 1e-5f);
        
        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        kernel.freeGPU(stream);
        gpuOutput.freeAsync(stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testKernelForwardWithUnitsGPU() {
        // Test unitSize=3, unitCount=2 (width=6)
        Kernel kernel = new Kernel(3, 2);
        kernel.setWeights(new float[]{0.5f, 1.0f, 1.5f, 2.0f, 0.5f, 1.2f});
        kernel.setBias(0.0f);

        CUstream stream = CudaUtil.createStream();

        // Prepare GPU
        kernel.prepareGPU(stream);

        // Input: 5 units of size 3 each = [1,2,3] [4,5,6] [7,8,9] [10,11,12] [13,14,15]
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};

        // Generate CPU output to compare with GPU results
        ConvolutionUnitForwardOutput cpuOutput = kernel.feedForward(input, 1);
        float[] expected = cpuOutput.output;

        // Create cuBLAS handle
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        // Get im2col representation for the input
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);

        // Perform GPU forward pass
        ConvolutionUnitForwardOutputGPU gpuOutput = kernel.feedForwardGPU(inputGPU, 1, stream, handle);

        // Get results back to CPU
        float[] output = CudaUtil.fromGPUFloatAsync(gpuOutput.output, expected.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        // Check results
        assertArrayEquals("GPU forward pass with units output incorrect", expected, output, 1e-5f);

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput.freeAsync(stream);
        kernel.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testKernelForwardMultiBatchGPU() {
        // Create kernel with width 3
        Kernel kernel = new Kernel(3);

        // Set weights and bias
        kernel.setWeights(new float[]{0.5f, 1.0f, 0.5f});
        kernel.setBias(1.0f);

        CUstream stream = CudaUtil.createStream();

        // Prepare GPU
        kernel.prepareGPU(stream);

        int batchSize = 4;
        int inputSizePerBatch = 7;
        int outputSizePerBatch = 5; // 7-3+1 = 5

        // Create batched input: 4 batches of size 7
        float[] input = {
            1, 2, 3, 4, 5, 6, 7,     // batch 1
            2, 3, 4, 5, 6, 7, 8,     // batch 2
            0, 1, 2, 3, 4, 5, 6,     // batch 3
            3, 4, 5, 6, 7, 8, 9      // batch 4
        };

        // Generate CPU output to compare with GPU results
        float[] expected = new float[batchSize * outputSizePerBatch];
        for (int b = 0; b < batchSize; b++) {
            float[] batchInput = Arrays.copyOfRange(input, b * inputSizePerBatch, (b + 1) * inputSizePerBatch);
            float[] batchOutput = kernel.feedForward(batchInput, 1).output;
            
            System.arraycopy(batchOutput, 0, expected, b * outputSizePerBatch, outputSizePerBatch);
        }

        // Create stream
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        // Move input to GPU
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);

        // Perform GPU forward pass
        ConvolutionUnitForwardOutputGPU gpuOutput = kernel.feedForwardGPU(inputGPU, batchSize, stream, handle);
        
        // Get results back to CPU
        float[] output = CudaUtil.fromGPUFloatAsync(gpuOutput.output, expected.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        // Check results
        assertArrayEquals("GPU forward pass output incorrect for multiple batches", expected, output, 1e-5f);

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput.freeAsync(stream);
        kernel.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testKernelForwardWithUnitsMultiBatchGPU() {
        // Test unitSize=3, unitCount=2 (width=6)
        Kernel kernel = new Kernel(3, 2);
        kernel.setWeights(new float[]{0.5f, 1.0f, 1.5f, 2.0f, 0.5f, 1.2f});
        kernel.setBias(0.0f);

        CUstream stream = CudaUtil.createStream();

        // Prepare GPU
        kernel.prepareGPU(stream);

        int batchSize = 2;
        int inputSizePerBatch = 12; // 4 units of size 3

        // Input: 2 batches, each with 4 units of size 3
        // Batch 1: [1,2,3] [4,5,6] [7,8,9] [10,11,12]
        // Batch 2: [13,14,15] [16,17,18] [19,20,21] [22,23,24]
        float[] input = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,      // Batch 1
            13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f // Batch 2
        };

        // Generate CPU output to compare with GPU results
        ConvolutionUnitForwardOutput cpuOutput = kernel.feedForward(input, batchSize);
        float[] expected = cpuOutput.output;

        // Create cuBLAS handle
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        // Move input to GPU
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);

        // Perform GPU forward pass
        ConvolutionUnitForwardOutputGPU gpuOutput = kernel.feedForwardGPU(inputGPU, batchSize, stream, handle);

        // Get results back to CPU
        float[] output = CudaUtil.fromGPUFloatAsync(gpuOutput.output, expected.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        // Check results
        assertArrayEquals("GPU forward pass with units multi-batch output incorrect", expected, output, 1e-5f);

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput.freeAsync(stream);
        kernel.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }
    
    
    @Test
    public void testKernelBackward() {
        // Create kernel with width 3
        Kernel kernel = new Kernel(3);
        
        // Set weights and bias
        kernel.setWeights(new float[]{0.5f, 1.0f, 0.5f});
        kernel.setBias(0.0f);
        
        // Input: [1, 2, 3, 4, 5]
        float[] input = {1, 2, 3, 4, 5};
        int inputSize = input.length;
        int batchSize = 1;

        ConvolutionUnitForwardOutput forwardOutput = kernel.feedForward(input, batchSize);

        // Gradient from next layer: [0.1, 0.2, 0.3]
        float[] gradOutput = {0.1f, 0.2f, 0.3f};
       
        // Perform backward pass
        ConvolutionUnitBackpropagateOutput backpropOutput = kernel.backpropagate(forwardOutput, gradOutput);

        // Expected weight gradients:
        // w[0]: 0.1*1 + 0.2*2 + 0.3*3 = 1.4
        // w[1]: 0.1*2 + 0.2*3 + 0.3*4 = 2.0
        // w[2]: 0.1*3 + 0.2*4 + 0.3*5 = 2.6
        float[] expectedWeightGrad = {1.4f, 2.0f, 2.6f};
        
        // Expected bias gradient: sum of output gradients = 0.6
        float expectedBiasGrad = 0.6f;
        
        // Expected input gradients: 
        // gradInput[0] = 0.1 * 0.5 = 0.05
        // gradInput[1] = 0.1 * 1.0 + 0.2 * 0.5 = 0.2
        // gradInput[2] = 0.1 * 0.5 + 0.2 * 1.0 + 0.3 * 0.5 = 0.4
        // gradInput[3] = 0.2 * 0.5 + 0.3 * 1.0 = 0.4
        // gradInput[4] = 0.3 * 0.5 = 0.15
        float[] expectedInputGrad = {0.05f, 0.2f, 0.4f, 0.4f, 0.15f};
        
        // Check results
        assertArrayEquals("Weight gradients incorrect", expectedWeightGrad, backpropOutput.weightGradients[0], 1e-5f);
        assertEquals("Bias gradient incorrect", expectedBiasGrad, backpropOutput.biasGradients[0], 1e-5f);
        assertArrayEquals("Input gradients incorrect", expectedInputGrad, backpropOutput.inputGradients, 1e-5f);
    }
    
    @Test
    public void testKernelBackwardMultiBatches() {
        // Create kernel with width 3
        Kernel kernel = new Kernel(3);
        kernel.setWeights(new float[]{0.5f, 1.0f, 0.5f});
        kernel.setBias(0.0f);

        // Input: batch 1: [1,2,3,4,5], batch 2: [6,7,8,9,10]
        float[] input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        int batchSize = 2;
        int inputSize = 5;

        ConvolutionUnitForwardOutput forwardOutput = kernel.feedForward(input, batchSize);

        // Forward pass manual calculation:
        // Batch 1: [1*0.5 + 2*1.0 + 3*0.5, 2*0.5 + 3*1.0 + 4*0.5, 3*0.5 + 4*1.0 + 5*0.5] = [4.0, 6.0, 8.0]
        // Batch 2: [6*0.5 + 7*1.0 + 8*0.5, 7*0.5 + 8*1.0 + 9*0.5, 8*0.5 + 9*1.0 + 10*0.5] = [14.0, 16.0, 18.0]
        float[] expectedForward = {4.0f, 6.0f, 8.0f, 14.0f, 16.0f, 18.0f};
        assertArrayEquals("Forward pass incorrect", expectedForward, forwardOutput.output, 1e-5f);

        // Gradient from next layer: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] (batch1: [0.1,0.2,0.3], batch2: [0.4,0.5,0.6])
        float[] gradOutput = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

        ConvolutionUnitBackpropagateOutput backpropOutput = kernel.backpropagate(forwardOutput, gradOutput);

        // Expected weight gradients (averaged over batches):
        // w[0]: (1*0.1 + 2*0.2 + 3*0.3 + 6*0.4 + 7*0.5 + 8*0.6) / 2 = (0.1 + 0.4 + 0.9 + 2.4 + 3.5 + 4.8) / 2 = 6.05
        // w[1]: (2*0.1 + 3*0.2 + 4*0.3 + 7*0.4 + 8*0.5 + 9*0.6) / 2 = (0.2 + 0.6 + 1.2 + 2.8 + 4.0 + 5.4) / 2 = 7.1
        // w[2]: (3*0.1 + 4*0.2 + 5*0.3 + 8*0.4 + 9*0.5 + 10*0.6) / 2 = (0.3 + 0.8 + 1.5 + 3.2 + 4.5 + 6.0) / 2 = 8.15
        float[] expectedWeightGrad = {6.05f, 7.1f, 8.15f};

        // Expected bias gradient: (0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6) / 2 = 1.05
        float expectedBiasGrad = 1.05f;

        // Expected input gradients: 
        // Batch 1:
        // input[0] = 0.5*0.1 = 0.05
        // input[1] = 1.0*0.1 + 0.5*0.2 = 0.2
        // input[2] = 0.5*0.1 + 1.0*0.2 + 0.5*0.3 = 0.4
        // input[3] = 0.5*0.2 + 1.0*0.3 = 0.4
        // input[4] = 0.5*0.3 = 0.15
        // Batch 2:
        // input[5] = 0.5*0.4 = 0.2
        // input[6] = 1.0*0.4 + 0.5*0.5 = 0.65
        // input[7] = 0.5*0.4 + 1.0*0.5 + 0.5*0.6 = 1.0
        // input[8] = 0.5*0.5 + 1.0*0.6 = 0.85
        // input[9] = 0.5*0.6 = 0.3
        float[] expectedInputGrad = {0.05f, 0.2f, 0.4f, 0.4f, 0.15f, 0.2f, 0.65f, 1.0f, 0.85f, 0.3f};

        assertArrayEquals("Weight gradients incorrect", expectedWeightGrad, backpropOutput.weightGradients[0], 1e-5f);
        assertEquals("Bias gradient incorrect", expectedBiasGrad, backpropOutput.biasGradients[0], 1e-5f);
        assertArrayEquals("Input gradients incorrect", expectedInputGrad, backpropOutput.inputGradients, 1e-5f);
    }
    
    @Test
    public void testKernelBackwardWithUnits() {
        // Test unitSize=3, unitCount=2 (width=6)
        Kernel kernel = new Kernel(3, 2);
        kernel.setWeights(new float[]{0.5f, 1.0f, 2.0f, 0.5f, 1.5f, 0.8f});
        kernel.setBias(0.0f);

        // Input: 4 units of size 3 = [1,2,3] [4,5,6] [7,8,9] [10,11,12]
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        int batchSize = 1;

        ConvolutionUnitForwardOutput forwardOutput = kernel.feedForward(input, batchSize);

        // Forward pass manual calculation (sliding by unitSize=3):
        // Position 0: [1,2,3,4,5,6] → 1×0.5 + 2×1.0 + 3×2.0 + 4×0.5 + 5×1.5 + 6×0.8 = 0.5+2.0+6.0+2.0+7.5+4.8 = 22.8
        // Position 1: [4,5,6,7,8,9] → 4×0.5 + 5×1.0 + 6×2.0 + 7×0.5 + 8×1.5 + 9×0.8 = 2.0+5.0+12.0+3.5+12.0+7.2 = 41.7
        // Position 2: [7,8,9,10,11,12] → 7×0.5 + 8×1.0 + 9×2.0 + 10×0.5 + 11×1.5 + 12×0.8 = 3.5+8.0+18.0+5.0+16.5+9.6 = 60.6
        float[] expectedForward = {22.8f, 41.7f, 60.6f};
        assertArrayEquals("Forward pass incorrect", expectedForward, forwardOutput.output, 1e-5f);

        // Gradients from next layer: [0.1, 0.2, 0.3]
        float[] gradOutput = {0.1f, 0.2f, 0.3f};

        ConvolutionUnitBackpropagateOutput backpropOutput = kernel.backpropagate(forwardOutput, gradOutput);

        // Expected weight gradients:
        // w[0]: 1×0.1 + 4×0.2 + 7×0.3 = 0.1 + 0.8 + 2.1 = 3.0
        // w[1]: 2×0.1 + 5×0.2 + 8×0.3 = 0.2 + 1.0 + 2.4 = 3.6
        // w[2]: 3×0.1 + 6×0.2 + 9×0.3 = 0.3 + 1.2 + 2.7 = 4.2
        // w[3]: 4×0.1 + 7×0.2 + 10×0.3 = 0.4 + 1.4 + 3.0 = 4.8
        // w[4]: 5×0.1 + 8×0.2 + 11×0.3 = 0.5 + 1.6 + 3.3 = 5.4
        // w[5]: 6×0.1 + 9×0.2 + 12×0.3 = 0.6 + 1.8 + 3.6 = 6.0
        float[] expectedWeightGrad = {3.0f, 3.6f, 4.2f, 4.8f, 5.4f, 6.0f};

        // Expected bias gradient: 0.1 + 0.2 + 0.3 = 0.6
        float expectedBiasGrad = 0.6f;

        // Expected input gradients (each input accumulates weight×gradient from outputs that use it):
        // input[0]: used by output 0 at weight pos 0 → w[0]×0.1 = 0.5×0.1 = 0.05
        // input[1]: used by output 0 at weight pos 1 → w[1]×0.1 = 1.0×0.1 = 0.1
        // input[2]: used by output 0 at weight pos 2 → w[2]×0.1 = 2.0×0.1 = 0.2
        // input[3]: used by output 0 at weight pos 3, output 1 at weight pos 0 → w[3]×0.1 + w[0]×0.2 = 0.5×0.1 + 0.5×0.2 = 0.15
        // input[4]: used by output 0 at weight pos 4, output 1 at weight pos 1 → w[4]×0.1 + w[1]×0.2 = 1.5×0.1 + 1.0×0.2 = 0.35
        // input[5]: used by output 0 at weight pos 5, output 1 at weight pos 2 → w[5]×0.1 + w[2]×0.2 = 0.8×0.1 + 2.0×0.2 = 0.48
        // input[6]: used by output 1 at weight pos 3, output 2 at weight pos 0 → w[3]×0.2 + w[0]×0.3 = 0.5×0.2 + 0.5×0.3 = 0.25
        // input[7]: used by output 1 at weight pos 4, output 2 at weight pos 1 → w[4]×0.2 + w[1]×0.3 = 1.5×0.2 + 1.0×0.3 = 0.6
        // input[8]: used by output 1 at weight pos 5, output 2 at weight pos 2 → w[5]×0.2 + w[2]×0.3 = 0.8×0.2 + 2.0×0.3 = 0.76
        // input[9]: used by output 2 at weight pos 3 → w[3]×0.3 = 0.5×0.3 = 0.15
        // input[10]: used by output 2 at weight pos 4 → w[4]×0.3 = 1.5×0.3 = 0.45
        // input[11]: used by output 2 at weight pos 5 → w[5]×0.3 = 0.8×0.3 = 0.24
        float[] expectedInputGrad = {0.05f, 0.1f, 0.2f, 0.15f, 0.35f, 0.48f, 0.25f, 0.6f, 0.76f, 0.15f, 0.45f, 0.24f};

        assertArrayEquals("Weight gradients incorrect", expectedWeightGrad, backpropOutput.weightGradients[0], 1e-5f);
        assertEquals("Bias gradient incorrect", expectedBiasGrad, backpropOutput.biasGradients[0], 1e-5f);
        assertArrayEquals("Input gradients incorrect", expectedInputGrad, backpropOutput.inputGradients, 1e-5f);
    }
    
    @Test
    public void testKernelBackwardWithUnitsMultiBatches() {
        // Test unitSize=2, unitCount=3 (width=6) with 2 batches
        Kernel kernel = new Kernel(2, 3);
        kernel.setWeights(new float[]{1.0f, 0.5f, 2.0f, 1.5f, 0.8f, 1.2f});
        kernel.setBias(0.0f);

        // Input: 2 batches, each with 4 units of size 2
        // Batch 1: [1,2] [3,4] [5,6] [7,8]
        // Batch 2: [9,10] [11,12] [13,14] [15,16]
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
        int batchSize = 2;

        ConvolutionUnitForwardOutput forwardOutput = kernel.feedForward(input, batchSize);

        // Forward pass manual calculation (sliding by unitSize=2):
        // Batch 1: Position 0: [1,2,3,4,5,6] → 1×1.0 + 2×0.5 + 3×2.0 + 4×1.5 + 5×0.8 + 6×1.2 = 1.0+1.0+6.0+6.0+4.0+7.2 = 25.2
        //         Position 1: [3,4,5,6,7,8] → 3×1.0 + 4×0.5 + 5×2.0 + 6×1.5 + 7×0.8 + 8×1.2 = 3.0+2.0+10.0+9.0+5.6+9.6 = 39.2
        // Batch 2: Position 0: [9,10,11,12,13,14] → 9×1.0 + 10×0.5 + 11×2.0 + 12×1.5 + 13×0.8 + 14×1.2 = 9.0+5.0+22.0+18.0+10.4+16.8 = 81.2
        //         Position 1: [11,12,13,14,15,16] → 11×1.0 + 12×0.5 + 13×2.0 + 14×1.5 + 15×0.8 + 16×1.2 = 11.0+6.0+26.0+21.0+12.0+19.2 = 95.2
        float[] expectedForward = {25.2f, 39.2f, 81.2f, 95.2f};
        assertArrayEquals("Forward pass incorrect", expectedForward, forwardOutput.output, 1e-5f);

        // Gradients from next layer: [0.1, 0.2, 0.3, 0.4] (batch1: [0.1,0.2], batch2: [0.3,0.4])
        float[] gradOutput = {0.1f, 0.2f, 0.3f, 0.4f};

        ConvolutionUnitBackpropagateOutput backpropOutput = kernel.backpropagate(forwardOutput, gradOutput);

        // Expected weight gradients (averaged over batches):
        // w[0]: (1×0.1 + 3×0.2 + 9×0.3 + 11×0.4) / 2 = (0.1 + 0.6 + 2.7 + 4.4) / 2 = 3.9
        // w[1]: (2×0.1 + 4×0.2 + 10×0.3 + 12×0.4) / 2 = (0.2 + 0.8 + 3.0 + 4.8) / 2 = 4.4
        // w[2]: (3×0.1 + 5×0.2 + 11×0.3 + 13×0.4) / 2 = (0.3 + 1.0 + 3.3 + 5.2) / 2 = 4.9
        // w[3]: (4×0.1 + 6×0.2 + 12×0.3 + 14×0.4) / 2 = (0.4 + 1.2 + 3.6 + 5.6) / 2 = 5.4
        // w[4]: (5×0.1 + 7×0.2 + 13×0.3 + 15×0.4) / 2 = (0.5 + 1.4 + 3.9 + 6.0) / 2 = 5.9
        // w[5]: (6×0.1 + 8×0.2 + 14×0.3 + 16×0.4) / 2 = (0.6 + 1.6 + 4.2 + 6.4) / 2 = 6.4
        float[] expectedWeightGrad = {3.9f, 4.4f, 4.9f, 5.4f, 5.9f, 6.4f};

        // Expected bias gradient: (0.1 + 0.2 + 0.3 + 0.4) / 2 = 0.5
        float expectedBiasGrad = 0.5f;

        // Expected input gradients:
        // Batch 1:
        // input[0] = 1.0×0.1 = 0.1
        // input[1] = 0.5×0.1 = 0.05
        // input[2] = 2.0×0.1 + 1.0×0.2 = 0.4
        // input[3] = 1.5×0.1 + 0.5×0.2 = 0.25
        // input[4] = 0.8×0.1 + 2.0×0.2 = 0.48
        // input[5] = 1.2×0.1 + 1.5×0.2 = 0.42
        // input[6] = 0.8×0.2 = 0.16
        // input[7] = 1.2×0.2 = 0.24
        // Batch 2:
        // input[8] = 1.0×0.3 = 0.3
        // input[9] = 0.5×0.3 = 0.15
        // input[10] = 2.0×0.3 + 1.0×0.4 = 1.0
        // input[11] = 1.5×0.3 + 0.5×0.4 = 0.65
        // input[12] = 0.8×0.3 + 2.0×0.4 = 1.04
        // input[13] = 1.2×0.3 + 1.5×0.4 = 0.96
        // input[14] = 0.8×0.4 = 0.32
        // input[15] = 1.2×0.4 = 0.48
        float[] expectedInputGrad = {0.1f, 0.05f, 0.4f, 0.25f, 0.48f, 0.42f, 0.16f, 0.24f, 0.3f, 0.15f, 1.0f, 0.65f, 1.04f, 0.96f, 0.32f, 0.48f};

        assertArrayEquals("Weight gradients incorrect", expectedWeightGrad, backpropOutput.weightGradients[0], 1e-5f);
        assertEquals("Bias gradient incorrect", expectedBiasGrad, backpropOutput.biasGradients[0], 1e-5f);
        assertArrayEquals("Input gradients incorrect", expectedInputGrad, backpropOutput.inputGradients, 1e-5f);
    }
    
    
    @Test
    public void testKernelBackwardGPU() {
        // Create kernel with unitSize=2, unitCount=3 (width=6)
        Kernel kernel = new Kernel(2, 3);
        kernel.setWeights(new float[]{0.5f, 1.0f, 0.5f, 0.8f, 1.2f, 0.3f});
        kernel.setBias(1.0f);

        CUstream stream = CudaUtil.createStream();
        kernel.prepareGPU(stream);

        // Input and forward pass: 6 units of size 2 = [1,2] [3,4] [5,6] [7,8] [9,10] [11,12]
        float[] input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        ConvolutionUnitForwardOutput forwardOutput = kernel.feedForward(input, 1);

        // Gradients from next layer (outputSize = 6-3+1 = 4)
        float[] deltaLoss = {0.1f, 0.2f, 0.3f, 0.4f};

        // CPU backprop for expected results
        ConvolutionUnitBackpropagateOutput expectedBackprop = kernel.backpropagate(forwardOutput, deltaLoss);

        // GPU backprop
        cublasHandle handle = CudaEngine.getCublasHandle(0);
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionUnitForwardOutputGPU gpuForwardOutput = kernel.feedForwardGPU(inputGPU, 1, stream, handle);
        CUdeviceptr deltaLossGPU = CudaUtil.toGPUAsync(deltaLoss, stream);

        ConvolutionUnitBackpropagateOutputGPU backpropGPU = kernel.backpropagateGPU(gpuForwardOutput, deltaLossGPU, stream, handle);

        // Get results back to CPU
        float[] weightGrads = CudaUtil.fromGPUFloatAsync(backpropGPU.weightGradients[0], kernel.width, stream);
        float[] biasGrads = CudaUtil.fromGPUFloatAsync(backpropGPU.biasGradients[0], 1, stream);
        float[] inputGrads = CudaUtil.fromGPUFloatAsync(backpropGPU.inputGradients, input.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        // Verify results
        assertArrayEquals("GPU weight gradients incorrect", expectedBackprop.weightGradients[0], weightGrads, 1e-5f);
        assertArrayEquals("GPU bias gradients incorrect", expectedBackprop.biasGradients, biasGrads, 1e-5f);
        assertArrayEquals("GPU input gradients incorrect", expectedBackprop.inputGradients, inputGrads, 1e-5f);

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeAsync(deltaLossGPU, stream);
        gpuForwardOutput.freeAsync(stream);
        backpropGPU.freeAsync(stream);
        kernel.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testKernelBackwardMultiBatchGPU() {
        // Create kernel with unitSize=2, unitCount=3 (width=6)
        Kernel kernel = new Kernel(2, 3);
        kernel.setWeights(new float[]{0.5f, 1.0f, 0.5f, 0.8f, 1.2f, 0.3f});
        kernel.setBias(1.0f);

        CUstream stream = CudaUtil.createStream();
        kernel.prepareGPU(stream);

        int batchSize = 2;
        int inputSizePerBatch = 10; // 5 units of size 2
        int outputSizePerBatch = 3; // 5-3+1 = 3

        // Input: 2 batches, each with 5 units of size 2
        // Batch 1: [1,2] [3,4] [5,6] [7,8] [9,10]
        // Batch 2: [11,12] [13,14] [15,16] [17,18] [19,20]
        float[] input = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,        // batch 1
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20 // batch 2
        };

        ConvolutionUnitForwardOutput forwardOutput = kernel.feedForward(input, batchSize);

        // Gradients from next layer (2 batches * 3 outputs each = 6 values)
        float[] deltaLoss = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

        // CPU backprop for expected results
        ConvolutionUnitBackpropagateOutput expectedBackprop = kernel.backpropagate(forwardOutput, deltaLoss);

        // GPU backprop
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionUnitForwardOutputGPU gpuForwardOutput = kernel.feedForwardGPU(inputGPU, batchSize, stream, handle);
        CUdeviceptr deltaLossGPU = CudaUtil.toGPUAsync(deltaLoss, stream);

        ConvolutionUnitBackpropagateOutputGPU backpropGPU = kernel.backpropagateGPU(gpuForwardOutput, deltaLossGPU, stream, handle);

        // Get results back to CPU
        float[] weightGrads = CudaUtil.fromGPUFloatAsync(backpropGPU.weightGradients[0], kernel.width, stream);
        float[] biasGrads = CudaUtil.fromGPUFloatAsync(backpropGPU.biasGradients[0], 1, stream);
        float[] inputGrads = CudaUtil.fromGPUFloatAsync(backpropGPU.inputGradients, input.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        // Verify results
        assertArrayEquals("GPU multi-batch weight gradients incorrect", expectedBackprop.weightGradients[0], weightGrads, 1e-5f);
        assertArrayEquals("GPU multi-batch bias gradients incorrect", expectedBackprop.biasGradients, biasGrads, 1e-5f);
        assertArrayEquals("GPU multi-batch input gradients incorrect", expectedBackprop.inputGradients, inputGrads, 1e-5f);

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeAsync(deltaLossGPU, stream);
        gpuForwardOutput.freeAsync(stream);
        backpropGPU.freeAsync(stream);
        kernel.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testKernelBackwardWithUnitsGPU() {
        // Test unitSize=3, unitCount=2 (width=6)
        Kernel kernel = new Kernel(3, 2);
        kernel.setWeights(new float[]{0.5f, 1.0f, 1.5f, 2.0f, 0.5f, 1.2f});
        kernel.setBias(0.0f);

        CUstream stream = CudaUtil.createStream();
        kernel.prepareGPU(stream);

        // Input: 5 units of size 3 each = [1,2,3] [4,5,6] [7,8,9] [10,11,12] [13,14,15]
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
        ConvolutionUnitForwardOutput forwardOutput = kernel.feedForward(input, 1);

        // Gradients from next layer (outputSize = 5-2+1 = 4)
        float[] deltaLoss = {0.1f, 0.2f, 0.3f, 0.4f};

        // CPU backprop for expected results
        ConvolutionUnitBackpropagateOutput expectedBackprop = kernel.backpropagate(forwardOutput, deltaLoss);

        // GPU backprop
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionUnitForwardOutputGPU gpuForwardOutput = kernel.feedForwardGPU(inputGPU, 1, stream, handle);
        CUdeviceptr deltaLossGPU = CudaUtil.toGPUAsync(deltaLoss, stream);

        ConvolutionUnitBackpropagateOutputGPU backpropGPU = kernel.backpropagateGPU(gpuForwardOutput, deltaLossGPU, stream, handle);

        // Get results back to CPU
        float[] weightGrads = CudaUtil.fromGPUFloatAsync(backpropGPU.weightGradients[0], kernel.width, stream);
        float[] biasGrads = CudaUtil.fromGPUFloatAsync(backpropGPU.biasGradients[0], 1, stream);
        float[] inputGrads = CudaUtil.fromGPUFloatAsync(backpropGPU.inputGradients, input.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        // Verify results
        assertArrayEquals("GPU with units weight gradients incorrect", expectedBackprop.weightGradients[0], weightGrads, 1e-5f);
        assertArrayEquals("GPU with units bias gradients incorrect", expectedBackprop.biasGradients, biasGrads, 1e-5f);
        assertArrayEquals("GPU with units input gradients incorrect", expectedBackprop.inputGradients, inputGrads, 1e-5f);

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeAsync(deltaLossGPU, stream);
        gpuForwardOutput.freeAsync(stream);
        backpropGPU.freeAsync(stream);
        kernel.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testKernelBackwardWithUnitsMultiBatchGPU() {
        // Test unitSize=3, unitCount=2 (width=6) - different values to catch mix-ups
        Kernel kernel = new Kernel(3, 2);
        kernel.setWeights(new float[]{0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f});
        kernel.setBias(0.0f);

        // Input: 2 batches, each with 5 units of size 3
        // Batch 1: [1,2,3] [4,5,6] [7,8,9] [10,11,12] [13,14,15]
        // Batch 2: [16,17,18] [19,20,21] [22,23,24] [25,26,27] [28,29,30]
        // Each batch gives 4 outputs: 5 - 2 + 1 = 4
        float[] input = {
            // Batch 1
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
            // Batch 2  
            16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f
        };

        // Forward pass with 2 batches
        ConvolutionUnitForwardOutput forwardOutput = kernel.feedForward(input, 2);

        // Expected forward outputs (sliding by unitSize=3):
        // Batch 1:
        // Position 0: [1,2,3,4,5,6] → 1*0.5 + 2*1.0 + 3*1.5 + 4*2.0 + 5*2.5 + 6*3.0 = 0.5+2+4.5+8+12.5+18 = 45.5
        // Position 1: [4,5,6,7,8,9] → 4*0.5 + 5*1.0 + 6*1.5 + 7*2.0 + 8*2.5 + 9*3.0 = 2+5+9+14+20+27 = 77.0
        // Position 2: [7,8,9,10,11,12] → 7*0.5 + 8*1.0 + 9*1.5 + 10*2.0 + 11*2.5 + 12*3.0 = 3.5+8+13.5+20+27.5+36 = 108.5
        // Position 3: [10,11,12,13,14,15] → 10*0.5 + 11*1.0 + 12*1.5 + 13*2.0 + 14*2.5 + 15*3.0 = 5+11+18+26+35+45 = 140.0
        // Batch 2:
        // Position 0: [16,17,18,19,20,21] → 16*0.5 + 17*1.0 + 18*1.5 + 19*2.0 + 20*2.5 + 21*3.0 = 8+17+27+38+50+63 = 203.0
        // Position 1: [19,20,21,22,23,24] → 19*0.5 + 20*1.0 + 21*1.5 + 22*2.0 + 23*2.5 + 24*3.0 = 9.5+20+31.5+44+57.5+72 = 234.5
        // Position 2: [22,23,24,25,26,27] → 22*0.5 + 23*1.0 + 24*1.5 + 25*2.0 + 26*2.5 + 27*3.0 = 11+23+36+50+65+81 = 266.0
        // Position 3: [25,26,27,28,29,30] → 25*0.5 + 26*1.0 + 27*1.5 + 28*2.0 + 29*2.5 + 30*3.0 = 12.5+26+40.5+56+72.5+90 = 297.5
        float[] expectedForward = {45.5f, 77.0f, 108.5f, 140.0f, 203.0f, 234.5f, 266.0f, 297.5f};
        assertArrayEquals("Forward pass incorrect", expectedForward, forwardOutput.output, 1e-5f);

        // Gradients from next layer (8 values for 2 batches * 4 outputs each)
        float[] deltaLoss = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

        // Backward pass
        ConvolutionUnitBackpropagateOutput backpropOutput = kernel.backpropagate(forwardOutput, deltaLoss);

        // Expected bias gradient: sum of deltaLoss = 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6 + 0.7 + 0.8 = 3.6
        // Averaged over batches: 3.6 / 2 = 1.8
        assertEquals("Bias gradient incorrect", 1.8f, backpropOutput.biasGradients[0], 1e-5f);

        // Expected weight gradients (summed over both batches, then averaged):
        // w[0]: (1*0.1 + 4*0.2 + 7*0.3 + 10*0.4 + 16*0.5 + 19*0.6 + 22*0.7 + 25*0.8) / 2 = 61.8/2 = 30.9
        // w[1]: (2*0.1 + 5*0.2 + 8*0.3 + 11*0.4 + 17*0.5 + 20*0.6 + 23*0.7 + 26*0.8) / 2 = 65.4/2 = 32.7
        // w[2]: (3*0.1 + 6*0.2 + 9*0.3 + 12*0.4 + 18*0.5 + 21*0.6 + 24*0.7 + 27*0.8) / 2 = 69.0/2 = 34.5
        // w[3]: (4*0.1 + 7*0.2 + 10*0.3 + 13*0.4 + 19*0.5 + 22*0.6 + 25*0.7 + 28*0.8) / 2 = 72.6/2 = 36.3
        // w[4]: (5*0.1 + 8*0.2 + 11*0.3 + 14*0.4 + 20*0.5 + 23*0.6 + 26*0.7 + 29*0.8) / 2 = 76.2/2 = 38.1
        // w[5]: (6*0.1 + 9*0.2 + 12*0.3 + 15*0.4 + 21*0.5 + 24*0.6 + 27*0.7 + 30*0.8) / 2 = 79.8/2 = 39.9
        float[] expectedWeightGrads = {30.9f, 32.7f, 34.5f, 36.3f, 38.1f, 39.9f};
        assertArrayEquals("Weight gradients incorrect", expectedWeightGrads, backpropOutput.weightGradients[0], 1e-5f);

        // Expected input gradients (30 values for 2 batches * 15 inputs each):
        // Each deltaLoss distributes back through kernel weights to contributing input positions

        // Batch 1 input gradients:
        // input[0]: 0.5*0.1 = 0.05
        // input[1]: 1.0*0.1 = 0.1
        // input[2]: 1.5*0.1 = 0.15
        // input[3]: 2.0*0.1 + 0.5*0.2 = 0.2 + 0.1 = 0.3
        // input[4]: 2.5*0.1 + 1.0*0.2 = 0.25 + 0.2 = 0.45
        // input[5]: 3.0*0.1 + 1.5*0.2 = 0.3 + 0.3 = 0.6
        // input[6]: 2.0*0.2 + 0.5*0.3 = 0.4 + 0.15 = 0.55
        // input[7]: 2.5*0.2 + 1.0*0.3 = 0.5 + 0.3 = 0.8
        // input[8]: 3.0*0.2 + 1.5*0.3 = 0.6 + 0.45 = 1.05
        // input[9]: 2.0*0.3 + 0.5*0.4 = 0.6 + 0.2 = 0.8
        // input[10]: 2.5*0.3 + 1.0*0.4 = 0.75 + 0.4 = 1.15
        // input[11]: 3.0*0.3 + 1.5*0.4 = 0.9 + 0.6 = 1.5
        // input[12]: 2.0*0.4 = 0.8
        // input[13]: 2.5*0.4 = 1.0
        // input[14]: 3.0*0.4 = 1.2

        // Batch 2 input gradients:
        // input[15]: 0.5*0.5 = 0.25
        // input[16]: 1.0*0.5 = 0.5
        // input[17]: 1.5*0.5 = 0.75
        // input[18]: 2.0*0.5 + 0.5*0.6 = 1.0 + 0.3 = 1.3
        // input[19]: 2.5*0.5 + 1.0*0.6 = 1.25 + 0.6 = 1.85
        // input[20]: 3.0*0.5 + 1.5*0.6 = 1.5 + 0.9 = 2.4
        // input[21]: 2.0*0.6 + 0.5*0.7 = 1.2 + 0.35 = 1.55
        // input[22]: 2.5*0.6 + 1.0*0.7 = 1.5 + 0.7 = 2.2
        // input[23]: 3.0*0.6 + 1.5*0.7 = 1.8 + 1.05 = 2.85
        // input[24]: 2.0*0.7 + 0.5*0.8 = 1.4 + 0.4 = 1.8
        // input[25]: 2.5*0.7 + 1.0*0.8 = 1.75 + 0.8 = 2.55
        // input[26]: 3.0*0.7 + 1.5*0.8 = 2.1 + 1.2 = 3.3
        // input[27]: 2.0*0.8 = 1.6
        // input[28]: 2.5*0.8 = 2.0
        // input[29]: 3.0*0.8 = 2.4

        float[] expectedInputGrads = {
            // Batch 1
            0.05f, 0.1f, 0.15f, 0.3f, 0.45f, 0.6f, 0.55f, 0.8f, 1.05f, 0.8f, 1.15f, 1.5f, 0.8f, 1.0f, 1.2f,
            // Batch 2
            0.25f, 0.5f, 0.75f, 1.3f, 1.85f, 2.4f, 1.55f, 2.2f, 2.85f, 1.8f, 2.55f, 3.3f, 1.6f, 2.0f, 2.4f
        };

        assertArrayEquals("Input gradients incorrect", expectedInputGrads, backpropOutput.inputGradients, 1e-5f);
    }

    
    @Test
    public void testKernelWeightUpdatesGPUvsCPU() {
        // Create identical kernels for CPU and GPU (no Adam)
        Kernel cpuKernel = new Kernel(3);
        cpuKernel.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        cpuKernel.setBias(0.5f);

        Kernel gpuKernel = new Kernel(3);
        gpuKernel.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        gpuKernel.setBias(0.5f);

        // Training data
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] target = {2.5f, 3.5f, 4.5f};
        float learningRate = 0.01f;
        int iterations = 5;

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);
        gpuKernel.prepareGPU(stream);

        for (int iter = 0; iter < iterations; iter++) {
            // CPU training step
            ConvolutionUnitForwardOutput cpuForward = cpuKernel.feedForward(input, 1);
            float[] cpuGradOutput = new float[3];
            for (int i = 0; i < 3; i++) {
                cpuGradOutput[i] = cpuForward.output[i] - target[i];
            }
            ConvolutionUnitBackpropagateOutput cpuBackprop = cpuKernel.backpropagate(cpuForward, cpuGradOutput);
            cpuKernel.updateWeights(cpuBackprop.weightGradients[0], cpuBackprop.biasGradients[0], learningRate);

            // GPU training step
            CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
            ConvolutionUnitForwardOutputGPU gpuForward = gpuKernel.feedForwardGPU(inputGPU, 1, stream, handle);

            float[] gpuOutput = CudaUtil.fromGPUFloatAsync(gpuForward.output, 3, stream);
            JCudaDriver.cuStreamSynchronize(stream);

            float[] gpuGradOutput = new float[3];
            for (int i = 0; i < 3; i++) {
                gpuGradOutput[i] = gpuOutput[i] - target[i];
            }

            CUdeviceptr gpuGradOutputGPU = CudaUtil.toGPUAsync(gpuGradOutput, stream);
            ConvolutionUnitBackpropagateOutputGPU gpuBackprop = gpuKernel.backpropagateGPU(gpuForward, gpuGradOutputGPU, stream, handle);
            gpuKernel.updateWeightsGPU(gpuBackprop.weightGradients[0], gpuBackprop.biasGradients[0], learningRate, stream);
            JCudaDriver.cuStreamSynchronize(stream);

            // Compare weights each iteration
            float[] gpuWeights = CudaUtil.fromGPUFloatAsync(gpuKernel.weightsGPU, 3, stream);
            float gpuBias = CudaUtil.fromGPUFloatAsync(gpuKernel.biasGPU, 1, stream)[0];
            JCudaDriver.cuStreamSynchronize(stream);

            // Assert consistency each iteration
            assertArrayEquals("SGD weights should match at iteration " + iter, 
                cpuKernel.getWeights(), gpuWeights, 1e-5f);
            assertEquals("SGD bias should match at iteration " + iter, 
                cpuKernel.getBias(), gpuBias, 1e-5f);

            // Cleanup iteration resources
            CudaUtil.freeAsync(inputGPU, stream);
            CudaUtil.freeAsync(gpuGradOutputGPU, stream);
            gpuForward.freeAsync(stream);
            gpuBackprop.freeAsync(stream);
        }

        // Cleanup
        gpuKernel.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testKernelWeightUpdatesAdamGPUvsCPU() {
        // Create identical kernels for CPU and GPU with Adam
        Kernel cpuKernel = new Kernel(3);
        cpuKernel.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        cpuKernel.setBias(0.5f);
        cpuKernel.setUseAdam(true);

        Kernel gpuKernel = new Kernel(3);
        gpuKernel.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        gpuKernel.setBias(0.5f);
        gpuKernel.setUseAdam(true);

        // Training data
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] target = {2.5f, 3.5f, 4.5f};
        float learningRate = 0.01f;
        int iterations = 5;

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);
        gpuKernel.prepareGPU(stream);

        for (int iter = 0; iter < iterations; iter++) {
            // CPU training step
            ConvolutionUnitForwardOutput cpuForward = cpuKernel.feedForward(input, 1);
            float[] cpuGradOutput = new float[3];
            for (int i = 0; i < 3; i++) {
                cpuGradOutput[i] = cpuForward.output[i] - target[i];
            }
            ConvolutionUnitBackpropagateOutput cpuBackprop = cpuKernel.backpropagate(cpuForward, cpuGradOutput);
            cpuKernel.updateWeights(cpuBackprop.weightGradients[0], cpuBackprop.biasGradients[0], learningRate);

            // GPU training step
            CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
            ConvolutionUnitForwardOutputGPU gpuForward = gpuKernel.feedForwardGPU(inputGPU, 1, stream, handle);

            float[] gpuOutput = CudaUtil.fromGPUFloatAsync(gpuForward.output, 3, stream);
            JCudaDriver.cuStreamSynchronize(stream);

            float[] gpuGradOutput = new float[3];
            for (int i = 0; i < 3; i++) {
                gpuGradOutput[i] = gpuOutput[i] - target[i];
            }

            CUdeviceptr gpuGradOutputGPU = CudaUtil.toGPUAsync(gpuGradOutput, stream);
            ConvolutionUnitBackpropagateOutputGPU gpuBackprop = gpuKernel.backpropagateGPU(gpuForward, gpuGradOutputGPU, stream, handle);
            gpuKernel.updateWeightsGPU(gpuBackprop.weightGradients[0], gpuBackprop.biasGradients[0], learningRate, stream);
            JCudaDriver.cuStreamSynchronize(stream);

            // Compare weights and Adam states each iteration
            float[] gpuWeights = CudaUtil.fromGPUFloatAsync(gpuKernel.weightsGPU, 3, stream);
            float gpuBias = CudaUtil.fromGPUFloatAsync(gpuKernel.biasGPU, 1, stream)[0];
            float[] gpuWeightMomentum = CudaUtil.fromGPUFloatAsync(gpuKernel.weightMomentumGPU, 3, stream);
            float[] gpuWeightVelocity = CudaUtil.fromGPUFloatAsync(gpuKernel.weightVelocityGPU, 3, stream);
            float gpuBiasMomentum = CudaUtil.fromGPUFloatAsync(gpuKernel.biasMomentumGPU, 1, stream)[0];
            float gpuBiasVelocity = CudaUtil.fromGPUFloatAsync(gpuKernel.biasVelocityGPU, 1, stream)[0];
            JCudaDriver.cuStreamSynchronize(stream);

            // Assert consistency each iteration
            assertArrayEquals("Adam weights should match at iteration " + iter, 
                cpuKernel.getWeights(), gpuWeights, 1e-5f);
            assertEquals("Adam bias should match at iteration " + iter, 
                cpuKernel.getBias(), gpuBias, 1e-5f);
            assertArrayEquals("Adam weight momentum should match at iteration " + iter,
                cpuKernel.weightMomentum, gpuWeightMomentum, 1e-5f);
            assertArrayEquals("Adam weight velocity should match at iteration " + iter,
                cpuKernel.weightVelocity, gpuWeightVelocity, 1e-5f);
            assertEquals("Adam bias momentum should match at iteration " + iter,
                cpuKernel.biasMomentum, gpuBiasMomentum, 1e-5f);
            assertEquals("Adam bias velocity should match at iteration " + iter,
                cpuKernel.biasVelocity, gpuBiasVelocity, 1e-5f);

            // Cleanup iteration resources
            CudaUtil.freeAsync(inputGPU, stream);
            CudaUtil.freeAsync(gpuGradOutputGPU, stream);
            gpuForward.freeAsync(stream);
            gpuBackprop.freeAsync(stream);
        }

        // Cleanup
        gpuKernel.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    
    @Test
    public void testKernelWithLargeInput() {
        // Create kernel with width 4
        Kernel kernel = new Kernel(4);
        kernel.setWeights(new float[]{0.25f, 0.25f, 0.25f, 0.25f});
        kernel.setBias(0.0f);
        
        // Generate large input (10000 elements)
        int inputSize = 10000;
        float[] input = new float[inputSize];
        for (int i = 0; i < inputSize; i++) {
            input[i] = i % 10;  // Repeating pattern
        }
        
        // Perform forward pass
        float[] output = kernel.feedForward(input, 1).output;
        
        // Validate random samples (avoiding checking everything)
        assertEquals("Large input sample 0", 0.25f * (0+1+2+3), output[0], 1e-5f);
        assertEquals("Large input sample 100", 0.25f * (100%10+(101%10)+(102%10)+(103%10)), output[100], 1e-5f);
        assertEquals("Large input sample 9996", 0.25f * (9996%10+(9997%10)+(9998%10)+(9999%10)), output[9996], 1e-5f);
    }
    
    @Test
    public void testKernelWithEvenSizedWidth() {
        // Create kernel with even width (4)
        Kernel kernel = new Kernel(4);
        kernel.setWeights(new float[]{0.25f, 0.25f, 0.25f, 0.25f});
        kernel.setBias(1.0f);
        
        // Input
        float[] input = {1, 2, 3, 4, 5, 6};
        
        // Expected: each output is average of 4 inputs + bias
        float[] expected = {1 + (1+2+3+4)*0.25f, 1 + (2+3+4+5)*0.25f, 1 + (3+4+5+6)*0.25f};
        
        // Output size = 6 - 4 + 1 = 3
        // Perform forward pass
        float[] output = kernel.feedForward(input, 1).output;
        
        // Check results
        assertArrayEquals("Even kernel width output incorrect", expected, output, 1e-5f);
    }
    
    @Test
    public void testEdgeCaseMinimalInput() {
        // Create kernel with width equal to input size
        Kernel kernel = new Kernel(3);
        kernel.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        kernel.setBias(0.5f);
        
        // Input exactly the size of kernel
        float[] input = {1.0f, 2.0f, 3.0f};
        
        // Expected: just one output value
        // 1.0*0.1 + 2.0*0.2 + 3.0*0.3 + 0.5 = 1.9
        float[] expected = {1.9f};
        
        // Output size = 3 - 3 + 1 = 1
        float[] output = kernel.feedForward(input, 1).output;
        
        // Check results
        assertArrayEquals("Minimal input case failed", expected, output, 1e-5f);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testInputValidation() {
        Kernel kernel = new Kernel(2, 2); // width = 4
        float[] input = {1.0f, 2.0f, 3.0f}; // length 3, not divisible by 4
        kernel.feedForward(input, 1);
    }    
}