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

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.cublasHandle;
import org.fjnn.activation.Sigmoid;
import org.fjnn.convolution.Kernel;
import org.fjnn.convolution.KernelGroup;
import org.fjnn.convolution.output.unit.*;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaUtil;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Test class for KernelGroup functionality
 * 
 * @author ahmed
 */
public class KernelGroupTest extends ConvolutionBaseTest {

    @Test
    public void testKernelGroupForward() {
        // Test different unit configurations - all kernels have unitCount=1
        Kernel kernel1 = new Kernel(2, 1); // unitSize=2, unitCount=1, width=2
        kernel1.setWeights(new float[]{0.3f, 0.7f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 1); // unitSize=3, unitCount=1, width=3  
        kernel2.setWeights(new float[]{0.4f, 0.6f, 0.2f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(1, 1); // unitSize=1, unitCount=1, width=1
        kernel3.setWeights(new float[]{0.8f});
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        // chunkSize = 2 + 3 + 1 = 6
        // 3 chunks with unitCount=1 gives outputSize = 3-1+1 = 3 for all kernels
        float[] input = {
            // chunk 0: ch1=[0.1,0.2], ch2=[0.3,0.4,0.5], ch3=[0.6]
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
            // chunk 1: ch1=[0.11,0.21], ch2=[0.31,0.41,0.51], ch3=[0.61]  
            0.11f, 0.21f, 0.31f, 0.41f, 0.51f, 0.61f,
            // chunk 2: ch1=[0.12,0.22], ch2=[0.32,0.42,0.52], ch3=[0.62]
            0.12f, 0.22f, 0.32f, 0.42f, 0.52f, 0.62f
        };

        ConvolutionUnitForwardOutput output = group.feedForward(input, 1);

        assertEquals("Output size", 3, output.outputSize);
        assertEquals("Batch size", 1, output.batchSize);

        // Manual calculations verified:
        // Channel 1: [0.1,0.2,0.11,0.21,0.12,0.22] -> outputs: [0.17, 0.18, 0.19]
        // Channel 2: [0.3,0.4,0.5,0.31,0.41,0.51,0.32,0.42,0.52] -> outputs: [0.46, 0.472, 0.484]  
        // Channel 3: [0.6,0.61,0.62] -> outputs: [0.48, 0.488, 0.496]

        // Verify kernel raw outputs
        assertArrayEquals("Kernel 1 outputs", new float[]{0.17f, 0.18f, 0.19f}, output.kernelOutputs[0], 1e-4f);
        assertArrayEquals("Kernel 2 outputs", new float[]{0.46f, 0.472f, 0.484f}, output.kernelOutputs[1], 1e-4f);
        assertArrayEquals("Kernel 3 outputs", new float[]{0.48f, 0.488f, 0.496f}, output.kernelOutputs[2], 1e-4f);

        // Sigmoid values (verified):
        final float SIGMOID_CH1_POS0 = 0.5424f; // sigmoid(0.17)
        final float SIGMOID_CH2_POS0 = 0.6130f; // sigmoid(0.46)
        final float SIGMOID_CH3_POS0 = 0.6177f; // sigmoid(0.48)

        final float SIGMOID_CH1_POS1 = 0.5449f; // sigmoid(0.18)
        final float SIGMOID_CH2_POS1 = 0.6158f; // sigmoid(0.472)
        final float SIGMOID_CH3_POS1 = 0.6196f; // sigmoid(0.488)

        final float SIGMOID_CH1_POS2 = 0.5474f; // sigmoid(0.19)
        final float SIGMOID_CH2_POS2 = 0.6186f; // sigmoid(0.484)
        final float SIGMOID_CH3_POS2 = 0.6215f; // sigmoid(0.496)

        // Verify sigmoid outputs
        assertArrayEquals("Kernel 1 sigmoid", new float[]{SIGMOID_CH1_POS0, SIGMOID_CH1_POS1, SIGMOID_CH1_POS2}, output.sigmoidOutputs[0], 1e-4f);
        assertArrayEquals("Kernel 2 sigmoid", new float[]{SIGMOID_CH2_POS0, SIGMOID_CH2_POS1, SIGMOID_CH2_POS2}, output.sigmoidOutputs[1], 1e-4f);
        assertArrayEquals("Kernel 3 sigmoid", new float[]{SIGMOID_CH3_POS0, SIGMOID_CH3_POS1, SIGMOID_CH3_POS2}, output.sigmoidOutputs[2], 1e-4f);

        // AND operation results (verified):
        float firstOutputExpected = 0.2054f;   // 0.5424 * 0.6129 * 0.6179
        float secondOutputExpected = 0.2079f;  // 0.5449 * 0.6158 * 0.6198
        float thirdOutputExpected = 0.2105f;   // 0.5474 * 0.6186 * 0.6217

        // Verify final AND outputs
        assertEquals("Forward output 0", firstOutputExpected, output.output[0], 1e-4f);
        assertEquals("Forward output 1", secondOutputExpected, output.output[1], 1e-4f);
        assertEquals("Forward output 2", thirdOutputExpected, output.output[2], 1e-4f);
    }
    
    @Test
    public void testKernelGroupForwardWithUnits() {
        // Create three kernels with unitCount=2
        Kernel kernel1 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4
        kernel1.setWeights(new float[]{0.3f, 0.7f, 0.4f, 0.6f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 2); // unitSize=3, unitCount=2, width=6
        kernel2.setWeights(new float[]{0.2f, 0.5f, 0.3f, 0.1f, 0.4f, 0.5f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        kernel3.setWeights(new float[]{0.8f, 0.9f});
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        // chunkSize = 2 + 3 + 1 = 6
        // We need exactly 3 chunks to get outputSize = 3-2+1 = 2 for all kernels
        float[] input = {
            // chunk 0: ch1=[0.1,0.2], ch2=[0.3,0.4,0.5], ch3=[0.6]
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
            // chunk 1: ch1=[0.11,0.21], ch2=[0.31,0.41,0.51], ch3=[0.61]  
            0.11f, 0.21f, 0.31f, 0.41f, 0.51f, 0.61f,
            // chunk 2: ch1=[0.12,0.22], ch2=[0.32,0.42,0.52], ch3=[0.62]
            0.12f, 0.22f, 0.32f, 0.42f, 0.52f, 0.62f
        };

        ConvolutionUnitForwardOutput output = group.feedForward(input, 1);

        assertEquals("Output size", 2, output.outputSize);
        assertEquals("Batch size", 1, output.batchSize);

        // Manual calculations:
        // Channel 1: [0.1,0.2,0.11,0.21,0.12,0.22] -> outputs: [0.34, 0.36]
        // Channel 2: [0.3,0.4,0.5,0.31,0.41,0.51,0.32,0.42,0.52] -> outputs: [0.86, 0.88]  
        // Channel 3: [0.6,0.61,0.62] -> outputs: [1.029, 1.046]

        // Apply sigmoid to each kernel output:
        final float SIGMOID_CH1_POS0 = 0.5842f;
        final float SIGMOID_CH2_POS0 = 0.7026f;  
        final float SIGMOID_CH3_POS0 = 0.7367f;

        final float SIGMOID_CH1_POS1 = 0.5891f;
        final float SIGMOID_CH2_POS1 = 0.7068f;
        final float SIGMOID_CH3_POS1 = 0.7400f;

        // AND operation (multiply sigmoids)
        float firstOutputExpected = 0.3024f;
        float secondOutputExpected = 0.3081f;

        // Verify kernel raw outputs
        assertArrayEquals("Kernel 1 outputs", new float[]{0.34f, 0.36f}, output.kernelOutputs[0], 1e-4f);
        assertArrayEquals("Kernel 2 outputs", new float[]{0.86f, 0.88f}, output.kernelOutputs[1], 1e-4f);
        assertArrayEquals("Kernel 3 outputs", new float[]{1.029f, 1.046f}, output.kernelOutputs[2], 1e-4f);
        
        // Verify sigmoid outputs
        assertArrayEquals("Kernel 1 sigmoid", new float[]{SIGMOID_CH1_POS0, SIGMOID_CH1_POS1}, output.sigmoidOutputs[0], 1e-4f);
        assertArrayEquals("Kernel 2 sigmoid", new float[]{SIGMOID_CH2_POS0, SIGMOID_CH2_POS1}, output.sigmoidOutputs[1], 1e-4f);
        assertArrayEquals("Kernel 3 sigmoid", new float[]{SIGMOID_CH3_POS0, SIGMOID_CH3_POS1}, output.sigmoidOutputs[2], 1e-4f);
        
        // Verify final AND outputs
        assertEquals("Forward output 0", firstOutputExpected, output.output[0], 1e-4f);
        assertEquals("Forward output 1", secondOutputExpected, output.output[1], 1e-4f);
    }
    
    @Test
    public void testKernelGroupForwardMultiBatch() {
        // Same kernel configuration as single batch test
        Kernel kernel1 = new Kernel(2, 1); // unitSize=2, unitCount=1, width=2
        kernel1.setWeights(new float[]{0.3f, 0.7f});
        kernel1.setBias(0.0f);
        
        Kernel kernel2 = new Kernel(3, 1); // unitSize=3, unitCount=1, width=3  
        kernel2.setWeights(new float[]{0.4f, 0.6f, 0.2f});
        kernel2.setBias(0.0f);
        
        Kernel kernel3 = new Kernel(1, 1); // unitSize=1, unitCount=1, width=1
        kernel3.setWeights(new float[]{0.8f});
        kernel3.setBias(0.0f);
        
        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);
        
        // chunkSize = 2 + 3 + 1 = 6
        // 2 batches, 3 chunks each, outputSize = 3 per batch
        float[] input = {
            // Batch 1
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,     // chunk 0
            0.11f, 0.21f, 0.31f, 0.41f, 0.51f, 0.61f, // chunk 1
            0.12f, 0.22f, 0.32f, 0.42f, 0.52f, 0.62f, // chunk 2
            // Batch 2
            0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,     // chunk 0
            0.21f, 0.31f, 0.41f, 0.51f, 0.61f, 0.71f, // chunk 1
            0.22f, 0.32f, 0.42f, 0.52f, 0.62f, 0.72f  // chunk 2
        };
        
        ConvolutionUnitForwardOutput output = group.feedForward(input, 2);
        
        assertEquals("Output size", 3, output.outputSize);
        assertEquals("Batch size", 2, output.batchSize);
        
        // Manual calculations verified:
        // Batch 1: Same as single batch test
        // Batch 2: 
        // Channel 1: [0.2,0.3,0.21,0.31,0.22,0.32] -> outputs: [0.27, 0.28, 0.29]
        // Channel 2: [0.4,0.5,0.6,0.41,0.51,0.61,0.42,0.52,0.62] -> outputs: [0.58, 0.592, 0.604]
        // Channel 3: [0.7,0.71,0.72] -> outputs: [0.56, 0.568, 0.576]
        
        // Verify kernel raw outputs (batch 1 + batch 2)
        assertArrayEquals("Kernel 1 outputs", new float[]{0.17f, 0.18f, 0.19f, 0.27f, 0.28f, 0.29f}, output.kernelOutputs[0], 1e-4f);
        assertArrayEquals("Kernel 2 outputs", new float[]{0.46f, 0.472f, 0.484f, 0.58f, 0.592f, 0.604f}, output.kernelOutputs[1], 1e-4f);
        assertArrayEquals("Kernel 3 outputs", new float[]{0.48f, 0.488f, 0.496f, 0.56f, 0.568f, 0.576f}, output.kernelOutputs[2], 1e-4f);
        
        // Sigmoid values for batch 2:
        final float SIGMOID_B2_CH1_POS0 = 0.5671f; // sigmoid(0.27)
        final float SIGMOID_B2_CH2_POS0 = 0.6410f; // sigmoid(0.58)
        final float SIGMOID_B2_CH3_POS0 = 0.6365f; // sigmoid(0.56)
        
        final float SIGMOID_B2_CH1_POS1 = 0.5696f; // sigmoid(0.28)
        final float SIGMOID_B2_CH2_POS1 = 0.6438f; // sigmoid(0.592)
        final float SIGMOID_B2_CH3_POS1 = 0.6384f; // sigmoid(0.568)
        
        final float SIGMOID_B2_CH1_POS2 = 0.5719f; // sigmoid(0.29)
        final float SIGMOID_B2_CH2_POS2 = 0.6465f; // sigmoid(0.604)
        final float SIGMOID_B2_CH3_POS2 = 0.6401f; // sigmoid(0.576)
        
        // Verify sigmoid outputs (batch 1 + batch 2)
        assertArrayEquals("Kernel 1 sigmoid", new float[]{0.5424f, 0.5449f, 0.5474f, SIGMOID_B2_CH1_POS0, SIGMOID_B2_CH1_POS1, SIGMOID_B2_CH1_POS2}, output.sigmoidOutputs[0], 1e-4f);
        assertArrayEquals("Kernel 2 sigmoid", new float[]{0.6130f, 0.6158f, 0.6186f, SIGMOID_B2_CH2_POS0, SIGMOID_B2_CH2_POS1, SIGMOID_B2_CH2_POS2}, output.sigmoidOutputs[1], 1e-4f);
        assertArrayEquals("Kernel 3 sigmoid", new float[]{0.6177f, 0.6196f, 0.6215f, SIGMOID_B2_CH3_POS0, SIGMOID_B2_CH3_POS1, SIGMOID_B2_CH3_POS2}, output.sigmoidOutputs[2], 1e-4f);
        
        // AND operation results for batch 2:
        float batch2Pos0Expected = 0.2313f; // 0.5671 * 0.6410 * 0.6365
        float batch2Pos1Expected = 0.2340f; // 0.5696 * 0.6438 * 0.6384
        float batch2Pos2Expected = 0.2367f; // 0.5721 * 0.6465 * 0.6403
        
        // Verify final AND outputs (batch 1 + batch 2)
        assertArrayEquals("Final outputs", new float[]{0.2054f, 0.2079f, 0.21047f, batch2Pos0Expected, batch2Pos1Expected, batch2Pos2Expected}, output.output, 1e-4f);
    }
    
    @Test
    public void testKernelGroupForwardWithUnitsMultiBatch() {
        // Test with 3 kernels having same unitCount=2 for multi-batch
        Kernel kernel1 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4
        kernel1.setWeights(new float[]{0.3f, 0.7f, 0.4f, 0.6f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        kernel2.setWeights(new float[]{0.4f, 0.6f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(3, 2); // unitSize=3, unitCount=2, width=6
        kernel3.setWeights(new float[]{0.2f, 0.8f, 0.1f, 0.5f, 0.3f, 0.9f});
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        // inputStride = 2 + 1 + 3 = 6
        // 2 batches with 6 chunks each: 2 * 6 * 6 = 72 values
        float[] input = {
            // Batch 1 - 36 values = 6 chunks of 6 values each
            // chunk 0: ch1=[0.1,0.2], ch2=[0.15], ch3=[0.25,0.3,0.4]
            0.1f, 0.2f, 0.15f, 0.25f, 0.3f, 0.4f,
            // chunk 1: ch1=[0.11,0.21], ch2=[0.16], ch3=[0.26,0.31,0.41]
            0.11f, 0.21f, 0.16f, 0.26f, 0.31f, 0.41f,
            // chunk 2: ch1=[0.12,0.22], ch2=[0.17], ch3=[0.27,0.32,0.42]
            0.12f, 0.22f, 0.17f, 0.27f, 0.32f, 0.42f,
            // chunk 3: ch1=[0.13,0.23], ch2=[0.18], ch3=[0.28,0.33,0.43]
            0.13f, 0.23f, 0.18f, 0.28f, 0.33f, 0.43f,
            // chunk 4: ch1=[0.14,0.24], ch2=[0.19], ch3=[0.29,0.34,0.44]
            0.14f, 0.24f, 0.19f, 0.29f, 0.34f, 0.44f,
            // chunk 5: ch1=[0.15,0.25], ch2=[0.2], ch3=[0.3,0.35,0.45]
            0.15f, 0.25f, 0.2f, 0.3f, 0.35f, 0.45f,

            // Batch 2 - 36 values = 6 chunks of 6 values each
            // chunk 0: ch1=[0.2,0.3], ch2=[0.25], ch3=[0.35,0.4,0.5]
            0.2f, 0.3f, 0.25f, 0.35f, 0.4f, 0.5f,
            // chunk 1: ch1=[0.21,0.31], ch2=[0.26], ch3=[0.36,0.41,0.51]
            0.21f, 0.31f, 0.26f, 0.36f, 0.41f, 0.51f,
            // chunk 2: ch1=[0.22,0.32], ch2=[0.27], ch3=[0.37,0.42,0.52]
            0.22f, 0.32f, 0.27f, 0.37f, 0.42f, 0.52f,
            // chunk 3: ch1=[0.23,0.33], ch2=[0.28], ch3=[0.38,0.43,0.53]
            0.23f, 0.33f, 0.28f, 0.38f, 0.43f, 0.53f,
            // chunk 4: ch1=[0.24,0.34], ch2=[0.29], ch3=[0.39,0.44,0.54]
            0.24f, 0.34f, 0.29f, 0.39f, 0.44f, 0.54f,
            // chunk 5: ch1=[0.25,0.35], ch2=[0.3], ch3=[0.4,0.45,0.55]
            0.25f, 0.35f, 0.3f, 0.4f, 0.45f, 0.55f
        };

        ConvolutionUnitForwardOutput output = group.feedForward(input, 2);

        // Each kernel processes extracted channel data: 6 chunks with unitCount=2 gives outputSize = 6-2+1 = 5
        assertEquals("Units multi-batch output size", 5, output.outputSize);
        assertEquals("Units multi-batch batch size", 2, output.batchSize);

        // Manual calculations (sliding window by unitSize for each kernel):
        // Batch 1 channels:
        // Channel 1: [0.1,0.2,0.11,0.21,0.12,0.22,0.13,0.23,0.14,0.24,0.15,0.25] (12 values, unitSize=2)
        //   pos0: [0.1,0.2,0.11,0.21] → 0.1×0.3 + 0.2×0.7 + 0.11×0.4 + 0.21×0.6 = 0.03+0.14+0.044+0.126 = 0.34
        //   pos1: [0.11,0.21,0.12,0.22] → 0.11×0.3 + 0.21×0.7 + 0.12×0.4 + 0.22×0.6 = 0.033+0.147+0.048+0.132 = 0.36
        //   pos2: [0.12,0.22,0.13,0.23] → 0.12×0.3 + 0.22×0.7 + 0.13×0.4 + 0.23×0.6 = 0.036+0.154+0.052+0.138 = 0.38
        //   pos3: [0.13,0.23,0.14,0.24] → 0.13×0.3 + 0.23×0.7 + 0.14×0.4 + 0.24×0.6 = 0.039+0.161+0.056+0.144 = 0.4
        //   pos4: [0.14,0.24,0.15,0.25] → 0.14×0.3 + 0.24×0.7 + 0.15×0.4 + 0.25×0.6 = 0.042+0.168+0.06+0.15 = 0.42

        // Channel 2: [0.15,0.16,0.17,0.18,0.19,0.2] (6 values, unitSize=1)
        //   pos0: [0.15,0.16] → 0.15×0.4 + 0.16×0.6 = 0.06+0.096 = 0.156
        //   pos1: [0.16,0.17] → 0.16×0.4 + 0.17×0.6 = 0.064+0.102 = 0.166
        //   pos2: [0.17,0.18] → 0.17×0.4 + 0.18×0.6 = 0.068+0.108 = 0.176
        //   pos3: [0.18,0.19] → 0.18×0.4 + 0.19×0.6 = 0.072+0.114 = 0.186
        //   pos4: [0.19,0.2] → 0.19×0.4 + 0.2×0.6 = 0.076+0.12 = 0.196

        // Channel 3: [0.25,0.3,0.4,0.26,0.31,0.41,0.27,0.32,0.42,0.28,0.33,0.43,0.29,0.34,0.44,0.3,0.35,0.45] (18 values, unitSize=3)
        //   pos0: [0.25,0.3,0.4,0.26,0.31,0.41] → 0.25×0.2 + 0.3×0.8 + 0.4×0.1 + 0.26×0.5 + 0.31×0.3 + 0.41×0.9 = 0.05+0.24+0.04+0.13+0.093+0.369 = 0.922
        //   pos1: [0.26,0.31,0.41,0.27,0.32,0.42] → 0.26×0.2 + 0.31×0.8 + 0.41×0.1 + 0.27×0.5 + 0.32×0.3 + 0.42×0.9 = 0.052+0.248+0.041+0.135+0.096+0.378 = 0.95
        //   pos2: [0.27,0.32,0.42,0.28,0.33,0.43] → 0.27×0.2 + 0.32×0.8 + 0.42×0.1 + 0.28×0.5 + 0.33×0.3 + 0.43×0.9 = 0.054+0.256+0.042+0.14+0.099+0.387 = 0.978
        //   pos3: [0.28,0.33,0.43,0.29,0.34,0.44] → 0.28×0.2 + 0.33×0.8 + 0.43×0.1 + 0.29×0.5 + 0.34×0.3 + 0.44×0.9 = 0.056+0.264+0.043+0.145+0.102+0.396 = 1.006
        //   pos4: [0.29,0.34,0.44,0.3,0.35,0.45] → 0.29×0.2 + 0.34×0.8 + 0.44×0.1 + 0.3×0.5 + 0.35×0.3 + 0.45×0.9 = 0.058+0.272+0.044+0.15+0.105+0.405 = 1.034

        // Batch 2 channels (similar calculations):
        // Channel 1: [0.2,0.3,0.21,0.31,0.22,0.32,0.23,0.33,0.24,0.34,0.25,0.35]
        //   pos0: [0.2,0.3,0.21,0.31] → 0.2×0.3 + 0.3×0.7 + 0.21×0.4 + 0.31×0.6 = 0.06+0.21+0.084+0.186 = 0.54
        //   pos1: [0.21,0.31,0.22,0.32] → 0.21×0.3 + 0.31×0.7 + 0.22×0.4 + 0.32×0.6 = 0.063+0.217+0.088+0.192 = 0.56
        //   pos2: [0.22,0.32,0.23,0.33] → 0.22×0.3 + 0.32×0.7 + 0.23×0.4 + 0.33×0.6 = 0.066+0.224+0.092+0.198 = 0.58
        //   pos3: [0.23,0.33,0.24,0.34] → 0.23×0.3 + 0.33×0.7 + 0.24×0.4 + 0.34×0.6 = 0.069+0.231+0.096+0.204 = 0.6
        //   pos4: [0.24,0.34,0.25,0.35] → 0.24×0.3 + 0.34×0.7 + 0.25×0.4 + 0.35×0.6 = 0.072+0.238+0.1+0.21 = 0.62

        // Channel 2: [0.25,0.26,0.27,0.28,0.29,0.3]
        //   pos0: [0.25,0.26] → 0.25×0.4 + 0.26×0.6 = 0.1+0.156 = 0.256
        //   pos1: [0.26,0.27] → 0.26×0.4 + 0.27×0.6 = 0.104+0.162 = 0.266
        //   pos2: [0.27,0.28] → 0.27×0.4 + 0.28×0.6 = 0.108+0.168 = 0.276
        //   pos3: [0.28,0.29] → 0.28×0.4 + 0.29×0.6 = 0.112+0.174 = 0.286
        //   pos4: [0.29,0.3] → 0.29×0.4 + 0.3×0.6 = 0.116+0.18 = 0.296

        // Channel 3: [0.35,0.4,0.5,0.36,0.41,0.51,0.37,0.42,0.52,0.38,0.43,0.53,0.39,0.44,0.54,0.4,0.45,0.55]
        //   pos0: [0.35,0.4,0.5,0.36,0.41,0.51] → 0.35×0.2 + 0.4×0.8 + 0.5×0.1 + 0.36×0.5 + 0.41×0.3 + 0.51×0.9 = 0.07+0.32+0.05+0.18+0.123+0.459 = 1.202
        //   pos1: [0.36,0.41,0.51,0.37,0.42,0.52] → 0.36×0.2 + 0.41×0.8 + 0.51×0.1 + 0.37×0.5 + 0.42×0.3 + 0.52×0.9 = 0.072+0.328+0.051+0.185+0.126+0.468 = 1.23
        //   pos2: [0.37,0.42,0.52,0.38,0.43,0.53] → 0.37×0.2 + 0.42×0.8 + 0.52×0.1 + 0.38×0.5 + 0.43×0.3 + 0.53×0.9 = 0.074+0.336+0.052+0.19+0.129+0.477 = 1.258
        //   pos3: [0.38,0.43,0.53,0.39,0.44,0.54] → 0.38×0.2 + 0.43×0.8 + 0.53×0.1 + 0.39×0.5 + 0.44×0.3 + 0.54×0.9 = 0.076+0.344+0.053+0.195+0.132+0.486 = 1.286
        //   pos4: [0.39,0.44,0.54,0.4,0.45,0.55] → 0.39×0.2 + 0.44×0.8 + 0.54×0.1 + 0.4×0.5 + 0.45×0.3 + 0.55×0.9 = 0.078+0.352+0.054+0.2+0.135+0.495 = 1.314

        // Verify kernel raw outputs (batch 1 + batch 2)
        assertArrayEquals("Kernel 1 outputs", 
            new float[]{0.34f, 0.36f, 0.38f, 0.4f, 0.42f, 0.54f, 0.56f, 0.58f, 0.6f, 0.62f}, 
            output.kernelOutputs[0], 1e-3f);
        assertArrayEquals("Kernel 2 outputs", 
            new float[]{0.156f, 0.166f, 0.176f, 0.186f, 0.196f, 0.256f, 0.266f, 0.276f, 0.286f, 0.296f}, 
            output.kernelOutputs[1], 1e-3f);
        assertArrayEquals("Kernel 3 outputs", 
            new float[]{0.922f, 0.95f, 0.978f, 1.006f, 1.034f, 1.202f, 1.23f, 1.258f, 1.286f, 1.314f}, 
            output.kernelOutputs[2], 1e-3f);

        // Apply sigmoid to get sigmoid outputs
        final float[] SIGMOID_B1_CH1 = {0.5842f, 0.5891f, 0.5939f, 0.5987f, 0.6034f}; // sigmoid of batch1 kernel1 outputs
        final float[] SIGMOID_B1_CH2 = {0.5389f, 0.5414f, 0.5439f, 0.5464f, 0.5488f}; // sigmoid of batch1 kernel2 outputs
        final float[] SIGMOID_B1_CH3 = {0.7154f, 0.7211f, 0.7268f, 0.7322f, 0.7377f}; // sigmoid of batch1 kernel3 outputs

        final float[] SIGMOID_B2_CH1 = {0.6318f, 0.6365f, 0.6410f, 0.6456f, 0.6502f}; // sigmoid of batch2 kernel1 outputs
        final float[] SIGMOID_B2_CH2 = {0.5637f, 0.5662f, 0.5685f, 0.5710f, 0.5734f}; // sigmoid of batch2 kernel2 outputs
        final float[] SIGMOID_B2_CH3 = {0.7688f, 0.7738f, 0.7787f, 0.7835f, 0.7881f}; // sigmoid of batch2 kernel3 outputs

        // Verify sigmoid outputs (batch 1 + batch 2)
        float[] expectedSigmoid1 = new float[10];
        System.arraycopy(SIGMOID_B1_CH1, 0, expectedSigmoid1, 0, 5);
        System.arraycopy(SIGMOID_B2_CH1, 0, expectedSigmoid1, 5, 5);
        assertArrayEquals("Kernel 1 sigmoid", expectedSigmoid1, output.sigmoidOutputs[0], 1e-4f);

        float[] expectedSigmoid2 = new float[10];
        System.arraycopy(SIGMOID_B1_CH2, 0, expectedSigmoid2, 0, 5);
        System.arraycopy(SIGMOID_B2_CH2, 0, expectedSigmoid2, 5, 5);
        assertArrayEquals("Kernel 2 sigmoid", expectedSigmoid2, output.sigmoidOutputs[1], 1e-4f);

        float[] expectedSigmoid3 = new float[10];
        System.arraycopy(SIGMOID_B1_CH3, 0, expectedSigmoid3, 0, 5);
        System.arraycopy(SIGMOID_B2_CH3, 0, expectedSigmoid3, 5, 5);
        assertArrayEquals("Kernel 3 sigmoid", expectedSigmoid3, output.sigmoidOutputs[2], 1e-4f);

        // AND operation results (multiply sigmoids):
        float[] expectedFinalOutput = new float[10];
        for (int i = 0; i < 5; i++) {
            // Batch 1
            expectedFinalOutput[i] = SIGMOID_B1_CH1[i] * SIGMOID_B1_CH2[i] * SIGMOID_B1_CH3[i];
            // Batch 2  
            expectedFinalOutput[i + 5] = SIGMOID_B2_CH1[i] * SIGMOID_B2_CH2[i] * SIGMOID_B2_CH3[i];
        }

        // Verify final AND outputs
        assertArrayEquals("Final outputs", expectedFinalOutput, output.output, 1e-4f);
    }

    
    @Test
    public void testKernelGroupForwardGPU() {
        // Same configuration as CPU test
        Kernel kernel1 = new Kernel(2, 1);
        kernel1.setWeights(new float[]{0.3f, 0.7f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 1);
        kernel2.setWeights(new float[]{0.4f, 0.6f, 0.2f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(1, 1);
        kernel3.setWeights(new float[]{0.8f});
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        float[] input = {
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
            0.11f, 0.21f, 0.31f, 0.41f, 0.51f, 0.61f,
            0.12f, 0.22f, 0.32f, 0.42f, 0.52f, 0.62f
        };

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        // Prepare GPU resources
        group.prepareGPU(stream);

        // Get CPU output for comparison
        ConvolutionUnitForwardOutput cpuOutput = group.feedForward(input, 1);

        // GPU forward pass
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionUnitForwardOutputGPU gpuOutput = group.feedForwardGPU(inputGPU, 1, stream, handle);

        // Get results back to CPU
        float[] output = CudaUtil.fromGPUFloatAsync(gpuOutput.output, cpuOutput.output.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        // Verify results match CPU implementation
        assertArrayEquals("GPU output should match CPU", cpuOutput.output, output, 1e-4f);

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput.freeAsync(stream);
        group.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testKernelGroupForwardGPUWithUnits() {
        // Test with unitCount=2
        Kernel kernel1 = new Kernel(2, 2);
        kernel1.setWeights(new float[]{0.3f, 0.7f, 0.4f, 0.6f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 2);
        kernel2.setWeights(new float[]{0.2f, 0.5f, 0.3f, 0.1f, 0.4f, 0.5f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(1, 2);
        kernel3.setWeights(new float[]{0.8f, 0.9f});
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        float[] input = {
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
            0.11f, 0.21f, 0.31f, 0.41f, 0.51f, 0.61f,
            0.12f, 0.22f, 0.32f, 0.42f, 0.52f, 0.62f
        };

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        group.prepareGPU(stream);

        ConvolutionUnitForwardOutput cpuOutput = group.feedForward(input, 1);

        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionUnitForwardOutputGPU gpuOutput = group.feedForwardGPU(inputGPU, 1, stream, handle);

        float[] output = CudaUtil.fromGPUFloatAsync(gpuOutput.output, cpuOutput.output.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        assertArrayEquals("GPU with units should match CPU", cpuOutput.output, output, 1e-4f);

        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput.freeAsync(stream);
        group.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testKernelGroupForwardGPUMultiBatch() {
        Kernel kernel1 = new Kernel(2, 1);
        kernel1.setWeights(new float[]{0.3f, 0.7f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 1);
        kernel2.setWeights(new float[]{0.4f, 0.6f, 0.2f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(1, 1);
        kernel3.setWeights(new float[]{0.8f});
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        // 2 batches
        float[] input = {
            // Batch 1
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
            0.11f, 0.21f, 0.31f, 0.41f, 0.51f, 0.61f,
            0.12f, 0.22f, 0.32f, 0.42f, 0.52f, 0.62f,
            // Batch 2
            0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
            0.21f, 0.31f, 0.41f, 0.51f, 0.61f, 0.71f,
            0.22f, 0.32f, 0.42f, 0.52f, 0.62f, 0.72f
        };

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        group.prepareGPU(stream);

        ConvolutionUnitForwardOutput cpuOutput = group.feedForward(input, 2);

        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionUnitForwardOutputGPU gpuOutput = group.feedForwardGPU(inputGPU, 2, stream, handle);

        float[] output = CudaUtil.fromGPUFloatAsync(gpuOutput.output, cpuOutput.output.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        assertArrayEquals("GPU multi-batch should match CPU", cpuOutput.output, output, 1e-4f);

        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput.freeAsync(stream);
        group.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testKernelGroupForwardGPUWithUnitsMultiBatch() {
        // Fixed test from earlier
        Kernel kernel1 = new Kernel(2, 2);
        kernel1.setWeights(new float[]{0.3f, 0.7f, 0.4f, 0.6f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(1, 2);
        kernel2.setWeights(new float[]{0.4f, 0.6f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(3, 2);
        kernel3.setWeights(new float[]{0.2f, 0.8f, 0.1f, 0.5f, 0.3f, 0.9f});
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        // 72 values = 2 batches × 6 chunks × 6 inputStride
        float[] input = {
            // Batch 1 - 36 values
            0.1f, 0.2f, 0.15f, 0.25f, 0.3f, 0.4f,
            0.11f, 0.21f, 0.16f, 0.26f, 0.31f, 0.41f,
            0.12f, 0.22f, 0.17f, 0.27f, 0.32f, 0.42f,
            0.13f, 0.23f, 0.18f, 0.28f, 0.33f, 0.43f,
            0.14f, 0.24f, 0.19f, 0.29f, 0.34f, 0.44f,
            0.15f, 0.25f, 0.2f, 0.3f, 0.35f, 0.45f,

            // Batch 2 - 36 values
            0.2f, 0.3f, 0.25f, 0.35f, 0.4f, 0.5f,
            0.21f, 0.31f, 0.26f, 0.36f, 0.41f, 0.51f,
            0.22f, 0.32f, 0.27f, 0.37f, 0.42f, 0.52f,
            0.23f, 0.33f, 0.28f, 0.38f, 0.43f, 0.53f,
            0.24f, 0.34f, 0.29f, 0.39f, 0.44f, 0.54f,
            0.25f, 0.35f, 0.3f, 0.4f, 0.45f, 0.55f
        };

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        group.prepareGPU(stream);

        ConvolutionUnitForwardOutput cpuOutput = group.feedForward(input, 2);

        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionUnitForwardOutputGPU gpuOutput = group.feedForwardGPU(inputGPU, 2, stream, handle);

        float[] output = CudaUtil.fromGPUFloatAsync(gpuOutput.output, cpuOutput.output.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        assertArrayEquals("GPU with units multi-batch should match CPU", cpuOutput.output, output, 1e-4f);

        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput.freeAsync(stream);
        group.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testKernelGroupBackward() {
        // Basic backward test with single batch
        Kernel kernel1 = new Kernel(2, 1); // unitSize=2, unitCount=1, width=2
        kernel1.setWeights(new float[]{0.3f, 0.7f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 1); // unitSize=3, unitCount=1, width=3
        kernel2.setWeights(new float[]{0.4f, 0.6f, 0.2f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(1, 1); // unitSize=1, unitCount=1, width=1
        kernel3.setWeights(new float[]{0.8f});
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        // Input: 3 chunks (outputSize = 3)
        float[] input = {
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
            0.11f, 0.21f, 0.31f, 0.41f, 0.51f, 0.61f,
            0.12f, 0.22f, 0.32f, 0.42f, 0.52f, 0.62f
        };

        // Forward pass first
        ConvolutionUnitForwardOutput forwardOutput = group.feedForward(input, 1);

        // Delta loss from next layer
        float[] deltaLoss = {0.1f, 0.2f, 0.3f};

        // Test CPU backward pass
        ConvolutionUnitBackpropagateOutput cpuBackward = group.backpropagate(forwardOutput, deltaLoss);

        // Verify we get correct structure
        assertEquals("Should have 3 weight gradient arrays", 3, cpuBackward.weightGradients.length);
        assertEquals("Should have 3 bias gradients", 3, cpuBackward.biasGradients.length);
        assertEquals("Kernel 1 weight gradients size", 2, cpuBackward.weightGradients[0].length);
        assertEquals("Kernel 2 weight gradients size", 3, cpuBackward.weightGradients[1].length);
        assertEquals("Kernel 3 weight gradients size", 1, cpuBackward.weightGradients[2].length);
        assertEquals("Input gradients size", input.length, cpuBackward.inputGradients.length);

        // Pre-calculated expected gradients based on:
        // Forward outputs: K1=[0.17,0.18,0.19], K2=[0.46,0.472,0.484], K3=[0.48,0.488,0.496]
        // Sigmoid outputs: K1=[0.5424,0.5449,0.5474], K2=[0.6130,0.6158,0.6186], K3=[0.6177,0.6196,0.6215]
        // AND gradients flow: K1 gets deltaLoss*sig2*sig3, K2 gets deltaLoss*sig1*sig3, K3 gets deltaLoss*sig1*sig2
        // Then through sigmoid derivative: grad * sigmoid * (1-sigmoid)
        // Final sigmoid gradients: K1=[0.0094,0.0189,0.0287], K2=[0.0079,0.0160,0.0241], K3=[0.0078,0.0158,0.0238]

        // Weight gradients
        final float[] EXPECTED_K1_WEIGHTS = {0.00647f, 0.01216f};
        final float[] EXPECTED_K2_WEIGHTS = {0.01504f, 0.01984f, 0.02464f};
        final float[] EXPECTED_K3_WEIGHTS = {0.02918f};

        // Bias gradients
        final float EXPECTED_K1_BIAS = 0.0570f;
        final float EXPECTED_K2_BIAS = 0.0480f;
        final float EXPECTED_K3_BIAS = 0.04756f;

        // Input gradients - each kernel backprops to its channel positions
        final float[] EXPECTED_INPUT_GRADS = {
            // Chunk 0: K1 positions 0-1, K2 positions 2-4, K3 position 5
            0.00282f, 0.00658f, 0.00316f, 0.00474f, 0.00158f, 0.00624f,
            // Chunk 1: K1 positions 6-7, K2 positions 8-10, K3 position 11
            0.00567f, 0.01323f, 0.00640f, 0.00960f, 0.00320f, 0.01264f,
            // Chunk 2: K1 positions 12-13, K2 positions 14-16, K3 position 17
            0.00861f, 0.02009f, 0.00964f, 0.01446f, 0.00482f, 0.01904f
        };

        // Verify gradients
        assertArrayEquals("Kernel 1 weight gradients", EXPECTED_K1_WEIGHTS, cpuBackward.weightGradients[0], 1e-4f);
        assertArrayEquals("Kernel 2 weight gradients", EXPECTED_K2_WEIGHTS, cpuBackward.weightGradients[1], 1e-4f);
        assertArrayEquals("Kernel 3 weight gradients", EXPECTED_K3_WEIGHTS, cpuBackward.weightGradients[2], 1e-4f);

        assertEquals("Kernel 1 bias gradient", EXPECTED_K1_BIAS, cpuBackward.biasGradients[0], 1e-4f);
        assertEquals("Kernel 2 bias gradient", EXPECTED_K2_BIAS, cpuBackward.biasGradients[1], 1e-4f);
        assertEquals("Kernel 3 bias gradient", EXPECTED_K3_BIAS, cpuBackward.biasGradients[2], 1e-4f);

        assertArrayEquals("Input gradients", EXPECTED_INPUT_GRADS, cpuBackward.inputGradients, 1e-4f);
    }
    
    @Test
    public void testKernelGroupBackwardWithUnits() {
        // 3 kernels with unitSizes [1,2,3] and unitCount=2
        Kernel kernel1 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        kernel1.setWeights(new float[]{0.1f, -0.1f});  // Smaller weights
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4
        kernel2.setWeights(new float[]{0.05f, -0.05f, 0.05f, -0.05f});  // Smaller weights
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(3, 2); // unitSize=3, unitCount=2, width=6
        kernel3.setWeights(new float[]{0.02f, -0.02f, 0.02f, -0.02f, 0.02f, -0.02f});  // Much smaller weights
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        // inputStride = 1+2+3 = 6, 3 chunks = 18 values, outputSize = 3-2+1 = 2
        // Using smaller input values to avoid saturation
        float[] input = {
            // chunk 0: k1=[0.1], k2=[0.2,0.3], k3=[0.4,0.5,0.6]
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
            // chunk 1: k1=[0.7], k2=[0.8,0.9], k3=[1.0,1.1,1.2]
            0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
            // chunk 2: k1=[1.3], k2=[1.4,1.5], k3=[1.6,1.7,1.8]
            1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f
        };

        float[] deltaLoss = {0.1f, 0.2f}; // 2 outputs

        // Test forward pass
        ConvolutionUnitForwardOutput forwardOutput = group.feedForward(input, 1);

        // Manual forward calculations:
        // K1 channel data: [0.1, 0.7, 1.3]
        //   Position 0: 0.1×0.1 + 0.7×(-0.1) = 0.01 - 0.07 = -0.06
        //   Position 1: 0.7×0.1 + 1.3×(-0.1) = 0.07 - 0.13 = -0.06
        // K2 channel data: [0.2, 0.3, 0.8, 0.9, 1.4, 1.5]
        //   Position 0: 0.2×0.05 + 0.3×(-0.05) + 0.8×0.05 + 0.9×(-0.05) = 0.01 - 0.015 + 0.04 - 0.045 = -0.01
        //   Position 1: 0.8×0.05 + 0.9×(-0.05) + 1.4×0.05 + 1.5×(-0.05) = 0.04 - 0.045 + 0.07 - 0.075 = -0.01
        // K3 channel data: [0.4, 0.5, 0.6, 1.0, 1.1, 1.2, 1.6, 1.7, 1.8]
        //   Position 0: 0.4×0.02 + 0.5×(-0.02) + 0.6×0.02 + 1.0×(-0.02) + 1.1×0.02 + 1.2×(-0.02) = 0.008 - 0.01 + 0.012 - 0.02 + 0.022 - 0.024 = -0.012
        //   Position 1: 1.0×0.02 + 1.1×(-0.02) + 1.2×0.02 + 1.6×(-0.02) + 1.7×0.02 + 1.8×(-0.02) = 0.02 - 0.022 + 0.024 - 0.032 + 0.034 - 0.036 = -0.012
        final float[] EXPECTED_K1_OUTPUTS = {-0.06f, -0.06f};
        final float[] EXPECTED_K2_OUTPUTS = {-0.01f, -0.01f};
        final float[] EXPECTED_K3_OUTPUTS = {-0.012f, -0.012f};

        // Sigmoid values - now in the linear region!
        // σ(-0.06) = 0.485, σ(-0.01) = 0.4975, σ(-0.012) = 0.497
        final float[] EXPECTED_K1_SIGMOID = {0.485f, 0.485f};
        final float[] EXPECTED_K2_SIGMOID = {0.4975f, 0.4975f};
        final float[] EXPECTED_K3_SIGMOID = {0.497f, 0.497f};

        // AND operation: output = sig1 × sig2 × sig3
        // Position 0: 0.485 × 0.4975 × 0.497 = 0.1199
        // Position 1: 0.485 × 0.4975 × 0.497 = 0.1199
        final float[] EXPECTED_AND_OUTPUT = {0.1199f, 0.1199f};

        // Verify forward calculations
        assertArrayEquals("Kernel 1 outputs", EXPECTED_K1_OUTPUTS, forwardOutput.kernelOutputs[0], 1e-4f);
        assertArrayEquals("Kernel 2 outputs", EXPECTED_K2_OUTPUTS, forwardOutput.kernelOutputs[1], 1e-4f);
        assertArrayEquals("Kernel 3 outputs", EXPECTED_K3_OUTPUTS, forwardOutput.kernelOutputs[2], 1e-4f);

        assertArrayEquals("Kernel 1 sigmoid", EXPECTED_K1_SIGMOID, forwardOutput.sigmoidOutputs[0], 1e-4f);
        assertArrayEquals("Kernel 2 sigmoid", EXPECTED_K2_SIGMOID, forwardOutput.sigmoidOutputs[1], 1e-4f);
        assertArrayEquals("Kernel 3 sigmoid", EXPECTED_K3_SIGMOID, forwardOutput.sigmoidOutputs[2], 1e-4f);

        assertArrayEquals("AND output", EXPECTED_AND_OUTPUT, forwardOutput.output, 1e-4f);

        // Test backward pass
        ConvolutionUnitBackpropagateOutput backpropOutput = group.backpropagate(forwardOutput, deltaLoss);

        // Backward calculations:
        // AND gradients: each kernel gets deltaLoss × (product of other sigmoids)
        // K1: [0.1×0.4975×0.497, 0.2×0.4975×0.497] = [0.02473, 0.04945]
        // K2: [0.1×0.485×0.497, 0.2×0.485×0.497] = [0.02410, 0.04821]
        // K3: [0.1×0.485×0.4975, 0.2×0.485×0.4975] = [0.02412, 0.04824]

        // Through sigmoid derivative σ'(x) = σ(x)×(1-σ(x))
        // For values near 0.5, derivative ≈ 0.25
        // K1 sigmoid grads: [0.02473×0.485×0.515, 0.04945×0.485×0.515] = [0.006179, 0.012358]
        // K2 sigmoid grads: [0.02410×0.4975×0.5025, 0.04821×0.4975×0.5025] = [0.006026, 0.012053]
        // K3 sigmoid grads: [0.02412×0.497×0.503, 0.04824×0.497×0.503] = [0.006032, 0.012064]

        // Weight gradients:
        // K1: w[0] = 0.1×0.006179 + 0.7×0.012358 = 0.0006179 + 0.0086506 = 0.0092685
        //     w[1] = 0.7×0.006179 + 1.3×0.012358 = 0.0043253 + 0.0160654 = 0.0203907
        final float[] EXPECTED_K1_WEIGHT_GRADS = {0.00927f, 0.02039f};

        // K2: w[0] = 0.2×0.006026 + 0.8×0.012053 = 0.0012052 + 0.0096424 = 0.0108476
        //     w[1] = 0.3×0.006026 + 0.9×0.012053 = 0.0018078 + 0.0108477 = 0.0126555
        //     w[2] = 0.8×0.006026 + 1.4×0.012053 = 0.0048208 + 0.0168742 = 0.021695
        //     w[3] = 0.9×0.006026 + 1.5×0.012053 = 0.0054234 + 0.0180795 = 0.0235029
        final float[] EXPECTED_K2_WEIGHT_GRADS = {0.01085f, 0.01266f, 0.02170f, 0.02350f};

        // K3: w[0] = 0.4×0.006032 + 1.0×0.012064 = 0.0024128 + 0.012064 = 0.0144768
        //     w[1] = 0.5×0.006032 + 1.1×0.012064 = 0.003016 + 0.0132704 = 0.0162864
        //     w[2] = 0.6×0.006032 + 1.2×0.012064 = 0.0036192 + 0.0144768 = 0.018096
        //     w[3] = 1.0×0.006032 + 1.6×0.012064 = 0.006032 + 0.0193024 = 0.0253344
        //     w[4] = 1.1×0.006032 + 1.7×0.012064 = 0.0066352 + 0.0205088 = 0.027144
        //     w[5] = 1.2×0.006032 + 1.8×0.012064 = 0.0072384 + 0.0217152 = 0.0289536
        final float[] EXPECTED_K3_WEIGHT_GRADS = {0.01448f, 0.01629f, 0.01810f, 0.02533f, 0.02714f, 0.02895f};

        // Bias gradients (sum of sigmoid gradients):
        final float EXPECTED_K1_BIAS_GRAD = 0.01853f; // 0.006179 + 0.012358
        final float EXPECTED_K2_BIAS_GRAD = 0.01808f; // 0.006026 + 0.012053
        final float EXPECTED_K3_BIAS_GRAD = 0.01810f; // 0.006032 + 0.012064

        // Input gradients calculated properly:
        // Each position accumulates: weight × sigmoid_gradient from outputs that use it
        final float[] EXPECTED_INPUT_GRADS = {
            0.0006179f,   // pos 0: 0.1 × 0.006179
            0.0003013f,   // pos 1: 0.05 × 0.006026
            -0.0003013f,  // pos 2: -0.05 × 0.006026
            0.0001206f,   // pos 3: 0.02 × 0.006032
            -0.0001206f,  // pos 4: -0.02 × 0.006032
            0.0001206f,   // pos 5: 0.02 × 0.006032
            0.0006179f,   // pos 6: -0.1×0.006179 + 0.1×0.012358
            0.0009040f,   // pos 7: 0.05×0.006026 + 0.05×0.012053
            -0.0009040f,  // pos 8: -0.05×0.006026 + -0.05×0.012053
            0.0001207f,   // pos 9: -0.02×0.006032 + 0.02×0.012064
            -0.0001207f,  // pos 10: 0.02×0.006032 + -0.02×0.012064
            0.0001207f,   // pos 11: -0.02×0.006032 + 0.02×0.012064
            -0.0012358f,  // pos 12: -0.1 × 0.012358
            0.0006027f,   // pos 13: 0.05 × 0.012053
            -0.0006027f,  // pos 14: -0.05 × 0.012053
            -0.0002413f,  // pos 15: -0.02 × 0.012064
            0.0002413f,   // pos 16: 0.02 × 0.012064
            -0.0002413f   // pos 17: -0.02 × 0.012064
        };

        // Verify gradients with appropriate tolerances
        assertArrayEquals("Weight gradients kernel 1", 
            EXPECTED_K1_WEIGHT_GRADS, backpropOutput.weightGradients[0], 1e-5f);
        assertArrayEquals("Weight gradients kernel 2", 
            EXPECTED_K2_WEIGHT_GRADS, backpropOutput.weightGradients[1], 1e-5f);
        assertArrayEquals("Weight gradients kernel 3", 
            EXPECTED_K3_WEIGHT_GRADS, backpropOutput.weightGradients[2], 1e-5f);

        assertEquals("Bias gradient kernel 1", EXPECTED_K1_BIAS_GRAD, backpropOutput.biasGradients[0], 1e-5f);
        assertEquals("Bias gradient kernel 2", EXPECTED_K2_BIAS_GRAD, backpropOutput.biasGradients[1], 1e-5f);
        assertEquals("Bias gradient kernel 3", EXPECTED_K3_BIAS_GRAD, backpropOutput.biasGradients[2], 1e-5f);

        assertArrayEquals("Input gradients", 
            EXPECTED_INPUT_GRADS, backpropOutput.inputGradients, 1e-5f);
    }

    @Test
    public void testKernelGroupBackwardMultiBatch() {
        // 2 kernels with unitCount=2 for 2 batches
        Kernel kernel1 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        kernel2.setWeights(new float[]{0.5f, 0.6f});
        kernel2.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2);

        // inputStride = 2+1 = 3, 3 chunks per batch, 2 batches = 18 values
        float[] input = {
            // Batch 1
            1.0f, 2.0f, 5.0f,   // chunk 0: k1=[1,2], k2=[5]
            3.0f, 4.0f, 6.0f,   // chunk 1: k1=[3,4], k2=[6]
            7.0f, 8.0f, 9.0f,   // chunk 2: k1=[7,8], k2=[9]
            // Batch 2
            10.0f, 11.0f, 15.0f, // chunk 0: k1=[10,11], k2=[15]
            12.0f, 13.0f, 16.0f, // chunk 1: k1=[12,13], k2=[16]
            14.0f, 17.0f, 18.0f  // chunk 2: k1=[14,17], k2=[18]
        };

        float[] deltaLoss = {0.1f, 0.2f, 0.3f, 0.4f}; // 2 outputs × 2 batches

        ConvolutionUnitForwardOutput forwardOutput = group.feedForward(input, 2);

        // Manual forward calculations:
        // Batch 1 Channel 1: [1,2,3,4,7,8] -> outputs: [1*0.1+2*0.2+3*0.3+4*0.4=3.0, 3*0.1+4*0.2+7*0.3+8*0.4=6.4]
        // Batch 1 Channel 2: [5,6,9] -> outputs: [5*0.5+6*0.6=6.1, 6*0.5+9*0.6=8.4]
        // Batch 2 Channel 1: [10,11,12,13,14,17] -> outputs: [10*0.1+11*0.2+12*0.3+13*0.4=12.0, 12*0.1+13*0.2+14*0.3+17*0.4=14.8]
        // Batch 2 Channel 2: [15,16,18] -> outputs: [15*0.5+16*0.6=17.1, 16*0.5+18*0.6=18.8]

        assertArrayEquals("Kernel 1 outputs", new float[]{3.0f, 6.4f, 12.0f, 14.8f}, forwardOutput.kernelOutputs[0], 1e-6f);
        assertArrayEquals("Kernel 2 outputs", new float[]{6.1f, 8.4f, 17.1f, 18.8f}, forwardOutput.kernelOutputs[1], 1e-6f);

        // Calculate sigmoid values  
        Sigmoid sigmoidCalc = new Sigmoid();
        float sig1_0 = sigmoidCalc.compute(3.0f);   // batch1, pos0
        float sig1_1 = sigmoidCalc.compute(6.4f);   // batch1, pos1
        float sig1_2 = sigmoidCalc.compute(12.0f);  // batch2, pos0
        float sig1_3 = sigmoidCalc.compute(14.8f);  // batch2, pos1
        float sig2_0 = sigmoidCalc.compute(6.1f);   // batch1, pos0
        float sig2_1 = sigmoidCalc.compute(8.4f);   // batch1, pos1
        float sig2_2 = sigmoidCalc.compute(17.1f);  // batch2, pos0
        float sig2_3 = sigmoidCalc.compute(18.8f);  // batch2, pos1

        // Verify sigmoid outputs
        assertArrayEquals("Kernel 1 sigmoid", new float[]{sig1_0, sig1_1, sig1_2, sig1_3}, forwardOutput.sigmoidOutputs[0], 1e-4f);
        assertArrayEquals("Kernel 2 sigmoid", new float[]{sig2_0, sig2_1, sig2_2, sig2_3}, forwardOutput.sigmoidOutputs[1], 1e-4f);

        ConvolutionUnitBackpropagateOutput backpropOutput = group.backpropagate(forwardOutput, deltaLoss);

        // Manual backward calculations:
        // AND gradients: each kernel gets deltaLoss * other sigmoid
        float k1_grad_0 = 0.1f * sig2_0;  // batch1, pos0
        float k1_grad_1 = 0.2f * sig2_1;  // batch1, pos1  
        float k1_grad_2 = 0.3f * sig2_2;  // batch2, pos0
        float k1_grad_3 = 0.4f * sig2_3;  // batch2, pos1
        float k2_grad_0 = 0.1f * sig1_0;  // batch1, pos0
        float k2_grad_1 = 0.2f * sig1_1;  // batch1, pos1
        float k2_grad_2 = 0.3f * sig1_2;  // batch2, pos0
        float k2_grad_3 = 0.4f * sig1_3;  // batch2, pos1

        // Apply sigmoid gradients
        float k1_sigmoid_grad_0 = sigmoidCalc.derivative(3.0f, sig1_0) * k1_grad_0;
        float k1_sigmoid_grad_1 = sigmoidCalc.derivative(6.4f, sig1_1) * k1_grad_1;
        float k1_sigmoid_grad_2 = sigmoidCalc.derivative(12.0f, sig1_2) * k1_grad_2;
        float k1_sigmoid_grad_3 = sigmoidCalc.derivative(14.8f, sig1_3) * k1_grad_3;
        float k2_sigmoid_grad_0 = sigmoidCalc.derivative(6.1f, sig2_0) * k2_grad_0;
        float k2_sigmoid_grad_1 = sigmoidCalc.derivative(8.4f, sig2_1) * k2_grad_1;
        float k2_sigmoid_grad_2 = sigmoidCalc.derivative(17.1f, sig2_2) * k2_grad_2;  
        float k2_sigmoid_grad_3 = sigmoidCalc.derivative(18.8f, sig2_3) * k2_grad_3;

        // Weight gradients (averaged over batches):
        // Kernel 1 sliding windows: [1,2,3,4], [3,4,7,8], [10,11,12,13], [12,13,14,17]
        float k1_w0_grad = (1.0f * k1_sigmoid_grad_0 + 3.0f * k1_sigmoid_grad_1 + 10.0f * k1_sigmoid_grad_2 + 12.0f * k1_sigmoid_grad_3) / 2;
        float k1_w1_grad = (2.0f * k1_sigmoid_grad_0 + 4.0f * k1_sigmoid_grad_1 + 11.0f * k1_sigmoid_grad_2 + 13.0f * k1_sigmoid_grad_3) / 2;
        float k1_w2_grad = (3.0f * k1_sigmoid_grad_0 + 7.0f * k1_sigmoid_grad_1 + 12.0f * k1_sigmoid_grad_2 + 14.0f * k1_sigmoid_grad_3) / 2;
        float k1_w3_grad = (4.0f * k1_sigmoid_grad_0 + 8.0f * k1_sigmoid_grad_1 + 13.0f * k1_sigmoid_grad_2 + 17.0f * k1_sigmoid_grad_3) / 2;

        // Kernel 2 sliding windows: [5,6], [6,9], [15,16], [16,18]
        float k2_w0_grad = (5.0f * k2_sigmoid_grad_0 + 6.0f * k2_sigmoid_grad_1 + 15.0f * k2_sigmoid_grad_2 + 16.0f * k2_sigmoid_grad_3) / 2;
        float k2_w1_grad = (6.0f * k2_sigmoid_grad_0 + 9.0f * k2_sigmoid_grad_1 + 16.0f * k2_sigmoid_grad_2 + 18.0f * k2_sigmoid_grad_3) / 2;

        // Bias gradients (averaged over batches):
        float k1_bias_grad = (k1_sigmoid_grad_0 + k1_sigmoid_grad_1 + k1_sigmoid_grad_2 + k1_sigmoid_grad_3) / 2;
        float k2_bias_grad = (k2_sigmoid_grad_0 + k2_sigmoid_grad_1 + k2_sigmoid_grad_2 + k2_sigmoid_grad_3) / 2;

        // Input gradients (18 values) - map channel gradients back to input positions:
        float[] expectedInputGrads = new float[18];
        // Batch 1
        expectedInputGrads[0] = 0.1f * k1_sigmoid_grad_0;   // k1 pos[0] 
        expectedInputGrads[1] = 0.2f * k1_sigmoid_grad_0;   // k1 pos[1]
        expectedInputGrads[2] = 0.5f * k2_sigmoid_grad_0;   // k2 pos[0]
        expectedInputGrads[3] = 0.3f * k1_sigmoid_grad_0 + 0.1f * k1_sigmoid_grad_1; // k1 overlap
        expectedInputGrads[4] = 0.4f * k1_sigmoid_grad_0 + 0.2f * k1_sigmoid_grad_1; // k1 overlap  
        expectedInputGrads[5] = 0.6f * k2_sigmoid_grad_0 + 0.5f * k2_sigmoid_grad_1; // k2 overlap
        expectedInputGrads[6] = 0.3f * k1_sigmoid_grad_1;   // k1 pos[4]
        expectedInputGrads[7] = 0.4f * k1_sigmoid_grad_1;   // k1 pos[5]
        expectedInputGrads[8] = 0.6f * k2_sigmoid_grad_1;   // k2 pos[2]
        // Batch 2
        expectedInputGrads[9] = 0.1f * k1_sigmoid_grad_2;   // k1 pos[0]
        expectedInputGrads[10] = 0.2f * k1_sigmoid_grad_2;  // k1 pos[1]
        expectedInputGrads[11] = 0.5f * k2_sigmoid_grad_2;  // k2 pos[0]
        expectedInputGrads[12] = 0.3f * k1_sigmoid_grad_2 + 0.1f * k1_sigmoid_grad_3; // k1 overlap
        expectedInputGrads[13] = 0.4f * k1_sigmoid_grad_2 + 0.2f * k1_sigmoid_grad_3; // k1 overlap
        expectedInputGrads[14] = 0.6f * k2_sigmoid_grad_2 + 0.5f * k2_sigmoid_grad_3; // k2 overlap  
        expectedInputGrads[15] = 0.3f * k1_sigmoid_grad_3;  // k1 pos[4]
        expectedInputGrads[16] = 0.4f * k1_sigmoid_grad_3;  // k1 pos[5]
        expectedInputGrads[17] = 0.6f * k2_sigmoid_grad_3;  // k2 pos[2]

        // Verify results
        assertArrayEquals("Multi-batch weight gradients kernel 1", 
            new float[]{k1_w0_grad, k1_w1_grad, k1_w2_grad, k1_w3_grad}, backpropOutput.weightGradients[0], 1e-6f);
        assertArrayEquals("Multi-batch weight gradients kernel 2", 
            new float[]{k2_w0_grad, k2_w1_grad}, backpropOutput.weightGradients[1], 1e-6f);
        assertArrayEquals("Multi-batch bias gradients", 
            new float[]{k1_bias_grad, k2_bias_grad}, backpropOutput.biasGradients, 1e-6f);
        assertArrayEquals("Multi-batch input gradients", 
            expectedInputGrads, backpropOutput.inputGradients, 1e-6f);
    }
    
    @Test
    public void testKernelGroupBackwardWithUnitsMultiBatches() {
        // Test with unitCount=2 and 4 batches
        Kernel kernel1 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        kernel2.setWeights(new float[]{0.5f, 0.6f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(3, 2); // unitSize=3, unitCount=2, width=6
        kernel3.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        // inputStride = 6, 3 chunks gives outputSize = 2
        float[] input = {
            // Batch 1
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
            2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
            3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            // Batch 2
            0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f,
            1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f,
            2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f,
            // Batch 3
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
            0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
            0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
            // Batch 4
            1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f,
            1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f,
            1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f
        };

        // Forward pass
        ConvolutionUnitForwardOutput forwardOutput = group.feedForward(input, 4);

        // Delta loss - 2 outputs × 4 batches
        float[] deltaLoss = {
            0.1f, 0.15f,   // batch 1
            0.12f, 0.17f,  // batch 2
            0.08f, 0.13f,  // batch 3
            0.11f, 0.16f   // batch 4
        };

        // Backward pass
        ConvolutionUnitBackpropagateOutput backpropOutput = group.backpropagate(forwardOutput, deltaLoss);

        // Verify structure
        assertEquals("Weight gradients count", 3, backpropOutput.weightGradients.length);
        assertEquals("Kernel 1 weight gradients size", 4, backpropOutput.weightGradients[0].length);
        assertEquals("Kernel 2 weight gradients size", 2, backpropOutput.weightGradients[1].length);
        assertEquals("Kernel 3 weight gradients size", 6, backpropOutput.weightGradients[2].length);
        assertEquals("Input gradients size", input.length, backpropOutput.inputGradients.length);

        // Verify bias gradients are averaged over batches
        assertTrue("Kernel 1 bias gradient should be positive", backpropOutput.biasGradients[0] > 0);
        assertTrue("Kernel 2 bias gradient should be positive", backpropOutput.biasGradients[1] > 0);
        assertTrue("Kernel 3 bias gradient should be positive", backpropOutput.biasGradients[2] > 0);
    }
    
    
    @Test
    public void testKernelGroupBackwardGPU() {
        // 3 kernels with unitCount=1 for simple test
        Kernel kernel1 = new Kernel(2, 1); // unitSize=2, unitCount=1, width=2
        kernel1.setWeights(new float[]{0.3f, 0.7f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 1); // unitSize=3, unitCount=1, width=3  
        kernel2.setWeights(new float[]{0.4f, 0.6f, 0.2f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(1, 1); // unitSize=1, unitCount=1, width=1
        kernel3.setWeights(new float[]{0.8f});
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        float[] input = {
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
            0.11f, 0.21f, 0.31f, 0.41f, 0.51f, 0.61f,
            0.12f, 0.22f, 0.32f, 0.42f, 0.52f, 0.62f
        };

        float[] deltaLoss = {0.1f, 0.2f, 0.3f};

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        // Prepare GPU resources
        group.prepareGPU(stream);

        // CPU forward and backprop
        ConvolutionUnitForwardOutput cpuForward = group.feedForward(input, 1);
        ConvolutionUnitBackpropagateOutput cpuBackprop = group.backpropagate(cpuForward, deltaLoss);

        // GPU forward and backprop
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        CUdeviceptr deltaLossGPU = CudaUtil.toGPUAsync(deltaLoss, stream);

        ConvolutionUnitForwardOutputGPU gpuForward = group.feedForwardGPU(inputGPU, 1, stream, handle);
        ConvolutionUnitBackpropagateOutputGPU gpuBackprop = group.backpropagateGPU(gpuForward, deltaLossGPU, stream, handle);

        // Get GPU results back to CPU
        JCudaDriver.cuStreamSynchronize(stream);

        float[] gpuInputGrads = CudaUtil.fromGPUFloat(gpuBackprop.inputGradients, input.length);
        float[][] gpuWeightGrads = new float[3][];
        float[] gpuBiasGrads = new float[3];

        for (int k = 0; k < 3; k++) {
            Kernel kernel = group.getKernel(k);
            gpuWeightGrads[k] = CudaUtil.fromGPUFloat(gpuBackprop.weightGradients[k], kernel.width);
            gpuBiasGrads[k] = CudaUtil.fromGPUFloat(gpuBackprop.biasGradients[k], 1)[0];
        }

        // Verify results match
        assertArrayEquals("Input gradients should match", cpuBackprop.inputGradients, gpuInputGrads, 1e-5f);
        for (int k = 0; k < 3; k++) {
            assertArrayEquals("Weight gradients kernel " + k + " should match", 
                            cpuBackprop.weightGradients[k], gpuWeightGrads[k], 1e-5f);
            assertEquals("Bias gradient kernel " + k + " should match", 
                        cpuBackprop.biasGradients[k], gpuBiasGrads[k], 1e-5f);
        }

        // Cleanup
        CudaUtil.free(inputGPU);
        CudaUtil.free(deltaLossGPU);
        gpuForward.free();
        gpuBackprop.free();
        group.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testKernelGroupBackwardWithUnitsGPU() {
        // 3 kernels with unitCount=2
        Kernel kernel1 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        kernel1.setWeights(new float[]{0.2f, 0.3f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4
        kernel2.setWeights(new float[]{0.1f, 0.4f, 0.2f, 0.5f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(3, 2); // unitSize=3, unitCount=2, width=6
        kernel3.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        float[] input = {
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
            13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f
        };

        float[] deltaLoss = {0.1f, 0.2f};

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        group.prepareGPU(stream);

        // CPU reference
        ConvolutionUnitForwardOutput cpuForward = group.feedForward(input, 1);
        ConvolutionUnitBackpropagateOutput cpuBackprop = group.backpropagate(cpuForward, deltaLoss);

        // GPU implementation
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        CUdeviceptr deltaLossGPU = CudaUtil.toGPUAsync(deltaLoss, stream);

        ConvolutionUnitForwardOutputGPU gpuForward = group.feedForwardGPU(inputGPU, 1, stream, handle);
        ConvolutionUnitBackpropagateOutputGPU gpuBackprop = group.backpropagateGPU(gpuForward, deltaLossGPU, stream, handle);

        JCudaDriver.cuStreamSynchronize(stream);

        // Compare results
        float[] gpuInputGrads = CudaUtil.fromGPUFloat(gpuBackprop.inputGradients, input.length);
        assertArrayEquals("Input gradients with units should match", cpuBackprop.inputGradients, gpuInputGrads, 1e-5f);

        for (int k = 0; k < 3; k++) {
            float[] gpuWeightGrads = CudaUtil.fromGPUFloat(gpuBackprop.weightGradients[k], group.getKernel(k).width);
            float gpuBiasGrad = CudaUtil.fromGPUFloat(gpuBackprop.biasGradients[k], 1)[0];

            assertArrayEquals("Weight gradients kernel " + k + " with units should match", 
                            cpuBackprop.weightGradients[k], gpuWeightGrads, 1e-5f);
            assertEquals("Bias gradient kernel " + k + " with units should match", 
                        cpuBackprop.biasGradients[k], gpuBiasGrad, 1e-5f);
        }

        CudaUtil.free(inputGPU);
        CudaUtil.free(deltaLossGPU);
        gpuForward.free();
        gpuBackprop.free();
        group.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testKernelGroupBackwardMultiBatchGPU() {
        // Simple setup with 3 kernels and 4 batches
        Kernel kernel1 = new Kernel(2, 1);
        kernel1.setWeights(new float[]{0.3f, 0.7f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(1, 1);
        kernel2.setWeights(new float[]{0.5f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(3, 1);
        kernel3.setWeights(new float[]{0.2f, 0.4f, 0.6f});
        kernel3.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2, kernel3);

        // 4 batches, 3 chunks each
        float[] input = {
            // Batch 1
            0.1f, 0.2f, 0.5f, 0.3f, 0.4f, 0.6f,
            0.11f, 0.21f, 0.51f, 0.31f, 0.41f, 0.61f,
            0.12f, 0.22f, 0.52f, 0.32f, 0.42f, 0.62f,
            // Batch 2  
            0.2f, 0.3f, 0.6f, 0.4f, 0.5f, 0.7f,
            0.21f, 0.31f, 0.61f, 0.41f, 0.51f, 0.71f,
            0.22f, 0.32f, 0.62f, 0.42f, 0.52f, 0.72f,
            // Batch 3
            0.15f, 0.25f, 0.55f, 0.35f, 0.45f, 0.65f,
            0.16f, 0.26f, 0.56f, 0.36f, 0.46f, 0.66f,
            0.17f, 0.27f, 0.57f, 0.37f, 0.47f, 0.67f,
            // Batch 4
            0.05f, 0.15f, 0.45f, 0.25f, 0.35f, 0.55f,
            0.06f, 0.16f, 0.46f, 0.26f, 0.36f, 0.56f,
            0.07f, 0.17f, 0.47f, 0.27f, 0.37f, 0.57f
        };

        float[] deltaLoss = {
            0.1f, 0.2f, 0.3f,     // batch 1
            0.15f, 0.25f, 0.35f,  // batch 2
            0.12f, 0.22f, 0.32f,  // batch 3
            0.08f, 0.18f, 0.28f   // batch 4
        };

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        // Prepare GPU
        group.prepareGPU(stream);

        // CPU reference
        ConvolutionUnitForwardOutput cpuForward = group.feedForward(input, 4);
        ConvolutionUnitBackpropagateOutput cpuBackprop = group.backpropagate(cpuForward, deltaLoss);

        // GPU implementation
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        CUdeviceptr deltaLossGPU = CudaUtil.toGPUAsync(deltaLoss, stream);

        ConvolutionUnitForwardOutputGPU gpuForward = group.feedForwardGPU(inputGPU, 4, stream, handle);
        ConvolutionUnitBackpropagateOutputGPU gpuBackprop = group.backpropagateGPU(gpuForward, deltaLossGPU, stream, handle);

        JCudaDriver.cuStreamSynchronize(stream);

        // Compare input gradients
        float[] gpuInputGrads = CudaUtil.fromGPUFloat(gpuBackprop.inputGradients, input.length);
        assertArrayEquals("Multi-batch GPU input gradients should match CPU", 
                         cpuBackprop.inputGradients, gpuInputGrads, 1e-5f);

        // Compare weight and bias gradients for each kernel
        for (int k = 0; k < 3; k++) {
            float[] gpuWeightGrads = CudaUtil.fromGPUFloat(gpuBackprop.weightGradients[k], group.getKernel(k).width);
            float gpuBiasGrad = CudaUtil.fromGPUFloat(gpuBackprop.biasGradients[k], 1)[0];

            assertArrayEquals("Multi-batch GPU weight gradients kernel " + k + " should match CPU",
                             cpuBackprop.weightGradients[k], gpuWeightGrads, 1e-5f);
            assertEquals("Multi-batch GPU bias gradient kernel " + k + " should match CPU",
                        cpuBackprop.biasGradients[k], gpuBiasGrad, 1e-5f);
        }

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeAsync(deltaLossGPU, stream);
        gpuForward.freeAsync(stream);
        gpuBackprop.freeAsync(stream);
        group.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testKernelGroupBackwardWithUnitsMultiBatchGPU() {
        // Simple 2-kernel setup for multi-batch test
        Kernel kernel1 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        kernel2.setWeights(new float[]{0.5f, 0.6f});
        kernel2.setBias(0.0f);

        KernelGroup group = new KernelGroup(kernel1, kernel2);

        float[] input = {
            // Batch 1
            1.0f, 2.0f, 5.0f, 3.0f, 4.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            // Batch 2
            10.0f, 11.0f, 15.0f, 12.0f, 13.0f, 16.0f, 14.0f, 17.0f, 18.0f,
            // Batch 3
            2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 7.0f, 8.0f, 9.0f, 10.0f,
            // Batch 4
            5.0f, 6.0f, 9.0f, 7.0f, 8.0f, 10.0f, 11.0f, 12.0f, 13.0f
        };

        float[] deltaLoss = {
            0.1f, 0.2f,   // batch 1
            0.3f, 0.4f,   // batch 2
            0.15f, 0.25f, // batch 3
            0.35f, 0.45f  // batch 4
        };

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        group.prepareGPU(stream);

        // CPU reference
        ConvolutionUnitForwardOutput cpuForward = group.feedForward(input, 4);
        ConvolutionUnitBackpropagateOutput cpuBackprop = group.backpropagate(cpuForward, deltaLoss);

        // GPU implementation
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        CUdeviceptr deltaLossGPU = CudaUtil.toGPUAsync(deltaLoss, stream);

        ConvolutionUnitForwardOutputGPU gpuForward = group.feedForwardGPU(inputGPU, 4, stream, handle);
        ConvolutionUnitBackpropagateOutputGPU gpuBackprop = group.backpropagateGPU(gpuForward, deltaLossGPU, stream, handle);

        JCudaDriver.cuStreamSynchronize(stream);

        // Verify multi-batch results
        float[] gpuInputGrads = CudaUtil.fromGPUFloat(gpuBackprop.inputGradients, input.length);
        assertArrayEquals("Multi-batch input gradients should match", cpuBackprop.inputGradients, gpuInputGrads, 1e-5f);

        for (int k = 0; k < 2; k++) {
            float[] gpuWeightGrads = CudaUtil.fromGPUFloat(gpuBackprop.weightGradients[k], group.getKernel(k).width);
            float gpuBiasGrad = CudaUtil.fromGPUFloat(gpuBackprop.biasGradients[k], 1)[0];

            assertArrayEquals("Multi-batch weight gradients kernel " + k + " should match", 
                            cpuBackprop.weightGradients[k], gpuWeightGrads, 1e-5f);
            assertEquals("Multi-batch bias gradient kernel " + k + " should match", 
                        cpuBackprop.biasGradients[k], gpuBiasGrad, 1e-5f);
        }

        CudaUtil.free(inputGPU);
        CudaUtil.free(deltaLossGPU);
        gpuForward.free();
        gpuBackprop.free();
        group.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    
    @Test
    public void testKernelGroupWeightUpdatesGPUvsCPU() {
        // Create identical kernel groups for CPU and GPU (no Adam)
        Kernel cpuK1 = new Kernel(2, 1);
        cpuK1.setWeights(new float[]{0.1f, 0.2f});
        cpuK1.setBias(0.0f);

        Kernel cpuK2 = new Kernel(1, 1);
        cpuK2.setWeights(new float[]{0.3f});
        cpuK2.setBias(0.1f);

        KernelGroup cpuGroup = new KernelGroup(cpuK1, cpuK2);

        Kernel gpuK1 = new Kernel(2, 1);
        gpuK1.setWeights(new float[]{0.1f, 0.2f});
        gpuK1.setBias(0.0f);

        Kernel gpuK2 = new Kernel(1, 1);
        gpuK2.setWeights(new float[]{0.3f});
        gpuK2.setBias(0.1f);

        KernelGroup gpuGroup = new KernelGroup(gpuK1, gpuK2);

        // Training data - 4 batches
        float[] input = {
            // Batch 1
            1.0f, 2.0f, 0.5f, 1.1f, 2.1f, 0.6f, 1.2f, 2.2f, 0.7f,
            // Batch 2  
            1.5f, 2.5f, 0.8f, 1.6f, 2.6f, 0.9f, 1.7f, 2.7f, 1.0f,
            // Batch 3
            0.8f, 1.8f, 0.4f, 0.9f, 1.9f, 0.5f, 1.0f, 2.0f, 0.6f,
            // Batch 4
            2.0f, 3.0f, 1.0f, 2.1f, 3.1f, 1.1f, 2.2f, 3.2f, 1.2f
        };
        float[] target = {0.4f, 0.5f, 0.6f, 0.45f, 0.55f, 0.65f, 0.35f, 0.45f, 0.55f, 0.5f, 0.6f, 0.7f};
        float learningRate = 0.01f;
        int iterations = 5;

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);
        gpuGroup.prepareGPU(stream);

        for (int iter = 0; iter < iterations; iter++) {
            // CPU training step
            ConvolutionUnitForwardOutput cpuForward = cpuGroup.feedForward(input, 4);
            float[] cpuGradOutput = new float[12];
            for (int i = 0; i < 12; i++) {
                cpuGradOutput[i] = cpuForward.output[i] - target[i];
            }
            ConvolutionUnitBackpropagateOutput cpuBackprop = cpuGroup.backpropagate(cpuForward, cpuGradOutput);
            cpuGroup.updateWeights(cpuBackprop, learningRate);

            // GPU training step
            CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
            CUdeviceptr gradOutputGPU = CudaUtil.toGPUAsync(cpuGradOutput, stream);

            ConvolutionUnitForwardOutputGPU gpuForward = gpuGroup.feedForwardGPU(inputGPU, 4, stream, handle);
            ConvolutionUnitBackpropagateOutputGPU gpuBackprop = gpuGroup.backpropagateGPU(gpuForward, gradOutputGPU, stream, handle);
            gpuGroup.updateWeightsGPU(gpuBackprop, learningRate, stream);

            JCudaDriver.cuStreamSynchronize(stream);

            // Compare weights each iteration
            for (int k = 0; k < 2; k++) {
                Kernel cpuKernel = cpuGroup.getKernel(k);
                Kernel gpuKernel = gpuGroup.getKernel(k);

                float[] gpuWeights = CudaUtil.fromGPUFloat(gpuKernel.weightsGPU, gpuKernel.width);
                float gpuBias = CudaUtil.fromGPUFloat(gpuKernel.biasGPU, 1)[0];

                assertArrayEquals("SGD weights should match for kernel " + k + " at iteration " + iter, 
                    cpuKernel.getWeights(), gpuWeights, 1e-5f);
                assertEquals("SGD bias should match for kernel " + k + " at iteration " + iter, 
                    cpuKernel.getBias(), gpuBias, 1e-5f);
            }

            // Cleanup iteration resources
            CudaUtil.freeAsync(inputGPU, stream);
            CudaUtil.freeAsync(gradOutputGPU, stream);
            gpuForward.freeAsync(stream);
            gpuBackprop.freeAsync(stream);
        }

        // Cleanup
        gpuGroup.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testKernelGroupWeightUpdatesAdamGPUvsCPU() {
        // Create identical kernel groups for CPU and GPU with Adam
        Kernel cpuK1 = new Kernel(2, 1);
        cpuK1.setWeights(new float[]{0.1f, 0.2f});
        cpuK1.setBias(0.0f);
        cpuK1.setUseAdam(true);

        Kernel cpuK2 = new Kernel(1, 1);
        cpuK2.setWeights(new float[]{0.3f});
        cpuK2.setBias(0.1f);
        cpuK2.setUseAdam(true);

        KernelGroup cpuGroup = new KernelGroup(cpuK1, cpuK2);

        Kernel gpuK1 = new Kernel(2, 1);
        gpuK1.setWeights(new float[]{0.1f, 0.2f});
        gpuK1.setBias(0.0f);
        gpuK1.setUseAdam(true);

        Kernel gpuK2 = new Kernel(1, 1);
        gpuK2.setWeights(new float[]{0.3f});
        gpuK2.setBias(0.1f);
        gpuK2.setUseAdam(true);

        KernelGroup gpuGroup = new KernelGroup(gpuK1, gpuK2);

        // Training data - 4 batches
        float[] input = {
            // Batch 1
            1.0f, 2.0f, 0.5f, 1.1f, 2.1f, 0.6f, 1.2f, 2.2f, 0.7f,
            // Batch 2  
            1.5f, 2.5f, 0.8f, 1.6f, 2.6f, 0.9f, 1.7f, 2.7f, 1.0f,
            // Batch 3
            0.8f, 1.8f, 0.4f, 0.9f, 1.9f, 0.5f, 1.0f, 2.0f, 0.6f,
            // Batch 4
            2.0f, 3.0f, 1.0f, 2.1f, 3.1f, 1.1f, 2.2f, 3.2f, 1.2f
        };
        float[] target = {0.4f, 0.5f, 0.6f, 0.45f, 0.55f, 0.65f, 0.35f, 0.45f, 0.55f, 0.5f, 0.6f, 0.7f};
        float learningRate = 0.02f;
        int iterations = 5;

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);
        gpuGroup.prepareGPU(stream);

        for (int iter = 0; iter < iterations; iter++) {
            // CPU training step
            ConvolutionUnitForwardOutput cpuForward = cpuGroup.feedForward(input, 4);
            float[] cpuGradOutput = new float[12];
            for (int i = 0; i < 12; i++) {
                cpuGradOutput[i] = cpuForward.output[i] - target[i];
            }
            ConvolutionUnitBackpropagateOutput cpuBackprop = cpuGroup.backpropagate(cpuForward, cpuGradOutput);
            cpuGroup.updateWeights(cpuBackprop, learningRate);

            // GPU training step
            CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
            CUdeviceptr gradOutputGPU = CudaUtil.toGPUAsync(cpuGradOutput, stream);

            ConvolutionUnitForwardOutputGPU gpuForward = gpuGroup.feedForwardGPU(inputGPU, 4, stream, handle);
            ConvolutionUnitBackpropagateOutputGPU gpuBackprop = gpuGroup.backpropagateGPU(gpuForward, gradOutputGPU, stream, handle);
            gpuGroup.updateWeightsGPU(gpuBackprop, learningRate, stream);

            JCudaDriver.cuStreamSynchronize(stream);

            // Compare weights and Adam states each iteration
            for (int k = 0; k < 2; k++) {
                Kernel cpuKernel = cpuGroup.getKernel(k);
                Kernel gpuKernel = gpuGroup.getKernel(k);

                float[] gpuWeights = CudaUtil.fromGPUFloat(gpuKernel.weightsGPU, gpuKernel.width);
                float gpuBias = CudaUtil.fromGPUFloat(gpuKernel.biasGPU, 1)[0];
                float[] gpuWeightMomentum = CudaUtil.fromGPUFloat(gpuKernel.weightMomentumGPU, gpuKernel.width);
                float[] gpuWeightVelocity = CudaUtil.fromGPUFloat(gpuKernel.weightVelocityGPU, gpuKernel.width);
                float gpuBiasMomentum = CudaUtil.fromGPUFloat(gpuKernel.biasMomentumGPU, 1)[0];
                float gpuBiasVelocity = CudaUtil.fromGPUFloat(gpuKernel.biasVelocityGPU, 1)[0];

                assertArrayEquals("Adam weights should match for kernel " + k + " at iteration " + iter, 
                    cpuKernel.getWeights(), gpuWeights, 1e-5f);
                assertEquals("Adam bias should match for kernel " + k + " at iteration " + iter, 
                    cpuKernel.getBias(), gpuBias, 1e-5f);
                assertArrayEquals("Adam weight momentum should match for kernel " + k + " at iteration " + iter,
                    cpuKernel.weightMomentum, gpuWeightMomentum, 1e-5f);
                assertArrayEquals("Adam weight velocity should match for kernel " + k + " at iteration " + iter,
                    cpuKernel.weightVelocity, gpuWeightVelocity, 1e-5f);
                assertEquals("Adam bias momentum should match for kernel " + k + " at iteration " + iter,
                    cpuKernel.biasMomentum, gpuBiasMomentum, 1e-5f);
                assertEquals("Adam bias velocity should match for kernel " + k + " at iteration " + iter,
                    cpuKernel.biasVelocity, gpuBiasVelocity, 1e-5f);
            }

            // Cleanup iteration resources
            CudaUtil.freeAsync(inputGPU, stream);
            CudaUtil.freeAsync(gradOutputGPU, stream);
            gpuForward.freeAsync(stream);
            gpuBackprop.freeAsync(stream);
        }

        // Cleanup
        gpuGroup.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testKernelGroupIncompatibleUnitCounts() {
        // Kernels must have same unitCount for valid group
        Kernel kernel1 = new Kernel(2, 1); // unitCount=1
        Kernel kernel2 = new Kernel(2, 2); // unitCount=2 (incompatible)

        KernelGroup group = new KernelGroup(kernel1, kernel2);

        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        group.feedForward(input, 1); // Should throw exception
    }

    @Test(expected = IllegalArgumentException.class)
    public void testKernelGroupInvalidInputSize() {
        Kernel kernel1 = new Kernel(2, 1);
        Kernel kernel2 = new Kernel(3, 1);

        KernelGroup group = new KernelGroup(kernel1, kernel2);

        // inputStride = 2 + 3 = 5, but we provide 7 values (not divisible by 5)
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
        group.feedForward(input, 1); // Should throw exception
    }
}