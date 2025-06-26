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

import org.fjnn.convolution.output.layer.ConvolutionLayerForwardOutput;
import org.fjnn.convolution.output.layer.ConvolutionLayerForwardOutputGPU;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.cublasHandle;
import org.fjnn.convolution.ConvolutionLayer;
import org.fjnn.convolution.Kernel;
import org.fjnn.convolution.KernelGroup;
import org.fjnn.convolution.output.layer.ConvolutionLayerBackpropagateOutput;
import org.fjnn.convolution.output.layer.ConvolutionLayerBackpropagateOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaUtil;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Test class for ConvolutionLayer functionality
 * 
 * @author ahmed
 */
public class ConvolutionLayerTest extends ConvolutionBaseTest {
    
    // ==================== FORWARD PASS TESTS ====================
    
    @Test
    public void testConvolutionLayerForward() {
        // ==================== Test Case 1: Only Individual Kernels ====================
        // Create 3 individual kernels with same stride size (unitSize=3, unitCount=1)
        Kernel kernel1 = new Kernel(3, 1);
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 1);
        kernel2.setWeights(new float[]{0.4f, 0.5f, 0.6f});
        kernel2.setBias(0.0f);

        Kernel kernel3 = new Kernel(3, 1);
        kernel3.setWeights(new float[]{0.7f, 0.8f, 0.9f});
        kernel3.setBias(0.0f);

        ConvolutionLayer layerKernelsOnly = new ConvolutionLayer(kernel1, kernel2, kernel3);

        // Input: 3 chunks of 3 values each = 9 values
        float[] input = {
            1.0f, 2.0f, 3.0f,  // chunk 0
            4.0f, 5.0f, 6.0f,  // chunk 1
            7.0f, 8.0f, 9.0f   // chunk 2
        };

        ConvolutionLayerForwardOutput output1 = layerKernelsOnly.feedForward(input, 1);

        // Expected outputs for kernels only:
        // Kernel 1: [1,2,3] → 1*0.1+2*0.2+3*0.3 = 1.4
        //           [4,5,6] → 4*0.1+5*0.2+6*0.3 = 3.2
        //           [7,8,9] → 7*0.1+8*0.2+9*0.3 = 5.0
        final float[] EXPECTED_KERNEL1 = {1.4f, 3.2f, 5.0f};

        // Kernel 2: [1,2,3] → 1*0.4+2*0.5+3*0.6 = 3.2
        //           [4,5,6] → 4*0.4+5*0.5+6*0.6 = 7.7
        //           [7,8,9] → 7*0.4+8*0.5+9*0.6 = 12.2
        final float[] EXPECTED_KERNEL2 = {3.2f, 7.7f, 12.2f};

        // Kernel 3: [1,2,3] → 1*0.7+2*0.8+3*0.9 = 5.0
        //           [4,5,6] → 4*0.7+5*0.8+6*0.9 = 12.2
        //           [7,8,9] → 7*0.7+8*0.8+9*0.9 = 19.4
        final float[] EXPECTED_KERNEL3 = {5.0f, 12.2f, 19.4f};

        assertEquals("Should have 3 units", 3, output1.unitOutputs.length);
        assertArrayEquals("Kernel 1 output", EXPECTED_KERNEL1, output1.unitOutputs[0].output, 1e-5f);
        assertArrayEquals("Kernel 2 output", EXPECTED_KERNEL2, output1.unitOutputs[1].output, 1e-5f);
        assertArrayEquals("Kernel 3 output", EXPECTED_KERNEL3, output1.unitOutputs[2].output, 1e-5f);

        // ==================== Test Case 2: Only KernelGroups ====================
        // Create 2 kernel groups
        Kernel g1k1 = new Kernel(2, 1);
        g1k1.setWeights(new float[]{0.1f, 0.2f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(1, 1);
        g1k2.setWeights(new float[]{0.3f});
        g1k2.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2); // strideSize = 3

        Kernel g2k1 = new Kernel(1, 1);
        g2k1.setWeights(new float[]{0.4f});
        g2k1.setBias(0.0f);

        Kernel g2k2 = new Kernel(2, 1);
        g2k2.setWeights(new float[]{0.5f, 0.6f});
        g2k2.setBias(0.0f);

        KernelGroup group2 = new KernelGroup(g2k1, g2k2); // strideSize = 3

        ConvolutionLayer layerGroupsOnly = new ConvolutionLayer(group1, group2);

        ConvolutionLayerForwardOutput output2 = layerGroupsOnly.feedForward(input, 1);

        // Expected outputs for groups only:
        // Group1 processing:
        // g1k1 channel: [1,2,4,5,7,8] → outputs: [0.5, 1.4, 2.3]
        // g1k2 channel: [3,6,9] → outputs: [0.9, 1.8, 2.7]
        // Group1 applies sigmoid then AND
        final float G1K1_OUT0 = 0.5f;
        final float G1K1_OUT1 = 1.4f;
        final float G1K1_OUT2 = 2.3f;
        final float G1K2_OUT0 = 0.9f;
        final float G1K2_OUT1 = 1.8f;
        final float G1K2_OUT2 = 2.7f;

        // Sigmoid values
        final float SIG_G1K1_0 = 0.6225f; // sigmoid(0.5)
        final float SIG_G1K1_1 = 0.8022f; // sigmoid(1.4)
        final float SIG_G1K1_2 = 0.9089f; // sigmoid(2.3)
        final float SIG_G1K2_0 = 0.7109f; // sigmoid(0.9)
        final float SIG_G1K2_1 = 0.8581f; // sigmoid(1.8)
        final float SIG_G1K2_2 = 0.9370f; // sigmoid(2.7)

        final float[] EXPECTED_GROUP1 = {
            SIG_G1K1_0 * SIG_G1K2_0, // 0.4426f
            SIG_G1K1_1 * SIG_G1K2_1, // 0.6884f
            SIG_G1K1_2 * SIG_G1K2_2  // 0.8516f
        };

        // Group2 processing:
        // g2k1 channel: [1,4,7] → outputs: [0.4, 1.6, 2.8]
        // g2k2 channel: [2,3,5,6,8,9] → outputs: [2.8, 6.1, 9.4]
        final float G2K1_OUT0 = 0.4f;
        final float G2K1_OUT1 = 1.6f;
        final float G2K1_OUT2 = 2.8f;
        final float G2K2_OUT0 = 2.8f;
        final float G2K2_OUT1 = 6.1f;
        final float G2K2_OUT2 = 9.4f;

        // Sigmoid values
        final float SIG_G2K1_0 = 0.5987f; // sigmoid(0.4)
        final float SIG_G2K1_1 = 0.8320f; // sigmoid(1.6)
        final float SIG_G2K1_2 = 0.9427f; // sigmoid(2.8)
        final float SIG_G2K2_0 = 0.9427f; // sigmoid(2.8)
        final float SIG_G2K2_1 = 0.9977f; // sigmoid(6.1)
        final float SIG_G2K2_2 = 0.9999f; // sigmoid(9.4)

        final float[] EXPECTED_GROUP2 = {
            SIG_G2K1_0 * SIG_G2K2_0, // 0.5644f
            SIG_G2K1_1 * SIG_G2K2_1, // 0.8301f
            SIG_G2K1_2 * SIG_G2K2_2  // 0.9426f
        };

        assertEquals("Should have 2 units", 2, output2.unitOutputs.length);
        assertArrayEquals("Group 1 output", EXPECTED_GROUP1, output2.unitOutputs[0].output, 1e-4f);
        assertArrayEquals("Group 2 output", EXPECTED_GROUP2, output2.unitOutputs[1].output, 1e-4f);

        // ==================== Test Case 3: Mixed (Kernels + Groups) ====================
        ConvolutionLayer layerMixed = new ConvolutionLayer(kernel1, group1, kernel2);

        ConvolutionLayerForwardOutput output3 = layerMixed.feedForward(input, 1);

        assertEquals("Should have 3 units", 3, output3.unitOutputs.length);
        assertArrayEquals("Mixed - Kernel 1 output", EXPECTED_KERNEL1, output3.unitOutputs[0].output, 1e-5f);
        assertArrayEquals("Mixed - Group 1 output", EXPECTED_GROUP1, output3.unitOutputs[1].output, 1e-4f);
        assertArrayEquals("Mixed - Kernel 2 output", EXPECTED_KERNEL2, output3.unitOutputs[2].output, 1e-5f);
    }

    @Test
    public void testConvolutionLayerForwardWithUnits() {
        // ==================== Test Case 1: Only Individual Kernels with unitCount > 1 ====================
        // Create 2 kernels with unitCount=2
        Kernel kernel1 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4
        kernel2.setWeights(new float[]{0.5f, 0.6f, 0.7f, 0.8f});
        kernel2.setBias(0.0f);

        ConvolutionLayer layerKernelsOnly = new ConvolutionLayer(kernel1, kernel2);

        // Input: 3 chunks of 4 values each = 12 values
        float[] input = {
            1.0f, 2.0f, 3.0f, 4.0f,    // chunk 0
            5.0f, 6.0f, 7.0f, 8.0f,    // chunk 1
            9.0f, 10.0f, 11.0f, 12.0f  // chunk 2
        };

        ConvolutionLayerForwardOutput output1 = layerKernelsOnly.feedForward(input, 1);

        // With 6 units and unitCount=2, outputSize = 6-2+1 = 5
        // Kernel 1: sliding window of width 4 (2 units) by unitSize=2
        // Position 0: [1,2,3,4] → 1*0.1+2*0.2+3*0.3+4*0.4 = 0.1+0.4+0.9+1.6 = 3.0
        // Position 1: [3,4,5,6] → 3*0.1+4*0.2+5*0.3+6*0.4 = 0.3+0.8+1.5+2.4 = 5.0
        // Position 2: [5,6,7,8] → 5*0.1+6*0.2+7*0.3+8*0.4 = 0.5+1.2+2.1+3.2 = 7.0
        // Position 3: [7,8,9,10] → 7*0.1+8*0.2+9*0.3+10*0.4 = 0.7+1.6+2.7+4.0 = 9.0
        // Position 4: [9,10,11,12] → 9*0.1+10*0.2+11*0.3+12*0.4 = 0.9+2.0+3.3+4.8 = 11.0
        final float[] EXPECTED_KERNEL1_UNITS = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f};

        // Kernel 2:
        // Position 0: [1,2,3,4] → 1*0.5+2*0.6+3*0.7+4*0.8 = 0.5+1.2+2.1+3.2 = 7.0
        // Position 1: [3,4,5,6] → 3*0.5+4*0.6+5*0.7+6*0.8 = 1.5+2.4+3.5+4.8 = 12.2
        // Position 2: [5,6,7,8] → 5*0.5+6*0.6+7*0.7+8*0.8 = 2.5+3.6+4.9+6.4 = 17.4
        // Position 3: [7,8,9,10] → 7*0.5+8*0.6+9*0.7+10*0.8 = 3.5+4.8+6.3+8.0 = 22.6
        // Position 4: [9,10,11,12] → 9*0.5+10*0.6+11*0.7+12*0.8 = 4.5+6.0+7.7+9.6 = 27.8
        final float[] EXPECTED_KERNEL2_UNITS = {7.0f, 12.2f, 17.4f, 22.6f, 27.8f};

        assertEquals("Should have 2 units", 2, output1.unitOutputs.length);
        assertArrayEquals("Kernel 1 with units output", EXPECTED_KERNEL1_UNITS, output1.unitOutputs[0].output, 1e-5f);
        assertArrayEquals("Kernel 2 with units output", EXPECTED_KERNEL2_UNITS, output1.unitOutputs[1].output, 1e-5f);

        // ==================== Test Case 2: Only KernelGroups with unitCount > 1 ====================
        // Create groups where each kernel has unitCount=2
        Kernel g1k1 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        g1k1.setWeights(new float[]{0.1f, 0.2f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4
        g1k2.setWeights(new float[]{0.3f, 0.4f, 0.5f, 0.6f});
        g1k2.setBias(0.0f);

        Kernel g1k3 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        g1k3.setWeights(new float[]{0.7f, 0.8f});
        g1k3.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2, g1k3); // inputStride = 1+2+1 = 4

        ConvolutionLayer layerGroupsOnly = new ConvolutionLayer(group1);
        ConvolutionLayerForwardOutput output2 = layerGroupsOnly.feedForward(input, 1);

        // With 3 chunks and unitCount=2, we have 2 output positions
        // Position 0: processes chunks 0,1 → [1,2,3,4, 5,6,7,8]
        // Position 1: processes chunks 1,2 → [5,6,7,8, 9,10,11,12]

        // Expected outputs for group with unitCount=2:
        // For Position 0:
        // g1k1 extracts position 0 from chunks 0,1: [1, 5]
        // g1k2 extracts positions 1,2 from chunks 0,1: [2,3, 6,7]
        // g1k3 extracts position 3 from chunks 0,1: [4, 8]

        // For Position 1:
        // g1k1 extracts position 0 from chunks 1,2: [5, 9]
        // g1k2 extracts positions 1,2 from chunks 1,2: [6,7, 10,11]
        // g1k3 extracts position 3 from chunks 1,2: [8, 12]

        // Calculations:
        // g1k1 Position 0: [1,5] → 1*0.1 + 5*0.2 = 0.1 + 1.0 = 1.1
        // g1k1 Position 1: [5,9] → 5*0.1 + 9*0.2 = 0.5 + 1.8 = 2.3

        // g1k2 Position 0: [2,3,6,7] → 2*0.3 + 3*0.4 + 6*0.5 + 7*0.6 = 0.6 + 1.2 + 3.0 + 4.2 = 9.0
        // g1k2 Position 1: [6,7,10,11] → 6*0.3 + 7*0.4 + 10*0.5 + 11*0.6 = 1.8 + 2.8 + 5.0 + 6.6 = 16.2

        // g1k3 Position 0: [4,8] → 4*0.7 + 8*0.8 = 2.8 + 6.4 = 9.2
        // g1k3 Position 1: [8,12] → 8*0.7 + 12*0.8 = 5.6 + 9.6 = 15.2

        final float G1K1_U0 = 1.1f;   // 1*0.1 + 5*0.2
        final float G1K1_U1 = 2.3f;   // 5*0.1 + 9*0.2
        final float G1K2_U0 = 9.0f;   // 2*0.3 + 3*0.4 + 6*0.5 + 7*0.6
        final float G1K2_U1 = 16.2f;  // 6*0.3 + 7*0.4 + 10*0.5 + 11*0.6
        final float G1K3_U0 = 9.2f;   // 4*0.7 + 8*0.8
        final float G1K3_U1 = 15.2f;  // 8*0.7 + 12*0.8

        // Sigmoid values
        final float SIG_G1K1_U0 = 0.7503f; // sigmoid(1.1)
        final float SIG_G1K1_U1 = 0.9089f; // sigmoid(2.3)
        final float SIG_G1K2_U0 = 0.9999f; // sigmoid(9.0)
        final float SIG_G1K2_U1 = 1.0000f; // sigmoid(16.2)
        final float SIG_G1K3_U0 = 0.9999f; // sigmoid(9.2)
        final float SIG_G1K3_U1 = 1.0000f; // sigmoid(15.2)

        final float[] EXPECTED_GROUP1_UNITS = {
            SIG_G1K1_U0 * SIG_G1K2_U0 * SIG_G1K3_U0, // 0.7503 * 0.9999 * 0.9999 = 0.7502
            SIG_G1K1_U1 * SIG_G1K2_U1 * SIG_G1K3_U1  // 0.9089 * 1.0000 * 1.0000 = 0.9089
        };

        assertEquals("Should have 1 unit", 1, output2.unitOutputs.length);
        assertArrayEquals("Group 1 with units output", EXPECTED_GROUP1_UNITS, output2.unitOutputs[0].output, 1e-4f);

        // ==================== Test Case 3: Mixed with unitCount > 1 ====================
        // We need a kernel with strideSize=4 to match group1's strideSize
        // For individual Kernel: strideSize = width
        // So we need a kernel with width=4
        Kernel mixedKernel = new Kernel(4, 1); // width=4, strideSize=4
        mixedKernel.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        mixedKernel.setBias(0.0f);

        // group1 from Test Case 2 already has strideSize=4
        ConvolutionLayer layerMixed = new ConvolutionLayer(mixedKernel, group1);

        ConvolutionLayerForwardOutput output3 = layerMixed.feedForward(input, 1);

        // Expected output for mixedKernel (unitCount=1, so 3 outputs):
        // Chunk 0: [1,2,3,4] → 1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 = 0.1 + 0.4 + 0.9 + 1.6 = 3.0
        // Chunk 1: [5,6,7,8] → 5*0.1 + 6*0.2 + 7*0.3 + 8*0.4 = 0.5 + 1.2 + 2.1 + 3.2 = 7.0
        // Chunk 2: [9,10,11,12] → 9*0.1 + 10*0.2 + 11*0.3 + 12*0.4 = 0.9 + 2.0 + 3.3 + 4.8 = 11.0
        final float[] EXPECTED_MIXED_KERNEL = {3.0f, 7.0f, 11.0f};

        assertEquals("Should have 2 units", 2, output3.unitOutputs.length);
        assertArrayEquals("Mixed - Kernel output", EXPECTED_MIXED_KERNEL, output3.unitOutputs[0].output, 1e-5f);
        assertArrayEquals("Mixed - Group 1 with units output", EXPECTED_GROUP1_UNITS, output3.unitOutputs[1].output, 1e-4f);
    }
    
    @Test
    public void testConvolutionLayerForwardMultiBatch() {
        int batchSize = 4;

        // ==================== Test Case 1: Only Individual Kernels ====================
        Kernel kernel1 = new Kernel(3, 1);
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 1);
        kernel2.setWeights(new float[]{0.4f, 0.5f, 0.6f});
        kernel2.setBias(0.0f);

        ConvolutionLayer layerKernelsOnly = new ConvolutionLayer(kernel1, kernel2);

        // Input: 4 batches, 3 chunks each
        float[] input = {
            // Batch 0
            1.0f, 2.0f, 3.0f,  4.0f, 5.0f, 6.0f,  7.0f, 8.0f, 9.0f,
            // Batch 1
            2.0f, 3.0f, 4.0f,  5.0f, 6.0f, 7.0f,  8.0f, 9.0f, 10.0f,
            // Batch 2
            0.5f, 1.0f, 1.5f,  2.0f, 2.5f, 3.0f,  3.5f, 4.0f, 4.5f,
            // Batch 3
            1.5f, 2.5f, 3.5f,  4.5f, 5.5f, 6.5f,  7.5f, 8.5f, 9.5f
        };

        ConvolutionLayerForwardOutput output1 = layerKernelsOnly.feedForward(input, batchSize);

        // Expected outputs - Kernel 1:
        // Batch 0: [1,2,3]→1.4, [4,5,6]→3.2, [7,8,9]→5.0
        // Batch 1: [2,3,4]→2.0, [5,6,7]→3.8, [8,9,10]→5.6
        // Batch 2: [0.5,1,1.5]→0.7, [2,2.5,3]→1.6, [3.5,4,4.5]→2.5
        // Batch 3: [1.5,2.5,3.5]→1.7, [4.5,5.5,6.5]→3.5, [7.5,8.5,9.5]→5.3
        final float[] EXPECTED_KERNEL1_BATCH = {
            1.4f, 3.2f, 5.0f,    // Batch 0
            2.0f, 3.8f, 5.6f,    // Batch 1
            0.7f, 1.6f, 2.5f,    // Batch 2
            1.7f, 3.5f, 5.3f     // Batch 3
        };

        // Expected outputs - Kernel 2:
        // Batch 0: [1,2,3]→3.2, [4,5,6]→7.7, [7,8,9]→12.2
        // Batch 1: [2,3,4]→2*0.4+3*0.5+4*0.6=0.8+1.5+2.4=4.7
        //          [5,6,7]→5*0.4+6*0.5+7*0.6=2.0+3.0+4.2=9.2
        //          [8,9,10]→8*0.4+9*0.5+10*0.6=3.2+4.5+6.0=13.7
        // Batch 2: [0.5,1,1.5]→0.5*0.4+1*0.5+1.5*0.6=0.2+0.5+0.9=1.6
        //          [2,2.5,3]→2*0.4+2.5*0.5+3*0.6=0.8+1.25+1.8=3.85
        //          [3.5,4,4.5]→3.5*0.4+4*0.5+4.5*0.6=1.4+2.0+2.7=6.1
        // Batch 3: [1.5,2.5,3.5]→1.5*0.4+2.5*0.5+3.5*0.6=0.6+1.25+2.1=3.95
        //          [4.5,5.5,6.5]→4.5*0.4+5.5*0.5+6.5*0.6=1.8+2.75+3.9=8.45
        //          [7.5,8.5,9.5]→7.5*0.4+8.5*0.5+9.5*0.6=3.0+4.25+5.7=12.95
        final float[] EXPECTED_KERNEL2_BATCH = {
            3.2f, 7.7f, 12.2f,     // Batch 0
            4.7f, 9.2f, 13.7f,     // Batch 1
            1.6f, 3.85f, 6.1f,     // Batch 2
            3.95f, 8.45f, 12.95f   // Batch 3
        };

        assertEquals("Batch size", batchSize, output1.batchSize);
        assertEquals("Output size per batch", 3, output1.unitOutputs[0].outputSize);
        assertArrayEquals("Kernel 1 multi-batch output", EXPECTED_KERNEL1_BATCH, output1.unitOutputs[0].output, 1e-5f);
        assertArrayEquals("Kernel 2 multi-batch output", EXPECTED_KERNEL2_BATCH, output1.unitOutputs[1].output, 1e-5f);

        // ==================== Test Case 2: Only KernelGroups ====================
        Kernel g1k1 = new Kernel(2, 1);
        g1k1.setWeights(new float[]{0.1f, 0.2f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(1, 1);
        g1k2.setWeights(new float[]{0.3f});
        g1k2.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2);

        ConvolutionLayer layerGroupsOnly = new ConvolutionLayer(group1);

        ConvolutionLayerForwardOutput output2 = layerGroupsOnly.feedForward(input, batchSize);

        // Group processing for each batch:
        // KernelGroup extracts channels: g1k1 gets [elem0, elem1], g1k2 gets [elem2]
        // Then applies sigmoid to each kernel output and performs AND operation (multiplication)
        //
        // Batch 0: 
        //   Chunk [1,2,3]: g1k1([1,2])→0.5, g1k2([3])→0.9 → sigmoid(0.5)*sigmoid(0.9) = 0.4425
        //   Chunk [4,5,6]: g1k1([4,5])→1.4, g1k2([6])→1.8 → sigmoid(1.4)*sigmoid(1.8) = 0.6884
        //   Chunk [7,8,9]: g1k1([7,8])→2.3, g1k2([9])→2.7 → sigmoid(2.3)*sigmoid(2.7) = 0.8516
        // 
        // Batch 1:
        //   Chunk [2,3,4]: g1k1([2,3])→0.8, g1k2([4])→1.2 → sigmoid(0.8)*sigmoid(1.2) = 0.5303
        //   Chunk [5,6,7]: g1k1([5,6])→1.7, g1k2([7])→2.1 → sigmoid(1.7)*sigmoid(2.1) = 0.7533
        //   Chunk [8,9,10]: g1k1([8,9])→2.6, g1k2([10])→3.0 → sigmoid(2.6)*sigmoid(3.0) = 0.8867
        //
        // Batch 2:
        //   Chunk [0.5,1,1.5]: g1k1([0.5,1])→0.25, g1k2([1.5])→0.45 → sigmoid(0.25)*sigmoid(0.45) = 0.3433
        //   Chunk [2,2.5,3]: g1k1([2,2.5])→0.7, g1k2([3])→0.9 → sigmoid(0.7)*sigmoid(0.9) = 0.4750
        //   Chunk [3.5,4,4.5]: g1k1([3.5,4])→1.15, g1k2([4.5])→1.35 → sigmoid(1.15)*sigmoid(1.35) = 0.6032
        //
        // Batch 3:
        //   Chunk [1.5,2.5,3.5]: g1k1([1.5,2.5])→0.65, g1k2([3.5])→1.05 → sigmoid(0.65)*sigmoid(1.05) = 0.4867
        //   Chunk [4.5,5.5,6.5]: g1k1([4.5,5.5])→1.55, g1k2([6.5])→1.95 → sigmoid(1.55)*sigmoid(1.95) = 0.7222
        //   Chunk [7.5,8.5,9.5]: g1k1([7.5,8.5])→2.45, g1k2([9.5])→2.85 → sigmoid(2.45)*sigmoid(2.85) = 0.8702

        final float[] EXPECTED_GROUP1_BATCH = {
            0.4425f, 0.6884f, 0.8516f,  // Batch 0
            0.5303f, 0.7533f, 0.8867f,  // Batch 1
            0.3433f, 0.4750f, 0.6032f,  // Batch 2
            0.4867f, 0.7222f, 0.8702f   // Batch 3
        };

        assertEquals("Group batch size", batchSize, output2.batchSize);
        assertArrayEquals("Group 1 multi-batch output", EXPECTED_GROUP1_BATCH, output2.unitOutputs[0].output, 1e-4f);

        // ==================== Test Case 3: Mixed ====================
        ConvolutionLayer layerMixed = new ConvolutionLayer(kernel1, group1);

        ConvolutionLayerForwardOutput output3 = layerMixed.feedForward(input, batchSize);

        assertEquals("Mixed should have 2 units", 2, output3.unitOutputs.length);
        assertArrayEquals("Mixed - Kernel 1 multi-batch output", EXPECTED_KERNEL1_BATCH, output3.unitOutputs[0].output, 1e-5f);
        assertArrayEquals("Mixed - Group 1 multi-batch output", EXPECTED_GROUP1_BATCH, output3.unitOutputs[1].output, 1e-4f);
    }
    
    @Test
    public void testConvolutionLayerForwardWithUnitsMultiBatch() {
        int batchSize = 4;

        // ==================== Test Case 1: Individual Kernels with unitCount=2 and strideSize=4 ====================
        // Note: To match KernelGroup strideSize=4, individual kernels must use Kernel(4, 2)
        Kernel kernel1 = new Kernel(4, 2); // unitSize=4, unitCount=2, width=8, strideSize=4
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(4, 2); // unitSize=4, unitCount=2, width=8, strideSize=4
        kernel2.setWeights(new float[]{0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});
        kernel2.setBias(0.0f);

        ConvolutionLayer layerKernelsOnly = new ConvolutionLayer(kernel1, kernel2);

        // Input: 4 batches, 12 values each (3 units of size 4)
        float[] input = {
            // Batch 0
            1.0f, 2.0f, 3.0f, 4.0f,  5.0f, 6.0f, 7.0f, 8.0f,  9.0f, 10.0f, 11.0f, 12.0f,
            // Batch 1
            2.0f, 3.0f, 4.0f, 5.0f,  6.0f, 7.0f, 8.0f, 9.0f,  10.0f, 11.0f, 12.0f, 13.0f,
            // Batch 2
            0.5f, 1.0f, 1.5f, 2.0f,  2.5f, 3.0f, 3.5f, 4.0f,  4.5f, 5.0f, 5.5f, 6.0f,
            // Batch 3
            1.5f, 2.5f, 3.5f, 4.5f,  5.5f, 6.5f, 7.5f, 8.5f,  9.5f, 10.5f, 11.5f, 12.5f
        };

        ConvolutionLayerForwardOutput output1 = layerKernelsOnly.feedForward(input, batchSize);

        // With strideSize=4, numUnits per batch = 12/4 = 3
        // With unitCount=2, outputSize = 3-2+1 = 2 per batch
        // 
        // Each position processes 2 consecutive units (8 values total):
        // Position 0: units 0+1 = [values 0-7]
        // Position 1: units 1+2 = [values 4-11]
        //
        // Batch 0:
        //   Position 0: [1,2,3,4,5,6,7,8] → 1×0.1+2×0.2+3×0.3+4×0.4+5×0.5+6×0.6+7×0.7+8×0.8 = 20.4
        //   Position 1: [5,6,7,8,9,10,11,12] → 5×0.1+6×0.2+7×0.3+8×0.4+9×0.5+10×0.6+11×0.7+12×0.8 = 34.8
        //
        // Batch 1:
        //   Position 0: [2,3,4,5,6,7,8,9] → 2×0.1+3×0.2+4×0.3+5×0.4+6×0.5+7×0.6+8×0.7+9×0.8 = 24.0
        //   Position 1: [6,7,8,9,10,11,12,13] → 6×0.1+7×0.2+8×0.3+9×0.4+10×0.5+11×0.6+12×0.7+13×0.8 = 38.4
        //
        // Batch 2:
        //   Position 0: [0.5,1,1.5,2,2.5,3,3.5,4] → 0.5×0.1+1×0.2+1.5×0.3+2×0.4+2.5×0.5+3×0.6+3.5×0.7+4×0.8 = 10.2
        //   Position 1: [2.5,3,3.5,4,4.5,5,5.5,6] → 2.5×0.1+3×0.2+3.5×0.3+4×0.4+4.5×0.5+5×0.6+5.5×0.7+6×0.8 = 17.4
        //
        // Batch 3:
        //   Position 0: [1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5] → 1.5×0.1+2.5×0.2+3.5×0.3+4.5×0.4+5.5×0.5+6.5×0.6+7.5×0.7+8.5×0.8 = 22.2
        //   Position 1: [5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5] → 5.5×0.1+6.5×0.2+7.5×0.3+8.5×0.4+9.5×0.5+10.5×0.6+11.5×0.7+12.5×0.8 = 36.6
        final float[] EXPECTED_KERNEL1_UNITS_BATCH = {
            20.4f, 34.8f,  // Batch 0
            24.0f, 38.4f,  // Batch 1
            10.2f, 17.4f,  // Batch 2
            22.2f, 36.6f   // Batch 3
        };

        // Kernel 2 with weights [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        // Batch 0:
        //   Position 0: [1,2,3,4,5,6,7,8] → 1×0.2+2×0.3+3×0.4+4×0.5+5×0.6+6×0.7+7×0.8+8×0.9 = 24.0
        //   Position 1: [5,6,7,8,9,10,11,12] → 5×0.2+6×0.3+7×0.4+8×0.5+9×0.6+10×0.7+11×0.8+12×0.9 = 41.6
        //
        // Batch 1:
        //   Position 0: [2,3,4,5,6,7,8,9] → 2×0.2+3×0.3+4×0.4+5×0.5+6×0.6+7×0.7+8×0.8+9×0.9 = 28.4
        //   Position 1: [6,7,8,9,10,11,12,13] → 6×0.2+7×0.3+8×0.4+9×0.5+10×0.6+11×0.7+12×0.8+13×0.9 = 46.0
        //
        // Batch 2:
        //   Position 0: [0.5,1,1.5,2,2.5,3,3.5,4] → 0.5×0.2+1×0.3+1.5×0.4+2×0.5+2.5×0.6+3×0.7+3.5×0.8+4×0.9 = 12.0
        //   Position 1: [2.5,3,3.5,4,4.5,5,5.5,6] → 2.5×0.2+3×0.3+3.5×0.4+4×0.5+4.5×0.6+5×0.7+5.5×0.8+6×0.9 = 20.8
        //
        // Batch 3:
        //   Position 0: [1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5] → 1.5×0.2+2.5×0.3+3.5×0.4+4.5×0.5+5.5×0.6+6.5×0.7+7.5×0.8+8.5×0.9 = 26.2
        //   Position 1: [5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5] → 5.5×0.2+6.5×0.3+7.5×0.4+8.5×0.5+9.5×0.6+10.5×0.7+11.5×0.8+12.5×0.9 = 43.8
        final float[] EXPECTED_KERNEL2_UNITS_BATCH = {
            24.0f, 41.6f,  // Batch 0
            28.4f, 46.0f,  // Batch 1
            12.0f, 20.8f,  // Batch 2
            26.2f, 43.8f   // Batch 3
        };

        assertEquals("Batch size", batchSize, output1.batchSize);
        assertEquals("Output size per batch", 2, output1.unitOutputs[0].outputSize);
        assertArrayEquals("Kernel 1 units multi-batch output", EXPECTED_KERNEL1_UNITS_BATCH, output1.unitOutputs[0].output, 1e-5f);
        assertArrayEquals("Kernel 2 units multi-batch output", EXPECTED_KERNEL2_UNITS_BATCH, output1.unitOutputs[1].output, 1e-5f);

        // ==================== Test Case 2: Only KernelGroups with unitCount=2 ====================
        Kernel g1k1 = new Kernel(1, 2);
        g1k1.setWeights(new float[]{0.1f, 0.2f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(2, 2);
        g1k2.setWeights(new float[]{0.3f, 0.4f, 0.5f, 0.6f});
        g1k2.setBias(0.0f);

        Kernel g1k3 = new Kernel(1, 2);
        g1k3.setWeights(new float[]{0.7f, 0.8f});
        g1k3.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2, g1k3); // strideSize = 1+2+1 = 4

        ConvolutionLayer layerGroupsOnly = new ConvolutionLayer(group1);

        ConvolutionLayerForwardOutput output2 = layerGroupsOnly.feedForward(input, batchSize);

        // KernelGroup processing with strideSize=4:
        // Each batch has 3 chunks of 4 values, outputSize = 3-2+1 = 2
        // Position 0: chunks 0+1, Position 1: chunks 1+2
        //
        // Channel extraction per chunk: ch1=[val0], ch2=[val1,val2], ch3=[val3]
        // Each kernel processes across 2 consecutive chunks with unitCount=2
        //
        // The exact values depend on sigmoid functions and AND operations between kernels.
        // These are computed based on the channel extraction and convolution operations.
        final float[] EXPECTED_GROUP1_UNITS_BATCH = {
            0.7501f, 0.9089f,  // Batch 0
            0.8021f, 0.9309f,  // Batch 1
            0.6209f, 0.7589f,  // Batch 2
            0.7772f, 0.9206f   // Batch 3
        };

        assertEquals("Group batch size", batchSize, output2.batchSize);
        assertEquals("Group output size per batch", 2, output2.unitOutputs[0].outputSize);
        assertArrayEquals("Group 1 units multi-batch output", EXPECTED_GROUP1_UNITS_BATCH, output2.unitOutputs[0].output, 1e-4f);

        // ==================== Test Case 3: Mixed ====================
        ConvolutionLayer layerMixed = new ConvolutionLayer(kernel1, group1);

        ConvolutionLayerForwardOutput output3 = layerMixed.feedForward(input, batchSize);

        assertEquals("Mixed should have 2 units", 2, output3.unitOutputs.length);
        assertArrayEquals("Mixed - Kernel 1 units multi-batch output", EXPECTED_KERNEL1_UNITS_BATCH, output3.unitOutputs[0].output, 1e-5f);
        assertArrayEquals("Mixed - Group 1 units multi-batch output", EXPECTED_GROUP1_UNITS_BATCH, output3.unitOutputs[1].output, 1e-4f);
    }
    
    // ==================== GPU FORWARD PASS TESTS ====================
    
    @Test
    public void testConvolutionLayerForwardGPU() {
        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        // ==================== Test Case 1: Only Individual Kernels ====================
        Kernel kernel1 = new Kernel(3, 1);
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 1);
        kernel2.setWeights(new float[]{0.4f, 0.5f, 0.6f});
        kernel2.setBias(0.0f);

        ConvolutionLayer layerKernelsOnly = new ConvolutionLayer(kernel1, kernel2);

        float[] input = {
            1.0f, 2.0f, 3.0f,  4.0f, 5.0f, 6.0f,  7.0f, 8.0f, 9.0f
        };

        // CPU reference
        ConvolutionLayerForwardOutput cpuOutput1 = layerKernelsOnly.feedForward(input, 1);

        // GPU computation
        layerKernelsOnly.prepareGPU(stream);
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuOutput1 = layerKernelsOnly.feedForwardGPU(inputGPU, 1, stream, handle);

        // Compare results
        for (int i = 0; i < 2; i++) {
            float[] gpuUnitOutput = CudaUtil.fromGPUFloatAsync(
                gpuOutput1.unitOutputs[i].output, 
                cpuOutput1.unitOutputs[i].output.length, 
                stream);
            JCudaDriver.cuStreamSynchronize(stream);

            assertArrayEquals("GPU Kernel " + (i+1) + " output should match CPU", 
                cpuOutput1.unitOutputs[i].output, gpuUnitOutput, 1e-5f);
        }

        // Cleanup case 1
        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput1.freeAsync(stream);
        layerKernelsOnly.freeGPU(stream);

        // ==================== Test Case 2: Only KernelGroups ====================
        Kernel g1k1 = new Kernel(2, 1);
        g1k1.setWeights(new float[]{0.1f, 0.2f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(1, 1);
        g1k2.setWeights(new float[]{0.3f});
        g1k2.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2);

        ConvolutionLayer layerGroupsOnly = new ConvolutionLayer(group1);

        // CPU reference
        ConvolutionLayerForwardOutput cpuOutput2 = layerGroupsOnly.feedForward(input, 1);

        // GPU computation
        layerGroupsOnly.prepareGPU(stream);
        inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuOutput2 = layerGroupsOnly.feedForwardGPU(inputGPU, 1, stream, handle);

        // Compare results
        float[] gpuGroupOutput = CudaUtil.fromGPUFloatAsync(
            gpuOutput2.unitOutputs[0].output, 
            cpuOutput2.unitOutputs[0].output.length, 
            stream);
        JCudaDriver.cuStreamSynchronize(stream);

        assertArrayEquals("GPU Group 1 output should match CPU", 
            cpuOutput2.unitOutputs[0].output, gpuGroupOutput, 1e-4f);

        // Cleanup case 2
        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput2.freeAsync(stream);
        layerGroupsOnly.freeGPU(stream);

        // ==================== Test Case 3: Mixed ====================
        ConvolutionLayer layerMixed = new ConvolutionLayer(kernel1, group1);

        // CPU reference
        ConvolutionLayerForwardOutput cpuOutput3 = layerMixed.feedForward(input, 1);

        // GPU computation
        layerMixed.prepareGPU(stream);
        inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuOutput3 = layerMixed.feedForwardGPU(inputGPU, 1, stream, handle);

        // Compare results
        float[] gpuKernelOutput = CudaUtil.fromGPUFloatAsync(
            gpuOutput3.unitOutputs[0].output, 
            cpuOutput3.unitOutputs[0].output.length, 
            stream);
        float[] gpuGroupMixedOutput = CudaUtil.fromGPUFloatAsync(
            gpuOutput3.unitOutputs[1].output, 
            cpuOutput3.unitOutputs[1].output.length, 
            stream);
        JCudaDriver.cuStreamSynchronize(stream);

        assertArrayEquals("GPU Mixed - Kernel output should match CPU", 
            cpuOutput3.unitOutputs[0].output, gpuKernelOutput, 1e-5f);
        assertArrayEquals("GPU Mixed - Group output should match CPU", 
            cpuOutput3.unitOutputs[1].output, gpuGroupMixedOutput, 1e-4f);

        // Final cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput3.freeAsync(stream);
        layerMixed.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testConvolutionLayerForwardWithUnitsGPU() {
        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        // ==================== Test Case 1: Individual Kernels with strideSize=4 ====================
        // Fixed: Use Kernel(4, 2) to match KernelGroup strideSize=4
        Kernel kernel1 = new Kernel(4, 2); // unitSize=4, unitCount=2, strideSize=4
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(4, 2); // unitSize=4, unitCount=2, strideSize=4
        kernel2.setWeights(new float[]{0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});
        kernel2.setBias(0.0f);

        ConvolutionLayer layerKernelsOnly = new ConvolutionLayer(kernel1, kernel2);

        // Input: 12 values = 3 units of size 4, outputSize = 3-2+1 = 2
        float[] input = {
            1.0f, 2.0f, 3.0f, 4.0f,  5.0f, 6.0f, 7.0f, 8.0f,  9.0f, 10.0f, 11.0f, 12.0f
        };

        // CPU reference
        ConvolutionLayerForwardOutput cpuOutput1 = layerKernelsOnly.feedForward(input, 1);

        // GPU computation
        layerKernelsOnly.prepareGPU(stream);
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuOutput1 = layerKernelsOnly.feedForwardGPU(inputGPU, 1, stream, handle);

        // Compare results
        for (int i = 0; i < 2; i++) {
            float[] gpuUnitOutput = CudaUtil.fromGPUFloatAsync(
                gpuOutput1.unitOutputs[i].output, 
                cpuOutput1.unitOutputs[i].output.length, 
                stream);
            JCudaDriver.cuStreamSynchronize(stream);

            assertArrayEquals("GPU Kernel " + (i+1) + " with units output should match CPU", 
                cpuOutput1.unitOutputs[i].output, gpuUnitOutput, 1e-5f);
        }

        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput1.freeAsync(stream);
        layerKernelsOnly.freeGPU(stream);

        // ==================== Test Case 2: KernelGroups with strideSize=4 ====================
        Kernel g1k1 = new Kernel(1, 2);
        g1k1.setWeights(new float[]{0.1f, 0.2f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(2, 2);
        g1k2.setWeights(new float[]{0.3f, 0.4f, 0.5f, 0.6f});
        g1k2.setBias(0.0f);

        Kernel g1k3 = new Kernel(1, 2);
        g1k3.setWeights(new float[]{0.7f, 0.8f});
        g1k3.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2, g1k3); // strideSize = 1+2+1 = 4

        ConvolutionLayer layerGroupsOnly = new ConvolutionLayer(group1);

        // CPU reference
        ConvolutionLayerForwardOutput cpuOutput2 = layerGroupsOnly.feedForward(input, 1);

        // GPU computation
        layerGroupsOnly.prepareGPU(stream);
        inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuOutput2 = layerGroupsOnly.feedForwardGPU(inputGPU, 1, stream, handle);

        // Compare results
        float[] gpuGroupOutput = CudaUtil.fromGPUFloatAsync(
            gpuOutput2.unitOutputs[0].output, 
            cpuOutput2.unitOutputs[0].output.length, 
            stream);
        JCudaDriver.cuStreamSynchronize(stream);

        assertArrayEquals("GPU Group with units output should match CPU", 
            cpuOutput2.unitOutputs[0].output, gpuGroupOutput, 1e-4f);

        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput2.freeAsync(stream);
        layerGroupsOnly.freeGPU(stream);

        // ==================== Test Case 3: Mixed (Compatible Stride Sizes) ====================
        // Now both kernel1 and group1 have strideSize=4, so they can be mixed
        ConvolutionLayer layerMixed = new ConvolutionLayer(kernel1, group1);

        // CPU reference
        ConvolutionLayerForwardOutput cpuOutput3 = layerMixed.feedForward(input, 1);

        // GPU computation
        layerMixed.prepareGPU(stream);
        inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuOutput3 = layerMixed.feedForwardGPU(inputGPU, 1, stream, handle);

        // Compare results
        for (int i = 0; i < 2; i++) {
            float[] gpuUnitOutput = CudaUtil.fromGPUFloatAsync(
                gpuOutput3.unitOutputs[i].output, 
                cpuOutput3.unitOutputs[i].output.length, 
                stream);
            JCudaDriver.cuStreamSynchronize(stream);

            assertArrayEquals("GPU Mixed unit " + i + " with units output should match CPU", 
                cpuOutput3.unitOutputs[i].output, gpuUnitOutput, 1e-4f);
        }

        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput3.freeAsync(stream);
        layerMixed.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testConvolutionLayerForwardMultiBatchGPU() {
        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);
        int batchSize = 4;

        // ==================== Test Case 1: Only Individual Kernels ====================
        Kernel kernel1 = new Kernel(3, 1);
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 1);
        kernel2.setWeights(new float[]{0.4f, 0.5f, 0.6f});
        kernel2.setBias(0.0f);

        ConvolutionLayer layerKernelsOnly = new ConvolutionLayer(kernel1, kernel2);

        float[] input = {
            // Batch 0
            1.0f, 2.0f, 3.0f,  4.0f, 5.0f, 6.0f,  7.0f, 8.0f, 9.0f,
            // Batch 1
            2.0f, 3.0f, 4.0f,  5.0f, 6.0f, 7.0f,  8.0f, 9.0f, 10.0f,
            // Batch 2
            0.5f, 1.0f, 1.5f,  2.0f, 2.5f, 3.0f,  3.5f, 4.0f, 4.5f,
            // Batch 3
            1.5f, 2.5f, 3.5f,  4.5f, 5.5f, 6.5f,  7.5f, 8.5f, 9.5f
        };

        // CPU reference
        ConvolutionLayerForwardOutput cpuOutput1 = layerKernelsOnly.feedForward(input, batchSize);

        // GPU computation
        layerKernelsOnly.prepareGPU(stream);
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuOutput1 = layerKernelsOnly.feedForwardGPU(inputGPU, batchSize, stream, handle);

        // Compare results
        for (int i = 0; i < 2; i++) {
            float[] gpuUnitOutput = CudaUtil.fromGPUFloatAsync(
                gpuOutput1.unitOutputs[i].output, 
                cpuOutput1.unitOutputs[i].output.length, 
                stream);
            JCudaDriver.cuStreamSynchronize(stream);

            assertArrayEquals("GPU Kernel " + (i+1) + " multi-batch output should match CPU", 
                cpuOutput1.unitOutputs[i].output, gpuUnitOutput, 1e-5f);
        }

        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput1.freeAsync(stream);
        layerKernelsOnly.freeGPU(stream);

        // ==================== Test Case 2: Only KernelGroups ====================
        Kernel g1k1 = new Kernel(2, 1);
        g1k1.setWeights(new float[]{0.1f, 0.2f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(1, 1);
        g1k2.setWeights(new float[]{0.3f});
        g1k2.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2);

        ConvolutionLayer layerGroupsOnly = new ConvolutionLayer(group1);

        // CPU reference
        ConvolutionLayerForwardOutput cpuOutput2 = layerGroupsOnly.feedForward(input, batchSize);

        // GPU computation
        layerGroupsOnly.prepareGPU(stream);
        inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuOutput2 = layerGroupsOnly.feedForwardGPU(inputGPU, batchSize, stream, handle);

        // Compare results
        float[] gpuGroupOutput = CudaUtil.fromGPUFloatAsync(
            gpuOutput2.unitOutputs[0].output, 
            cpuOutput2.unitOutputs[0].output.length, 
            stream);
        JCudaDriver.cuStreamSynchronize(stream);

        assertArrayEquals("GPU Group multi-batch output should match CPU", 
            cpuOutput2.unitOutputs[0].output, gpuGroupOutput, 1e-4f);

        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput2.freeAsync(stream);
        layerGroupsOnly.freeGPU(stream);

        // ==================== Test Case 3: Mixed ====================
        ConvolutionLayer layerMixed = new ConvolutionLayer(kernel1, group1);

        // CPU reference
        ConvolutionLayerForwardOutput cpuOutput3 = layerMixed.feedForward(input, batchSize);

        // GPU computation
        layerMixed.prepareGPU(stream);
        inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuOutput3 = layerMixed.feedForwardGPU(inputGPU, batchSize, stream, handle);

        // Compare results
        for (int i = 0; i < 2; i++) {
            float[] gpuUnitOutput = CudaUtil.fromGPUFloatAsync(
                gpuOutput3.unitOutputs[i].output, 
                cpuOutput3.unitOutputs[i].output.length, 
                stream);
            JCudaDriver.cuStreamSynchronize(stream);

            assertArrayEquals("GPU Mixed unit " + i + " multi-batch output should match CPU", 
                cpuOutput3.unitOutputs[i].output, gpuUnitOutput, 1e-4f);
        }

        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput3.freeAsync(stream);
        layerMixed.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testConvolutionLayerForwardWithUnitsMultiBatchGPU() {
        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);
        int batchSize = 4;

        // ==================== Test Case 1: Individual Kernels with strideSize=4 ====================
        // Fixed: Use Kernel(4, 2) to match KernelGroup strideSize=4
        Kernel kernel1 = new Kernel(4, 2); // unitSize=4, unitCount=2, strideSize=4
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(4, 2); // unitSize=4, unitCount=2, strideSize=4
        kernel2.setWeights(new float[]{0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f});
        kernel2.setBias(0.0f);

        ConvolutionLayer layerKernelsOnly = new ConvolutionLayer(kernel1, kernel2);

        // Input: 4 batches, 12 values each (3 units of size 4), outputSize = 3-2+1 = 2 per batch
        float[] input = {
            // Batch 0
            1.0f, 2.0f, 3.0f, 4.0f,  5.0f, 6.0f, 7.0f, 8.0f,  9.0f, 10.0f, 11.0f, 12.0f,
            // Batch 1
            2.0f, 3.0f, 4.0f, 5.0f,  6.0f, 7.0f, 8.0f, 9.0f,  10.0f, 11.0f, 12.0f, 13.0f,
            // Batch 2
            0.5f, 1.0f, 1.5f, 2.0f,  2.5f, 3.0f, 3.5f, 4.0f,  4.5f, 5.0f, 5.5f, 6.0f,
            // Batch 3
            1.5f, 2.5f, 3.5f, 4.5f,  5.5f, 6.5f, 7.5f, 8.5f,  9.5f, 10.5f, 11.5f, 12.5f
        };

        // CPU reference
        ConvolutionLayerForwardOutput cpuOutput1 = layerKernelsOnly.feedForward(input, batchSize);

        // GPU computation
        layerKernelsOnly.prepareGPU(stream);
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuOutput1 = layerKernelsOnly.feedForwardGPU(inputGPU, batchSize, stream, handle);

        // Compare results
        for (int i = 0; i < 2; i++) {
            float[] gpuUnitOutput = CudaUtil.fromGPUFloatAsync(
                gpuOutput1.unitOutputs[i].output, 
                cpuOutput1.unitOutputs[i].output.length, 
                stream);
            JCudaDriver.cuStreamSynchronize(stream);

            assertArrayEquals("GPU Kernel " + (i+1) + " units multi-batch output should match CPU", 
                cpuOutput1.unitOutputs[i].output, gpuUnitOutput, 1e-5f);
        }

        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput1.freeAsync(stream);
        layerKernelsOnly.freeGPU(stream);

        // ==================== Test Case 2: KernelGroups with strideSize=4 ====================
        Kernel g1k1 = new Kernel(1, 2);
        g1k1.setWeights(new float[]{0.1f, 0.2f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(2, 2);
        g1k2.setWeights(new float[]{0.3f, 0.4f, 0.5f, 0.6f});
        g1k2.setBias(0.0f);

        Kernel g1k3 = new Kernel(1, 2);
        g1k3.setWeights(new float[]{0.7f, 0.8f});
        g1k3.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2, g1k3); // strideSize = 1+2+1 = 4

        ConvolutionLayer layerGroupsOnly = new ConvolutionLayer(group1);

        // CPU reference
        ConvolutionLayerForwardOutput cpuOutput2 = layerGroupsOnly.feedForward(input, batchSize);

        // GPU computation
        layerGroupsOnly.prepareGPU(stream);
        inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuOutput2 = layerGroupsOnly.feedForwardGPU(inputGPU, batchSize, stream, handle);

        // Compare results
        float[] gpuGroupOutput = CudaUtil.fromGPUFloatAsync(
            gpuOutput2.unitOutputs[0].output, 
            cpuOutput2.unitOutputs[0].output.length, 
            stream);
        JCudaDriver.cuStreamSynchronize(stream);

        assertArrayEquals("GPU Group units multi-batch output should match CPU", 
            cpuOutput2.unitOutputs[0].output, gpuGroupOutput, 1e-4f);

        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput2.freeAsync(stream);
        layerGroupsOnly.freeGPU(stream);

        // ==================== Test Case 3: Mixed (Compatible Stride Sizes) ====================
        // Now both kernel1 and group1 have strideSize=4, so they can be mixed
        ConvolutionLayer layerMixed = new ConvolutionLayer(kernel1, group1);

        // CPU reference
        ConvolutionLayerForwardOutput cpuOutput3 = layerMixed.feedForward(input, batchSize);

        // GPU computation
        layerMixed.prepareGPU(stream);
        inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuOutput3 = layerMixed.feedForwardGPU(inputGPU, batchSize, stream, handle);

        // Compare results
        for (int i = 0; i < 2; i++) {
            float[] gpuUnitOutput = CudaUtil.fromGPUFloatAsync(
                gpuOutput3.unitOutputs[i].output, 
                cpuOutput3.unitOutputs[i].output.length, 
                stream);
            JCudaDriver.cuStreamSynchronize(stream);

            assertArrayEquals("GPU Mixed unit " + i + " units multi-batch output should match CPU", 
                cpuOutput3.unitOutputs[i].output, gpuUnitOutput, 1e-4f);
        }

        CudaUtil.freeAsync(inputGPU, stream);
        gpuOutput3.freeAsync(stream);
        layerMixed.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }
    
    // ==================== BACKWARD PASS TESTS ====================
    
    @Test
    public void testConvolutionLayerBackward() {
        // Basic backward test with individual kernels + kernel group (mixed)
        Kernel kernel1 = new Kernel(3, 1); // unitSize=3, unitCount=1, width=3
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        kernel1.setBias(0.0f);

        Kernel g1k1 = new Kernel(2, 1); // unitSize=2, unitCount=1, width=2
        g1k1.setWeights(new float[]{0.4f, 0.5f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(1, 1); // unitSize=1, unitCount=1, width=1
        g1k2.setWeights(new float[]{0.6f});
        g1k2.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2); // strideSize = 2+1 = 3

        ConvolutionLayer layer = new ConvolutionLayer(kernel1, group1);

        // Input: 3 chunks of 3 values each = 9 values
        float[] input = {
            1.0f, 2.0f, 3.0f,  // chunk 0
            4.0f, 5.0f, 6.0f,  // chunk 1
            7.0f, 8.0f, 9.0f   // chunk 2
        };

        ConvolutionLayerForwardOutput forwardOutput = layer.feedForward(input, 1);

        // Delta loss from next layer
        float[][] unitDeltaLoss = new float[2][];
        unitDeltaLoss[0] = new float[]{0.1f, 0.2f, 0.3f}; // kernel1 outputs
        unitDeltaLoss[1] = new float[]{0.05f, 0.15f, 0.25f}; // group1 outputs

        ConvolutionLayerBackpropagateOutput backpropOutput = layer.backpropagate(forwardOutput, unitDeltaLoss);

        // Expected structure verification
        assertEquals("Should have 2 unit backprops", 2, backpropOutput.unitBackprops.length);
        assertEquals("Input gradients size", input.length, backpropOutput.inputGradients.length);

        // Expected kernel1 weight gradients: [1*0.1+4*0.2+7*0.3, 2*0.1+5*0.2+8*0.3, 3*0.1+6*0.2+9*0.3] = [3.0, 3.6, 4.2]
        final float[] EXPECTED_KERNEL1_WEIGHT_GRADS = {3.0f, 3.6f, 4.2f};
        assertArrayEquals("Kernel1 weight gradients", EXPECTED_KERNEL1_WEIGHT_GRADS, 
            backpropOutput.unitBackprops[0].weightGradients[0], 1e-4f);

        // Expected kernel1 bias gradient: 0.1+0.2+0.3 = 0.6
        final float EXPECTED_KERNEL1_BIAS_GRAD = 0.6f;
        assertEquals("Kernel1 bias gradient", EXPECTED_KERNEL1_BIAS_GRAD, 
            backpropOutput.unitBackprops[0].biasGradients[0], 1e-4f);

        // Group1 gradients computed through sigmoid derivative and AND operation
        // The exact values depend on sigmoid calculations of g1k1 outputs [1.4, 4.1, 6.8] and g1k2 outputs [1.8, 3.6, 5.4]

        // g1k1 weight gradients (computed through sigmoid backprop)
        final float[] EXPECTED_G1K1_WEIGHT_GRADS = {0.018111f, 0.027538f};
        assertArrayEquals("Group1 k1 weight gradients", EXPECTED_G1K1_WEIGHT_GRADS, 
            backpropOutput.unitBackprops[1].weightGradients[0], 1e-4f);

        // g1k2 weight gradients (computed through sigmoid backprop)
        final float[] EXPECTED_G1K2_WEIGHT_GRADS = {0.047628f};
        assertArrayEquals("Group1 k2 weight gradients", EXPECTED_G1K2_WEIGHT_GRADS, 
            backpropOutput.unitBackprops[1].weightGradients[1], 1e-4f);

        // g1k1 bias gradient (sum of sigmoid gradients)
        final float EXPECTED_G1K1_BIAS_GRAD = 0.009427f;
        assertEquals("Group1 k1 bias gradient", EXPECTED_G1K1_BIAS_GRAD, 
            backpropOutput.unitBackprops[1].biasGradients[0], 1e-4f);

        // g1k2 bias gradient (sum of sigmoid gradients)
        final float EXPECTED_G1K2_BIAS_GRAD = 0.009820f;
        assertEquals("Group1 k2 bias gradient", EXPECTED_G1K2_BIAS_GRAD, 
            backpropOutput.unitBackprops[1].biasGradients[1], 1e-4f);

        // Expected input gradients (aggregated from both kernel1 and group1):
        // Each position receives gradients from units that process it
        final float[] EXPECTED_INPUT_GRADS = {
            0.012724f,  // pos 0: kernel1(0.1*0.1) + group1_k1(0.4*sigmoid_grad[0])  
            0.023404f,  // pos 1: kernel1(0.2*0.1) + group1_k1(0.5*sigmoid_grad[0])
            0.032929f,  // pos 2: kernel1(0.3*0.1) + group1_k2(0.6*sigmoid_grad[0])
            0.020937f,  // pos 3: kernel1(0.1*0.2) + group1_k1(0.4*sigmoid_grad[1])
            0.041171f,  // pos 4: kernel1(0.2*0.2) + group1_k1(0.5*sigmoid_grad[1])
            0.062292f,  // pos 5: kernel1(0.3*0.2) + group1_k2(0.6*sigmoid_grad[1])
            0.030111f,  // pos 6: kernel1(0.1*0.3) + group1_k1(0.4*sigmoid_grad[2])
            0.060138f,  // pos 7: kernel1(0.2*0.3) + group1_k1(0.5*sigmoid_grad[2])
            0.090671f   // pos 8: kernel1(0.3*0.3) + group1_k2(0.6*sigmoid_grad[2])
        };

        // Verify aggregated input gradients
        assertArrayEquals("Input gradients", EXPECTED_INPUT_GRADS, backpropOutput.inputGradients, 1e-4f);
    }
    
    @Test
    public void testConvolutionLayerBackwardWithUnits() {
        // Test with kernels and groups having unitCount=2
        Kernel kernel1 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        kernel1.setBias(0.0f);

        Kernel g1k1 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        g1k1.setWeights(new float[]{0.5f, 0.6f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        g1k2.setWeights(new float[]{0.7f, 0.8f});
        g1k2.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2); // strideSize = 1+1 = 2, but unitCount=2 → width=4

        ConvolutionLayer layer = new ConvolutionLayer(kernel1, group1);

        // Input: 3 units of size 4 = 12 values, outputSize = 3-2+1 = 2 per unit
        float[] input = {
            1.0f, 2.0f, 3.0f, 4.0f,    // unit 0
            5.0f, 6.0f, 7.0f, 8.0f,    // unit 1
            9.0f, 10.0f, 11.0f, 12.0f  // unit 2
        };

        ConvolutionLayerForwardOutput forwardOutput = layer.feedForward(input, 1);

        // Delta loss from next layer
        float[][] unitDeltaLoss = new float[2][];
        unitDeltaLoss[0] = new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f}; // kernel1 outputs (5 values)
        unitDeltaLoss[1] = new float[]{0.15f, 0.25f, 0.35f, 0.45f, 0.55f}; // group1 outputs (5 values)

        ConvolutionLayerBackpropagateOutput backpropOutput = layer.backpropagate(forwardOutput, unitDeltaLoss);

        // Expected structure verification
        assertEquals("Should have 2 unit backprops", 2, backpropOutput.unitBackprops.length);
        assertEquals("Input gradients size", input.length, backpropOutput.inputGradients.length);

        // Expected input gradients (aggregated from both units)
        // Values computed through backward pass with unitCount=2
        final float[] EXPECTED_INPUT_GRADS = {
            0.0161f, 0.0209f, 0.0587f, 0.0812f, 0.0918f, 0.1401f, 0.1303f, 0.2000f, 
            0.1700f, 0.2600f, 0.1500f, 0.2000f
        };

        assertArrayEquals("Input gradients with units", EXPECTED_INPUT_GRADS, 
            backpropOutput.inputGradients, 1e-4f);

        // Kernel1 weight gradients
        final float[] EXPECTED_KERNEL1_WEIGHT_GRADS = {9.5f, 11.0f, 12.5f, 14.0f};
        assertArrayEquals("Kernel1 weight gradients with units", EXPECTED_KERNEL1_WEIGHT_GRADS, 
            backpropOutput.unitBackprops[0].weightGradients[0], 1e-4f);

        // Group gradients through sigmoid and AND operations with unitCount=2
        final float[] EXPECTED_G1K1_WEIGHT_GRADS = {0.023098f, 0.054126f};
        final float[] EXPECTED_G1K2_WEIGHT_GRADS = {0.003238f, 0.006190f};

        assertArrayEquals("Group1 k1 weight gradients with units", EXPECTED_G1K1_WEIGHT_GRADS, 
            backpropOutput.unitBackprops[1].weightGradients[0], 1e-4f);
        assertArrayEquals("Group1 k2 weight gradients with units", EXPECTED_G1K2_WEIGHT_GRADS, 
            backpropOutput.unitBackprops[1].weightGradients[1], 1e-4f);
    }
    
    @Test
    public void testConvolutionLayerBackwardMultiBatch() {
        int batchSize = 4;

        // Create layer with 2 individual kernels
        Kernel kernel1 = new Kernel(3, 1);
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 1);
        kernel2.setWeights(new float[]{0.4f, 0.5f, 0.6f});
        kernel2.setBias(0.0f);

        ConvolutionLayer layer = new ConvolutionLayer(kernel1, kernel2);

        // Input: 4 batches, 3 chunks each
        float[] input = {
            // Batch 0
            1.0f, 2.0f, 3.0f,  4.0f, 5.0f, 6.0f,  7.0f, 8.0f, 9.0f,
            // Batch 1
            2.0f, 3.0f, 4.0f,  5.0f, 6.0f, 7.0f,  8.0f, 9.0f, 10.0f,
            // Batch 2
            0.5f, 1.0f, 1.5f,  2.0f, 2.5f, 3.0f,  3.5f, 4.0f, 4.5f,
            // Batch 3
            1.5f, 2.5f, 3.5f,  4.5f, 5.5f, 6.5f,  7.5f, 8.5f, 9.5f
        };

        ConvolutionLayerForwardOutput forwardOutput = layer.feedForward(input, batchSize);

        // Delta loss: 4 batches × 3 outputs per unit × 2 units = 24 total values
        float[][] unitDeltaLoss = new float[2][];
        unitDeltaLoss[0] = new float[]{
            0.1f, 0.2f, 0.3f,    // batch 0
            0.15f, 0.25f, 0.35f, // batch 1
            0.05f, 0.1f, 0.15f,  // batch 2
            0.2f, 0.3f, 0.4f     // batch 3
        };
        unitDeltaLoss[1] = new float[]{
            0.08f, 0.12f, 0.16f,   // batch 0
            0.1f, 0.15f, 0.2f,     // batch 1
            0.06f, 0.09f, 0.12f,   // batch 2
            0.14f, 0.18f, 0.22f    // batch 3
        };

        ConvolutionLayerBackpropagateOutput backpropOutput = layer.backpropagate(forwardOutput, unitDeltaLoss);

        // Expected structure verification
        assertEquals("Should have 2 unit backprops", 2, backpropOutput.unitBackprops.length);
        assertEquals("Input gradients size", input.length, backpropOutput.inputGradients.length);

        // Expected kernel1 weight gradients (recalculated and averaged over batches):
        // w[0]: (1×0.1 + 4×0.2 + 7×0.3 + 2×0.15 + 5×0.25 + 8×0.35 + 0.5×0.05 + 2×0.1 + 3.5×0.15 + 1.5×0.2 + 4.5×0.3 + 7.5×0.4)/4 = 3.1875f
        // w[1]: (2×0.1 + 5×0.2 + 8×0.3 + 3×0.15 + 6×0.25 + 9×0.35 + 1×0.05 + 2.5×0.1 + 4×0.15 + 2.5×0.2 + 5.5×0.3 + 8.5×0.4)/4 = 3.7875f
        // w[2]: (3×0.1 + 6×0.2 + 9×0.3 + 4×0.15 + 7×0.25 + 10×0.35 + 1.5×0.05 + 3×0.1 + 4.5×0.15 + 3.5×0.2 + 6.5×0.3 + 9.5×0.4)/4 = 4.3875f
        final float[] EXPECTED_KERNEL1_WEIGHT_GRADS = {3.1875f, 3.7875f, 4.3875f};
        assertArrayEquals("Kernel1 weight gradients multi-batch", EXPECTED_KERNEL1_WEIGHT_GRADS, 
            backpropOutput.unitBackprops[0].weightGradients[0], 1e-4f);

        // Expected kernel2 weight gradients (recalculated and averaged over batches):
        // w[0]: (1×0.08 + 4×0.12 + 7×0.16 + 2×0.1 + 5×0.15 + 8×0.2 + 0.5×0.06 + 2×0.09 + 3.5×0.12 + 1.5×0.14 + 4.5×0.18 + 7.5×0.22)/4 = 1.8875f
        // w[1]: (2×0.08 + 5×0.12 + 8×0.16 + 3×0.1 + 6×0.15 + 9×0.2 + 1×0.06 + 2.5×0.09 + 4×0.12 + 2.5×0.14 + 5.5×0.18 + 8.5×0.22)/4 = 2.26625f
        // w[2]: (3×0.08 + 6×0.12 + 9×0.16 + 4×0.1 + 7×0.15 + 10×0.2 + 1.5×0.06 + 3×0.09 + 4.5×0.12 + 3.5×0.14 + 6.5×0.18 + 9.5×0.22)/4 = 2.62f
        final float[] EXPECTED_KERNEL2_WEIGHT_GRADS = {1.8825f, 2.25375f, 2.625f};
        assertArrayEquals("Kernel2 weight gradients multi-batch", EXPECTED_KERNEL2_WEIGHT_GRADS, 
            backpropOutput.unitBackprops[1].weightGradients[0], 1e-4f);

        // Expected bias gradients (averaged over batches):
        final float EXPECTED_KERNEL1_BIAS_GRAD = 0.6375f; // (0.1+0.2+0.3+0.15+0.25+0.35+0.05+0.1+0.15+0.2+0.3+0.4)/4
        final float EXPECTED_KERNEL2_BIAS_GRAD = 0.405f;  // (0.08+0.12+0.16+0.1+0.15+0.2+0.06+0.09+0.12+0.14+0.18+0.22)/4

        assertEquals("Kernel1 bias gradient multi-batch", EXPECTED_KERNEL1_BIAS_GRAD, 
            backpropOutput.unitBackprops[0].biasGradients[0], 1e-4f);
        assertEquals("Kernel2 bias gradient multi-batch", EXPECTED_KERNEL2_BIAS_GRAD, 
            backpropOutput.unitBackprops[1].biasGradients[0], 1e-4f);

        // Input gradients verification (aggregated from both kernels across all batches)
        // Each batch contributes to input gradients, accumulated per position
        assertTrue("Input gradients should be non-zero", backpropOutput.inputGradients[0] != 0.0f);
        assertTrue("Input gradients should be accumulated", 
            Math.abs(backpropOutput.inputGradients[9]) > Math.abs(backpropOutput.inputGradients[0]));
    }

    @Test
    public void testConvolutionLayerBackwardWithUnitsMultiBatch() {
        int batchSize = 4;

        // Create layer with compatible stride sizes for units
        Kernel kernel1 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4, strideSize=2
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        kernel1.setBias(0.0f);

        Kernel g1k1 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        g1k1.setWeights(new float[]{0.5f, 0.6f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        g1k2.setWeights(new float[]{0.7f, 0.8f});
        g1k2.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2); // strideSize = 1+1 = 2

        ConvolutionLayer layer = new ConvolutionLayer(kernel1, group1);

        // Input: 4 batches, 3 units of size 2 each = 6 values per batch
        float[] input = {
            // Batch 0
            1.0f, 2.0f,  3.0f, 4.0f,  5.0f, 6.0f,
            // Batch 1
            2.0f, 3.0f,  4.0f, 5.0f,  6.0f, 7.0f,
            // Batch 2
            0.5f, 1.0f,  1.5f, 2.0f,  2.5f, 3.0f,
            // Batch 3
            1.5f, 2.5f,  3.5f, 4.5f,  5.5f, 6.5f
        };

        ConvolutionLayerForwardOutput forwardOutput = layer.feedForward(input, batchSize);

        // Delta loss: 4 batches × 2 outputs per unit × 2 units = 16 total values
        float[][] unitDeltaLoss = new float[2][];
        unitDeltaLoss[0] = new float[]{
            0.1f, 0.2f,  // batch 0
            0.15f, 0.25f, // batch 1
            0.05f, 0.1f,  // batch 2
            0.2f, 0.3f    // batch 3
        };
        unitDeltaLoss[1] = new float[]{
            0.08f, 0.12f,  // batch 0
            0.1f, 0.15f,   // batch 1
            0.06f, 0.09f,  // batch 2
            0.14f, 0.18f   // batch 3
        };

        ConvolutionLayerBackpropagateOutput backpropOutput = layer.backpropagate(forwardOutput, unitDeltaLoss);

        // Expected structure verification
        assertEquals("Should have 2 unit backprops", 2, backpropOutput.unitBackprops.length);
        assertEquals("Input gradients size", input.length, backpropOutput.inputGradients.length);

        // Expected kernel1 weight gradients (recalculated and averaged over batches):
        // w[0]: (1×0.1 + 3×0.2 + 2×0.15 + 4×0.25 + 0.5×0.05 + 1.5×0.1 + 1.5×0.2 + 3.5×0.3)/4 = 0.88125f
        // w[1]: (2×0.1 + 4×0.2 + 3×0.15 + 5×0.25 + 1×0.05 + 2×0.1 + 2.5×0.2 + 4.5×0.3)/4 = 1.2f  
        // w[2]: (3×0.1 + 5×0.2 + 4×0.15 + 6×0.25 + 1.5×0.05 + 2.5×0.1 + 3.5×0.2 + 5.5×0.3)/4 = 1.51875f
        // w[3]: (4×0.1 + 6×0.2 + 5×0.15 + 7×0.25 + 2×0.05 + 3×0.1 + 4.5×0.2 + 6.5×0.3)/4 = 1.8375f
        final float[] EXPECTED_KERNEL1_WEIGHT_GRADS = {0.88125f, 1.2f, 1.51875f, 1.8375f};
        assertArrayEquals("Kernel1 weight gradients with units multi-batch", EXPECTED_KERNEL1_WEIGHT_GRADS, 
            backpropOutput.unitBackprops[0].weightGradients[0], 1e-4f);

        // Bias gradients (averaged over batches)
        final float EXPECTED_KERNEL1_BIAS_GRAD = 0.3375f; // (0.1+0.2+0.15+0.25+0.05+0.1+0.2+0.3)/4
        assertEquals("Kernel1 bias gradient with units multi-batch", EXPECTED_KERNEL1_BIAS_GRAD, 
            backpropOutput.unitBackprops[0].biasGradients[0], 1e-4f);

        // Group gradients verification (structure check since sigmoid calculations are complex)
        assertTrue("Group k1 should have weight gradients", 
            backpropOutput.unitBackprops[1].weightGradients[0].length == 2);
        assertTrue("Group k2 should have weight gradients", 
            backpropOutput.unitBackprops[1].weightGradients[1].length == 2);
        assertTrue("Group k1 weight gradients should be computed", 
            backpropOutput.unitBackprops[1].weightGradients[0][0] != 0.0f);
        assertTrue("Group k2 weight gradients should be computed", 
            backpropOutput.unitBackprops[1].weightGradients[1][0] != 0.0f);
        assertTrue("Group k1 bias gradient should be computed", 
            backpropOutput.unitBackprops[1].biasGradients[0] != 0.0f);
        assertTrue("Group k2 bias gradient should be computed", 
            backpropOutput.unitBackprops[1].biasGradients[1] != 0.0f);

        // Input gradients verification
        assertTrue("Input gradients should be computed", backpropOutput.inputGradients[0] != 0.0f);
    }

    // ==================== GPU BACKWARD PASS TESTS ====================

    @Test
    public void testConvolutionLayerBackwardGPU() {
        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        // Create mixed layer (kernel + group)
        Kernel kernel1 = new Kernel(3, 1);
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        kernel1.setBias(0.0f);

        Kernel g1k1 = new Kernel(2, 1);
        g1k1.setWeights(new float[]{0.4f, 0.5f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(1, 1);
        g1k2.setWeights(new float[]{0.6f});
        g1k2.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2);

        ConvolutionLayer layer = new ConvolutionLayer(kernel1, group1);

        float[] input = {
            1.0f, 2.0f, 3.0f,  4.0f, 5.0f, 6.0f,  7.0f, 8.0f, 9.0f
        };

        float[][] unitDeltaLoss = new float[2][];
        unitDeltaLoss[0] = new float[]{0.1f, 0.2f, 0.3f};
        unitDeltaLoss[1] = new float[]{0.05f, 0.15f, 0.25f};

        // CPU reference
        ConvolutionLayerForwardOutput cpuForward = layer.feedForward(input, 1);
        ConvolutionLayerBackpropagateOutput cpuBackprop = layer.backpropagate(cpuForward, unitDeltaLoss);

        // GPU computation
        layer.prepareGPU(stream);
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuForward = layer.feedForwardGPU(inputGPU, 1, stream, handle);

        CUdeviceptr[] unitDeltaLossGPU = {
            CudaUtil.toGPUAsync(unitDeltaLoss[0], stream),
            CudaUtil.toGPUAsync(unitDeltaLoss[1], stream)
        };

        ConvolutionLayerBackpropagateOutputGPU gpuBackprop = layer.backpropagateGPU(gpuForward, unitDeltaLossGPU, stream, handle);

        // Get GPU results
        float[] gpuInputGrads = CudaUtil.fromGPUFloatAsync(gpuBackprop.inputGradients, input.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        // Compare input gradients
        assertArrayEquals("GPU input gradients should match CPU", 
            cpuBackprop.inputGradients, gpuInputGrads, 1e-5f);

        // Compare unit weight gradients
        for (int unitIdx = 0; unitIdx < 2; unitIdx++) {
            int numKernels = (unitIdx == 0) ? 1 : 2; // kernel1 has 1 kernel, group1 has 2 kernels
            for (int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++) {
                float[] gpuWeightGrads = CudaUtil.fromGPUFloatAsync(
                    gpuBackprop.unitBackprops[unitIdx].weightGradients[kernelIdx], 
                    cpuBackprop.unitBackprops[unitIdx].weightGradients[kernelIdx].length, 
                    stream);
                JCudaDriver.cuStreamSynchronize(stream);

                assertArrayEquals("GPU unit " + unitIdx + " kernel " + kernelIdx + " weight gradients should match CPU", 
                    cpuBackprop.unitBackprops[unitIdx].weightGradients[kernelIdx], gpuWeightGrads, 1e-5f);
            }

            // Compare bias gradients
            for (int kernelIdx = 0; kernelIdx < numKernels; kernelIdx++) {
                float gpuBiasGrad = CudaUtil.fromGPUFloatAsync(
                    gpuBackprop.unitBackprops[unitIdx].biasGradients[kernelIdx], 1, stream)[0];
                JCudaDriver.cuStreamSynchronize(stream);

                assertEquals("GPU unit " + unitIdx + " kernel " + kernelIdx + " bias gradient should match CPU", 
                    cpuBackprop.unitBackprops[unitIdx].biasGradients[kernelIdx], gpuBiasGrad, 1e-5f);
            }
        }

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeAsync(unitDeltaLossGPU[0], stream);
        CudaUtil.freeAsync(unitDeltaLossGPU[1], stream);
        gpuForward.freeAsync(stream);
        gpuBackprop.freeAsync(stream);
        layer.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testConvolutionLayerBackwardWithUnitsGPU() {
        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);

        // Create layer with units (unitCount=2)
        Kernel kernel1 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        kernel1.setBias(0.0f);

        Kernel g1k1 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        g1k1.setWeights(new float[]{0.5f, 0.6f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        g1k2.setWeights(new float[]{0.7f, 0.8f});
        g1k2.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2); // strideSize = 2

        ConvolutionLayer layer = new ConvolutionLayer(kernel1, group1);

        float[] input = {
            1.0f, 2.0f, 3.0f, 4.0f,    // unit 0
            5.0f, 6.0f, 7.0f, 8.0f,    // unit 1
            9.0f, 10.0f, 11.0f, 12.0f  // unit 2
        };

        float[][] unitDeltaLoss = new float[2][];
        unitDeltaLoss[0] = new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f}; // kernel1 outputs (5 values)
        unitDeltaLoss[1] = new float[]{0.15f, 0.25f, 0.35f, 0.45f, 0.55f}; // group1 outputs (5 values)

        // CPU reference
        ConvolutionLayerForwardOutput cpuForward = layer.feedForward(input, 1);
        ConvolutionLayerBackpropagateOutput cpuBackprop = layer.backpropagate(cpuForward, unitDeltaLoss);

        // GPU computation
        layer.prepareGPU(stream);
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuForward = layer.feedForwardGPU(inputGPU, 1, stream, handle);

        CUdeviceptr[] unitDeltaLossGPU = {
            CudaUtil.toGPUAsync(unitDeltaLoss[0], stream),
            CudaUtil.toGPUAsync(unitDeltaLoss[1], stream)
        };

        ConvolutionLayerBackpropagateOutputGPU gpuBackprop = layer.backpropagateGPU(gpuForward, unitDeltaLossGPU, stream, handle);

        // Compare results
        float[] gpuInputGrads = CudaUtil.fromGPUFloatAsync(gpuBackprop.inputGradients, input.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        assertArrayEquals("GPU input gradients with units should match CPU", 
            cpuBackprop.inputGradients, gpuInputGrads, 1e-5f);

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeAsync(unitDeltaLossGPU[0], stream);
        CudaUtil.freeAsync(unitDeltaLossGPU[1], stream);
        gpuForward.freeAsync(stream);
        gpuBackprop.freeAsync(stream);
        layer.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testConvolutionLayerBackwardMultiBatchGPU() {
        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);
        int batchSize = 4;

        // Create layer with 2 kernels
        Kernel kernel1 = new Kernel(3, 1);
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(3, 1);
        kernel2.setWeights(new float[]{0.4f, 0.5f, 0.6f});
        kernel2.setBias(0.0f);

        ConvolutionLayer layer = new ConvolutionLayer(kernel1, kernel2);

        float[] input = {
            // Batch 0
            1.0f, 2.0f, 3.0f,  4.0f, 5.0f, 6.0f,  7.0f, 8.0f, 9.0f,
            // Batch 1
            2.0f, 3.0f, 4.0f,  5.0f, 6.0f, 7.0f,  8.0f, 9.0f, 10.0f,
            // Batch 2
            0.5f, 1.0f, 1.5f,  2.0f, 2.5f, 3.0f,  3.5f, 4.0f, 4.5f,
            // Batch 3
            1.5f, 2.5f, 3.5f,  4.5f, 5.5f, 6.5f,  7.5f, 8.5f, 9.5f
        };

        float[][] unitDeltaLoss = new float[2][];
        unitDeltaLoss[0] = new float[]{
            0.1f, 0.2f, 0.3f,    // batch 0
            0.15f, 0.25f, 0.35f, // batch 1
            0.05f, 0.1f, 0.15f,  // batch 2
            0.2f, 0.3f, 0.4f     // batch 3
        };
        unitDeltaLoss[1] = new float[]{
            0.08f, 0.12f, 0.16f,   // batch 0
            0.1f, 0.15f, 0.2f,     // batch 1
            0.06f, 0.09f, 0.12f,   // batch 2
            0.14f, 0.18f, 0.22f    // batch 3
        };

        // CPU reference
        ConvolutionLayerForwardOutput cpuForward = layer.feedForward(input, batchSize);
        ConvolutionLayerBackpropagateOutput cpuBackprop = layer.backpropagate(cpuForward, unitDeltaLoss);

        // GPU computation
        layer.prepareGPU(stream);
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuForward = layer.feedForwardGPU(inputGPU, batchSize, stream, handle);

        CUdeviceptr[] unitDeltaLossGPU = {
            CudaUtil.toGPUAsync(unitDeltaLoss[0], stream),
            CudaUtil.toGPUAsync(unitDeltaLoss[1], stream)
        };

        ConvolutionLayerBackpropagateOutputGPU gpuBackprop = layer.backpropagateGPU(gpuForward, unitDeltaLossGPU, stream, handle);

        // Compare results
        float[] gpuInputGrads = CudaUtil.fromGPUFloatAsync(gpuBackprop.inputGradients, input.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        assertArrayEquals("GPU input gradients multi-batch should match CPU", 
            cpuBackprop.inputGradients, gpuInputGrads, 1e-5f);

        // Compare weight and bias gradients for each unit
        for (int unitIdx = 0; unitIdx < 2; unitIdx++) {
            float[] gpuWeightGrads = CudaUtil.fromGPUFloatAsync(
                gpuBackprop.unitBackprops[unitIdx].weightGradients[0], 
                cpuBackprop.unitBackprops[unitIdx].weightGradients[0].length, 
                stream);
            float gpuBiasGrad = CudaUtil.fromGPUFloatAsync(
                gpuBackprop.unitBackprops[unitIdx].biasGradients[0], 1, stream)[0];
            JCudaDriver.cuStreamSynchronize(stream);

            assertArrayEquals("GPU unit " + unitIdx + " weight gradients multi-batch should match CPU", 
                cpuBackprop.unitBackprops[unitIdx].weightGradients[0], gpuWeightGrads, 1e-5f);
            assertEquals("GPU unit " + unitIdx + " bias gradient multi-batch should match CPU", 
                cpuBackprop.unitBackprops[unitIdx].biasGradients[0], gpuBiasGrad, 1e-5f);
        }

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeAsync(unitDeltaLossGPU[0], stream);
        CudaUtil.freeAsync(unitDeltaLossGPU[1], stream);
        gpuForward.freeAsync(stream);
        gpuBackprop.freeAsync(stream);
        layer.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testConvolutionLayerBackwardWithUnitsMultiBatchGPU() {
        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);
        int batchSize = 4;

        // Create layer with compatible stride sizes
        Kernel kernel1 = new Kernel(2, 2); // unitSize=2, unitCount=2, width=4, strideSize=2
        kernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        kernel1.setBias(0.0f);

        Kernel g1k1 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        g1k1.setWeights(new float[]{0.5f, 0.6f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(1, 2); // unitSize=1, unitCount=2, width=2
        g1k2.setWeights(new float[]{0.7f, 0.8f});
        g1k2.setBias(0.0f);

        KernelGroup group1 = new KernelGroup(g1k1, g1k2); // strideSize = 2

        ConvolutionLayer layer = new ConvolutionLayer(kernel1, group1);

        float[] input = {
            // Batch 0
            1.0f, 2.0f,  3.0f, 4.0f,  5.0f, 6.0f,
            // Batch 1
            2.0f, 3.0f,  4.0f, 5.0f,  6.0f, 7.0f,
            // Batch 2
            0.5f, 1.0f,  1.5f, 2.0f,  2.5f, 3.0f,
            // Batch 3
            1.5f, 2.5f,  3.5f, 4.5f,  5.5f, 6.5f
        };

        float[][] unitDeltaLoss = new float[2][];
        unitDeltaLoss[0] = new float[]{
            0.1f, 0.2f,  // batch 0
            0.15f, 0.25f, // batch 1
            0.05f, 0.1f,  // batch 2
            0.2f, 0.3f    // batch 3
        };
        unitDeltaLoss[1] = new float[]{
            0.08f, 0.12f,  // batch 0
            0.1f, 0.15f,   // batch 1
            0.06f, 0.09f,  // batch 2
            0.14f, 0.18f   // batch 3
        };

        // CPU reference
        ConvolutionLayerForwardOutput cpuForward = layer.feedForward(input, batchSize);
        ConvolutionLayerBackpropagateOutput cpuBackprop = layer.backpropagate(cpuForward, unitDeltaLoss);

        // GPU computation
        layer.prepareGPU(stream);
        CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
        ConvolutionLayerForwardOutputGPU gpuForward = layer.feedForwardGPU(inputGPU, batchSize, stream, handle);

        CUdeviceptr[] unitDeltaLossGPU = {
            CudaUtil.toGPUAsync(unitDeltaLoss[0], stream),
            CudaUtil.toGPUAsync(unitDeltaLoss[1], stream)
        };

        ConvolutionLayerBackpropagateOutputGPU gpuBackprop = layer.backpropagateGPU(gpuForward, unitDeltaLossGPU, stream, handle);

        // Compare results
        float[] gpuInputGrads = CudaUtil.fromGPUFloatAsync(gpuBackprop.inputGradients, input.length, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        assertArrayEquals("GPU input gradients with units multi-batch should match CPU", 
            cpuBackprop.inputGradients, gpuInputGrads, 1e-5f);

        // Cleanup
        CudaUtil.freeAsync(inputGPU, stream);
        CudaUtil.freeAsync(unitDeltaLossGPU[0], stream);
        CudaUtil.freeAsync(unitDeltaLossGPU[1], stream);
        gpuForward.freeAsync(stream);
        gpuBackprop.freeAsync(stream);
        layer.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    // ==================== WEIGHT UPDATE TESTS ====================

    @Test
    public void testConvolutionLayerWeightUpdatesGPUvsCPU() {
        // Create identical layers for CPU and GPU (no Adam)
        Kernel cpuKernel1 = new Kernel(3, 1);
        cpuKernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        cpuKernel1.setBias(0.5f);

        Kernel cpuG1K1 = new Kernel(2, 1);
        cpuG1K1.setWeights(new float[]{0.4f, 0.6f});
        cpuG1K1.setBias(0.1f);

        Kernel cpuG1K2 = new Kernel(1, 1);
        cpuG1K2.setWeights(new float[]{0.8f});
        cpuG1K2.setBias(0.2f);

        KernelGroup cpuGroup = new KernelGroup(cpuG1K1, cpuG1K2);
        ConvolutionLayer cpuLayer = new ConvolutionLayer(cpuKernel1, cpuGroup);

        // GPU versions with identical weights
        Kernel gpuKernel1 = new Kernel(3, 1);
        gpuKernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        gpuKernel1.setBias(0.5f);

        Kernel gpuG1K1 = new Kernel(2, 1);
        gpuG1K1.setWeights(new float[]{0.4f, 0.6f});
        gpuG1K1.setBias(0.1f);

        Kernel gpuG1K2 = new Kernel(1, 1);
        gpuG1K2.setWeights(new float[]{0.8f});
        gpuG1K2.setBias(0.2f);

        KernelGroup gpuGroup = new KernelGroup(gpuG1K1, gpuG1K2);
        ConvolutionLayer gpuLayer = new ConvolutionLayer(gpuKernel1, gpuGroup);

        // Training data - 4 batches
        float[] input = {
            // Batch 1
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            // Batch 2  
            1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f,
            // Batch 3
            0.8f, 1.8f, 2.8f, 3.8f, 4.8f, 5.8f, 6.8f, 7.8f, 8.8f,
            // Batch 4
            2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f
        };
        
        float[][] target = {
            {0.4f, 0.5f, 0.6f}, // kernel1 targets for all batches
            {0.3f, 0.4f, 0.5f}  // group targets for all batches
        };
        
        float learningRate = 0.01f;
        int iterations = 5;

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);
        gpuLayer.prepareGPU(stream);

        for (int iter = 0; iter < iterations; iter++) {
            // CPU training step
            ConvolutionLayerForwardOutput cpuForward = cpuLayer.feedForward(input, 4);
            float[][] cpuGradOutput = new float[2][];
            cpuGradOutput[0] = new float[12]; // 4 batches * 3 outputs
            cpuGradOutput[1] = new float[12]; // 4 batches * 3 outputs
            
            // Calculate gradients (simulating MSE derivative)
            for (int b = 0; b < 4; b++) {
                for (int o = 0; o < 3; o++) {
                    int idx = b * 3 + o;
                    cpuGradOutput[0][idx] = cpuForward.unitOutputs[0].output[idx] - target[0][o];
                    cpuGradOutput[1][idx] = cpuForward.unitOutputs[1].output[idx] - target[1][o];
                }
            }
            
            ConvolutionLayerBackpropagateOutput cpuBackprop = cpuLayer.backpropagate(cpuForward, cpuGradOutput);
            cpuLayer.updateWeights(cpuBackprop, learningRate);

            // GPU training step
            CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
            ConvolutionLayerForwardOutputGPU gpuForward = gpuLayer.feedForwardGPU(inputGPU, 4, stream, handle);
            
            CUdeviceptr[] gpuGradOutputGPU = {
                CudaUtil.toGPUAsync(cpuGradOutput[0], stream),
                CudaUtil.toGPUAsync(cpuGradOutput[1], stream)
            };
            
            ConvolutionLayerBackpropagateOutputGPU gpuBackprop = gpuLayer.backpropagateGPU(gpuForward, gpuGradOutputGPU, stream, handle);
            gpuLayer.updateWeightsGPU(gpuBackprop, learningRate, stream);
            
            JCudaDriver.cuStreamSynchronize(stream);

            // Compare weights each iteration
            // Kernel1 weights
            float[] gpuKernel1Weights = CudaUtil.fromGPUFloat(gpuKernel1.weightsGPU, 3);
            float gpuKernel1Bias = CudaUtil.fromGPUFloat(gpuKernel1.biasGPU, 1)[0];
            
            assertArrayEquals("SGD Kernel1 weights should match at iteration " + iter, 
                cpuKernel1.getWeights(), gpuKernel1Weights, 1e-5f);
            assertEquals("SGD Kernel1 bias should match at iteration " + iter, 
                cpuKernel1.getBias(), gpuKernel1Bias, 1e-5f);

            // Group kernel weights
            float[] gpuG1K1Weights = CudaUtil.fromGPUFloat(gpuG1K1.weightsGPU, 2);
            float gpuG1K1Bias = CudaUtil.fromGPUFloat(gpuG1K1.biasGPU, 1)[0];
            float[] gpuG1K2Weights = CudaUtil.fromGPUFloat(gpuG1K2.weightsGPU, 1);
            float gpuG1K2Bias = CudaUtil.fromGPUFloat(gpuG1K2.biasGPU, 1)[0];

            assertArrayEquals("SGD Group K1 weights should match at iteration " + iter, 
                cpuG1K1.getWeights(), gpuG1K1Weights, 1e-5f);
            assertEquals("SGD Group K1 bias should match at iteration " + iter, 
                cpuG1K1.getBias(), gpuG1K1Bias, 1e-5f);
            assertArrayEquals("SGD Group K2 weights should match at iteration " + iter, 
                cpuG1K2.getWeights(), gpuG1K2Weights, 1e-5f);
            assertEquals("SGD Group K2 bias should match at iteration " + iter, 
                cpuG1K2.getBias(), gpuG1K2Bias, 1e-5f);

            // Cleanup iteration resources
            CudaUtil.freeAsync(inputGPU, stream);
            CudaUtil.freeAsync(gpuGradOutputGPU[0], stream);
            CudaUtil.freeAsync(gpuGradOutputGPU[1], stream);
            gpuForward.freeAsync(stream);
            gpuBackprop.freeAsync(stream);
        }

        // Cleanup
        gpuLayer.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    @Test
    public void testConvolutionLayerWeightUpdatesAdamGPUvsCPU() {
        // Create identical layers for CPU and GPU with Adam
        Kernel cpuKernel1 = new Kernel(3, 1);
        cpuKernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        cpuKernel1.setBias(0.5f);
        cpuKernel1.setUseAdam(true);

        Kernel cpuG1K1 = new Kernel(2, 1);
        cpuG1K1.setWeights(new float[]{0.4f, 0.6f});
        cpuG1K1.setBias(0.1f);
        cpuG1K1.setUseAdam(true);

        Kernel cpuG1K2 = new Kernel(1, 1);
        cpuG1K2.setWeights(new float[]{0.8f});
        cpuG1K2.setBias(0.2f);
        cpuG1K2.setUseAdam(true);

        KernelGroup cpuGroup = new KernelGroup(cpuG1K1, cpuG1K2);
        ConvolutionLayer cpuLayer = new ConvolutionLayer(cpuKernel1, cpuGroup);

        // GPU versions with identical weights and Adam
        Kernel gpuKernel1 = new Kernel(3, 1);
        gpuKernel1.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        gpuKernel1.setBias(0.5f);
        gpuKernel1.setUseAdam(true);

        Kernel gpuG1K1 = new Kernel(2, 1);
        gpuG1K1.setWeights(new float[]{0.4f, 0.6f});
        gpuG1K1.setBias(0.1f);
        gpuG1K1.setUseAdam(true);

        Kernel gpuG1K2 = new Kernel(1, 1);
        gpuG1K2.setWeights(new float[]{0.8f});
        gpuG1K2.setBias(0.2f);
        gpuG1K2.setUseAdam(true);

        KernelGroup gpuGroup = new KernelGroup(gpuG1K1, gpuG1K2);
        ConvolutionLayer gpuLayer = new ConvolutionLayer(gpuKernel1, gpuGroup);

        // Training data - 4 batches
        float[] input = {
            // Batch 1
            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
            // Batch 2  
            1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f, 9.5f,
            // Batch 3
            0.8f, 1.8f, 2.8f, 3.8f, 4.8f, 5.8f, 6.8f, 7.8f, 8.8f,
            // Batch 4
            2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f
        };
        
        float[][] target = {
            {0.4f, 0.5f, 0.6f}, // kernel1 targets for all batches
            {0.3f, 0.4f, 0.5f}  // group targets for all batches
        };
        
        float learningRate = 0.01f;
        int iterations = 5;

        CUstream stream = CudaUtil.createStream();
        cublasHandle handle = CudaEngine.getCublasHandle(0);
        gpuLayer.prepareGPU(stream);

        for (int iter = 0; iter < iterations; iter++) {
            // CPU training step
            ConvolutionLayerForwardOutput cpuForward = cpuLayer.feedForward(input, 4);
            float[][] cpuGradOutput = new float[2][];
            cpuGradOutput[0] = new float[12]; // 4 batches * 3 outputs
            cpuGradOutput[1] = new float[12]; // 4 batches * 3 outputs
            
            // Calculate gradients (simulating MSE derivative)
            for (int b = 0; b < 4; b++) {
                for (int o = 0; o < 3; o++) {
                    int idx = b * 3 + o;
                    cpuGradOutput[0][idx] = cpuForward.unitOutputs[0].output[idx] - target[0][o];
                    cpuGradOutput[1][idx] = cpuForward.unitOutputs[1].output[idx] - target[1][o];
                }
            }
            
            ConvolutionLayerBackpropagateOutput cpuBackprop = cpuLayer.backpropagate(cpuForward, cpuGradOutput);
            cpuLayer.updateWeights(cpuBackprop, learningRate);

            // GPU training step
            CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);
            ConvolutionLayerForwardOutputGPU gpuForward = gpuLayer.feedForwardGPU(inputGPU, 4, stream, handle);
            
            CUdeviceptr[] gpuGradOutputGPU = {
                CudaUtil.toGPUAsync(cpuGradOutput[0], stream),
                CudaUtil.toGPUAsync(cpuGradOutput[1], stream)
            };
            
            ConvolutionLayerBackpropagateOutputGPU gpuBackprop = gpuLayer.backpropagateGPU(gpuForward, gpuGradOutputGPU, stream, handle);
            gpuLayer.updateWeightsGPU(gpuBackprop, learningRate, stream);
            
            JCudaDriver.cuStreamSynchronize(stream);

            // Compare weights and Adam states each iteration
            // Kernel1 
            float[] gpuKernel1Weights = CudaUtil.fromGPUFloat(gpuKernel1.weightsGPU, 3);
            float gpuKernel1Bias = CudaUtil.fromGPUFloat(gpuKernel1.biasGPU, 1)[0];
            float[] gpuKernel1WeightMomentum = CudaUtil.fromGPUFloat(gpuKernel1.weightMomentumGPU, 3);
            float[] gpuKernel1WeightVelocity = CudaUtil.fromGPUFloat(gpuKernel1.weightVelocityGPU, 3);
            float gpuKernel1BiasMomentum = CudaUtil.fromGPUFloat(gpuKernel1.biasMomentumGPU, 1)[0];
            float gpuKernel1BiasVelocity = CudaUtil.fromGPUFloat(gpuKernel1.biasVelocityGPU, 1)[0];
            
            assertArrayEquals("Adam Kernel1 weights should match at iteration " + iter, 
                cpuKernel1.getWeights(), gpuKernel1Weights, 1e-5f);
            assertEquals("Adam Kernel1 bias should match at iteration " + iter, 
                cpuKernel1.getBias(), gpuKernel1Bias, 1e-5f);
            assertArrayEquals("Adam Kernel1 weight momentum should match at iteration " + iter,
                cpuKernel1.weightMomentum, gpuKernel1WeightMomentum, 1e-5f);
            assertArrayEquals("Adam Kernel1 weight velocity should match at iteration " + iter,
                cpuKernel1.weightVelocity, gpuKernel1WeightVelocity, 1e-5f);
            assertEquals("Adam Kernel1 bias momentum should match at iteration " + iter,
                cpuKernel1.biasMomentum, gpuKernel1BiasMomentum, 1e-5f);
            assertEquals("Adam Kernel1 bias velocity should match at iteration " + iter,
                cpuKernel1.biasVelocity, gpuKernel1BiasVelocity, 1e-5f);

            // Group kernels Adam states
            float[] gpuG1K1Weights = CudaUtil.fromGPUFloat(gpuG1K1.weightsGPU, 2);
            float[] gpuG1K1WeightMomentum = CudaUtil.fromGPUFloat(gpuG1K1.weightMomentumGPU, 2);
            float[] gpuG1K1WeightVelocity = CudaUtil.fromGPUFloat(gpuG1K1.weightVelocityGPU, 2);

            assertArrayEquals("Adam Group K1 weights should match at iteration " + iter, 
                cpuG1K1.getWeights(), gpuG1K1Weights, 1e-5f);
            assertArrayEquals("Adam Group K1 weight momentum should match at iteration " + iter,
                cpuG1K1.weightMomentum, gpuG1K1WeightMomentum, 1e-5f);
            assertArrayEquals("Adam Group K1 weight velocity should match at iteration " + iter,
                cpuG1K1.weightVelocity, gpuG1K1WeightVelocity, 1e-5f);

            // Cleanup iteration resources
            CudaUtil.freeAsync(inputGPU, stream);
            CudaUtil.freeAsync(gpuGradOutputGPU[0], stream);
            CudaUtil.freeAsync(gpuGradOutputGPU[1], stream);
            gpuForward.freeAsync(stream);
            gpuBackprop.freeAsync(stream);
        }

        // Cleanup
        gpuLayer.freeGPU(stream);
        CudaUtil.freeStream(stream);
    }

    // ==================== EDGE CASE & ERROR HANDLING TESTS ====================

    @Test(expected = IllegalArgumentException.class)
    public void testConvolutionLayerIncompatibleStrideSizes() {
        // Mix kernels and groups with incompatible stride sizes
        Kernel kernel1 = new Kernel(3, 1); // strideSize = 3
        
        Kernel g1k1 = new Kernel(2, 1); // 
        Kernel g1k2 = new Kernel(1, 1); // 
        KernelGroup group1 = new KernelGroup(g1k1, g1k2); // strideSize = 2+1 = 3 (compatible)
        
        Kernel g2k1 = new Kernel(1, 1); // 
        Kernel g2k2 = new Kernel(1, 1); // 
        KernelGroup group2 = new KernelGroup(g2k1, g2k2); // strideSize = 1+1 = 2 (incompatible)

        // This should throw exception due to mixed stride sizes (3 and 2)
        ConvolutionLayer layer = new ConvolutionLayer(kernel1, group1, group2);
        
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        layer.feedForward(input, 1); // Should throw exception
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConvolutionLayerInvalidInputSize() {
        Kernel kernel1 = new Kernel(3, 1);
        Kernel kernel2 = new Kernel(3, 1);
        
        ConvolutionLayer layer = new ConvolutionLayer(kernel1, kernel2);

        // strideSize = 3, but we provide 10 values (not divisible by 3)
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
        layer.feedForward(input, 1); // Should throw exception
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConvolutionLayerIncompatibleUnitSizes() {
        // Mix kernels with different unitSizes (which means different strideSizes)
        Kernel kernel1 = new Kernel(3, 1); // unitSize=3, strideSize=3
        Kernel kernel2 = new Kernel(4, 1); // unitSize=4, strideSize=4 (incompatible)

        ConvolutionLayer layer = new ConvolutionLayer(kernel1, kernel2);

        // This should throw exception due to incompatible stride sizes
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
        layer.feedForward(input, 1); // Should throw exception
    }
    
    @Test
    public void testConvolutionLayerWithLargeInput() {
        // Create layer with small kernels
        Kernel kernel1 = new Kernel(4, 1);
        kernel1.setWeights(new float[]{0.25f, 0.25f, 0.25f, 0.25f});
        kernel1.setBias(0.0f);

        Kernel kernel2 = new Kernel(4, 1);
        kernel2.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        kernel2.setBias(0.0f);

        ConvolutionLayer layer = new ConvolutionLayer(kernel1, kernel2);

        // Generate large input (10000 elements)
        int inputSize = 10000;
        float[] input = new float[inputSize];
        for (int i = 0; i < inputSize; i++) {
            input[i] = (i % 10) * 0.1f;  // Repeating pattern
        }

        // Perform forward pass
        ConvolutionLayerForwardOutput output = layer.feedForward(input, 1);

        // Validate structure
        assertEquals("Should have 2 units", 2, output.unitOutputs.length);

        // Expected output size = numChunks = 10000/4 = 2500 per unit
        assertEquals("Large input output size", 2500, output.unitOutputs[0].output.length);
        assertEquals("Large input output size", 2500, output.unitOutputs[1].output.length);

        // Validate samples - each chunk is processed independently
        // Chunk 0: input[0:4] = [0.0, 0.1, 0.2, 0.3]
        // kernel1: average = 0.25×0.6 = 0.15
        // kernel2: 0×0.1 + 0.1×0.2 + 0.2×0.3 + 0.3×0.4 = 0.20
        assertEquals("Large input sample 0", 0.15f, output.unitOutputs[0].output[0], 1e-5f);
        assertEquals("Large input sample 0", 0.20f, output.unitOutputs[1].output[0], 1e-5f);

        // Chunk 100: input[400:404] = [0.0, 0.1, 0.2, 0.3] (pattern repeats every 10)
        assertEquals("Large input sample 100", 0.15f, output.unitOutputs[0].output[100], 1e-5f);
        assertEquals("Large input sample 100", 0.20f, output.unitOutputs[1].output[100], 1e-5f);

        // Chunk 2499 (last chunk): input[9996:10000] = [0.6, 0.7, 0.8, 0.9]
        // kernel1: average = 0.25×3.0 = 0.75
        // kernel2: 0.6×0.1 + 0.7×0.2 + 0.8×0.3 + 0.9×0.4 = 0.80
        assertEquals("Large input sample 2499", 0.75f, output.unitOutputs[0].output[2499], 1e-5f);
        assertEquals("Large input sample 2499", 0.80f, output.unitOutputs[1].output[2499], 1e-5f);
    }

    @Test
    public void testConvolutionLayerEdgeCases() {
        // Test minimal valid configuration
        Kernel kernel = new Kernel(3, 1);
        kernel.setWeights(new float[]{0.1f, 0.2f, 0.3f});
        kernel.setBias(0.5f);
        
        ConvolutionLayer layer = new ConvolutionLayer(kernel);
        
        // Input exactly the size of one chunk
        float[] input = {1.0f, 2.0f, 3.0f};
        
        ConvolutionLayerForwardOutput output = layer.feedForward(input, 1);
        
        // Expected: just one output value per unit
        // 1.0*0.1 + 2.0*0.2 + 3.0*0.3 + 0.5 = 1.9
        float[] expected = {1.9f};
        
        assertEquals("Should have 1 unit", 1, output.unitOutputs.length);
        assertEquals("Output size should be 1", 1, output.unitOutputs[0].output.length);
        assertArrayEquals("Minimal input case failed", expected, output.unitOutputs[0].output, 1e-5f);
    }
    
    @Test
    public void testConvolutionLayerWithEvenSizedKernels() {
        // Create layer with even-width kernels having different stride sizes
        Kernel kernel1 = new Kernel(4, 1); // unitSize=4, strideSize=4
        kernel1.setWeights(new float[]{0.25f, 0.25f, 0.25f, 0.25f});
        kernel1.setBias(1.0f);

        Kernel kernel2 = new Kernel(2, 1); // unitSize=2, strideSize=2  
        kernel2.setWeights(new float[]{0.5f, 0.5f});
        kernel2.setBias(0.5f);

        // These have different stride sizes (4 vs 2), so constructor should throw immediately
        try {
            ConvolutionLayer layer = new ConvolutionLayer(kernel1, kernel2);
            fail("Should have thrown exception for incompatible stride sizes");
        } catch (IllegalArgumentException e) {
            // Expected - incompatible stride sizes
            assertTrue("Error message should mention stride", e.getMessage().contains("stride"));
        }

        // Test valid even-sized kernels with same stride
        Kernel kernel3 = new Kernel(4, 1); // unitSize=4, strideSize=4 - matches kernel1
        kernel3.setWeights(new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        kernel3.setBias(0.0f);

        ConvolutionLayer validLayer = new ConvolutionLayer(kernel1, kernel3);

        float[] input = {1, 2, 3, 4, 5, 6, 7, 8};
        ConvolutionLayerForwardOutput output = validLayer.feedForward(input, 1);

        // Expected output size = (8 / 4) - 1 + 1 = 2 per unit (2 chunks, unitCount=1)
        assertEquals("Should have 2 units", 2, output.unitOutputs.length);
        assertEquals("Even kernel output size", 2, output.unitOutputs[0].output.length);

        // kernel1: 
        // Chunk 0: (1+2+3+4)*0.25 + 1.0 = 2.5 + 1.0 = 3.5
        // Chunk 1: (5+6+7+8)*0.25 + 1.0 = 6.5 + 1.0 = 7.5
        assertEquals("Even kernel first output", 3.5f, output.unitOutputs[0].output[0], 1e-5f);
        assertEquals("Even kernel second output", 7.5f, output.unitOutputs[0].output[1], 1e-5f);

        // kernel3:
        // Chunk 0: 1*0.1+2*0.2+3*0.3+4*0.4 = 0.1+0.4+0.9+1.6 = 3.0
        // Chunk 1: 5*0.1+6*0.2+7*0.3+8*0.4 = 0.5+1.2+2.1+3.2 = 7.0
        assertEquals("Even kernel3 first output", 3.0f, output.unitOutputs[1].output[0], 1e-5f);
        assertEquals("Even kernel3 second output", 7.0f, output.unitOutputs[1].output[1], 1e-5f);
    }
    
    @Test
    public void testConvolutionLayerTrainingConvergence() {
        // Test actual learning convergence with realistic scenario
        Kernel kernel1 = new Kernel(3, 1);
        kernel1.setWeights(new float[]{0.1f, 0.1f, 0.1f});
        kernel1.setBias(0.0f);
        kernel1.setUseAdam(true);

        Kernel g1k1 = new Kernel(2, 1);
        g1k1.setWeights(new float[]{0.1f, 0.1f});
        g1k1.setBias(0.0f);

        Kernel g1k2 = new Kernel(1, 1);
        g1k2.setWeights(new float[]{0.1f});
        g1k2.setBias(0.0f);

        KernelGroup group = new KernelGroup(g1k1, g1k2);
        ConvolutionLayer layer = new ConvolutionLayer(kernel1, group);

        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        float[][] target = {
            {0.5f, 0.6f, 0.7f}, // kernel1 targets
            {0.4f, 0.5f, 0.6f}  // group targets
        };

        float learningRate = 0.01f;
        int iterations = 100;

        float initialLoss = Float.MAX_VALUE;
        float finalLoss = 0.0f;

        for (int iter = 0; iter < iterations; iter++) {
            ConvolutionLayerForwardOutput forwardOutput = layer.feedForward(input, 1);

            // Calculate loss and gradients
            float[][] gradOutput = new float[2][];
            gradOutput[0] = new float[3];
            gradOutput[1] = new float[3];
            
            float loss = 0.0f;
            for (int i = 0; i < 3; i++) {
                float diff1 = forwardOutput.unitOutputs[0].output[i] - target[0][i];
                float diff2 = forwardOutput.unitOutputs[1].output[i] - target[1][i];
                loss += diff1 * diff1 + diff2 * diff2;
                gradOutput[0][i] = 2 * diff1;
                gradOutput[1][i] = 2 * diff2;
            }
            
            if (iter == 0) initialLoss = loss;
            if (iter == iterations - 1) finalLoss = loss;

            ConvolutionLayerBackpropagateOutput backpropOutput = layer.backpropagate(forwardOutput, gradOutput);
            layer.updateWeights(backpropOutput, learningRate);
        }

        // Verify learning occurred
        assertTrue("Loss should decrease during training", finalLoss < initialLoss);
        assertTrue("Final loss should be reasonable", finalLoss < initialLoss * 0.8f);
        
        // Verify weights changed from initial values
        assertFalse("Kernel1 weights should change", 
            java.util.Arrays.equals(kernel1.getWeights(), new float[]{0.1f, 0.1f, 0.1f}));
        assertFalse("Group kernel weights should change", 
            java.util.Arrays.equals(g1k1.getWeights(), new float[]{0.1f, 0.1f}));
    }
}