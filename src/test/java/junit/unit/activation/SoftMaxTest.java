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
package junit.unit.activation;

import jcuda.driver.CUdeviceptr;
import org.fjnn.activation.Activation;
import org.fjnn.activation.SoftMax;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;
import org.fjnn.loss.CrossEntropy;
import org.junit.Test;

/**
 * Unit test for SoftMax activation function.
 * 
 * @author ahmed
 */
public class SoftMaxTest extends ActivationTest {
    
    @Override
    public void testCompute() {
        SoftMax softmax = new SoftMax();
        
        // Test batch (4 batches, 5 units each)
        float[] input = {
            1.0f, 2.0f, 3.0f, 4.0f, 1.0f,
            0.0f, -1.0f, 2.0f, 0.5f, -0.5f,
            5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
            -2.0f, 0.0f, 2.0f, 4.0f, 6.0f
        };
        
        float[] expected = {
            0.031063f, 0.084437f, 0.229525f, 0.623912f, 0.031063f,
            0.090808f, 0.033407f, 0.670989f, 0.149718f, 0.055078f,
            0.200000f, 0.200000f, 0.200000f, 0.200000f, 0.200000f,
            0.000290f, 0.002143f, 0.015838f, 0.117025f, 0.864704f
        };
        
        float[] output = input.clone();
        softmax.compute(output, 5, 4);
        assertArrayEquals(expected, output, EPSILON);
        
        // Verify each row sums to 1
        for (int i = 0; i < 4; i++) {
            float sum = 0;
            for (int j = 0; j < 5; j++) {
                sum += output[i * 5 + j];
            }
            assertEquals(1.0f, sum, EPSILON);
        }
    }
    
    @Override
    public void testDerivative() {
        // SoftMax derivative throws UnsupportedOperationException
        SoftMax softmax = new SoftMax();
        
        float[] preActivation = {1.0f, 2.0f, 3.0f};
        float[] postActivation = {0.09f, 0.24f, 0.67f};
        float[] derivative = new float[3];
        
        try {
            softmax.derivative(preActivation, postActivation, derivative, 3, 1);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage().contains("Use gradient"));
        }
    }
    
    @Override
    public void testGradient() {
        SoftMax softmax = new SoftMax();
        
        float[] preActivation = {
            1.0f, 2.0f, 3.0f, 4.0f, 1.0f,
            0.0f, -1.0f, 2.0f, 0.5f, -0.5f,
            5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
            -2.0f, 0.0f, 2.0f, 4.0f, 6.0f
        };
        
        // Compute softmax first
        float[] postActivation = preActivation.clone();
        softmax.compute(postActivation, 5, 4);
        
        // Use one-hot gradient (common in classification)
        float[] gradient = {
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // target class 2
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // target class 0
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // target class 4
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f   // target class 4
        };
        
        float[] expected = {
            -0.007130f, -0.019380f, 0.176843f, -0.143203f, -0.007130f,
            0.082562f, -0.003034f, -0.060931f, -0.013596f, -0.005002f,
            -0.040000f, -0.040000f, -0.040000f, -0.040000f, 0.160000f,
            -0.000251f, -0.001853f, -0.013695f, -0.101192f, 0.116991f
        };
        
        gradient = gradient.clone();
        softmax.gradient(preActivation, postActivation, gradient, 5, 4);
        assertArrayEquals(expected, gradient, EPSILON);
    }
    
    @Override
    public void testEdgeCases() {
        SoftMax softmax = new SoftMax();
        
        // Test numerical stability with large values
        float[] largeValues = {
            1000f, 1001f, 1002f, 999f,
            -1000f, -999f, -1001f, -1002f,
            100f, 0f, -100f, 50f,
            5.0f, 5.0f, 5.0f, 5.0f
        };
        float[] output = largeValues.clone();
        softmax.compute(output, 4, 4);
        
        // Should not produce NaN or Inf
        for (float v : output) {
            assertFalse(Float.isNaN(v));
            assertFalse(Float.isInfinite(v));
            assertTrue(v >= 0.0f && v <= 1.0f);
        }
        
        // Verify each row sums to 1
        for (int i = 0; i < 4; i++) {
            float sum = 0;
            for (int j = 0; j < 4; j++) {
                sum += output[i * 4 + j];
            }
            assertEquals(1.0f, sum, EPSILON);
        }
    }
    
    @Override
    public void testSerialization() {
        SoftMax softmax = new SoftMax();
        
        Map<String, Object> serialized = softmax.serialize();
        assertNotNull(serialized);
        assertEquals("SoftMax", serialized.get("type"));
        
        Activation deserialized = Activation.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof SoftMax);
        
        // Test functionality
        float[] input = {
            1.0f, 2.0f, 3.0f, 4.0f, 1.0f,
            0.0f, -1.0f, 2.0f, 0.5f, -0.5f
        };
        
        float[] cpuOutput = input.clone();
        softmax.compute(cpuOutput, 5, 2);
        
        float[] deserializedOutput = input.clone();
        deserialized.compute(deserializedOutput, 5, 2);
        
        assertArrayEquals(cpuOutput, deserializedOutput, EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        SoftMax softmax = new SoftMax();
        
        float[] input = {
            1.0f, 2.0f, 3.0f, 4.0f, 1.0f,
            0.0f, -1.0f, 2.0f, 0.5f, -0.5f,
            5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
            -2.0f, 0.0f, 2.0f, 4.0f, 6.0f
        };
        
        // CPU computation
        float[] cpuOutput = input.clone();
        softmax.compute(cpuOutput, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(input);
        softmax.computeGPU(gpuPtr, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, input.length);
        
        assertArrayEquals(cpuOutput, gpuOutput, RELAXED_EPSILON);
        CudaUtil.free(gpuPtr);
    }
    
    @Override
    public void testDerivativeGPU() {
        // SoftMax derivative throws UnsupportedOperationException
        SoftMax softmax = new SoftMax();
        
        float[] preActivation = {1.0f, 2.0f, 3.0f};
        float[] postActivation = {0.09f, 0.24f, 0.67f};
        float[] derivative = new float[3];
        
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(derivative);
        
        try {
            softmax.derivativeGPU(gpuPre, gpuPost, gpuDeriv, 3, 1, null);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage().contains("Use gradient"));
        } finally {
            CudaUtil.free(gpuPre);
            CudaUtil.free(gpuPost);
            CudaUtil.free(gpuDeriv);
        }
    }
    
    @Override
    public void testGradientGPU() {
        SoftMax softmax = new SoftMax();
        
        float[] preActivation = {
            1.0f, 2.0f, 3.0f, 4.0f, 1.0f,
            0.0f, -1.0f, 2.0f, 0.5f, -0.5f,
            5.0f, 5.0f, 5.0f, 5.0f, 5.0f,
            -2.0f, 0.0f, 2.0f, 4.0f, 6.0f
        };
        
        // CPU computation
        float[] postActivation = preActivation.clone();
        softmax.compute(postActivation, 5, 4);
        
        float[] gradient = {
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f
        };
        
        float[] cpuGradient = gradient.clone();
        softmax.gradient(preActivation, postActivation, cpuGradient, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuGrad = CudaUtil.toGPU(gradient);
        softmax.gradientGPU(gpuPre, gpuPost, gpuGrad, 5, 4, null);
        float[] gpuGradient = CudaUtil.fromGPUFloat(gpuGrad, gradient.length);
        
        assertArrayEquals(cpuGradient, gpuGradient, RELAXED_EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuGrad);
    }
    
    @Override
    public void testEdgeCasesGPU() {
        SoftMax softmax = new SoftMax();
        
        // Test numerical stability with extreme values
        float[] extremeValues = {
            -100f, 0f, 100f, 50f,
            1000f, 1001f, 999f, 1000f,
            -1000f, -999f, -1001f, -1000f,
            Float.MIN_VALUE, Float.MAX_VALUE/2, 0f, 1f
        };
        
        // CPU computation
        float[] cpuOutput = extremeValues.clone();
        softmax.compute(cpuOutput, 4, 4);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(extremeValues);
        softmax.computeGPU(gpuPtr, 4, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, extremeValues.length);
        
        // Compare CPU vs GPU
        assertArrayEquals(cpuOutput, gpuOutput, RELAXED_EPSILON);
        
        // Verify no NaN or Inf values
        for (float v : gpuOutput) {
            assertFalse(Float.isNaN(v));
            assertFalse(Float.isInfinite(v));
        }
        
        CudaUtil.free(gpuPtr);
    }
    
    @Test
    public void testCrossEntropyFusedGradient() {
        SoftMax softmax = new SoftMax();

        // Test batch (4 batches, 5 units each)
        float[] logits = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };

        // One-hot encoded targets for multi-class classification
        float[] targets = {
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // class 2
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // class 0
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // class 4
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f   // class 3
        };

        // Apply softmax
        float[] softmaxOutput = logits.clone();
        softmax.compute(softmaxOutput, 5, 4);

        // Test fused gradient computation
        float[] gradient = new float[20];
        softmax.gradientCrossEntropy(softmaxOutput, targets, gradient, 5, 4);

        // Expected: softmax - target
        // Manually calculated constants from softmax outputs minus targets
        float[] expected = {
            0.52693f, 0.02624f, -0.88243f, 0.00965f, 0.31960f,
            -0.90376f, 0.43130f, 0.03540f, 0.31951f, 0.11754f,
            0.38285f, 0.05181f, 0.17203f, 0.04689f, -0.65358f,
            0.03810f, 0.25480f, 0.11449f, -0.43294f, 0.02554f
        };

        assertArrayEquals(expected, gradient, EPSILON);
    }    
    @Test
    public void testCrossEntropyFusedGradientGPU() {
        SoftMax softmax = new SoftMax();
        
        // Test batch (4 batches, 5 units each)
        float[] logits = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] targets = {
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f
        };
        
        // Apply softmax on CPU
        float[] softmaxOutput = logits.clone();
        softmax.compute(softmaxOutput, 5, 4);
        
        // Compute CPU gradient
        float[] cpuGradient = new float[20];
        softmax.gradientCrossEntropy(softmaxOutput, targets, cpuGradient, 5, 4);
        
        // Test GPU fused gradient
        CUdeviceptr gpuSoftmax = CudaUtil.toGPU(softmaxOutput);
        CUdeviceptr gpuTargets = CudaUtil.toGPU(targets);
        CUdeviceptr gpuGradient = CudaUtil.toGPU(new float[20]);
        softmax.gradientGPUCrossEntropy(gpuSoftmax, gpuTargets, gpuGradient, 5, 4, null);
        float[] gpuResult = CudaUtil.fromGPUFloat(gpuGradient, 20);
        
        assertArrayEquals(cpuGradient, gpuResult, RELAXED_EPSILON);
        
        CudaUtil.free(gpuSoftmax);
        CudaUtil.free(gpuTargets);
        CudaUtil.free(gpuGradient);
    }
    
    @Test
    public void testCrossEntropyFusedVsSeparate() {
        SoftMax softmax = new SoftMax();
        CrossEntropy ce = new CrossEntropy();
        
        // Test batch (4 batches, 5 units each)
        float[] logits = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] targets = {
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f
        };
        
        // Method 1: Separate operations
        float[] softmaxOutput = logits.clone();
        softmax.compute(softmaxOutput, 5, 4);
        
        float[] gradients = ce.derivative(softmaxOutput, targets);
        softmax.gradient(logits, softmaxOutput, gradients, 5, 4);
        
        // Method 2: Fused operation
        float[] fusedGradient = new float[20];
        softmax.gradientCrossEntropy(softmaxOutput, targets, fusedGradient, 5, 4);
        
        assertArrayEquals(gradients, fusedGradient, RELAXED_EPSILON);
    }
    
    @Test
    public void testCrossEntropyFusedVsSeparateGPU() {
        SoftMax softmax = new SoftMax();
        CrossEntropy ce = new CrossEntropy();
        
        // Test batch (4 batches, 5 units each)
        float[] logits = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] targets = {
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f
        };
        
        // Method 1: Separate GPU operations
        CUdeviceptr gpuLogits = CudaUtil.toGPU(logits);
        CUdeviceptr gpuTargets = CudaUtil.toGPU(targets);
        CUdeviceptr gpuSoftmax = CudaUtil.toGPU(logits.clone());
        
        softmax.computeGPU(gpuSoftmax, 5, 4, null);
        
        CUdeviceptr gpuCEDerivative = CudaUtil.toGPU(new float[20]);
        ce.derivativeGPU(gpuSoftmax, gpuTargets, gpuCEDerivative, 20, null);
        softmax.gradientGPU(gpuLogits, gpuSoftmax, gpuCEDerivative, 5, 4, null);
        
        // Method 2: Fused GPU operation
        CUdeviceptr gpuFusedGradient = CudaUtil.toGPU(new float[20]);
        softmax.gradientGPUCrossEntropy(gpuSoftmax, gpuTargets, gpuFusedGradient, 5, 4, null);
        
        // Compare results
        float[] separateResult = CudaUtil.fromGPUFloat(gpuCEDerivative, 20);
        float[] fusedResult = CudaUtil.fromGPUFloat(gpuFusedGradient, 20);
        
        assertArrayEquals(separateResult, fusedResult, RELAXED_EPSILON);
        
        CudaUtil.free(gpuLogits);
        CudaUtil.free(gpuTargets);
        CudaUtil.free(gpuSoftmax);
        CudaUtil.free(gpuCEDerivative);
        CudaUtil.free(gpuFusedGradient);
    }
    
    @Test
    public void testCrossEntropyFusedNumericalStability() {
        SoftMax softmax = new SoftMax();

        // Test batch (3 batches, 5 units each) with extreme values
        float[] logits = {
            1000f, 999f, 998f, 997f, 996f,      // Very large values
            -1000f, -999f, -998f, -997f, -996f, // Very small values
            0f, 100f, -100f, 50f, -50f          // Mixed values
        };

        float[] targets = {
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // class 0
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // class 4
            0.0f, 1.0f, 0.0f, 0.0f, 0.0f   // class 1
        };

        // Apply softmax (should handle large values via max subtraction)
        float[] softmaxOutput = logits.clone();
        softmax.compute(softmaxOutput, 5, 3);

        // Test fused gradient - should be numerically stable
        float[] gradient = new float[15];
        softmax.gradientCrossEntropy(softmaxOutput, targets, gradient, 5, 3);

        // Expected gradients:
        // Batch 0: softmax of [1000,999,998,997,996] = [0.6365, 0.2341, 0.0861, 0.0317, 0.0117]
        //          gradient = softmax - [1,0,0,0,0] = [-0.3635, 0.2341, 0.0861, 0.0317, 0.0117]
        // Batch 1: softmax of [-1000,-999,-998,-997,-996] = [0.0117, 0.0317, 0.0861, 0.2341, 0.6365]
        //          gradient = softmax - [0,0,0,0,1] = [0.0117, 0.0317, 0.0861, 0.2341, -0.3635]
        // Batch 2: softmax of [0,100,-100,50,-50] ≈ [0, 1, 0, 0, 0]
        //          gradient = softmax - [0,1,0,0,0] ≈ [0, 0, 0, 0, 0]
        float[] expected = {
            -0.36359f, 0.23412f, 0.08612f, 0.03168f, 0.01166f,  // batch 0
            0.01166f, 0.03168f, 0.08613f, 0.23412f, -0.36359f,  // batch 1
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f                    // batch 2
        };

        // Batch 2 should have near-zero gradients, others won't
        for (int i = 0; i < 10; i++) {
            assertEquals(expected[i], gradient[i], EPSILON);
        }
        // Batch 2 (indices 10-14) with relaxed tolerance
        for (int i = 10; i < 15; i++) {
            assertEquals(expected[i], gradient[i], 0.0001f);
        }

        // Verify no NaN or Inf values
        for (float g : gradient) {
            assertFalse("Gradient contains NaN", Float.isNaN(g));
            assertFalse("Gradient contains Inf", Float.isInfinite(g));
        }
}
}