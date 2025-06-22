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
import org.fjnn.activation.Step;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;

/**
 * Unit test for Step activation function.
 * 
 * @author ahmed
 */
public class StepTest extends ActivationTest {
    
    @Override
    public void testCompute() {
        Step step = new Step();

        // Test batch (4 batches, 5 units each)
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };

        float[] expected = {
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f,  // 0.0f should return 1.0f
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f
        };

        float[] output = input.clone();
        step.compute(output, 5, 4);
        assertArrayEquals(expected, output, EPSILON);
    }
    
    @Override
    public void testDerivative() {
        Step step = new Step();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        step.compute(postActivation, 5, 4);
        
        float[] derivative = new float[20];
        
        try {
            step.derivative(preActivation, postActivation, derivative, 5, 4);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage().contains("does not support derivatives"));
        }
    }
    
    @Override
    public void testGradient() {
        Step step = new Step();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        step.compute(postActivation, 5, 4);
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        gradient = gradient.clone();
        
        try {
            step.gradient(preActivation, postActivation, gradient, 5, 4);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage().contains("does not support gradients"));
        }
    }
    
    @Override
    public void testEdgeCases() {
        Step step = new Step();
        
        float[] edgeValues = {
            -1e10f, -Float.MAX_VALUE, Float.MIN_VALUE, 1e10f,
            Float.MAX_VALUE, Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY,
            -0.0001f, 0.0001f, 0.0f, -0.0f
        };
        
        float[] output = edgeValues.clone();
        step.compute(output, 4, 3);
        
        // Verify outputs
        assertEquals(0.0f, output[0], EPSILON); // negative
        assertEquals(0.0f, output[1], EPSILON); // negative
        assertEquals(1.0f, output[2], EPSILON); // positive
        assertEquals(1.0f, output[3], EPSILON); // positive
        assertEquals(1.0f, output[4], EPSILON); // positive
        assertEquals(0.0f, output[5], EPSILON); // NaN -> 0
        assertEquals(0.0f, output[6], EPSILON); // -Inf -> 0
        assertEquals(1.0f, output[7], EPSILON); // +Inf -> 1
        assertEquals(0.0f, output[8], EPSILON); // negative
        assertEquals(1.0f, output[9], EPSILON); // positive
        assertEquals(1.0f, output[10], EPSILON); // 0 -> 1
        assertEquals(1.0f, output[11], EPSILON); // -0 -> 1
    }
    
    @Override
    public void testSerialization() {
        Step step = new Step();
        
        Map<String, Object> serialized = step.serialize();
        assertNotNull(serialized);
        assertEquals("Step", serialized.get("type"));
        
        Activation deserialized = Activation.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof Step);
        
        // Test functionality
        float[] input = {-1.0f, 0.0f, 1.0f, 2.0f};
        float[] cpuOutput = input.clone();
        step.compute(cpuOutput, 4, 1);
        
        float[] deserializedOutput = input.clone();
        deserialized.compute(deserializedOutput, 4, 1);
        
        assertArrayEquals(cpuOutput, deserializedOutput, EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        Step step = new Step();
        
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        // CPU computation
        float[] cpuOutput = input.clone();
        step.compute(cpuOutput, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(input);
        step.computeGPU(gpuPtr, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, input.length);
        
        assertArrayEquals(cpuOutput, gpuOutput, EPSILON);
        CudaUtil.free(gpuPtr);
    }
    
    @Override
    public void testDerivativeGPU() {
        Step step = new Step();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        step.compute(postActivation, 5, 4);
        
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        
        try {
            step.derivativeGPU(gpuPre, gpuPost, gpuDeriv, 5, 4, null);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage().contains("does not support derivatives"));
        } finally {
            CudaUtil.free(gpuPre);
            CudaUtil.free(gpuPost);
            CudaUtil.free(gpuDeriv);
        }
    }
    
    @Override
    public void testGradientGPU() {
        Step step = new Step();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        step.compute(postActivation, 5, 4);
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuGrad = CudaUtil.toGPU(gradient);
        
        try {
            step.gradientGPU(gpuPre, gpuPost, gpuGrad, 5, 4, null);
            fail("Expected UnsupportedOperationException");
        } catch (UnsupportedOperationException e) {
            assertTrue(e.getMessage().contains("does not support gradients"));
        } finally {
            CudaUtil.free(gpuPre);
            CudaUtil.free(gpuPost);
            CudaUtil.free(gpuGrad);
        }
    }
    
    @Override
    public void testEdgeCasesGPU() {
        Step step = new Step();
        
        float[] edgeValues = {
            -1e10f, -Float.MAX_VALUE, Float.MIN_VALUE, 1e10f,
            Float.MAX_VALUE, Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY,
            -0.0001f, 0.0001f, 0.0f, -0.0f
        };
        
        // CPU computation
        float[] cpuOutput = edgeValues.clone();
        step.compute(cpuOutput, 4, 3);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(edgeValues);
        step.computeGPU(gpuPtr, 4, 3, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, edgeValues.length);
        
        assertArrayEquals(cpuOutput, gpuOutput, EPSILON);
        CudaUtil.free(gpuPtr);
    }
}