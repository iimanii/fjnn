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
import org.fjnn.activation.GeLU;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;

/**
 * Unit test for GeLU activation function.
 * 
 * @author ahmed
 */
public class GeLUTest extends ActivationTest {
    
    @Override
    public void testCompute() {
        GeLU gelu = new GeLU();
        
        // Test batch (4 batches, 5 units each)
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            -0.04540f, -0.15881f, 0.00000f, 0.84119f, 1.95460f,
            -0.10043f, -0.15428f, 0.34572f, 1.39957f, 2.48492f,
            -0.16942f, -0.11462f, 0.18537f, 0.53057f, 1.62410f,
            -0.01508f, -0.07589f, 0.11585f, 1.06170f, 2.16967f
        };
        
        float[] output = input.clone();
        gelu.compute(output, 5, 4);
        assertArrayEquals(expected, output, EPSILON);
    }
    
    @Override
    public void testDerivative() {
        GeLU gelu = new GeLU();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        gelu.compute(postActivation, 5, 4);
        
        // Compute derivative using CPU
        float[] derivative = new float[20];
        gelu.derivative(preActivation, postActivation, derivative, 5, 4);
        
        // Verify derivative values are reasonable
        for (int i = 0; i < 20; i++) {
            assertTrue(derivative[i] >= -0.2f && derivative[i] <= 1.2f);
        }
    }
    
    @Override
    public void testGradient() {
        GeLU gelu = new GeLU();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        gelu.compute(postActivation, 5, 4);
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] derivative = new float[20];
        gelu.derivative(preActivation, postActivation, derivative, 5, 4);
        
        float[] expected = gradient.clone();
        for (int i = 0; i < 20; i++) {
            expected[i] *= derivative[i];
        }
        
        gradient = gradient.clone();
        gelu.gradient(preActivation, postActivation, gradient, 5, 4);
        assertArrayEquals(expected, gradient, EPSILON);
    }
    
    @Override
    public void testEdgeCases() {
        GeLU gelu = new GeLU();
        
        float[] edgeValues = {
            -10f, -5f, -1f, -0.1f,
            0f, 0.1f, 1f, 5f,
            10f, -1e-10f, 1e-10f, 100f
        };
        
        float[] output = edgeValues.clone();
        gelu.compute(output, 4, 3);
        
        // Verify behavior
        assertTrue(Math.abs(output[0]) < 0.01f); // GeLU(-10) ≈ 0
        assertEquals(0.0f, output[4], EPSILON); // GeLU(0) = 0
        assertTrue(output[8] > 9.99f); // GeLU(10) ≈ 10
        assertTrue(Math.abs(output[9]) < 1e-9f); // GeLU(-1e-10) ≈ 0
        assertTrue(output[10] > 0.0f); // GeLU(1e-10) > 0
    }
    
    @Override
    public void testSerialization() {
        GeLU gelu = new GeLU();
        
        Map<String, Object> serialized = gelu.serialize();
        assertNotNull(serialized);
        assertEquals("GeLU", serialized.get("type"));
        
        Activation deserialized = Activation.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof GeLU);
        
        // Test functionality
        float[] input = {-1.0f, 0.0f, 1.0f, 2.0f};
        float[] cpuOutput = input.clone();
        gelu.compute(cpuOutput, 4, 1);
        
        float[] deserializedOutput = input.clone();
        deserialized.compute(deserializedOutput, 4, 1);
        
        assertArrayEquals(cpuOutput, deserializedOutput, EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        GeLU gelu = new GeLU();
        
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        // CPU computation
        float[] cpuOutput = input.clone();
        gelu.compute(cpuOutput, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(input);
        gelu.computeGPU(gpuPtr, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, input.length);
        
        assertArrayEquals(cpuOutput, gpuOutput, RELAXED_EPSILON);
        CudaUtil.free(gpuPtr);
    }
    
    @Override
    public void testDerivativeGPU() {
        GeLU gelu = new GeLU();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        gelu.compute(postActivation, 5, 4);
        
        // CPU computation
        float[] cpuDerivative = new float[20];
        gelu.derivative(preActivation, postActivation, cpuDerivative, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        gelu.derivativeGPU(gpuPre, gpuPost, gpuDeriv, 5, 4, null);
        float[] gpuDerivative = CudaUtil.fromGPUFloat(gpuDeriv, 20);
        
        assertArrayEquals(cpuDerivative, gpuDerivative, RELAXED_EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuDeriv);
    }
    
    @Override
    public void testGradientGPU() {
        GeLU gelu = new GeLU();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        gelu.compute(postActivation, 5, 4);
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        // CPU computation
        float[] cpuGradient = gradient.clone();
        gelu.gradient(preActivation, postActivation, cpuGradient, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuGrad = CudaUtil.toGPU(gradient);
        gelu.gradientGPU(gpuPre, gpuPost, gpuGrad, 5, 4, null);
        float[] gpuGradient = CudaUtil.fromGPUFloat(gpuGrad, gradient.length);
        
        assertArrayEquals(cpuGradient, gpuGradient, RELAXED_EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuGrad);
    }
    
    @Override
    public void testEdgeCasesGPU() {
        GeLU gelu = new GeLU();
        
        float[] edgeValues = {
            -10f, -5f, -1f, -0.1f,
            0f, 0.1f, 1f, 5f,
            10f, -1e-10f, 1e-10f, 100f
        };
        
        // CPU computation
        float[] cpuOutput = edgeValues.clone();
        gelu.compute(cpuOutput, 4, 3);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(edgeValues);
        gelu.computeGPU(gpuPtr, 4, 3, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, edgeValues.length);
        
        assertArrayEquals(cpuOutput, gpuOutput, RELAXED_EPSILON);
        CudaUtil.free(gpuPtr);
    }
}