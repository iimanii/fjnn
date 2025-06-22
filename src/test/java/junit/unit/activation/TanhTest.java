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
import org.fjnn.activation.Tanh;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;

/**
 * Unit test for Tanh activation function.
 * 
 * @author ahmed
 */
public class TanhTest extends ActivationTest {
    
    @Override
    public void testCompute() {
        Tanh tanh = new Tanh();
        
        // Test batch (4 batches, 5 units each)
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            -0.96403f, -0.76159f, 0.00000f, 0.76159f, 0.96403f,
            -0.90515f, -0.46212f, 0.46212f, 0.90515f, 0.98661f,
            -0.60437f, -0.29131f, 0.29131f, 0.60437f, 0.93541f,
            -0.98661f, -0.93541f, 0.19738f, 0.83365f, 0.97574f
        };
        
        float[] output = input.clone();
        tanh.compute(output, 5, 4);
        assertArrayEquals(expected, output, EPSILON);
    }
    
    @Override
    public void testDerivative() {
        Tanh tanh = new Tanh();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        tanh.compute(postActivation, 5, 4);
        
        // Compute derivative
        float[] derivative = new float[20];
        tanh.derivative(preActivation, postActivation, derivative, 5, 4);
        
        // Verify derivative = 1 - tanh²(x)
        for (int i = 0; i < 20; i++) {
            float expected = 1.0f - postActivation[i] * postActivation[i];
            assertEquals(expected, derivative[i], EPSILON);
        }
    }
    
    @Override
    public void testGradient() {
        Tanh tanh = new Tanh();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        tanh.compute(postActivation, 5, 4);
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] derivative = new float[20];
        tanh.derivative(preActivation, postActivation, derivative, 5, 4);
        
        float[] expected = gradient.clone();
        for (int i = 0; i < 20; i++) {
            expected[i] *= derivative[i];
        }
        
        gradient = gradient.clone();
        tanh.gradient(preActivation, postActivation, gradient, 5, 4);
        assertArrayEquals(expected, gradient, EPSILON);
    }
    
    @Override
    public void testEdgeCases() {
        Tanh tanh = new Tanh();
        
        float[] edgeValues = {
            -100f, -10f, -1f, -0.1f,
            0f, 0.1f, 1f, 10f,
            100f, Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY
        };
        
        float[] output = edgeValues.clone();
        tanh.compute(output, 4, 3);
        
        // Verify saturation
        assertTrue(Math.abs(output[0] + 1.0f) < 0.001f); // tanh(-100) ≈ -1
        assertTrue(Math.abs(output[8] - 1.0f) < 0.001f); // tanh(100) ≈ 1
        assertEquals(0.0f, output[4], EPSILON); // tanh(0) = 0
        assertTrue(Float.isNaN(output[9])); // NaN input
        assertEquals(-1.0f, output[10], EPSILON); // tanh(-∞) = -1
        assertEquals(1.0f, output[11], EPSILON); // tanh(+∞) = 1
    }
    
    @Override
    public void testSerialization() {
        Tanh tanh = new Tanh();
        
        Map<String, Object> serialized = tanh.serialize();
        assertNotNull(serialized);
        assertEquals("Tanh", serialized.get("type"));
        
        Activation deserialized = Activation.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof Tanh);
        
        // Test functionality
        float[] input = {-1.0f, 0.0f, 1.0f, 2.0f};
        float[] cpuOutput = input.clone();
        tanh.compute(cpuOutput, 4, 1);
        
        float[] deserializedOutput = input.clone();
        deserialized.compute(deserializedOutput, 4, 1);
        
        assertArrayEquals(cpuOutput, deserializedOutput, EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        Tanh tanh = new Tanh();
        
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        // CPU computation
        float[] cpuOutput = input.clone();
        tanh.compute(cpuOutput, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(input);
        tanh.computeGPU(gpuPtr, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, input.length);
        
        assertArrayEquals(cpuOutput, gpuOutput, RELAXED_EPSILON);
        CudaUtil.free(gpuPtr);
    }
    
    @Override
    public void testDerivativeGPU() {
        Tanh tanh = new Tanh();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        tanh.compute(postActivation, 5, 4);
        
        // CPU computation
        float[] cpuDerivative = new float[20];
        tanh.derivative(preActivation, postActivation, cpuDerivative, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        tanh.derivativeGPU(gpuPre, gpuPost, gpuDeriv, 5, 4, null);
        float[] gpuDerivative = CudaUtil.fromGPUFloat(gpuDeriv, 20);
        
        assertArrayEquals(cpuDerivative, gpuDerivative, RELAXED_EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuDeriv);
    }
    
    @Override
    public void testGradientGPU() {
        Tanh tanh = new Tanh();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        tanh.compute(postActivation, 5, 4);
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        // CPU computation
        float[] cpuGradient = gradient.clone();
        tanh.gradient(preActivation, postActivation, cpuGradient, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuGrad = CudaUtil.toGPU(gradient);
        tanh.gradientGPU(gpuPre, gpuPost, gpuGrad, 5, 4, null);
        float[] gpuGradient = CudaUtil.fromGPUFloat(gpuGrad, gradient.length);
        
        assertArrayEquals(cpuGradient, gpuGradient, RELAXED_EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuGrad);
    }
    
    @Override
    public void testEdgeCasesGPU() {
        Tanh tanh = new Tanh();
        
        float[] edgeValues = {
            -100f, -10f, -1f, -0.1f,
            0f, 0.1f, 1f, 10f,
            100f, Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY
        };
        
        // CPU computation
        float[] cpuOutput = edgeValues.clone();
        tanh.compute(cpuOutput, 4, 3);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(edgeValues);
        tanh.computeGPU(gpuPtr, 4, 3, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, edgeValues.length);
        
        for (int i = 0; i < cpuOutput.length; i++) {
            if (Float.isNaN(cpuOutput[i])) {
                assertTrue("GPU should produce NaN", Float.isNaN(gpuOutput[i]));
            } else {
                assertEquals(cpuOutput[i], gpuOutput[i], RELAXED_EPSILON);
            }
        }
        
        CudaUtil.free(gpuPtr);
    }
}