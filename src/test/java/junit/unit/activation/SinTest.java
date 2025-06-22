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
import org.fjnn.activation.Sin;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;

/**
 * Unit test for Sin activation function.
 * 
 * @author ahmed
 */
public class SinTest extends ActivationTest {
    
    @Override
    public void testCompute() {
        Sin sin = new Sin();
        
        // Test batch (4 batches, 5 units each)
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            -0.90930f, -0.84147f, 0.00000f, 0.84147f, 0.90930f,
            -0.99749f, -0.47943f, 0.47943f, 0.99749f, 0.59847f,
            -0.64422f, -0.29552f, 0.29552f, 0.64422f, 0.99166f,
            -0.59847f, -0.99166f, 0.19867f, 0.93204f, 0.80850f
        };
        
        float[] output = input.clone();
        sin.compute(output, 5, 4);
        assertArrayEquals(expected, output, EPSILON);
    }
    
    @Override
    public void testDerivative() {
        Sin sin = new Sin();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        sin.compute(postActivation, 5, 4);
        
        // Compute derivative
        float[] derivative = new float[20];
        sin.derivative(preActivation, postActivation, derivative, 5, 4);
        
        // Verify derivative = cos(x)
        for (int i = 0; i < 20; i++) {
            float expected = (float)Math.cos(preActivation[i]);
            assertEquals(expected, derivative[i], EPSILON);
        }
    }
    
    @Override
    public void testGradient() {
        Sin sin = new Sin();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        sin.compute(postActivation, 5, 4);
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] derivative = new float[20];
        sin.derivative(preActivation, postActivation, derivative, 5, 4);
        
        float[] expected = gradient.clone();
        for (int i = 0; i < 20; i++) {
            expected[i] *= derivative[i];
        }
        
        gradient = gradient.clone();
        sin.gradient(preActivation, postActivation, gradient, 5, 4);
        assertArrayEquals(expected, gradient, EPSILON);
    }
    
    @Override
    public void testEdgeCases() {
        Sin sin = new Sin();
        
        float[] edgeValues = {
            0.0f, (float)Math.PI, (float)(Math.PI/2), (float)(-Math.PI/2),
            -100f, -10f, -1f, -0.1f, 
            0.1f, 1f, 10f, 100f
        };
        
        float[] output = edgeValues.clone();
        sin.compute(output, 4, 3);
        
        // Verify all outputs are in [-1, 1]
        for (float v : output) {
            assertTrue(v >= -1.0f && v <= 1.0f);
        }
        
        // Test key points
        assertEquals(0.0f, output[0], EPSILON); // sin(0) = 0
        assertEquals(0.0f, output[1], 0.001f);  // sin(π) ≈ 0
        assertEquals(1.0f, output[2], 0.001f);  // sin(π/2) ≈ 1
        assertEquals(-1.0f, output[3], 0.001f); // sin(-π/2) ≈ -1
    }
    
    @Override
    public void testSerialization() {
        Sin sin = new Sin();
        
        Map<String, Object> serialized = sin.serialize();
        assertNotNull(serialized);
        assertEquals("Sin", serialized.get("type"));
        
        Activation deserialized = Activation.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof Sin);
        
        // Test functionality
        float[] input = {0.0f, 1.0f, -1.0f, 2.0f};
        float[] cpuOutput = input.clone();
        sin.compute(cpuOutput, 4, 1);
        
        float[] deserializedOutput = input.clone();
        deserialized.compute(deserializedOutput, 4, 1);
        
        assertArrayEquals(cpuOutput, deserializedOutput, EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        Sin sin = new Sin();
        
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        // CPU computation
        float[] cpuOutput = input.clone();
        sin.compute(cpuOutput, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(input);
        sin.computeGPU(gpuPtr, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, input.length);
        
        assertArrayEquals(cpuOutput, gpuOutput, RELAXED_EPSILON);
        CudaUtil.free(gpuPtr);
    }
    
    @Override
    public void testDerivativeGPU() {
        Sin sin = new Sin();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        sin.compute(postActivation, 5, 4);
        
        // CPU computation
        float[] cpuDerivative = new float[20];
        sin.derivative(preActivation, postActivation, cpuDerivative, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        sin.derivativeGPU(gpuPre, gpuPost, gpuDeriv, 5, 4, null);
        float[] gpuDerivative = CudaUtil.fromGPUFloat(gpuDeriv, 20);
        
        assertArrayEquals(cpuDerivative, gpuDerivative, RELAXED_EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuDeriv);
    }
    
    @Override
    public void testGradientGPU() {
        Sin sin = new Sin();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        sin.compute(postActivation, 5, 4);
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        // CPU computation
        float[] cpuGradient = gradient.clone();
        sin.gradient(preActivation, postActivation, cpuGradient, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuGrad = CudaUtil.toGPU(gradient);
        sin.gradientGPU(gpuPre, gpuPost, gpuGrad, 5, 4, null);
        float[] gpuGradient = CudaUtil.fromGPUFloat(gpuGrad, gradient.length);
        
        assertArrayEquals(cpuGradient, gpuGradient, RELAXED_EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuGrad);
    }
    
    @Override
    public void testEdgeCasesGPU() {
        Sin sin = new Sin();
        
        float[] edgeValues = {
            0.0f, (float)Math.PI, (float)(Math.PI/2), (float)(-Math.PI/2),
            -100f, -10f, -1f, -0.1f,
            0.1f, 1f, 10f, 100f
        };
        
        // CPU computation
        float[] cpuOutput = edgeValues.clone();
        sin.compute(cpuOutput, 4, 3);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(edgeValues);
        sin.computeGPU(gpuPtr, 4, 3, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, edgeValues.length);
        
        assertArrayEquals(cpuOutput, gpuOutput, RELAXED_EPSILON);
        CudaUtil.free(gpuPtr);
    }
}