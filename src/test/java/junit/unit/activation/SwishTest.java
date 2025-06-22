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
import org.fjnn.activation.Swish;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;

/**
 * Unit test for Swish activation function.
 * Swish(x) = x * sigmoid(x)
 * 
 * @author ahmed
 */
public class SwishTest extends ActivationTest {
    
    @Override
    public void testCompute() {
        Swish swish = new Swish();
        
        // Test batch (4 batches, 5 units each)
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            -0.23841f, -0.26894f, 0.00000f, 0.73106f, 1.76159f,
            -0.27364f, -0.18877f, 0.31123f, 1.22636f, 2.31035f,
            -0.23227f, -0.12767f, 0.17233f, 0.46773f, 1.43740f,
            -0.18965f, -0.26260f, 0.10997f, 0.92223f, 1.98055f
        };
        
        float[] output = input.clone();
        swish.compute(output, 5, 4);
        assertArrayEquals(expected, output, EPSILON);
    }
    
    @Override
    public void testDerivative() {
        Swish swish = new Swish();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        swish.compute(postActivation, 5, 4);
        
        // Compute derivative
        float[] derivative = new float[20];
        swish.derivative(preActivation, postActivation, derivative, 5, 4);
        
        // Verify derivative = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        for (int i = 0; i < 20; i++) {
            float x = preActivation[i];
            float sigmoid = 1.0f / (1.0f + (float)Math.exp(-x));
            float expected = sigmoid + x * sigmoid * (1.0f - sigmoid);
            assertEquals(expected, derivative[i], EPSILON);
        }
    }
    
    @Override
    public void testGradient() {
        Swish swish = new Swish();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        swish.compute(postActivation, 5, 4);
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] derivative = new float[20];
        swish.derivative(preActivation, postActivation, derivative, 5, 4);
        
        float[] expected = gradient.clone();
        for (int i = 0; i < 20; i++) {
            expected[i] *= derivative[i];
        }
        
        gradient = gradient.clone();
        swish.gradient(preActivation, postActivation, gradient, 5, 4);
        assertArrayEquals(expected, gradient, EPSILON);
    }
    
    @Override
    public void testEdgeCases() {
        Swish swish = new Swish();
        
        float[] edgeValues = {
            -100f, -50f, -10f, -1f,
            0f, 1f, 10f, 50f,
            100f, Float.MIN_VALUE, -Float.MIN_VALUE, Float.NaN
        };
        
        float[] output = edgeValues.clone();
        swish.compute(output, 4, 3);
        
        // Verify behavior
        assertTrue(output[0] < 0.0f && output[0] > -1.0f); // -100 * sigmoid(-100) ≈ 0
        assertEquals(0.0f, output[4], EPSILON); // 0 * sigmoid(0) = 0
        assertTrue(output[8] > 99.9f && output[8] <= 100.0f); // 100 * sigmoid(100) ≈ 100
        assertTrue(Float.isNaN(output[11])); // NaN input
    }
    
    @Override
    public void testSerialization() {
        Swish swish = new Swish();
        
        Map<String, Object> serialized = swish.serialize();
        assertNotNull(serialized);
        assertEquals("Swish", serialized.get("type"));
        
        Activation deserialized = Activation.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof Swish);
        
        // Test functionality
        float[] input = {-1.0f, 0.0f, 1.0f, 2.0f};
        float[] cpuOutput = input.clone();
        swish.compute(cpuOutput, 4, 1);
        
        float[] deserializedOutput = input.clone();
        deserialized.compute(deserializedOutput, 4, 1);
        
        assertArrayEquals(cpuOutput, deserializedOutput, EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        Swish swish = new Swish();
        
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        // CPU computation
        float[] cpuOutput = input.clone();
        swish.compute(cpuOutput, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(input);
        swish.computeGPU(gpuPtr, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, input.length);
        
        assertArrayEquals(cpuOutput, gpuOutput, RELAXED_EPSILON);
        CudaUtil.free(gpuPtr);
    }
    
    @Override
    public void testDerivativeGPU() {
        Swish swish = new Swish();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        swish.compute(postActivation, 5, 4);
        
        // CPU computation
        float[] cpuDerivative = new float[20];
        swish.derivative(preActivation, postActivation, cpuDerivative, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        swish.derivativeGPU(gpuPre, gpuPost, gpuDeriv, 5, 4, null);
        float[] gpuDerivative = CudaUtil.fromGPUFloat(gpuDeriv, 20);
        
        assertArrayEquals(cpuDerivative, gpuDerivative, RELAXED_EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuDeriv);
    }
    
    @Override
    public void testGradientGPU() {
        Swish swish = new Swish();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        swish.compute(postActivation, 5, 4);
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        // CPU computation
        float[] cpuGradient = gradient.clone();
        swish.gradient(preActivation, postActivation, cpuGradient, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuGrad = CudaUtil.toGPU(gradient);
        swish.gradientGPU(gpuPre, gpuPost, gpuGrad, 5, 4, null);
        float[] gpuGradient = CudaUtil.fromGPUFloat(gpuGrad, gradient.length);
        
        assertArrayEquals(cpuGradient, gpuGradient, RELAXED_EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuGrad);
    }
    
    @Override
    public void testEdgeCasesGPU() {
        Swish swish = new Swish();
        
        float[] edgeValues = {
            -100f, -50f, -10f, -1f,
            0f, 1f, 10f, 50f,
            100f, Float.MIN_VALUE, -Float.MIN_VALUE, Float.NaN
        };
        
        // CPU computation
        float[] cpuOutput = edgeValues.clone();
        swish.compute(cpuOutput, 4, 3);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(edgeValues);
        swish.computeGPU(gpuPtr, 4, 3, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, edgeValues.length);
        
        for (int i = 0; i < cpuOutput.length; i++) {
            if (Float.isNaN(cpuOutput[i])) {
                assertTrue("GPU should produce NaN", Float.isNaN(gpuOutput[i]));
            } else if (Float.isInfinite(cpuOutput[i])) {
                assertEquals(cpuOutput[i], gpuOutput[i], EPSILON);
            } else {
                assertEquals(cpuOutput[i], gpuOutput[i], RELAXED_EPSILON);
            }
        }
        
        CudaUtil.free(gpuPtr);
    }
}