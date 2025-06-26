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
import org.fjnn.activation.LeakyReLU;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;

/**
 * Unit test for LeakyReLU activation function.
 * 
 * @author ahmed
 */
public class LeakyReLUTest extends ActivationBaseTest {
    
    @Override
    public void testCompute() {
        LeakyReLU leakyRelu = new LeakyReLU(); // Default alpha = 0.3f
        
        // Test single values
        assertEquals(-0.6f, leakyRelu.compute(-2.0f), EPSILON);
        assertEquals(-0.3f, leakyRelu.compute(-1.0f), EPSILON);
        assertEquals(0.0f, leakyRelu.compute(0.0f), EPSILON);
        assertEquals(1.0f, leakyRelu.compute(1.0f), EPSILON);
        assertEquals(2.0f, leakyRelu.compute(2.0f), EPSILON);
        
        // Test batch
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            -0.6f, -0.3f, 0.0f, 1.0f, 2.0f,
            -0.45f, -0.15f, 0.5f, 1.5f, 2.5f,
            -0.21f, -0.09f, 0.3f, 0.7f, 1.7f,
            -0.75f, -0.51f, 0.2f, 1.2f, 2.2f
        };
        
        float[] output = input.clone();
        leakyRelu.compute(output, 5, 4);
        assertArrayEquals(expected, output, EPSILON);
    }
    
    @Override
    public void testDerivative() {
        LeakyReLU leakyRelu = new LeakyReLU();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = {
            -0.6f, -0.3f, 0.0f, 1.0f, 2.0f,
            -0.45f, -0.15f, 0.5f, 1.5f, 2.5f,
            -0.21f, -0.09f, 0.3f, 0.7f, 1.7f,
            -0.75f, -0.51f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            0.3f, 0.3f, 1.0f, 1.0f, 1.0f,
            0.3f, 0.3f, 1.0f, 1.0f, 1.0f,
            0.3f, 0.3f, 1.0f, 1.0f, 1.0f,
            0.3f, 0.3f, 1.0f, 1.0f, 1.0f
        };
        
        float[] derivative = new float[20];
        leakyRelu.derivative(preActivation, postActivation, derivative, 5, 4);
        assertArrayEquals(expected, derivative, EPSILON);
    }
    
    @Override
    public void testGradient() {
        LeakyReLU leakyRelu = new LeakyReLU();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = {
            -0.6f, -0.3f, 0.0f, 1.0f, 2.0f,
            -0.45f, -0.15f, 0.5f, 1.5f, 2.5f,
            -0.21f, -0.09f, 0.3f, 0.7f, 1.7f,
            -0.75f, -0.51f, 0.2f, 1.2f, 2.2f
        };
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] expected = {
            0.6f, -0.3f, 0.5f, -2.0f, 1.5f,
            -0.15f, 0.3f, -1.5f, 0.7f, -0.3f,
            0.36f, -0.24f, 0.4f, -0.9f, 1.1f,
            -0.39f, 0.18f, -0.2f, 1.4f, -1.7f
        };
        
        gradient = gradient.clone();
        leakyRelu.gradient(preActivation, postActivation, gradient, 5, 4);
        assertArrayEquals(expected, gradient, EPSILON);
    }
    
    @Override
    public void testEdgeCases() {
        LeakyReLU leakyRelu = new LeakyReLU();
        
        // Test extreme values
        assertEquals(-3e9f, leakyRelu.compute(-1e10f), 1e8f);
        assertEquals(1e10f, leakyRelu.compute(1e10f), EPSILON);
        
        // Test special values
        assertTrue(Float.isNaN(leakyRelu.compute(Float.NaN)));
        assertEquals(Float.NEGATIVE_INFINITY, leakyRelu.compute(Float.NEGATIVE_INFINITY), EPSILON);
        assertEquals(Float.POSITIVE_INFINITY, leakyRelu.compute(Float.POSITIVE_INFINITY), EPSILON);
        
        // Test custom alpha
        LeakyReLU customAlpha = new LeakyReLU(0.01f);
        assertEquals(-0.02f, customAlpha.compute(-2.0f), EPSILON);
        assertEquals(2.0f, customAlpha.compute(2.0f), EPSILON);
    }
    
    @Override
    public void testSerialization() {
        LeakyReLU leakyRelu = new LeakyReLU(0.2f);
        
        Map<String, Object> serialized = leakyRelu.serialize();
        assertNotNull(serialized);
        assertEquals("LeakyReLU", serialized.get("type"));
        assertEquals(0.2f, (Float)serialized.get("alpha"), EPSILON);
        
        Activation deserialized = Activation.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof LeakyReLU);
        
        // Test functionality
        assertEquals(-0.2f, deserialized.compute(-1.0f), EPSILON);
        assertEquals(1.0f, deserialized.compute(1.0f), EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        LeakyReLU leakyRelu = new LeakyReLU();
        
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            -0.6f, -0.3f, 0.0f, 1.0f, 2.0f,
            -0.45f, -0.15f, 0.5f, 1.5f, 2.5f,
            -0.21f, -0.09f, 0.3f, 0.7f, 1.7f,
            -0.75f, -0.51f, 0.2f, 1.2f, 2.2f
        };
        
        CUdeviceptr gpuPtr = CudaUtil.toGPU(input);
        leakyRelu.computeGPU(gpuPtr, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, input.length);
        assertArrayEquals(expected, gpuOutput, EPSILON);
        CudaUtil.free(gpuPtr);
    }
    
    @Override
    public void testDerivativeGPU() {
        LeakyReLU leakyRelu = new LeakyReLU();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = {
            -0.6f, -0.3f, 0.0f, 1.0f, 2.0f,
            -0.45f, -0.15f, 0.5f, 1.5f, 2.5f,
            -0.21f, -0.09f, 0.3f, 0.7f, 1.7f,
            -0.75f, -0.51f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            0.3f, 0.3f, 1.0f, 1.0f, 1.0f,
            0.3f, 0.3f, 1.0f, 1.0f, 1.0f,
            0.3f, 0.3f, 1.0f, 1.0f, 1.0f,
            0.3f, 0.3f, 1.0f, 1.0f, 1.0f
        };
        
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        leakyRelu.derivativeGPU(gpuPre, gpuPost, gpuDeriv, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuDeriv, 20);
        assertArrayEquals(expected, gpuOutput, EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuDeriv);
    }
    
    @Override
    public void testGradientGPU() {
        LeakyReLU leakyRelu = new LeakyReLU();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = {
            -0.6f, -0.3f, 0.0f, 1.0f, 2.0f,
            -0.45f, -0.15f, 0.5f, 1.5f, 2.5f,
            -0.21f, -0.09f, 0.3f, 0.7f, 1.7f,
            -0.75f, -0.51f, 0.2f, 1.2f, 2.2f
        };
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] expected = {
            0.6f, -0.3f, 0.5f, -2.0f, 1.5f,
            -0.15f, 0.3f, -1.5f, 0.7f, -0.3f,
            0.36f, -0.24f, 0.4f, -0.9f, 1.1f,
            -0.39f, 0.18f, -0.2f, 1.4f, -1.7f
        };
        
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuGrad = CudaUtil.toGPU(gradient);
        leakyRelu.gradientGPU(gpuPre, gpuPost, gpuGrad, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuGrad, gradient.length);
        assertArrayEquals(expected, gpuOutput, EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuGrad);
    }
    
    @Override
    public void testEdgeCasesGPU() {
        LeakyReLU leakyRelu = new LeakyReLU();
        
        float[] edgeValues = {
            -1e10f, -Float.MAX_VALUE, Float.MIN_VALUE, 1e10f, 
            Float.MAX_VALUE, Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY
        };
        
        float[] cpuOutput = edgeValues.clone();
        for (int i = 0; i < cpuOutput.length; i++) {
            cpuOutput[i] = leakyRelu.compute(edgeValues[i]);
        }
        
        CUdeviceptr gpuPtr = CudaUtil.toGPU(edgeValues);
        leakyRelu.computeGPU(gpuPtr, edgeValues.length, 1, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, edgeValues.length);
        
        for (int i = 0; i < cpuOutput.length; i++) {
            if (Float.isNaN(cpuOutput[i])) {
                assertTrue("GPU should produce NaN", Float.isNaN(gpuOutput[i]));
            } else if (Float.isInfinite(cpuOutput[i])) {
                assertEquals(cpuOutput[i], gpuOutput[i], EPSILON);
            } else {
                assertEquals(cpuOutput[i], gpuOutput[i], Math.abs(cpuOutput[i]) * 0.01f);
            }
        }
        
        CudaUtil.free(gpuPtr);
    }
}