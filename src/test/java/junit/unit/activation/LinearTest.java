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
import org.fjnn.activation.Linear;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;

/**
 * Unit test for Linear activation function.
 * 
 * @author ahmed
 */
public class LinearTest extends ActivationBaseTest {
    
    @Override
    public void testCompute() {
        Linear linear = new Linear();
        
        // Test single values
        assertEquals(-2.0f, linear.compute(-2.0f), EPSILON);
        assertEquals(-1.0f, linear.compute(-1.0f), EPSILON);
        assertEquals(0.0f, linear.compute(0.0f), EPSILON);
        assertEquals(1.0f, linear.compute(1.0f), EPSILON);
        assertEquals(2.0f, linear.compute(2.0f), EPSILON);
        
        // Test batch
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = input.clone();
        
        float[] output = input.clone();
        linear.compute(output, 5, 4);
        assertArrayEquals(expected, output, EPSILON);
    }
    
    @Override
    public void testDerivative() {
        Linear linear = new Linear();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        
        float[] expected = new float[20];
        for (int i = 0; i < 20; i++) {
            expected[i] = 1.0f;
        }
        
        float[] derivative = new float[20];
        linear.derivative(preActivation, postActivation, derivative, 5, 4);
        assertArrayEquals(expected, derivative, EPSILON);
    }
    
    @Override
    public void testGradient() {
        Linear linear = new Linear();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] expected = gradient.clone();
        
        gradient = gradient.clone();
        linear.gradient(preActivation, postActivation, gradient, 5, 4);
        assertArrayEquals(expected, gradient, EPSILON);
    }
    
    @Override
    public void testEdgeCases() {
        Linear linear = new Linear();
        
        // Test extreme values
        assertEquals(-1e10f, linear.compute(-1e10f), EPSILON);
        assertEquals(1e10f, linear.compute(1e10f), EPSILON);
        assertEquals(Float.MAX_VALUE, linear.compute(Float.MAX_VALUE), EPSILON);
        assertEquals(-Float.MAX_VALUE, linear.compute(-Float.MAX_VALUE), EPSILON);
        assertEquals(Float.MIN_VALUE, linear.compute(Float.MIN_VALUE), EPSILON);
        
        // Test special values
        assertTrue(Float.isNaN(linear.compute(Float.NaN)));
        assertEquals(Float.NEGATIVE_INFINITY, linear.compute(Float.NEGATIVE_INFINITY), EPSILON);
        assertEquals(Float.POSITIVE_INFINITY, linear.compute(Float.POSITIVE_INFINITY), EPSILON);
    }
    
    @Override
    public void testSerialization() {
        Linear linear = new Linear();
        
        Map<String, Object> serialized = linear.serialize();
        assertNotNull(serialized);
        assertEquals("Linear", serialized.get("type"));
        
        Activation deserialized = Activation.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof Linear);
        
        // Test functionality
        assertEquals(-1.0f, deserialized.compute(-1.0f), EPSILON);
        assertEquals(1.0f, deserialized.compute(1.0f), EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        Linear linear = new Linear();
        
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = input.clone();
        
        CUdeviceptr gpuPtr = CudaUtil.toGPU(input);
        linear.computeGPU(gpuPtr, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, input.length);
        assertArrayEquals(expected, gpuOutput, EPSILON);
        CudaUtil.free(gpuPtr);
    }
    
    @Override
    public void testDerivativeGPU() {
        Linear linear = new Linear();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        
        float[] expected = new float[20];
        for (int i = 0; i < 20; i++) {
            expected[i] = 1.0f;
        }
        
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        linear.derivativeGPU(gpuPre, gpuPost, gpuDeriv, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuDeriv, 20);
        assertArrayEquals(expected, gpuOutput, EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuDeriv);
    }
    
    @Override
    public void testGradientGPU() {
        Linear linear = new Linear();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] expected = gradient.clone();
        
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuGrad = CudaUtil.toGPU(gradient);
        linear.gradientGPU(gpuPre, gpuPost, gpuGrad, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuGrad, gradient.length);
        assertArrayEquals(expected, gpuOutput, EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuGrad);
    }
    
    @Override
    public void testEdgeCasesGPU() {
        Linear linear = new Linear();
        
        float[] edgeValues = {
            -1e10f, -Float.MAX_VALUE, Float.MIN_VALUE, 1e10f, 
            Float.MAX_VALUE, Float.NaN, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY
        };
        
        float[] cpuOutput = edgeValues.clone();
        for (int i = 0; i < cpuOutput.length; i++) {
            cpuOutput[i] = linear.compute(edgeValues[i]);
        }
        
        CUdeviceptr gpuPtr = CudaUtil.toGPU(edgeValues);
        linear.computeGPU(gpuPtr, edgeValues.length, 1, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, edgeValues.length);
        
        for (int i = 0; i < cpuOutput.length; i++) {
            if (Float.isNaN(cpuOutput[i])) {
                assertTrue("GPU should produce NaN", Float.isNaN(gpuOutput[i]));
            } else {
                assertEquals(cpuOutput[i], gpuOutput[i], EPSILON);
            }
        }
        
        CudaUtil.free(gpuPtr);
    }
}