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
import org.fjnn.activation.ReLU;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;

/**
 * Unit test for ReLU activation function.
 * ReLU(x) = max(0, x)
 * 
 * @author ahmed
 */
public class ReLUTest extends ActivationBaseTest {
    
    @Override
    public void testCompute() {
        ReLU relu = new ReLU();
        
        // Test single values
        assertEquals(0.0f, relu.compute(-2.0f), EPSILON);
        assertEquals(0.0f, relu.compute(-1.0f), EPSILON);
        assertEquals(0.0f, relu.compute(-0.5f), EPSILON);
        assertEquals(0.0f, relu.compute(0.0f), EPSILON);
        assertEquals(0.5f, relu.compute(0.5f), EPSILON);
        assertEquals(1.0f, relu.compute(1.0f), EPSILON);
        assertEquals(2.0f, relu.compute(2.0f), EPSILON);
        
        // Test batch
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            0.0f, 0.0f, 0.0f, 1.0f, 2.0f,
            0.0f, 0.0f, 0.5f, 1.5f, 2.5f,
            0.0f, 0.0f, 0.3f, 0.7f, 1.7f,
            0.0f, 0.0f, 0.2f, 1.2f, 2.2f
        };
        
        float[] output = input.clone();
        relu.compute(output, 5, 4);
        assertArrayEquals(expected, output, EPSILON);
    }
    
    @Override
    public void testDerivative() {
        ReLU relu = new ReLU();
        
        // Test single value derivative
        assertEquals(0.0f, relu.derivative(-1.0f, 0.0f), EPSILON);
        assertEquals(0.0f, relu.derivative(0.0f, 0.0f), EPSILON);
        assertEquals(1.0f, relu.derivative(1.0f, 1.0f), EPSILON);
        
        // Test batch derivative
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = {
            0.0f, 0.0f, 0.0f, 1.0f, 2.0f,
            0.0f, 0.0f, 0.5f, 1.5f, 2.5f,
            0.0f, 0.0f, 0.3f, 0.7f, 1.7f,
            0.0f, 0.0f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f
        };
        
        float[] derivative = new float[20];
        relu.derivative(preActivation, postActivation, derivative, 5, 4);
        assertArrayEquals(expected, derivative, EPSILON);
    }
    
    @Override
    public void testGradient() {
        ReLU relu = new ReLU();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = {
            0.0f, 0.0f, 0.0f, 1.0f, 2.0f,
            0.0f, 0.0f, 0.5f, 1.5f, 2.5f,
            0.0f, 0.0f, 0.3f, 0.7f, 1.7f,
            0.0f, 0.0f, 0.2f, 1.2f, 2.2f
        };
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] expected = {
            0.0f, 0.0f, 0.0f, -2.0f, 1.5f,
            0.0f, 0.0f, -1.5f, 0.7f, -0.3f,
            0.0f, 0.0f, 0.4f, -0.9f, 1.1f,
            0.0f, 0.0f, -0.2f, 1.4f, -1.7f
        };
        
        gradient = gradient.clone();
        relu.gradient(preActivation, postActivation, gradient, 5, 4);
        assertArrayEquals(expected, gradient, EPSILON);
    }
    
    @Override
    public void testEdgeCases() {
        ReLU relu = new ReLU();
        
        // Test extreme values
        assertEquals(0.0f, relu.compute(-1e10f), EPSILON);
        assertEquals(0.0f, relu.compute(-1000f), EPSILON);
        assertEquals(1000f, relu.compute(1000f), EPSILON);
        assertEquals(1e10f, relu.compute(1e10f), EPSILON);
        
        // Test zero and near-zero
        assertEquals(0.0f, relu.compute(0.0f), EPSILON);
        assertEquals(0.0f, relu.compute(-1e-10f), EPSILON);
        assertEquals(1e-10f, relu.compute(1e-10f), EPSILON);
    }
    
    @Override
    public void testSerialization() {
        ReLU relu = new ReLU();
        
        Map<String, Object> serialized = relu.serialize();
        assertNotNull(serialized);
        assertEquals("ReLU", serialized.get("type"));
        
        Activation deserialized = Activation.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof ReLU);
        
        // Test functionality
        assertEquals(0.0f, deserialized.compute(-1.0f), EPSILON);
        assertEquals(1.0f, deserialized.compute(1.0f), EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        ReLU relu = new ReLU();
        
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            0.0f, 0.0f, 0.0f, 1.0f, 2.0f,
            0.0f, 0.0f, 0.5f, 1.5f, 2.5f,
            0.0f, 0.0f, 0.3f, 0.7f, 1.7f,
            0.0f, 0.0f, 0.2f, 1.2f, 2.2f
        };
        
        CUdeviceptr gpuPtr = CudaUtil.toGPU(input);
        relu.computeGPU(gpuPtr, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, input.length);
        assertArrayEquals(expected, gpuOutput, EPSILON);
        CudaUtil.free(gpuPtr);
    }
    
    @Override
    public void testDerivativeGPU() {
        ReLU relu = new ReLU();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = {
            0.0f, 0.0f, 0.0f, 1.0f, 2.0f,
            0.0f, 0.0f, 0.5f, 1.5f, 2.5f,
            0.0f, 0.0f, 0.3f, 0.7f, 1.7f,
            0.0f, 0.0f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            0.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f, 1.0f
        };
        
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        relu.derivativeGPU(gpuPre, gpuPost, gpuDeriv, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuDeriv, 20);
        assertArrayEquals(expected, gpuOutput, EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuDeriv);
    }
    
    @Override
    public void testGradientGPU() {
        ReLU relu = new ReLU();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = {
            0.0f, 0.0f, 0.0f, 1.0f, 2.0f,
            0.0f, 0.0f, 0.5f, 1.5f, 2.5f,
            0.0f, 0.0f, 0.3f, 0.7f, 1.7f,
            0.0f, 0.0f, 0.2f, 1.2f, 2.2f
        };
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] expected = {
            0.0f, 0.0f, 0.0f, -2.0f, 1.5f,
            0.0f, 0.0f, -1.5f, 0.7f, -0.3f,
            0.0f, 0.0f, 0.4f, -0.9f, 1.1f,
            0.0f, 0.0f, -0.2f, 1.4f, -1.7f
        };
        
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuGrad = CudaUtil.toGPU(gradient);
        relu.gradientGPU(gpuPre, gpuPost, gpuGrad, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuGrad, gradient.length);
        assertArrayEquals(expected, gpuOutput, EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuGrad);
    }
    
    @Override
    public void testEdgeCasesGPU() {
        ReLU relu = new ReLU();
        
        float[] edgeValues = {
            -1e10f, -1000f, -1f, -1e-10f, 0.0f, 1e-10f, 1f, 1000f, 1e10f
        };
        
        float[] cpuOutput = edgeValues.clone();
        for (int i = 0; i < cpuOutput.length; i++) {
            cpuOutput[i] = relu.compute(edgeValues[i]);
        }
        
        CUdeviceptr gpuPtr = CudaUtil.toGPU(edgeValues);
        relu.computeGPU(gpuPtr, edgeValues.length, 1, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, edgeValues.length);
        
        assertArrayEquals(cpuOutput, gpuOutput, EPSILON);
        CudaUtil.free(gpuPtr);
    }
}