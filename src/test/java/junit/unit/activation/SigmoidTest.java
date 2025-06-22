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
import org.fjnn.activation.Sigmoid;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;
import org.fjnn.loss.BinaryCrossEntropy;
import org.junit.Test;

/**
 * Unit test for Sigmoid activation function.
 * 
 * @author ahmed
 */
public class SigmoidTest extends ActivationTest {
    
    @Override
    public void testCompute() {
        Sigmoid sigmoid = new Sigmoid();
        
        // Test batch (4 batches, 5 units each)
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] expected = {
            0.11920f, 0.26894f, 0.50000f, 0.73106f, 0.88080f,
            0.18243f, 0.37754f, 0.62246f, 0.81757f, 0.92414f,
            0.33181f, 0.42556f, 0.57444f, 0.66819f, 0.84553f,
            0.07586f, 0.15447f, 0.54983f, 0.76852f, 0.90025f
        };
        
        float[] output = input.clone();
        sigmoid.compute(output, 5, 4);
        assertArrayEquals(expected, output, EPSILON);
    }
    
    @Override
    public void testDerivative() {
        Sigmoid sigmoid = new Sigmoid();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        // Compute actual sigmoid values
        float[] postActivation = preActivation.clone();
        sigmoid.compute(postActivation, 5, 4);
        
        // Compute derivative
        float[] derivative = new float[20];
        sigmoid.derivative(preActivation, postActivation, derivative, 5, 4);
        
        // Verify derivative = sigmoid * (1 - sigmoid)
        for (int i = 0; i < 20; i++) {
            float expected = postActivation[i] * (1.0f - postActivation[i]);
            assertEquals(expected, derivative[i], EPSILON);
        }
    }
    
    @Override
    public void testGradient() {
        Sigmoid sigmoid = new Sigmoid();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        sigmoid.compute(postActivation, 5, 4);
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] derivative = new float[20];
        sigmoid.derivative(preActivation, postActivation, derivative, 5, 4);
        
        float[] expected = gradient.clone();
        for (int i = 0; i < 20; i++) {
            expected[i] *= derivative[i];
        }
        
        gradient = gradient.clone();
        sigmoid.gradient(preActivation, postActivation, gradient, 5, 4);
        assertArrayEquals(expected, gradient, EPSILON);
    }
    
    @Override
    public void testEdgeCases() {
        Sigmoid sigmoid = new Sigmoid();
        
        float[] edgeValues = {
            -100f, -50f, -10f, -1f,
            0f, 1f, 10f, 50f,
            100f, -0.0001f, 0.0001f, 1000f
        };
        
        float[] output = edgeValues.clone();
        sigmoid.compute(output, 4, 3);
        
        // Verify all outputs are in [0, 1]
        for (float v : output) {
            assertTrue(v >= 0.0f && v <= 1.0f);
        }
        
        // Test saturation
        assertTrue(output[0] < 0.01f); // sigmoid(-100) ≈ 0
        assertTrue(output[8] > 0.99f);  // sigmoid(100) ≈ 1
        assertEquals(0.5f, output[4], EPSILON); // sigmoid(0) = 0.5
    }
    
    @Override
    public void testSerialization() {
        Sigmoid sigmoid = new Sigmoid();
        
        Map<String, Object> serialized = sigmoid.serialize();
        assertNotNull(serialized);
        assertEquals("Sigmoid", serialized.get("type"));
        
        Activation deserialized = Activation.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof Sigmoid);
        
        // Test functionality
        float[] input = {-1.0f, 0.0f, 1.0f, 2.0f};
        float[] cpuOutput = input.clone();
        sigmoid.compute(cpuOutput, 4, 1);
        
        float[] deserializedOutput = input.clone();
        deserialized.compute(deserializedOutput, 4, 1);
        
        assertArrayEquals(cpuOutput, deserializedOutput, EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        Sigmoid sigmoid = new Sigmoid();
        
        float[] input = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        // CPU computation
        float[] cpuOutput = input.clone();
        sigmoid.compute(cpuOutput, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(input);
        sigmoid.computeGPU(gpuPtr, 5, 4, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, input.length);
        
        assertArrayEquals(cpuOutput, gpuOutput, RELAXED_EPSILON);
        CudaUtil.free(gpuPtr);
    }
    
    @Override
    public void testDerivativeGPU() {
        Sigmoid sigmoid = new Sigmoid();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        sigmoid.compute(postActivation, 5, 4);
        
        // CPU computation
        float[] cpuDerivative = new float[20];
        sigmoid.derivative(preActivation, postActivation, cpuDerivative, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        sigmoid.derivativeGPU(gpuPre, gpuPost, gpuDeriv, 5, 4, null);
        float[] gpuDerivative = CudaUtil.fromGPUFloat(gpuDeriv, 20);
        
        assertArrayEquals(cpuDerivative, gpuDerivative, RELAXED_EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuDeriv);
    }
    
    @Override
    public void testGradientGPU() {
        Sigmoid sigmoid = new Sigmoid();
        
        float[] preActivation = {
            -2.0f, -1.0f, 0.0f, 1.0f, 2.0f,
            -1.5f, -0.5f, 0.5f, 1.5f, 2.5f,
            -0.7f, -0.3f, 0.3f, 0.7f, 1.7f,
            -2.5f, -1.7f, 0.2f, 1.2f, 2.2f
        };
        
        float[] postActivation = preActivation.clone();
        sigmoid.compute(postActivation, 5, 4);
        
        float[] gradient = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        // CPU computation
        float[] cpuGradient = gradient.clone();
        sigmoid.gradient(preActivation, postActivation, cpuGradient, 5, 4);
        
        // GPU computation
        CUdeviceptr gpuPre = CudaUtil.toGPU(preActivation);
        CUdeviceptr gpuPost = CudaUtil.toGPU(postActivation);
        CUdeviceptr gpuGrad = CudaUtil.toGPU(gradient);
        sigmoid.gradientGPU(gpuPre, gpuPost, gpuGrad, 5, 4, null);
        float[] gpuGradient = CudaUtil.fromGPUFloat(gpuGrad, gradient.length);
        
        assertArrayEquals(cpuGradient, gpuGradient, RELAXED_EPSILON);
        CudaUtil.free(gpuPre);
        CudaUtil.free(gpuPost);
        CudaUtil.free(gpuGrad);
    }
    
    @Override
    public void testEdgeCasesGPU() {
        Sigmoid sigmoid = new Sigmoid();
        
        float[] edgeValues = {
            -100f, -50f, -10f, -1f,
            0f, 1f, 10f, 50f,
            100f, -0.0001f, 0.0001f, 1000f
        };
        
        // CPU computation
        float[] cpuOutput = edgeValues.clone();
        sigmoid.compute(cpuOutput, 4, 3);
        
        // GPU computation
        CUdeviceptr gpuPtr = CudaUtil.toGPU(edgeValues);
        sigmoid.computeGPU(gpuPtr, 4, 3, null);
        float[] gpuOutput = CudaUtil.fromGPUFloat(gpuPtr, edgeValues.length);
        
        assertArrayEquals(cpuOutput, gpuOutput, RELAXED_EPSILON);
        CudaUtil.free(gpuPtr);
    }
    
    @Test
    public void testBinaryCrossEntropyGradient() {
        Sigmoid sigmoid = new Sigmoid();
        float[] logits = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 1.0f, 1.0f, 0.0f
        };
        // Apply sigmoid
        float[] sigmoidOutput = logits.clone();
        sigmoid.compute(sigmoidOutput, 5, 4);
        // Test fused gradient computation with alpha=2.0, beta=1.0
        float[] gradient = new float[20];
        sigmoid.gradientBinaryCrossEntropy(sigmoidOutput, targets, gradient, 2.0f, 1.0f, 5, 4);
        // Expected: weight * (sigmoid - target) where weight = alpha for target=1, beta for target=0
        float[] expected = {
            -0.23840f, 0.26894f, -0.75508f, 0.11920f, -0.36486f,
            0.37754f, -0.53788f, 0.18243f, -0.66362f, 0.42556f,
            -0.46296f, 0.31002f, 0.59868f, -1.42190f, -0.49948f,
            0.21417f, -0.70868f, -1.09966f, -0.39564f, 0.15447f
        };
        assertArrayEquals(expected, gradient, EPSILON);
    }

    @Test
    public void testBinaryCrossEntropyGradientGPU() {
        Sigmoid sigmoid = new Sigmoid();

        float[] logits = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };

        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 1.0f, 1.0f, 0.0f
        };

        // Apply sigmoid on CPU
        float[] sigmoidOutput = logits.clone();
        sigmoid.compute(sigmoidOutput, 5, 4);

        // Compute CPU gradient
        float[] cpuGradient = new float[20];
        sigmoid.gradientBinaryCrossEntropy(sigmoidOutput, targets, cpuGradient, 2.0f, 1.0f, 5, 4);

        // Test GPU fused gradient
        CUdeviceptr gpuSigmoid = CudaUtil.toGPU(sigmoidOutput);
        CUdeviceptr gpuTargets = CudaUtil.toGPU(targets);
        CUdeviceptr gpuGradient = CudaUtil.toGPU(new float[20]);
        sigmoid.gradientBinaryCrossEntropyGPU(gpuSigmoid, gpuTargets, gpuGradient, 2.0f, 1.0f, 5, 4, null);
        float[] gpuResult = CudaUtil.fromGPUFloat(gpuGradient, 20);

        assertArrayEquals(cpuGradient, gpuResult, RELAXED_EPSILON);

        CudaUtil.free(gpuSigmoid);
        CudaUtil.free(gpuTargets);
        CudaUtil.free(gpuGradient);
    }

    @Test
    public void testBinaryCrossEntropyFusedVsSeparate() {
        Sigmoid sigmoid = new Sigmoid();
        BinaryCrossEntropy bce = new BinaryCrossEntropy(2.0f, 1.0f);

        float[] logits = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };

        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 1.0f, 1.0f, 0.0f
        };

        // Method 1: Separate operations
        float[] sigmoidOutput = logits.clone();
        sigmoid.compute(sigmoidOutput, 5, 4);

        float[] gradients = bce.derivative(sigmoidOutput, targets);
        sigmoid.gradient(logits, sigmoidOutput, gradients, 5, 4);

        // Method 2: Fused operation
        float[] fusedGradient = new float[20];
        sigmoid.gradientBinaryCrossEntropy(sigmoidOutput, targets, fusedGradient, 2.0f, 1.0f, 5, 4);

        assertArrayEquals(gradients, fusedGradient, EPSILON);
    }

    @Test
    public void testBinaryCrossEntropyFusedVsSeparateGPU() {
        Sigmoid sigmoid = new Sigmoid();
        BinaryCrossEntropy bce = new BinaryCrossEntropy(2.0f, 1.0f);

        float[] logits = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };

        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 1.0f, 1.0f, 0.0f
        };

        // Method 1: Separate GPU operations
        CUdeviceptr gpuLogits = CudaUtil.toGPU(logits);
        CUdeviceptr gpuTargets = CudaUtil.toGPU(targets);
        CUdeviceptr gpuSigmoid = CudaUtil.toGPU(logits.clone());

        sigmoid.computeGPU(gpuSigmoid, 5, 4, null);

        CUdeviceptr gpuBCEDerivative = CudaUtil.toGPU(new float[20]);
        bce.derivativeGPU(gpuSigmoid, gpuTargets, gpuBCEDerivative, 20, null);
        sigmoid.gradientGPU(gpuLogits, gpuSigmoid, gpuBCEDerivative, 5, 4, null);

        // Method 2: Fused GPU operation
        CUdeviceptr gpuFusedGradient = CudaUtil.toGPU(new float[20]);
        sigmoid.gradientBinaryCrossEntropyGPU(gpuSigmoid, gpuTargets, gpuFusedGradient, 2.0f, 1.0f, 5, 4, null);

        // Compare results
        float[] separateResult = CudaUtil.fromGPUFloat(gpuBCEDerivative, 20);
        float[] fusedResult = CudaUtil.fromGPUFloat(gpuFusedGradient, 20);

        assertArrayEquals(separateResult, fusedResult, RELAXED_EPSILON);

        CudaUtil.free(gpuLogits);
        CudaUtil.free(gpuTargets);
        CudaUtil.free(gpuSigmoid);
        CudaUtil.free(gpuBCEDerivative);
        CudaUtil.free(gpuFusedGradient);
    }
}