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
package junit.unit.loss;

import jcuda.driver.CUdeviceptr;
import org.fjnn.loss.CrossEntropy;
import org.fjnn.loss.Loss;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.activation.SoftMax;
import org.junit.Test;

/**
 * Unit test for CrossEntropy loss function.
 * 
 * @author ahmed
 */
public class CrossEntropyTest extends LossTest {
    
    @Override
    public void testCompute() {
        CrossEntropy ce = new CrossEntropy();
        
        // Test batch (4 batches, 5 classes each)
        // These should be softmax outputs (probabilities that sum to 1)
        float[] predictions = {
            0.03106f, 0.08444f, 0.22953f, 0.62391f, 0.03106f,
            0.09081f, 0.03341f, 0.67099f, 0.14972f, 0.05507f,
            0.20000f, 0.20000f, 0.20000f, 0.20000f, 0.20000f,
            0.00029f, 0.00214f, 0.01584f, 0.11702f, 0.86470f
        };
        
        // One-hot encoded targets
        float[] targets = {
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // class 2
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // class 0
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // class 4
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f   // class 3
        };
        
        // Expected: sum of -t_i * log(y_i) / total_elements
        // Only non-zero targets contribute:
        // -1.0 * log(0.22953) = 1.47209
        // -1.0 * log(0.09081) = 2.39894
        // -1.0 * log(0.20000) = 1.60944
        // -1.0 * log(0.11702) = 2.14507
        // Total = 7.62554, Average = 7.62554 / 20 = 0.38128
        float expectedLoss = 0.38128f;
        
        float loss = ce.compute(predictions, targets);
        assertEquals(expectedLoss, loss, EPSILON);
    }
    
    @Override
    public void testDerivative() {
        CrossEntropy ce = new CrossEntropy();
        
        float[] predictions = {
            0.03106f, 0.08444f, 0.22953f, 0.62391f, 0.03106f,
            0.09081f, 0.03341f, 0.67099f, 0.14972f, 0.05507f,
            0.20000f, 0.20000f, 0.20000f, 0.20000f, 0.20000f,
            0.00029f, 0.00214f, 0.01584f, 0.11702f, 0.86470f
        };
        
        float[] targets = {
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f
        };
        
        // For CrossEntropy, derivative = -t_i / y_i (clipped)
        // Only non-zero targets have non-zero derivatives
        float[] expected = {
            0.0f, 0.0f, -4.35672f, 0.0f, 0.0f,
            -11.012f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, -5.00000f,
            0.0f, 0.0f, 0.0f, -8.54554f, 0.0f
        };
        
        float[] derivatives = ce.derivative(predictions, targets);
        assertArrayEquals(expected, derivatives, EPSILON);
    }
    
    @Override
    public void testBatchReduction() {
        CrossEntropy ce = new CrossEntropy();
        
        // Test different batch sizes
        float[] predictions1 = {
            0.25f, 0.25f, 0.25f, 0.25f
        };
        float[] targets1 = {
            1.0f, 0.0f, 0.0f, 0.0f  // class 0
        };
        float loss1 = ce.compute(predictions1, targets1);
        float expected1 = 1.38629f / 4.0f;  // -log(0.25) / 4
        assertEquals(expected1, loss1, EPSILON);
        
        // Test with multiple samples
        float[] predictions2 = {
            0.25f, 0.25f, 0.25f, 0.25f,
            0.10f, 0.20f, 0.30f, 0.40f,
            0.70f, 0.10f, 0.10f, 0.10f,
            0.05f, 0.05f, 0.05f, 0.85f
        };
        
        float[] targets2 = {
            1.0f, 0.0f, 0.0f, 0.0f,  // class 0
            0.0f, 0.0f, 1.0f, 0.0f,  // class 2
            1.0f, 0.0f, 0.0f, 0.0f,  // class 0
            0.0f, 0.0f, 0.0f, 1.0f   // class 3
        };
        
        float loss2 = ce.compute(predictions2, targets2);
        // Sum: -log(0.25) - log(0.30) - log(0.70) - log(0.85)
        // = 1.38629 + 1.20397 + 0.35667 + 0.16252 = 3.10945
        float expected2 = 3.10945f / 16.0f;
        assertEquals(expected2, loss2, EPSILON);
    }
    
    @Override
    public void testEdgeCases() {
        CrossEntropy ce = new CrossEntropy();
        
        // Test with very small probabilities (near 0)
        float[] smallProbs = {
            0.00001f, 0.49999f, 0.49999f, 0.00001f,
            0.99998f, 0.00001f, 0.00000f, 0.00001f,
            0.33333f, 0.33333f, 0.33334f, 0.00000f
        };
        
        float[] targets = {
            1.0f, 0.0f, 0.0f, 0.0f,  // class 0 with tiny probability
            1.0f, 0.0f, 0.0f, 0.0f,  // class 0 with high probability
            0.0f, 0.0f, 0.0f, 1.0f   // class 3 with zero probability
        };
        
        float loss = ce.compute(smallProbs, targets);
        
        // Loss should be finite due to epsilon clamping
        assertFalse(Float.isInfinite(loss));
        assertFalse(Float.isNaN(loss));
        
        // Expected calculation with epsilon clamping:
        // -log(max(1e-7, 0.00001)) - log(max(1e-7, 0.99998)) - log(max(1e-7, 0.00000))
        // = -log(0.00001) - log(0.99998) - log(1e-7)
        // = 11.51293 + 0.00002 + 16.11810 = 27.63105 / 12
        float expectedLoss = 2.30259f;
        assertEquals(expectedLoss, loss, 0.001f);
        
        // Test derivative with edge cases
        float[] derivatives = ce.derivative(smallProbs, targets);
        
        // Check that derivatives are finite
        for (float deriv : derivatives) {
            assertFalse(Float.isNaN(deriv));
            assertFalse(Float.isInfinite(deriv));
        }
    }
    
    @Override
    public void testSerialization() {
        CrossEntropy ce = new CrossEntropy();
        
        Map<String, Object> serialized = ce.serialize();
        assertNotNull(serialized);
        assertEquals("CrossEntropy", serialized.get("type"));
        
        Loss deserialized = Loss.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof CrossEntropy);
        
        // Test functionality
        float[] predictions = {
            0.25f, 0.25f, 0.25f, 0.25f,
            0.10f, 0.20f, 0.30f, 0.40f
        };
        float[] targets = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f
        };
        
        float originalLoss = ce.compute(predictions, targets);
        float deserializedLoss = deserialized.compute(predictions, targets);
        
        assertEquals(originalLoss, deserializedLoss, EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        CrossEntropy ce = new CrossEntropy();
        
        float[] predictions = {
            0.03106f, 0.08444f, 0.22953f, 0.62391f, 0.03106f,
            0.09081f, 0.03341f, 0.67099f, 0.14972f, 0.05507f,
            0.20000f, 0.20000f, 0.20000f, 0.20000f, 0.20000f,
            0.00029f, 0.00214f, 0.01584f, 0.11702f, 0.86470f
        };
        
        float[] targets = {
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f
        };
        
        // CPU computation
        float cpuLoss = ce.compute(predictions, targets);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        ce.computeGPU(gpuPred, gpuTarget, gpuResult, 20, stream);
        float[] gpuLoss = CudaUtil.fromGPUFloatAsync(gpuResult, 1, stream);
        JCudaDriver.cuStreamSynchronize(stream);
        
        assertEquals(cpuLoss, gpuLoss[0], RELAXED_EPSILON);
        
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuResult);
        CudaUtil.freeStream(stream);
    }
    
    @Override
    public void testDerivativeGPU() {
        CrossEntropy ce = new CrossEntropy();
        
        float[] predictions = {
            0.03106f, 0.08444f, 0.22953f, 0.62391f, 0.03106f,
            0.09081f, 0.03341f, 0.67099f, 0.14972f, 0.05507f,
            0.20000f, 0.20000f, 0.20000f, 0.20000f, 0.20000f,
            0.00029f, 0.00214f, 0.01584f, 0.11702f, 0.86470f
        };
        
        float[] targets = {
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f
        };
        
        // CPU computation
        float[] cpuDerivatives = ce.derivative(predictions, targets);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        ce.derivativeGPU(gpuPred, gpuTarget, gpuDeriv, 20, stream);
        float[] gpuDerivatives = CudaUtil.fromGPUFloatAsync(gpuDeriv, 20, stream);
        JCudaDriver.cuStreamSynchronize(stream);
        
        assertArrayEquals(cpuDerivatives, gpuDerivatives, RELAXED_EPSILON);
        
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuDeriv);
        CudaUtil.freeStream(stream);
    }
    
    @Override
    public void testBatchReductionGPU() {
        CrossEntropy ce = new CrossEntropy();
        
        float[] predictions = {
            0.25f, 0.25f, 0.25f, 0.25f,
            0.10f, 0.20f, 0.30f, 0.40f,
            0.70f, 0.10f, 0.10f, 0.10f,
            0.05f, 0.05f, 0.05f, 0.85f
        };
        
        float[] targets = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f
        };
        
        // CPU computation
        float cpuLoss = ce.compute(predictions, targets);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        ce.computeGPU(gpuPred, gpuTarget, gpuResult, 16, stream);
        float[] gpuLoss = CudaUtil.fromGPUFloatAsync(gpuResult, 1, stream);
        JCudaDriver.cuStreamSynchronize(stream);
        
        assertEquals(cpuLoss, gpuLoss[0], RELAXED_EPSILON);
        
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuResult);
        CudaUtil.freeStream(stream);
    }
    
    @Override
    public void testEdgeCasesGPU() {
        CrossEntropy ce = new CrossEntropy();
        
        // Test with extreme values
        float[] extremeProbs = {
            0.00001f, 0.49999f, 0.49999f, 0.00001f,
            0.99998f, 0.00001f, 0.00000f, 0.00001f,
            0.33333f, 0.33333f, 0.33334f, 0.00000f,
            1.00000f, 0.00000f, 0.00000f, 0.00000f
        };
        
        float[] targets = {
            1.0f, 0.0f, 0.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 0.0f
        };
        
        // CPU computation - returns single loss value
        float cpuLoss = ce.compute(extremeProbs, targets);
        
        // GPU computation - also returns single loss value
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(extremeProbs);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        ce.computeGPU(gpuPred, gpuTarget, gpuResult, 16, stream);
        float[] gpuLoss = CudaUtil.fromGPUFloatAsync(gpuResult, 1, stream);
        JCudaDriver.cuStreamSynchronize(stream);
        
        // Verify no NaN or Inf values
        assertFalse("GPU result contains NaN", Float.isNaN(gpuLoss[0]));
        assertFalse("GPU result contains Inf", Float.isInfinite(gpuLoss[0]));
        assertFalse("CPU result contains NaN", Float.isNaN(cpuLoss));
        assertFalse("CPU result contains Inf", Float.isInfinite(cpuLoss));
        
        // Compare CPU and GPU results
        assertEquals(cpuLoss, gpuLoss[0], RELAXED_EPSILON);
        
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuResult);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testFusedGradient() {
        CrossEntropy ce = new CrossEntropy();
        SoftMax softmax = new SoftMax();
        
        // Test that we can fuse with SoftMax
        assertTrue(ce.canFuseWith(softmax));
        
        float[] logits = {
            2.0f, -1.0f, 0.5f, -2.0f, 1.5f,
            -0.5f, 1.0f, -1.5f, 0.7f, -0.3f,
            1.2f, -0.8f, 0.4f, -0.9f, 1.1f,
            -1.3f, 0.6f, -0.2f, 1.4f, -1.7f
        };
        
        float[] targets = {
            0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // class 2
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // class 0
            0.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // class 4
            0.0f, 0.0f, 0.0f, 1.0f, 0.0f   // class 3
        };
        
        // Compute softmax
        float[] softmaxOutput = logits.clone();
        softmax.compute(softmaxOutput, 5, 4);
        
        // Test fused gradient
        float[] fusedGradient = new float[20];
        ce.fusedGradient(softmaxOutput, targets, fusedGradient, softmax, 5, 4);
        
        // Fused gradient should be softmax - targets
        float[] expected = {
            0.52693f, 0.02624f, -0.88243f, 0.00965f, 0.31960f,
            -0.90376f, 0.43130f, 0.03540f, 0.31951f, 0.11754f,
            0.38285f, 0.05181f, 0.17203f, 0.04689f, -0.65358f,
            0.03810f, 0.25480f, 0.11449f, -0.43294f, 0.02554f
        };
        
        assertArrayEquals(expected, fusedGradient, EPSILON);
    }
    
    @Test
    public void testFusedGradientGPU() {
        CrossEntropy ce = new CrossEntropy();
        SoftMax softmax = new SoftMax();
        
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
        
        // Compute softmax on CPU for comparison
        float[] softmaxOutput = logits.clone();
        softmax.compute(softmaxOutput, 5, 4);
        
        // CPU fused gradient
        float[] cpuGradient = new float[20];
        ce.fusedGradient(softmaxOutput, targets, cpuGradient, softmax, 5, 4);
        
        // GPU fused gradient
        CUdeviceptr gpuSoftmax = CudaUtil.toGPU(softmaxOutput);
        CUdeviceptr gpuTargets = CudaUtil.toGPU(targets);
        CUdeviceptr gpuGradient = CudaUtil.toGPU(new float[20]);
        ce.fusedGradientGPU(gpuSoftmax, gpuTargets, gpuGradient, softmax, 5, 4, null);
        float[] gpuResult = CudaUtil.fromGPUFloat(gpuGradient, 20);
        
        assertArrayEquals(cpuGradient, gpuResult, RELAXED_EPSILON);
        
        CudaUtil.free(gpuSoftmax);
        CudaUtil.free(gpuTargets);
        CudaUtil.free(gpuGradient);
    }
}