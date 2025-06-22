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
import jcuda.driver.CUstream;
import org.fjnn.loss.BinaryCrossEntropy;
import org.fjnn.loss.Loss;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;
import org.junit.Test;

/**
 * Unit test for BinaryCrossEntropy loss function.
 * 
 * Note: BinaryCrossEntropy.compute() returns a single float value (average loss over all elements),
 * while derivative() returns an array of gradients for each element.
 * 
 * @author ahmed
 */
public class BinaryCrossEntropyTest extends LossTest {
    
    @Override
    public void testCompute() {
        BinaryCrossEntropy bce = new BinaryCrossEntropy();
        
        // Test batch (4 batches, 5 outputs each)
        // Predictions should be probabilities in [0, 1]
        float[] predictions = {
            0.88080f, 0.26894f, 0.62246f, 0.11920f, 0.81757f,
            0.37754f, 0.73106f, 0.18243f, 0.66819f, 0.42556f,
            0.76852f, 0.31002f, 0.59868f, 0.15447f, 0.74974f,
            0.21417f, 0.64566f, 0.54983f, 0.80218f, 0.15447f
        };
        
        // Binary targets (0 or 1)
        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 1.0f, 1.0f, 0.0f
        };
        
        // Expected total loss averaged over all elements
        // BCE = -[t*log(p) + (1-t)*log(1-p)]
        float expectedLoss = 0.42785f;
        
        float loss = bce.compute(predictions, targets);
        assertEquals(expectedLoss, loss, EPSILON);
    }
    
    @Override
    @Test
    public void testDerivative() {
        BinaryCrossEntropy bce = new BinaryCrossEntropy();

        float[] predictions = {
            0.88080f, 0.26894f, 0.62246f, 0.11920f, 0.81757f,
            0.37754f, 0.73106f, 0.18243f, 0.66819f, 0.42556f,
            0.76852f, 0.31002f, 0.59868f, 0.15447f, 0.74974f,
            0.21417f, 0.64566f, 0.54983f, 0.80218f, 0.15447f
        };

        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 1.0f, 1.0f, 0.0f
        };

        float[] expected = {
            -1.13533f, 1.36788f, -1.60653f, 1.13533f, -1.22314f,
            1.60653f, -1.36788f, 1.22314f, -1.49658f, 1.74083f,
            -1.30120f, 1.44932f, 2.49178f, -6.47375f, -1.33380f,
            1.27254f, -1.54880f, -1.81874f, -1.24660f, 1.18269f
        };

        float[] derivatives = bce.derivative(predictions, targets);
        assertArrayEquals(expected, derivatives, EPSILON);
    }
    
    @Override
    public void testBatchReduction() {
        BinaryCrossEntropy bce = new BinaryCrossEntropy();
        
        float[] predictions = {
            0.9f, 0.1f, 0.8f, 0.2f,
            0.3f, 0.7f, 0.6f, 0.4f,
            0.95f, 0.05f, 0.85f, 0.15f,
            0.01f, 0.99f, 0.5f, 0.5f
        };
        
        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 1.0f, 1.0f
        };
        
        // Expected average loss
        float loss = bce.compute(predictions, targets);
        float expectedLoss = 0.31481f;
        assertEquals(expectedLoss, loss, EPSILON);
        
        // Test with different batch configurations
        float[] smallBatch = {0.9f, 0.1f};
        float[] smallTargets = {1.0f, 0.0f};
        float smallLoss = bce.compute(smallBatch, smallTargets);
        float expectedSmall = 0.10536f; // Average of -log(0.9) and -log(0.9)
        assertEquals(expectedSmall, smallLoss, EPSILON);
    }
    
    @Override
    public void testEdgeCases() {
        BinaryCrossEntropy bce = new BinaryCrossEntropy();
        
        // Test with extreme probabilities
        float[] extremeProbs = {
            0.00001f, 0.99999f, 0.5f, 1.0f,
            0.0f, 0.00001f, 0.99999f, 0.5f,
            0.1f, 0.9f, 0.0f, 1.0f
        };
        
        float[] targets = {
            1.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 0.0f
        };
        
        float loss = bce.compute(extremeProbs, targets);
        
        // Verify no NaN or Inf values
        assertFalse("Loss contains NaN", Float.isNaN(loss));
        assertFalse("Loss contains Inf", Float.isInfinite(loss));
        assertTrue("Loss should be non-negative", loss >= 0.0f);
        
        // Test derivative edge cases
        float[] derivatives = bce.derivative(extremeProbs, targets);
        for (float deriv : derivatives) {
            assertFalse("Derivative contains NaN", Float.isNaN(deriv));
            assertFalse("Derivative contains Inf", Float.isInfinite(deriv));
        }
    }
    
    @Override
    public void testSerialization() {
        BinaryCrossEntropy bce = new BinaryCrossEntropy();

        Map<String, Object> serialized = bce.serialize();
        assertNotNull(serialized);
        assertEquals("BinaryCrossEntropy", serialized.get("type"));
        assertEquals(1.0f, serialized.get("alpha"));
        assertEquals(1.0f, serialized.get("beta"));

        Loss deserialized = Loss.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof BinaryCrossEntropy);

        // Test functionality
        float[] predictions = {0.7f, 0.3f, 0.9f, 0.1f};
        float[] targets = {1.0f, 0.0f, 1.0f, 0.0f};

        float originalLoss = bce.compute(predictions, targets);
        float deserializedLoss = deserialized.compute(predictions, targets);

        assertEquals(originalLoss, deserializedLoss, EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        BinaryCrossEntropy bce = new BinaryCrossEntropy();
        
        float[] predictions = {
            0.88080f, 0.26894f, 0.62246f, 0.11920f, 0.81757f,
            0.37754f, 0.73106f, 0.18243f, 0.66819f, 0.42556f,
            0.76852f, 0.31002f, 0.59868f, 0.15447f, 0.74974f,
            0.21417f, 0.64566f, 0.54983f, 0.80218f, 0.15447f
        };
        
        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 1.0f, 1.0f, 0.0f
        };
        
        // CPU computation
        float cpuLoss = bce.compute(predictions, targets);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        bce.computeGPU(gpuPred, gpuTarget, gpuResult, 20, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gpuLoss = CudaUtil.fromGPUFloat(gpuResult, 1);
        
        assertEquals(cpuLoss, gpuLoss[0], RELAXED_EPSILON);
        
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuResult);
        CudaUtil.freeStream(stream);
    }
    
    @Override
    public void testDerivativeGPU() {
        BinaryCrossEntropy bce = new BinaryCrossEntropy();
        
        float[] predictions = {
            0.88080f, 0.26894f, 0.62246f, 0.11920f, 0.81757f,
            0.37754f, 0.73106f, 0.18243f, 0.66819f, 0.42556f,
            0.76852f, 0.31002f, 0.59868f, 0.15447f, 0.74974f,
            0.21417f, 0.64566f, 0.54983f, 0.80218f, 0.15447f
        };
        
        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 1.0f, 1.0f, 1.0f, 0.0f
        };
        
        // CPU computation
        float[] cpuDerivatives = bce.derivative(predictions, targets);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        bce.derivativeGPU(gpuPred, gpuTarget, gpuDeriv, 20, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gpuDerivatives = CudaUtil.fromGPUFloat(gpuDeriv, 20);
        
        assertArrayEquals(cpuDerivatives, gpuDerivatives, RELAXED_EPSILON);
        
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuDeriv);
        CudaUtil.freeStream(stream);
    }
    
    @Override
    public void testBatchReductionGPU() {
        BinaryCrossEntropy bce = new BinaryCrossEntropy();
        
        float[] predictions = {
            0.9f, 0.1f, 0.8f, 0.2f,
            0.3f, 0.7f, 0.6f, 0.4f,
            0.95f, 0.05f, 0.85f, 0.15f,
            0.01f, 0.99f, 0.5f, 0.5f
        };
        
        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 1.0f, 1.0f
        };
        
        // CPU computation
        float cpuLoss = bce.compute(predictions, targets);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        bce.computeGPU(gpuPred, gpuTarget, gpuResult, 16, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gpuLoss = CudaUtil.fromGPUFloat(gpuResult, 1);
        
        assertEquals(cpuLoss, gpuLoss[0], RELAXED_EPSILON);
        
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuResult);
        CudaUtil.freeStream(stream);
    }
    
    @Override
    public void testEdgeCasesGPU() {
        BinaryCrossEntropy bce = new BinaryCrossEntropy();
        
        float[] extremeProbs = {
            0.00001f, 0.99999f, 0.5f, 1.0f,
            0.0f, 0.00001f, 0.99999f, 0.5f,
            0.1f, 0.9f, 0.0f, 1.0f
        };
        
        float[] targets = {
            1.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 0.0f
        };
        
        // CPU computation
        float cpuLoss = bce.compute(extremeProbs, targets);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(extremeProbs);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        bce.computeGPU(gpuPred, gpuTarget, gpuResult, 12, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gpuLoss = CudaUtil.fromGPUFloat(gpuResult, 1);
        
        // Verify results match and contain no invalid values
        assertFalse("GPU result contains NaN", Float.isNaN(gpuLoss[0]));
        assertFalse("GPU result contains Inf", Float.isInfinite(gpuLoss[0]));
        assertEquals(cpuLoss, gpuLoss[0], RELAXED_EPSILON);
        
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuResult);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testWeightedBinaryCrossEntropy() {
        // Test with different positive/negative weights
        BinaryCrossEntropy weightedBce = new BinaryCrossEntropy(2.0f, 1.0f);

        float[] predictions = {
            0.8f, 0.3f, 0.9f, 0.2f,
            0.7f, 0.4f, 0.6f, 0.1f
        };

        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 1.0f, 1.0f
        };

        float loss = weightedBce.compute(predictions, targets);

        // Expected weighted losses for each element:
        // Pos 0: target=1, weight=2.0, loss = 2.0 * -log(0.8) = 2.0 * 0.22314 = 0.44629
        // Pos 1: target=0, weight=1.0, loss = 1.0 * -log(1-0.3) = 1.0 * -log(0.7) = 0.35667
        // Pos 2: target=1, weight=2.0, loss = 2.0 * -log(0.9) = 2.0 * 0.10536 = 0.21072
        // Pos 3: target=0, weight=1.0, loss = 1.0 * -log(1-0.2) = 1.0 * -log(0.8) = 0.22314
        // Pos 4: target=1, weight=2.0, loss = 2.0 * -log(0.7) = 2.0 * 0.35667 = 0.71334
        // Pos 5: target=0, weight=1.0, loss = 1.0 * -log(1-0.4) = 1.0 * -log(0.6) = 0.51083
        // Pos 6: target=1, weight=2.0, loss = 2.0 * -log(0.6) = 2.0 * 0.51083 = 1.02166
        // Pos 7: target=1, weight=2.0, loss = 2.0 * -log(0.1) = 2.0 * 2.30259 = 4.60518
        // Total = 8.08783, Average = 8.08783 / 8 = 1.01098
        float expectedLoss = 1.01098f;

        assertEquals(expectedLoss, loss, EPSILON);

        // Test derivative with weights
        float[] derivatives = weightedBce.derivative(predictions, targets);

        // For BCE derivative with weights: weight * (prediction - target) / (prediction * (1 - prediction))
        // Pos 0: 2.0 * (0.8-1.0) / (0.8*0.2) = 2.0 * (-0.2) / 0.16 = -2.5
        // Pos 1: 1.0 * (0.3-0.0) / (0.3*0.7) = 0.3 / 0.21 = 1.42857
        // Pos 2: 2.0 * (0.9-1.0) / (0.9*0.1) = 2.0 * (-0.1) / 0.09 = -2.22222
        // Pos 3: 1.0 * (0.2-0.0) / (0.2*0.8) = 0.2 / 0.16 = 1.25
        // Pos 4: 2.0 * (0.7-1.0) / (0.7*0.3) = 2.0 * (-0.3) / 0.21 = -2.85714
        // Pos 5: 1.0 * (0.4-0.0) / (0.4*0.6) = 0.4 / 0.24 = 1.66667
        // Pos 6: 2.0 * (0.6-1.0) / (0.6*0.4) = 2.0 * (-0.4) / 0.24 = -3.33333
        // Pos 7: 2.0 * (0.1-1.0) / (0.1*0.9) = 2.0 * (-0.9) / 0.09 = -20.0
        float[] expectedDerivatives = {
            -2.50000f, 1.42857f, -2.22222f, 1.25000f,
            -2.85714f, 1.66667f, -3.33333f, -20.00000f
        };

        assertArrayEquals(expectedDerivatives, derivatives, EPSILON);
}
    
    @Test
    public void testWeightedBinaryCrossEntropyGPU() {
        BinaryCrossEntropy weightedBce = new BinaryCrossEntropy(2.0f, 1.0f);
        
        float[] predictions = {
            0.8f, 0.3f, 0.9f, 0.2f,
            0.7f, 0.4f, 0.6f, 0.1f
        };
        
        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f,
            1.0f, 0.0f, 1.0f, 1.0f
        };
        
        // CPU computation
        float cpuLoss = weightedBce.compute(predictions, targets);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        weightedBce.computeGPU(gpuPred, gpuTarget, gpuResult, 8, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gpuLoss = CudaUtil.fromGPUFloat(gpuResult, 1);
        
        assertEquals(cpuLoss, gpuLoss[0], RELAXED_EPSILON);
        
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuResult);
        CudaUtil.freeStream(stream);
    }
}
