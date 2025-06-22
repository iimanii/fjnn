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
import org.fjnn.loss.WeightedMeanSquareError;
import org.fjnn.loss.Loss;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;
import org.junit.Test;

/**
 * Unit test for WeightedMeanSquareError loss function.
 * 
 * @author ahmed
 */
public class WeightedMeanSquareErrorTest extends LossTest {
    
    @Override
    public void testCompute() {
        float[] predictions = {
            0.5f, -1.2f, 2.3f, -0.8f, 1.5f,
            -0.3f, 1.7f, -2.1f, 0.9f, -1.4f,
            2.2f, -0.5f, 0.7f, -1.9f, 1.1f,
            -1.3f, 0.4f, -0.2f, 1.8f, -0.7f
        };

        float[] targets = {
            0.8f, -1.0f, 2.0f, -1.0f, 1.8f,
            -0.5f, 2.0f, -2.0f, 1.0f, -1.0f,
            2.0f, -0.8f, 1.0f, -2.0f, 1.0f,
            -1.0f, 0.5f, 0.0f, 2.0f, -1.0f
        };
        
        float[] weights = {
            2.0f, 1.0f, 1.5f, 0.5f, 3.0f,
            1.0f, 2.5f, 0.8f, 1.2f, 0.7f,
            1.5f, 2.0f, 1.0f, 0.6f, 2.2f,
            0.9f, 1.8f, 1.1f, 2.0f, 1.3f
        };
        
        CUdeviceptr weightsGPU = CudaUtil.toGPU(weights);
        WeightedMeanSquareError wmse = new WeightedMeanSquareError(weights, weightsGPU);
        
        float expectedLoss = 0.087f;
        float loss = wmse.compute(predictions, targets);
        assertEquals(expectedLoss, loss, EPSILON);
        
        CudaUtil.free(weightsGPU);
    }
    
    @Override
    public void testDerivative() {
        float[] predictions = {
            0.5f, -1.2f, 2.3f, -0.8f, 1.5f,
            -0.3f, 1.7f, -2.1f, 0.9f, -1.4f,
            2.2f, -0.5f, 0.7f, -1.9f, 1.1f,
            -1.3f, 0.4f, -0.2f, 1.8f, -0.7f
        };
        
        float[] targets = {
            0.8f, -1.0f, 2.0f, -1.0f, 1.8f,
            -0.5f, 2.0f, -2.0f, 1.0f, -1.0f,
            2.0f, -0.8f, 1.0f, -2.0f, 1.0f,
            -1.0f, 0.5f, 0.0f, 2.0f, -1.0f
        };
        
        float[] weights = {
            2.0f, 1.0f, 1.5f, 0.5f, 3.0f,
            1.0f, 2.5f, 0.8f, 1.2f, 0.7f,
            1.5f, 2.0f, 1.0f, 0.6f, 2.2f,
            0.9f, 1.8f, 1.1f, 2.0f, 1.3f
        };
        
        CUdeviceptr weightsGPU = CudaUtil.toGPU(weights);
        WeightedMeanSquareError wmse = new WeightedMeanSquareError(weights, weightsGPU);
        
        float[] expected = {
            -0.06f, -0.02f, 0.045f, 0.01f, -0.09f,
            0.02f, -0.075f, -0.008f, -0.012f, -0.028f,
            0.03f, 0.06f, -0.03f, 0.006f, 0.022f,
            -0.027f, -0.018f, -0.022f, -0.04f, 0.039f
        };
        
        float[] derivatives = wmse.derivative(predictions, targets);
        assertArrayEquals(expected, derivatives, EPSILON);
        
        CudaUtil.free(weightsGPU);
    }
    
    @Override
    public void testBatchReduction() {
        float[] predictions = {
            1.0f, 2.0f, 3.0f, 4.0f,
            -1.0f, -2.0f, -3.0f, -4.0f,
            0.5f, 1.5f, 2.5f, 3.5f,
            -0.5f, -1.5f, -2.5f, -3.5f
        };
        
        float[] targets = {
            1.5f, 2.5f, 3.5f, 4.5f,
            -0.5f, -1.5f, -2.5f, -3.5f,
            0.0f, 1.0f, 2.0f, 3.0f,
            0.0f, -1.0f, -2.0f, -3.0f
        };
        
        float[] weights = new float[16];
        for (int i = 0; i < 16; i++) {
            weights[i] = 2.0f;
        }
        
        CUdeviceptr weightsGPU = CudaUtil.toGPU(weights);
        WeightedMeanSquareError wmse = new WeightedMeanSquareError(weights, weightsGPU);
        
        float loss = wmse.compute(predictions, targets);
        float expectedLoss = 0.5f;
        assertEquals(expectedLoss, loss, EPSILON);
        
        float[] smallPred = {1.0f, 2.0f};
        float[] smallTarget = {1.5f, 2.5f};
        float[] smallWeights = {3.0f, 1.0f};
        
        CUdeviceptr smallWeightsGPU = CudaUtil.toGPU(smallWeights);
        WeightedMeanSquareError smallWmse = new WeightedMeanSquareError(smallWeights, smallWeightsGPU);
        
        float smallLoss = smallWmse.compute(smallPred, smallTarget);
        assertEquals(0.5f, smallLoss, EPSILON);
        
        CudaUtil.free(weightsGPU);
        CudaUtil.free(smallWeightsGPU);
    }
    
    @Override
    public void testEdgeCases() {
        float[] predictions = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] targets = {2.0f, 3.0f, 4.0f, 5.0f};
        float[] zeroWeights = {0.0f, 1.0f, 0.0f, 2.0f};
        
        CUdeviceptr zeroWeightsGPU = CudaUtil.toGPU(zeroWeights);
        WeightedMeanSquareError wmse = new WeightedMeanSquareError(zeroWeights, zeroWeightsGPU);
        
        float loss = wmse.compute(predictions, targets);
        assertEquals(0.75f, loss, EPSILON);
        
        float[] largeDiffs = {1000.0f, -1000.0f, 0.0f, 500.0f};
        float[] largeTargets = {0.0f, 0.0f, 0.0f, 0.0f};
        float[] largeWeights = {0.001f, 0.001f, 1.0f, 0.002f};
        
        CUdeviceptr largeWeightsGPU = CudaUtil.toGPU(largeWeights);
        WeightedMeanSquareError largeWmse = new WeightedMeanSquareError(largeWeights, largeWeightsGPU);
        
        float largeLoss = largeWmse.compute(largeDiffs, largeTargets);
        
        assertFalse("Loss contains NaN", Float.isNaN(largeLoss));
        assertFalse("Loss contains Inf", Float.isInfinite(largeLoss));
        assertTrue("Loss should be non-negative", largeLoss >= 0.0f);
        
        float expectedLoss = 625.0f;
        assertEquals(expectedLoss, largeLoss, 1.0f);
        
        float[] identical = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] anyWeights = {1.5f, 2.0f, 0.5f, 3.0f};
        
        CUdeviceptr anyWeightsGPU = CudaUtil.toGPU(anyWeights);
        WeightedMeanSquareError identicalWmse = new WeightedMeanSquareError(anyWeights, anyWeightsGPU);
        
        float zeroLoss = identicalWmse.compute(identical, identical);
        assertEquals(0.0f, zeroLoss, EPSILON);
        
        CudaUtil.free(zeroWeightsGPU);
        CudaUtil.free(largeWeightsGPU);
        CudaUtil.free(anyWeightsGPU);
    }
    
    @Override
    public void testSerialization() {
        float[] weights = {1.0f, 2.0f, 3.0f, 4.0f};
        CUdeviceptr weightsGPU = CudaUtil.toGPU(weights);
        WeightedMeanSquareError wmse = new WeightedMeanSquareError(weights, weightsGPU);
        
        Map<String, Object> serialized = wmse.serialize();
        assertNotNull(serialized);
        assertEquals("WeightedMeanSquareError", serialized.get("type"));
        assertArrayEquals(weights, (float[])serialized.get("weights"), EPSILON);
        
        try {
            Loss.deserialize(serialized);
            fail("Expected RuntimeException for unimplemented deserialization");
        } catch (RuntimeException e) {
            assertEquals("WeightedMeanSquareError deserialization not implemented", e.getMessage());
        }
        
        CudaUtil.free(weightsGPU);
    }
    
    @Override
    public void testComputeGPU() {
        float[] predictions = {
            0.5f, -1.2f, 2.3f, -0.8f, 1.5f,
            -0.3f, 1.7f, -2.1f, 0.9f, -1.4f,
            2.2f, -0.5f, 0.7f, -1.9f, 1.1f,
            -1.3f, 0.4f, -0.2f, 1.8f, -0.7f
        };
        
        float[] targets = {
            0.8f, -1.0f, 2.0f, -1.0f, 1.8f,
            -0.5f, 2.0f, -2.0f, 1.0f, -1.0f,
            2.0f, -0.8f, 1.0f, -2.0f, 1.0f,
            -1.0f, 0.5f, 0.0f, 2.0f, -1.0f
        };
        
        float[] weights = {
            2.0f, 1.0f, 1.5f, 0.5f, 3.0f,
            1.0f, 2.5f, 0.8f, 1.2f, 0.7f,
            1.5f, 2.0f, 1.0f, 0.6f, 2.2f,
            0.9f, 1.8f, 1.1f, 2.0f, 1.3f
        };
        
        CUdeviceptr weightsGPU = CudaUtil.toGPU(weights);
        WeightedMeanSquareError wmse = new WeightedMeanSquareError(weights, weightsGPU);
        
        float cpuLoss = wmse.compute(predictions, targets);
        
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        wmse.computeGPU(gpuPred, gpuTarget, gpuResult, 20, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gpuLoss = CudaUtil.fromGPUFloat(gpuResult, 1);
        
        assertEquals(cpuLoss, gpuLoss[0], RELAXED_EPSILON);
        
        CudaUtil.free(weightsGPU);
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuResult);
        CudaUtil.freeStream(stream);
    }
    
    @Override
    public void testDerivativeGPU() {
        float[] predictions = {
            0.5f, -1.2f, 2.3f, -0.8f, 1.5f,
            -0.3f, 1.7f, -2.1f, 0.9f, -1.4f,
            2.2f, -0.5f, 0.7f, -1.9f, 1.1f,
            -1.3f, 0.4f, -0.2f, 1.8f, -0.7f
        };
        
        float[] targets = {
            0.8f, -1.0f, 2.0f, -1.0f, 1.8f,
            -0.5f, 2.0f, -2.0f, 1.0f, -1.0f,
            2.0f, -0.8f, 1.0f, -2.0f, 1.0f,
            -1.0f, 0.5f, 0.0f, 2.0f, -1.0f
        };
        
        float[] weights = {
            2.0f, 1.0f, 1.5f, 0.5f, 3.0f,
            1.0f, 2.5f, 0.8f, 1.2f, 0.7f,
            1.5f, 2.0f, 1.0f, 0.6f, 2.2f,
            0.9f, 1.8f, 1.1f, 2.0f, 1.3f
        };
        
        CUdeviceptr weightsGPU = CudaUtil.toGPU(weights);
        WeightedMeanSquareError wmse = new WeightedMeanSquareError(weights, weightsGPU);
        
        float[] cpuDerivatives = wmse.derivative(predictions, targets);
        
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        wmse.derivativeGPU(gpuPred, gpuTarget, gpuDeriv, 20, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gpuDerivatives = CudaUtil.fromGPUFloat(gpuDeriv, 20);
        
        assertArrayEquals(cpuDerivatives, gpuDerivatives, RELAXED_EPSILON);
        
        CudaUtil.free(weightsGPU);
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuDeriv);
        CudaUtil.freeStream(stream);
    }
    
    @Override
    public void testBatchReductionGPU() {
        float[] predictions = {
            1.0f, 2.0f, 3.0f, 4.0f,
            -1.0f, -2.0f, -3.0f, -4.0f,
            0.5f, 1.5f, 2.5f, 3.5f,
            -0.5f, -1.5f, -2.5f, -3.5f
        };
        
        float[] targets = {
            1.5f, 2.5f, 3.5f, 4.5f,
            -0.5f, -1.5f, -2.5f, -3.5f,
            0.0f, 1.0f, 2.0f, 3.0f,
            0.0f, -1.0f, -2.0f, -3.0f
        };
        
        float[] weights = new float[16];
        for (int i = 0; i < 16; i++) {
            weights[i] = 2.0f;
        }
        
        CUdeviceptr weightsGPU = CudaUtil.toGPU(weights);
        WeightedMeanSquareError wmse = new WeightedMeanSquareError(weights, weightsGPU);
        
        float cpuLoss = wmse.compute(predictions, targets);
        
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        wmse.computeGPU(gpuPred, gpuTarget, gpuResult, 16, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gpuLoss = CudaUtil.fromGPUFloat(gpuResult, 1);
        
        assertEquals(cpuLoss, gpuLoss[0], RELAXED_EPSILON);
        
        CudaUtil.free(weightsGPU);
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuResult);
        CudaUtil.freeStream(stream);
    }
    
    @Override
    public void testEdgeCasesGPU() {
        float[] extremeValues = {
            Float.MAX_VALUE / 2, -Float.MAX_VALUE / 2, 0.0f, 1000000.0f,
            -1000000.0f, Float.MIN_VALUE, -Float.MIN_VALUE, 0.0f,
            1e10f, -1e10f, 1e-10f, -1e-10f
        };
        
        float[] targets = new float[12];
        
        float[] weights = new float[12];
        for (int i = 0; i < 12; i++) {
            weights[i] = 1e-20f;
        }
        weights[2] = 1.0f;
        weights[5] = 1.0f;
        weights[6] = 1.0f;
        weights[7] = 1.0f;
        weights[10] = 1.0f;
        weights[11] = 1.0f;
        
        CUdeviceptr weightsGPU = CudaUtil.toGPU(weights);
        WeightedMeanSquareError wmse = new WeightedMeanSquareError(weights, weightsGPU);
        
        float cpuLoss = wmse.compute(extremeValues, targets);
        
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(extremeValues);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        wmse.computeGPU(gpuPred, gpuTarget, gpuResult, 12, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gpuLoss = CudaUtil.fromGPUFloat(gpuResult, 1);
        
        assertFalse("GPU result contains NaN", Float.isNaN(gpuLoss[0]));
        if (!Float.isInfinite(cpuLoss)) {
            assertFalse("GPU result contains Inf", Float.isInfinite(gpuLoss[0]));
            if (Math.abs(cpuLoss) > 1.0f) {
                assertEquals(cpuLoss, gpuLoss[0], cpuLoss * 0.001f);
            } else {
                assertEquals(cpuLoss, gpuLoss[0], RELAXED_EPSILON);
            }
        }
        
        CudaUtil.free(weightsGPU);
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuResult);
        CudaUtil.freeStream(stream);
    }
}