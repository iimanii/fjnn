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
import org.fjnn.loss.MeanSquareError;
import org.fjnn.loss.Loss;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;
import org.junit.Test;

/**
 * Unit test for MeanSquaredError loss function.
 * 
 * Note: MeanSquaredError.compute() returns a single float value (average loss over all elements),
 * while derivative() returns an array of gradients for each element.
 * 
 * @author ahmed
 */
public class MeanSquaredErrorTest extends LossTest {
    
    @Override
    public void testCompute() {
        MeanSquareError mse = new MeanSquareError();

        // Test batch (4 batches, 5 outputs each)
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

        // Calculate differences and squared differences:
        // Row 1: (-0.3)²=0.09, (-0.2)²=0.04, (0.3)²=0.09, (0.2)²=0.04, (-0.3)²=0.09  Sum=0.35
        // Row 2: (0.2)²=0.04, (-0.3)²=0.09, (-0.1)²=0.01, (-0.1)²=0.01, (-0.4)²=0.16  Sum=0.31
        // Row 3: (0.2)²=0.04, (0.3)²=0.09, (-0.3)²=0.09, (0.1)²=0.01, (0.1)²=0.01    Sum=0.24
        // Row 4: (-0.3)²=0.09, (-0.1)²=0.01, (-0.2)²=0.04, (-0.2)²=0.04, (0.3)²=0.09  Sum=0.27
        // Total sum = 1.17, Average = 1.17/20 = 0.0585
        float expectedLoss = 0.0585f;

        float loss = mse.compute(predictions, targets);
        assertEquals(expectedLoss, loss, EPSILON);
    }
    
    @Override
    public void testDerivative() {
        MeanSquareError mse = new MeanSquareError();
        
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
        
        // For MSE, derivative = (2/n) * (prediction - target) where n=20
        // multiplier = 2/20 = 0.1
        float[] expected = {
            -0.03f, -0.02f, 0.03f, 0.02f, -0.03f,
            0.02f, -0.03f, -0.01f, -0.01f, -0.04f,
            0.02f, 0.03f, -0.03f, 0.01f, 0.01f,
            -0.03f, -0.01f, -0.02f, -0.02f, 0.03f
        };
        
        float[] derivatives = mse.derivative(predictions, targets);
        assertArrayEquals(expected, derivatives, EPSILON);
    }
    
    @Override
    public void testBatchReduction() {
        MeanSquareError mse = new MeanSquareError();
        
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
        
        // All differences are 0.5, so squared error = 0.25 for each element
        // Average = 0.25
        float loss = mse.compute(predictions, targets);
        float expectedLoss = 0.25f;
        assertEquals(expectedLoss, loss, EPSILON);
        
        // Test with smaller batch
        float[] smallPred = {1.0f, 2.0f};
        float[] smallTarget = {1.5f, 2.5f};
        float smallLoss = mse.compute(smallPred, smallTarget);
        assertEquals(0.25f, smallLoss, EPSILON);
    }
    
    @Override
    public void testEdgeCases() {
        MeanSquareError mse = new MeanSquareError();
        
        // Test with large differences
        float[] largeDiffs = {
            1000.0f, -1000.0f, 0.0f, 500.0f,
            -500.0f, 0.0f, 750.0f, -750.0f,
            100.0f, -100.0f, 50.0f, -50.0f
        };
        
        float[] targets = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f
        };
        
            float loss = mse.compute(largeDiffs, targets);

        // Verify no NaN or Inf values
        assertFalse("Loss contains NaN", Float.isNaN(loss));
        assertFalse("Loss contains Inf", Float.isInfinite(loss));
        assertTrue("Loss should be non-negative", loss >= 0.0f);

        // Expected average: (1000²+1000²+0²+500²+500²+0²+750²+750²+100²+100²+50²+50²)/12
        // = (1000000+1000000+0+250000+250000+0+562500+562500+10000+10000+2500+2500)/12
        // = 3650000/12 = 304166.67
        float expectedLoss = 304166.67f;
        assertEquals(expectedLoss, loss, 1.0f);

        // Test with identical predictions and targets
        float[] identical = {1.0f, 2.0f, 3.0f, 4.0f};
        float zeroLoss = mse.compute(identical, identical);
        assertEquals(0.0f, zeroLoss, EPSILON);
    }
    
    @Override
    public void testSerialization() {
        MeanSquareError mse = new MeanSquareError();
        
        Map<String, Object> serialized = mse.serialize();
        assertNotNull(serialized);
        assertEquals("MeanSquareError", serialized.get("type"));
        
        Loss deserialized = Loss.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof MeanSquareError);
        
        // Test functionality
        float[] predictions = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] targets = {1.5f, 2.5f, 2.5f, 3.5f};
        
        float originalLoss = mse.compute(predictions, targets);
        float deserializedLoss = deserialized.compute(predictions, targets);
        
        assertEquals(originalLoss, deserializedLoss, EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        MeanSquareError mse = new MeanSquareError();
        
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
        
        // CPU computation
        float cpuLoss = mse.compute(predictions, targets);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        mse.computeGPU(gpuPred, gpuTarget, gpuResult, 20, stream);
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
        MeanSquareError mse = new MeanSquareError();
        
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
        
        // CPU computation
        float[] cpuDerivatives = mse.derivative(predictions, targets);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        mse.derivativeGPU(gpuPred, gpuTarget, gpuDeriv, 20, stream);
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
        MeanSquareError mse = new MeanSquareError();
        
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
        
        // CPU computation
        float cpuLoss = mse.compute(predictions, targets);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        mse.computeGPU(gpuPred, gpuTarget, gpuResult, 16, stream);
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
        MeanSquareError mse = new MeanSquareError();
        
        // Test with extreme values
        float[] extremeValues = {
            Float.MAX_VALUE / 2, -Float.MAX_VALUE / 2, 0.0f, 1000000.0f,
            -1000000.0f, Float.MIN_VALUE, -Float.MIN_VALUE, 0.0f,
            1e10f, -1e10f, 1e-10f, -1e-10f
        };
        
        float[] targets = new float[12]; // All zeros
        
        // CPU computation
        float cpuLoss = mse.compute(extremeValues, targets);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(extremeValues);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        mse.computeGPU(gpuPred, gpuTarget, gpuResult, 12, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gpuLoss = CudaUtil.fromGPUFloat(gpuResult, 1);
        
        // Verify results contain no invalid values
        assertFalse("GPU result contains NaN", Float.isNaN(gpuLoss[0]));
        if (!Float.isInfinite(cpuLoss)) {
            assertFalse("GPU result contains Inf", Float.isInfinite(gpuLoss[0]));
            assertEquals(cpuLoss, gpuLoss[0], cpuLoss * 0.001f); // Relative tolerance for large values
        }
        
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuResult);
        CudaUtil.freeStream(stream);
    }
}