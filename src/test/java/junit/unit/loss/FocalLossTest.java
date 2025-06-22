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
import org.fjnn.loss.FocalLoss;
import org.fjnn.loss.Loss;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;
import org.junit.Test;

/**
 * Unit test for FocalLoss function.
 * 
 * Note: The current FocalLoss implementation uses a simplified derivative formula
 * that differs from the exact mathematical derivative of focal loss.
 * 
 * @author ahmed
 */
public class FocalLossTest extends LossTest {
    
    @Override
    public void testCompute() {
        FocalLoss fl = new FocalLoss(2.0f);
        
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
        
        float expectedLoss = 0.11362f;
        float loss = fl.compute(predictions, targets);
        assertEquals(expectedLoss, loss, EPSILON);
    }
    
    @Override
    public void testDerivative() {
        FocalLoss fl = new FocalLoss(2.0f);
        
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
            -0.02023f, 0.03695f, -0.44611f, 0.01204f, -0.05711f,
            0.01187f, -0.16092f, 0.02431f, -0.29764f, -0.03428f,
            -0.10644f, 0.03591f, -0.73769f, -21.91698f, -0.13166f,
            0.03023f, -0.36461f, -0.80950f, -0.07029f, 0.01875f
        };
        
        float[] derivatives = fl.derivative(predictions, targets);
        assertArrayEquals(expected, derivatives, 0.001f);
    }
    
    @Override
    public void testBatchReduction() {
        FocalLoss fl = new FocalLoss(2.0f);
        
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
        
        float loss = fl.compute(predictions, targets);
        float expectedLoss = 0.06863f;
        assertEquals(expectedLoss, loss, EPSILON);
        
        FocalLoss fl0 = new FocalLoss(0.0f);
        FocalLoss fl1 = new FocalLoss(1.0f);
        FocalLoss fl5 = new FocalLoss(5.0f);
        
        float loss0 = fl0.compute(predictions, targets);
        float loss1 = fl1.compute(predictions, targets);
        float loss5 = fl5.compute(predictions, targets);
        
        float expectedLoss0 = 0.31481f;
        assertEquals(expectedLoss0, loss0, EPSILON);
        
        assertTrue("Loss should decrease with higher gamma for easy examples", loss1 < loss0);
        assertTrue("Loss should decrease with higher gamma for easy examples", loss5 < loss1);
    }
    
    @Override
    public void testEdgeCases() {
        FocalLoss fl = new FocalLoss(2.0f);
        
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
        
        float loss = fl.compute(extremeProbs, targets);
        
        assertFalse("Loss contains NaN", Float.isNaN(loss));
        assertFalse("Loss contains Inf", Float.isInfinite(loss));
        assertTrue("Loss should be non-negative", loss >= 0.0f);
        
        float[] derivatives = fl.derivative(extremeProbs, targets);
        for (float deriv : derivatives) {
            assertFalse("Derivative contains NaN", Float.isNaN(deriv));
            assertFalse("Derivative contains Inf", Float.isInfinite(deriv));
        }
        
        FocalLoss fl0 = new FocalLoss(0.0f);
        float[] simpleProbs = {0.7f, 0.3f, 0.9f, 0.1f};
        float[] simpleTargets = {1.0f, 0.0f, 1.0f, 0.0f};
        
        float fl0Loss = fl0.compute(simpleProbs, simpleTargets);
        float expectedBCE = 0.23102f;
        assertEquals(expectedBCE, fl0Loss, EPSILON);
    }
    
    @Override
    public void testSerialization() {
        FocalLoss fl = new FocalLoss(2.5f);
        
        Map<String, Object> serialized = fl.serialize();
        assertNotNull(serialized);
        assertEquals("FocalLoss", serialized.get("type"));
        assertEquals(2.5f, serialized.get("gamma"));
        
        Loss deserialized = Loss.deserialize(serialized);
        assertNotNull(deserialized);
        assertTrue(deserialized instanceof FocalLoss);
        assertEquals(2.5f, ((FocalLoss)deserialized).gamma, EPSILON);
        
        float[] predictions = {0.7f, 0.3f, 0.9f, 0.1f};
        float[] targets = {1.0f, 0.0f, 1.0f, 0.0f};
        
        float originalLoss = fl.compute(predictions, targets);
        float deserializedLoss = deserialized.compute(predictions, targets);
        
        assertEquals(originalLoss, deserializedLoss, EPSILON);
    }
    
    @Override
    public void testComputeGPU() {
        FocalLoss fl = new FocalLoss(2.0f);
        
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
        
        float cpuLoss = fl.compute(predictions, targets);
        
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        fl.computeGPU(gpuPred, gpuTarget, gpuResult, 20, stream);
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
        FocalLoss fl = new FocalLoss(2.0f);
        
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
        
        float[] cpuDerivatives = fl.derivative(predictions, targets);
        
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuDeriv = CudaUtil.toGPU(new float[20]);
        fl.derivativeGPU(gpuPred, gpuTarget, gpuDeriv, 20, stream);
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
        FocalLoss fl = new FocalLoss(2.0f);
        
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
        
        float cpuLoss = fl.compute(predictions, targets);
        
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(predictions);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        fl.computeGPU(gpuPred, gpuTarget, gpuResult, 16, stream);
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
        FocalLoss fl = new FocalLoss(2.0f);
        
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
        
        float cpuLoss = fl.compute(extremeProbs, targets);
        
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr gpuPred = CudaUtil.toGPU(extremeProbs);
        CUdeviceptr gpuTarget = CudaUtil.toGPU(targets);
        CUdeviceptr gpuResult = CudaUtil.toGPU(new float[1]);
        fl.computeGPU(gpuPred, gpuTarget, gpuResult, 12, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gpuLoss = CudaUtil.fromGPUFloat(gpuResult, 1);
        
        assertFalse("GPU result contains NaN", Float.isNaN(gpuLoss[0]));
        assertFalse("GPU result contains Inf", Float.isInfinite(gpuLoss[0]));
        assertEquals(cpuLoss, gpuLoss[0], RELAXED_EPSILON);
        
        CudaUtil.free(gpuPred);
        CudaUtil.free(gpuTarget);
        CudaUtil.free(gpuResult);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    public void testDifferentGammaValues() {
        float[] predictions = {
            0.9f, 0.8f, 0.7f, 0.6f,
            0.1f, 0.2f, 0.3f, 0.4f
        };
        
        float[] targets = {
            1.0f, 1.0f, 1.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 0.0f
        };
        
        float[] gammaValues = {0.0f, 0.5f, 1.0f, 2.0f, 5.0f};
        float[] losses = new float[gammaValues.length];
        
        for (int i = 0; i < gammaValues.length; i++) {
            FocalLoss fl = new FocalLoss(gammaValues[i]);
            losses[i] = fl.compute(predictions, targets);
        }
        
        for (int i = 1; i < losses.length; i++) {
            assertTrue("Loss should decrease with higher gamma for well-classified examples", 
                      losses[i] < losses[i-1]);
        }
    }
    
    @Test
    public void testFocalLossVsBinaryCrossEntropy() {
        FocalLoss fl0 = new FocalLoss(0.0f);
        
        float[] predictions = {
            0.7f, 0.3f, 0.9f, 0.1f,
            0.6f, 0.4f, 0.8f, 0.2f
        };
        
        float[] targets = {
            1.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 1.0f
        };
        
        float flLoss = fl0.compute(predictions, targets);
        float expectedBCE = 0.74694f;
        
        assertEquals("Focal loss with gamma=0 should equal BCE", expectedBCE, flLoss, EPSILON);
    }
}