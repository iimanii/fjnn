/*
 * The MIT License
 *
 * Copyright 2024 ahmed.
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
import jcuda.driver.JCudaDriver;
import org.fjnn.loss.BirdLoss;
import org.fjnn.loss.Loss;
import org.fjnn.cuda.CudaUtil;
import static org.junit.Assert.*;
import java.util.Map;
import org.junit.Test;

/**
 * Unit test for BirdLoss function.
 * 
 * Tests the property that Bird Loss maintains higher values than MSE
 * for small errors, forcing networks to achieve higher precision.
 * 
 * @author ahmed
 */
public class BirdLossTest extends LossTest {
    
    @Test
    @Override
    public void testCompute() {
        BirdLoss loss = new BirdLoss(0.5f, 100.0f);
        
        // Test case 1: Perfect prediction (should be 0)
        float[] output1 = {0.5f, 0.3f, 0.8f};
        float[] expected1 = {0.5f, 0.3f, 0.8f};
        float result1 = loss.compute(output1, expected1);
        assertEquals("Perfect prediction should yield 0 loss", 0.0f, result1, EPSILON);
        
        // Test case 2: Small differences
        float[] output2 = {0.5f, 0.3f, 0.8f};
        float[] expected2 = {0.51f, 0.31f, 0.79f};
        float result2 = loss.compute(output2, expected2);
        assertTrue("Small differences should yield small positive loss", result2 > 0 && result2 < 0.1f);
        
        // Test case 3: Large differences (testing logarithmic saturation)
        float[] output3 = {1.0f, 0.0f};
        float[] expected3 = {0.0f, 1.0f};
        float result3 = loss.compute(output3, expected3);
        // With alpha=0.5 and beta=100, loss should be bounded by logarithmic growth
        assertTrue("Large differences should be bounded", result3 > 1.0f && result3 < 3.0f);
    }
    
    @Test
    @Override
    public void testDerivative() {
        BirdLoss loss = new BirdLoss(0.5f, 100.0f);
        
        // Test case 1: Zero difference (gradient should be 0)
        float[] output1 = {0.5f, 0.5f};
        float[] expected1 = {0.5f, 0.5f};
        float[] grad1 = loss.derivative(output1, expected1);
        assertArrayEquals("Zero difference should yield zero gradient", 
                         new float[]{0.0f, 0.0f}, grad1, EPSILON);
        
        // Test case 2: Positive difference
        float[] output2 = {0.6f};
        float[] expected2 = {0.5f};
        float[] grad2 = loss.derivative(output2, expected2);
        assertTrue("Positive difference should yield positive gradient", grad2[0] > 0);
        
        // Test case 3: Negative difference
        float[] output3 = {0.4f};
        float[] expected3 = {0.5f};
        float[] grad3 = loss.derivative(output3, expected3);
        assertTrue("Negative difference should yield negative gradient", grad3[0] < 0);
        
        // Test case 4: Gradient magnitude decreases with larger differences
        float[] output4 = {1.0f, 2.0f};
        float[] expected4 = {0.5f, 0.5f};
        float[] grad4 = loss.derivative(output4, expected4);
        assertTrue("Gradient magnitude should decrease for larger differences",
                  Math.abs(grad4[1]) < Math.abs(grad4[0]));
    }
    
    @Test
    @Override
    public void testBatchReduction() {
        BirdLoss loss = new BirdLoss(0.5f, 100.0f);
        
        // Test averaging over batch
        float[] output = {0.1f, 0.2f, 0.3f, 0.4f};
        float[] expected = {0.15f, 0.25f, 0.35f, 0.45f};
        
        float batchLoss = loss.compute(output, expected);
        
        // Compute individual losses and average manually
        float sum = 0.0f;
        for (int i = 0; i < output.length; i++) {
            float diff = output[i] - expected[i];
            sum += 0.5f * Math.log(100.0f * diff * diff + 1);  // alpha=0.5, beta=100
        }
        float expectedAvg = sum / output.length;
        
        assertEquals("Batch loss should be average of individual losses", 
                    expectedAvg, batchLoss, EPSILON);
    }
    
    @Test
    @Override
    public void testEdgeCases() {
        BirdLoss loss = new BirdLoss(0.5f, 100.0f);
        
        // Test with very small alpha
//        BirdLoss smallAlphaLoss = new BirdLoss(0.01f, 100.0f);
//        float[] output = {1.0f};
//        float[] expected = {0.0f};
//        float result = smallAlphaLoss.compute(output, expected);
//        assertTrue("Small alpha should scale down the loss", result < 0.1f);
//        
//        // Test with very small beta
//        BirdLoss smallBetaLoss = new BirdLoss(0.5f, 0.1f);
//        float result2 = smallBetaLoss.compute(output, expected);
//        assertTrue("Small beta should reduce steepness", result2 < result);
        
        // Test parameter validation
        try {
            new BirdLoss(-0.5f, 100.0f);
            fail("Should throw exception for negative alpha");
        } catch (IllegalArgumentException e) {
            // Expected
        }
        
        try {
            new BirdLoss(0.5f, -100.0f);
            fail("Should throw exception for negative beta");
        } catch (IllegalArgumentException e) {
            // Expected
        }
        
        // Test extreme values
        float[] extremeOutput = {10.0f, -10.0f, 0.0f};
        float[] extremeExpected = {0.0f, 0.0f, 0.0f};
        float extremeResult = loss.compute(extremeOutput, extremeExpected);
        
        assertFalse("Result should not be NaN", Float.isNaN(extremeResult));
        assertFalse("Result should not be infinite", Float.isInfinite(extremeResult));
        assertTrue("Result should be positive", extremeResult >= 0);
    }
    
    @Test
    @Override
    public void testSerialization() {
        BirdLoss loss = new BirdLoss(0.7f, 250.0f);
        
        Map serialized = loss.serialize();
        assertEquals("BirdLoss", serialized.get("type"));
        assertEquals(0.7f, serialized.get("alpha"));
        assertEquals(250.0f, serialized.get("beta"));
        
        BirdLoss deserialized = BirdLoss.deserialize(serialized);
        assertEquals(0.7f, deserialized.alpha, EPSILON);
        assertEquals(250.0f, deserialized.beta, EPSILON);
    }
    
    @Test
    @Override
    public void testComputeGPU() {
        BirdLoss loss = new BirdLoss(0.5f, 100.0f);
        
        float[] output = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        float[] expected = {0.15f, 0.18f, 0.35f, 0.38f, 0.52f};
        
        // CPU computation for comparison
        float cpuLoss = loss.compute(output, expected);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr outputGPU = CudaUtil.toGPU(output);
        CUdeviceptr expectedGPU = CudaUtil.toGPU(expected);
        CUdeviceptr resultGPU = CudaUtil.toGPU(new float[1]);
        
        loss.computeGPU(outputGPU, expectedGPU, resultGPU, output.length, stream);
        CudaUtil.synchronizeStream(stream);
        float[] resultFromGPU = CudaUtil.fromGPUFloat(resultGPU, 1);
        
        assertEquals("GPU computation should match CPU", cpuLoss, resultFromGPU[0], RELAXED_EPSILON);
        
        // Cleanup
        CudaUtil.free(outputGPU);
        CudaUtil.free(expectedGPU);
        CudaUtil.free(resultGPU);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    @Override
    public void testDerivativeGPU() {
        BirdLoss loss = new BirdLoss(0.5f, 100.0f);
        
        float[] output = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
        float[] expected = {0.15f, 0.18f, 0.35f, 0.38f, 0.52f};
        
        // CPU computation for comparison
        float[] gradCPU = loss.derivative(output, expected);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr outputGPU = CudaUtil.toGPU(output);
        CUdeviceptr expectedGPU = CudaUtil.toGPU(expected);
        CUdeviceptr gradGPU = CudaUtil.toGPU(new float[output.length]);
        
        loss.derivativeGPU(outputGPU, expectedGPU, gradGPU, output.length, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gradFromGPU = CudaUtil.fromGPUFloat(gradGPU, output.length);
        
        assertArrayEquals("GPU derivative should match CPU", gradCPU, gradFromGPU, RELAXED_EPSILON);
        
        // Cleanup
        CudaUtil.free(outputGPU);
        CudaUtil.free(expectedGPU);
        CudaUtil.free(gradGPU);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    @Override
    public void testBatchReductionGPU() {
        BirdLoss loss = new BirdLoss(0.5f, 100.0f);
        
        float[] output = {
            0.1f, 0.2f, 0.3f, 0.4f,
            0.5f, 0.6f, 0.7f, 0.8f,
            0.9f, 0.15f, 0.25f, 0.35f,
            0.45f, 0.55f, 0.65f, 0.75f
        };
        
        float[] expected = {
            0.15f, 0.25f, 0.35f, 0.45f,
            0.55f, 0.65f, 0.75f, 0.85f,
            0.95f, 0.1f, 0.2f, 0.3f,
            0.4f, 0.5f, 0.6f, 0.7f
        };
        
        // CPU computation
        float cpuLoss = loss.compute(output, expected);
        
        // GPU computation
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr outputGPU = CudaUtil.toGPU(output);
        CUdeviceptr expectedGPU = CudaUtil.toGPU(expected);
        CUdeviceptr resultGPU = CudaUtil.toGPU(new float[1]);
        
        loss.computeGPU(outputGPU, expectedGPU, resultGPU, output.length, stream);
        CudaUtil.synchronizeStream(stream);
        float[] gpuLoss = CudaUtil.fromGPUFloat(resultGPU, 1);
        
        assertEquals("GPU batch reduction should match CPU", cpuLoss, gpuLoss[0], RELAXED_EPSILON);
        
        // Cleanup
        CudaUtil.free(outputGPU);
        CudaUtil.free(expectedGPU);
        CudaUtil.free(resultGPU);
        CudaUtil.freeStream(stream);
    }
    
    @Test
    @Override
    public void testEdgeCasesGPU() {
        BirdLoss loss = new BirdLoss(0.1f, 1000.0f);
        
        float[] output = {10.0f, -10.0f, 0.0f};
        float[] expected = {0.0f, 0.0f, 0.0f};
        
        // CPU computation for comparison
        float cpuLoss = loss.compute(output, expected);
        
        CUstream stream = CudaUtil.createStream();
        CUdeviceptr outputGPU = CudaUtil.toGPU(output);
        CUdeviceptr expectedGPU = CudaUtil.toGPU(expected);
        CUdeviceptr resultGPU = CudaUtil.toGPU(new float[1]);
        
        loss.computeGPU(outputGPU, expectedGPU, resultGPU, output.length, stream);
        CudaUtil.synchronizeStream(stream);
        float[] result = CudaUtil.fromGPUFloat(resultGPU, 1);
        
        // Verify no NaN or infinity
        assertFalse("Result should not be NaN", Float.isNaN(result[0]));
        assertFalse("Result should not be infinite", Float.isInfinite(result[0]));
        assertTrue("Result should be positive", result[0] >= 0);
        assertEquals("GPU extreme values should match CPU", cpuLoss, result[0], RELAXED_EPSILON);
        
        // Test derivative edge cases
        CUdeviceptr gradGPU = CudaUtil.toGPU(new float[output.length]);
        loss.derivativeGPU(outputGPU, expectedGPU, gradGPU, output.length, stream);
        CudaUtil.synchronizeStream(stream);
        float[] derivatives = CudaUtil.fromGPUFloat(gradGPU, output.length);
        
        for (float deriv : derivatives) {
            assertFalse("Derivative should not be NaN", Float.isNaN(deriv));
            assertFalse("Derivative should not be infinite", Float.isInfinite(deriv));
        }
        
        // Cleanup
        CudaUtil.free(outputGPU);
        CudaUtil.free(expectedGPU);
        CudaUtil.free(resultGPU);
        CudaUtil.free(gradGPU);
        CudaUtil.freeStream(stream);
    }
}