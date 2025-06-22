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

import org.fjnn.cuda.CudaEngine;
import org.junit.*;

/**
 * Base test class for loss functions.
 * 
 * @author ahmed
 */
public abstract class LossTest {

    protected static final float EPSILON = 1e-5f;
    protected static final float RELAXED_EPSILON = 1e-3f;

    @BeforeClass
    public static void setupCuda() {
        CudaEngine.init();
    }
    
    @Before
    public void prepareCudaThread() {
        CudaEngine.prepareThread(0);
    }
    
    @After
    public void finalizeCudaThread() {
        CudaEngine.finalizeThread();
    }
    
    @Test
    public abstract void testCompute();
    
    @Test
    public abstract void testDerivative();
    
    @Test
    public abstract void testBatchReduction();
    
    @Test
    public abstract void testEdgeCases();
    
    @Test
    public abstract void testSerialization();
    
    @Test
    public abstract void testComputeGPU();
    
    @Test
    public abstract void testDerivativeGPU();
    
    @Test
    public abstract void testBatchReductionGPU();
    
    @Test
    public abstract void testEdgeCasesGPU();
}