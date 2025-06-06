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
package org.fjnn.convolution.output.unit;

/**
 *
 * @author ahmed
 */
public class ConvolutionUnitBackpropagateOutput {
    public final float[] inputGradients;
    public final int inputSize;
    public final int batchSize;
    
    // Weight/bias gradients
    public final float[][] weightGradients;  // [kernel_index][weight_index]
    public final float[] biasGradients;      // [kernel_index]
    
    // Single kernel constructor
    public ConvolutionUnitBackpropagateOutput(float[] inputGradients, int inputSize, int batchSize,
                                              float[] weightGradients, float biasGradient) {
        this.inputGradients = inputGradients;
        this.inputSize = inputSize;
        this.batchSize = batchSize;
        this.weightGradients = new float[][]{weightGradients};
        this.biasGradients = new float[]{biasGradient};
    }
    
    // Multi-kernel constructor  
    public ConvolutionUnitBackpropagateOutput(float[] inputGradients, int inputSize, int batchSize,
                                              float[][] weightGradients, float[] biasGradients) {
        this.inputGradients = inputGradients;
        this.inputSize = inputSize;
        this.batchSize = batchSize;
        this.weightGradients = weightGradients;
        this.biasGradients = biasGradients;
    }
}
