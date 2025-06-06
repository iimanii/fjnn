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
package org.fjnn.transformer.output;

import org.fjnn.base.output.BackpropagateOutput;

/**
 *
 * @author ahmed
 */
public class AttentionWeightsBackpropOutput extends BackpropagateOutput {
    public final float[] weightGradients;       // [vectorDim × vectorDim]
    public final float[] inputGradients;        // [batchSize * sequenceLen * vectorDim]

    public AttentionWeightsBackpropOutput(float[] input, int vectorDim, int sequenceLen, int batchSize) {
        super(sequenceLen * vectorDim, batchSize);  // superclass takes total input dim and batch size
        
        // Initialize gradient arrays
        this.weightGradients = new float[vectorDim * vectorDim];
        this.inputGradients = new float[batchSize * sequenceLen * vectorDim];
    }

    @Override
    public float[] deltaLoss() {
        return inputGradients;  // Pass gradients to previous layer
    }
}
