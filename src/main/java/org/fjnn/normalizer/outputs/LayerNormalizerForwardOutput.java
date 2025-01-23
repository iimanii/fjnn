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
package org.fjnn.normalizer.outputs;

import org.fjnn.base.output.FeedForwardOutput;

/**
 *
 * @author ahmed
 */
public class LayerNormalizerForwardOutput extends FeedForwardOutput {
    // Fields for layer normalization
    public final float[] preNormalization;      // input values
    public final float[] normalized;            // x̂ values (after normalization, before γ and β)
    public final float[] postNormalization;     // final output (after γ and β)
    public final float[] stds;                  // standard deviation per layer
    
    public LayerNormalizerForwardOutput(float[] preNormalization, int outputDim, int batchSize) {
        super(outputDim, batchSize);
        
        int size = outputDim * batchSize;
        
        this.preNormalization = preNormalization;
        this.normalized = new float[size];
        this.postNormalization = new float[size];
        this.stds = new float[batchSize];
    }

    @Override
    public float[] output() {
        return postNormalization;
    }
}
