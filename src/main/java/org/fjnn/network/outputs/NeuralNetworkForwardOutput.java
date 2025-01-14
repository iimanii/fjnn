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
package org.fjnn.network.outputs;

import org.fjnn.base.output.FeedForwardOutput;

/**
 *
 * @author ahmed
 */
public class NeuralNetworkForwardOutput extends FeedForwardOutput {
    public final float[][] preActivations; // Stores z_x (pre-activation) for each layer
    public final float[][] postActivations; // Stores a_x (post-activation) for each layer

    public NeuralNetworkForwardOutput(int batchSize, int batchCount, int layerCount) {
        super(batchSize, batchCount);
        
        this.preActivations = new float[layerCount][];
        this.postActivations = new float[layerCount][];
    }

    @Override
    public float[] output() {
        return postActivations[postActivations.length - 1];
    }
    
}
