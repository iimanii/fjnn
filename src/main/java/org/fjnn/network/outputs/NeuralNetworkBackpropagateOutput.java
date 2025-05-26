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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.network.gradient.ConnectionGradient;

public class NeuralNetworkBackpropagateOutput extends BackpropagateOutput {
    public final float[][] preActivationDeltas;
    public final BackpropagateOutput[] normalizerGradients;
    public final List<Map<Integer, ConnectionGradient>> layerConnectionGradients;

    public NeuralNetworkBackpropagateOutput(int deltaLossDim, int batchSize, int layerCount) {
        super(deltaLossDim, batchSize);
        
        this.preActivationDeltas = new float[layerCount][];
        this.normalizerGradients = new BackpropagateOutput[layerCount];
        this.layerConnectionGradients = new ArrayList<>();        
        
        // Initialize the ConnectionGradient Map to store gradients for all connections
        for (int i = 0; i < layerCount; i++) {
            layerConnectionGradients.add(new HashMap<>());
        }
    }

    @Override
    public float[] deltaLoss() {
        return preActivationDeltas[0];
    }
}
