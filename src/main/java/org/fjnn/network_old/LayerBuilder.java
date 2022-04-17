/*
 * The MIT License
 *
 * Copyright 2018 Ahmed Tarek.
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
package org.fjnn.network_old;

import org.fjnn.activation.Activation;

/**
 *
 * @author ahmed
 */
public class LayerBuilder {
    int neurons;
    Activation activation;
    boolean hasBias;
    boolean[] condition;
    float[][] weights;
    float[] biases;
    
    LayerBuilder(int neurons, Activation activation, boolean hasBias, boolean[] condition) {
        this.neurons = neurons;
        this.activation = activation;
        this.hasBias = hasBias;
        this.condition = condition;
    }
    
    LayerBuilder(LayerStub stub) {
        this.neurons = stub.neurons;
        this.hasBias = stub.hasBias;
        this.activation = stub.activation;
        this.weights = stub.weights;
        this.biases = stub.biases;
        this.condition = stub.condition;
    }

    Layer build(int links, boolean threadSafe) {
        Layer layer = threadSafe ? new ConcurrentLayer(activation, neurons, links, hasBias, condition) :
                                   new FastLayer(activation, neurons, links, hasBias, condition);
        
        if(weights != null)
            layer.setWeights(weights);
        
        if(biases != null)
            layer.setBiases(biases);
        
        return layer;
    }

    Layer buildOutput(boolean threadSafe) {
        return threadSafe ? new ConcurrentLayer(activation, neurons) :
                            new FastLayer(activation, neurons);
    }

    int size() {
        return neurons;
    }
}
