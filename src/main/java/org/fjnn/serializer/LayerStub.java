/*
 * The MIT License
 *
 * Copyright 2018 ahmed.
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
package org.fjnn.serializer;

import java.io.Serializable;
import org.fjnn.activation.Activation;

/**
 *
 * @author ahmed
 */
public class LayerStub implements Serializable {
    /* number of neurons in this layer */
    public final int neurons;

    /* activation function for this layer */
    public final Activation activation;
    
    /* does this layer has a bias */
    public final boolean hasBias;
    
    /* [to][from] */
    public final float[][] weights;
    
    /* list of biases of size "links" */
    public final float[] biases;
    
    public LayerStub(int neurons, float[][] weights, Activation activation, boolean hasBias, float[] biases) {
        this.neurons = neurons;
        this.weights = weights;
        this.activation = activation;
        this.hasBias = hasBias;
        this.biases = biases;
    }
}
