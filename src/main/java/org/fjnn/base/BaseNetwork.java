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
package org.fjnn.base;

import org.fjnn.activation.Activation;

/**
 *
 * @author ahmed
 */
public abstract class BaseNetwork extends Network {    

    /**
     * Randomize the network in the range -1[inclusive], 1[exclusive]
     * @return 
     */
    public BaseNetwork randomize() {
        return randomize(-1, 1);
    }

    /**
     * Randomize the network in the range min[inclusive] to max[exclusive]
     * @param min
     * @param max
     * @return 
     */
    public abstract BaseNetwork randomize(float min, float max);
    
    /**
     * Does this layer has a bias
     * @param layer
     * @return 
     */
    public abstract boolean hasBias(int layer);
    
    /**
     * 
     * @param layer
     * @return 
     */
    public abstract Activation getActivation(int layer);
    
    /**
     * 
     * @param layer
     * @return 
     */
    public abstract boolean[] getCondition(int layer);

    /**
     * 
     * @param layer
     * @return Neuron count for a specific layer
     */
    public abstract int getLayerSize(int layer);
    
    /**
     * @return Number of hidden layers
     */
    public abstract int getHiddenLayerCount();
    
    /**
     * @return Total number of layers
     */
    public abstract int getLayerCount();
}
