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
package org.fjnn.base;

import org.fjnn.serializer.NetworkStub;

/**
 *
 * @author ahmed
 */
public abstract class LayeredNetwork extends BaseNetwork {

    /**
     * @param layer
     * @param from
     * @param to
     * @return 
     */
    public abstract float getWeight(int layer, int from, int to);
    
    /**
     * 
     * @param layer
     * @param from
     * @param to
     * @param value 
     */
    public abstract void setWeight(int layer, int from, int to, float value);

    /**
     * 
     * @return 
     */
    public abstract float[][][] getWeights();
    
    /**
     * Update all weights at once
     * Must be in the format [layer][to][from]
     * @param values 
     */
    public abstract void setWeights(float[][][] values);
    
    /**
     * 
     * @param layer
     * @param to
     * @return 
     */
    public abstract float getBias(int layer, int to);
    
    /**
     * 
     * @param layer
     * @param to
     * @param value 
     */
    public abstract void setBias(int layer, int to, float value);
    
    /**
     * 
     * @return 
     */
    public abstract float[][] getBiases();
    
    /**
     * Update all biases
     * Must be in the format [layer][to]
     * @param values 
     */
    public abstract void setBiases(float[][] values);

    /**
     * 
     * @param n 
     */
    public void fromNetwork(LayeredNetwork n) {
        fromNetwork(n, false);
    }
    
    /**
     * 
     * @param n
     * @param copyProperties 
     */
    public void fromNetwork(LayeredNetwork n, boolean copyProperties) {
        for(int i=0; i < n.getLayerCount()-1; i++) {
            int current = n.getLayerSize(i);
            int next = n.getLayerSize(i + 1);
            
            for(int j=0; j < next; j++) {
                for(int k=0; k < current; k++) {
                    float weight = n.getWeight(i, k, j);
                    setWeight(i, k, j, weight);
                }
                
                float bias = n.getBias(i, j);
                setBias(i, j, bias);
            }
        }
        
        if(copyProperties) {
            properties.clear();
            properties.putAll(n.properties);
        }        
    }
    
    /**
     * 
     * @return 
     */
    public abstract NetworkStub getStub();
}
