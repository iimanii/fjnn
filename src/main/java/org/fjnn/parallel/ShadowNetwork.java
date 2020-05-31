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
package org.fjnn.parallel;

import java.util.Map;
import org.fjnn.activation.Activation;
import org.fjnn.base.LayeredNetwork;
import org.fjnn.serializer.NetworkStub;

/**
 *
 * @author ahmed
 */
public class ShadowNetwork extends LayeredNetwork {
    final MultiNetwork parent;
    final int index;
    
    ShadowNetwork(MultiNetwork parent, int index) {
        this.index = index;
        this.parent = parent;
    }
    
    @Override
    public float getWeight(int layer, int from, int to) {
        return parent.getWeight(index, layer, from, to);
    }

    @Override
    public void setWeight(int layer, int from, int to, float value) {
        parent.setWeight(index, layer, from, to, value);
    }

    @Override
    public void setWeights(float[][][] values) {
        parent.setWeights(index, values);
    }

    @Override
    public float getBias(int layer, int to) {
        return parent.getBias(index, layer, to);
    }

    @Override
    public void setBias(int layer, int to, float value) {
        parent.setBias(index, layer, to, value);
    }

    @Override
    public void setBiases(float[][] values) {
        parent.setBiases(index, values);
    }

    @Override
    public LayeredNetwork randomize(float min, float max) {
        parent.randomize(index, min, max);
        return this;
    }

    @Override
    public boolean hasBias(int layer) {
        return parent.hasBias(layer);
    }

    @Override
    public Activation getActivation(int layer) {
        return parent.getActivation(layer);
    }

    @Override
    public int getInputSize() {
        return parent.getInputSize();
    }

    @Override
    public int getOutputSize() {
        return parent.getOutputSize();
    }

    @Override
    public int getLayerSize(int layer) {
        return parent.getLayerSize(layer);
    }

    @Override
    public int getHiddenLayerCount() {
        return parent.getHiddenLayerCount();
    }

    @Override
    public int getLayerCount() {
        return parent.getLayerCount();
    }

    @Override
    public float[][][] getWeights() {
        return parent.getWeights(index);
    }

    @Override
    public float[][] getBiases() {
        return parent.getBiases(index);
    }
    
    @Override
    public NetworkStub getStub() {
        return parent.getStub(index);
    }
    
    /**
     * 
     * @return 
     */
    public Map<String, Object> getProperties() {
        return properties;
    }    

    @Override
    public boolean[] getCondition(int layer) {
        return parent.getCondition(layer);
    }
}
