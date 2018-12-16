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
package org.fjnn.network;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.activation.Activation;
import org.fjnn.base.BaseLayer;
import org.fjnn.serializer.LayerStub;

/**
 *
 * @author ahmed
 */
public abstract class Layer extends BaseLayer {
    
    public Layer(Activation activation, int neurons, int links, boolean hasBias, boolean[] conditional) {
        super(activation, neurons, links, hasBias, conditional);
    }
    

    protected abstract float getWeight(int from, int to);

    protected abstract void setWeight(int from, int to, float value);
    
    protected abstract float[][] getWeights();

    protected abstract void setWeights(float[][] values);
    
    
    protected abstract float getBias(int to);

    protected abstract void setBias(int to, float value);
    
    protected abstract float[] getBiases();
    
    protected abstract void setBiases(float[] values);
    
    
    protected abstract float[] feedForward(float[] input);
    
    protected abstract CUdeviceptr feedForwardGPU(CUdeviceptr ptr, CUstream stream);
    
    
    protected abstract LayerStub getStub();
}
