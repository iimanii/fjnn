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

import jcuda.driver.CUstream;
import org.fjnn.activation.Activation;

/**
 *
 * @author ahmed
 */
public abstract class BaseLayer {
    /* number of neurons in this layer */
    protected final int neurons;
    
    /* number of neurons in the next layer */
    protected final int links;

    /* activation function for this layer */
    protected final Activation activation;
    
    /* conditional activation */
    protected final boolean[] condition;
    
    /* wether or not this layer is the output layer */
    protected final boolean isOutput;
    
    /* does this layer has a bias */
    protected final boolean hasBias;
    
    /* can we call computeGPU */
    protected boolean gpuReady;
    
    protected BaseLayer(Activation activation, int neurons, int links, boolean hasBias, boolean[] condition) {        
        this.neurons = neurons;
        this.links = links;
        this.hasBias = hasBias;
        this.activation = activation;

        this.isOutput = links == 0;
        
        this.gpuReady = false;
        
        this.condition = condition;
    }
    
    /**
     * Gets activation function for this layer
     * @return 
     */
    public Activation getActivation() {
        return activation;
    }
    
    /**
     * 
     * @return 
     */
    public boolean[] getCondition() {
        return condition;
    }

    /**
     * @return Number of neurons
     */
    public int size() {
        return neurons;
    }
    
    /**
     * 
     * @return Number of neurons + bias
     */    
    public int totalSize() {
        return neurons + (hasBias ? 1 : 0);
    }
    
    /**
     * @return Whether or not this layer has bias node
     */
    public boolean hasBias() {
        return hasBias;
    }
    
    public void randomize() {
        randomize(-1, 1);
    }
    
    public abstract void randomize(float min, float max);

    
    public abstract void prepareGPU(CUstream stream);
        
    public abstract void freeCPU();
    
    public abstract void freeGPU();
}
