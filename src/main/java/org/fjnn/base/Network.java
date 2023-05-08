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


import java.util.HashMap;
import java.util.Map;
import org.fjnn.activation.Activation;

/**
 *
 * @author ahmed
 */

public abstract class Network {
    protected final Map<String, Object> properties;
    protected final int inputSize;
    protected final int outputSize;
    protected final Activation outputActivation;
    
    public Network(int inputSize, int outputSize, Activation outputActivation) {
        this.properties = new HashMap<>();
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.outputActivation = outputActivation;
    }
    
    /**
     * 
     * @param name
     * @return 
     */
    public final Object getProperty(String name) {
        return properties.get(name);
    }
    
    public final Object getOrDefault(String name, Object defaultValue) {
        return properties.getOrDefault(name, defaultValue);
    }
    
    /**
     * 
     * @param name
     * @param object 
     */
    public final void setProperty(String name, Object object) {
        synchronized(properties) {
            properties.put(name, object);
        }
    }
    
    /**
     * 
     * @param name
     * @return 
     */
    public final boolean hasProperty(String name) {
        return properties.containsKey(name);
    }
    
    /**
     * 
     * @param name
     * @return 
     */
    public final Object removeProperty(String name) {
        return properties.remove(name);
    }
    
    /**
     * 
     * @return Number of inputs
     */
    public final int getInputSize() {
        return inputSize;
    }
    
    /**
     * 
     * @return Number of outputs
     */
    public final int getOutputSize() {
        return outputSize;
    }    
    
    /**
     * Randomize the network in the range -1[inclusive], 1[exclusive]
     */
    public void randomize() {
        randomize(-1, 1);
    }

    /**
     * Randomize the network in the range min[inclusive] to max[exclusive]
     * @param min
     * @param max
     */
    public abstract void randomize(float min, float max);
    
}
