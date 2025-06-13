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


import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import org.fjnn.activation.Activation;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.normalizer.Normalizer;

/**
 *
 * @author ahmed
 * @param <T>
 * @param <K>
 */

public abstract class Network <T extends Network> extends ModelComponent {
    protected final Map<String, Object> properties;
    protected final int inputSize;
    protected final int outputSize;
    protected final Activation outputActivation;
    protected final Normalizer outputNormalizer;
    
    /* a network can only be built once */
    protected boolean finalized;

    public Network(int inputSize, int outputSize, Activation outputActivation, Normalizer outputNormalizer) {
        this.properties = new ConcurrentHashMap<>();
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.outputActivation = outputActivation;
        this.outputNormalizer = outputNormalizer;
        
        if(inputSize == 0 || outputSize == 0)
            throw new RuntimeException("invalid input / output: " + inputSize + " " + outputSize);
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
    @Override
    public final int getInputDim() {
        return inputSize;
    }
    
    /**
     * 
     * @return Number of outputs
     */
    @Override
    public final int getOutputDim() {
        return outputSize;
    }    

    public abstract long getGPUPrepareMemoryRequired();
    
    public abstract long getGPUTrainingMemoryRequired(int count);
    
    public final boolean isFinalized() {
        return finalized;
    }
    
    public HashMap serialize(Set<String> ignoreProperties) {
        HashMap result = new HashMap();
        
        Map<String, Object> map = new HashMap<>();
        
        for(Map.Entry<String, Object> e : properties.entrySet())
            if(!ignoreProperties.contains(e.getKey()))
                map.put(e.getKey(), e.getValue());
        
        result.put("properties", map);
        result.put("type", this.getClass().getName());
        
        return result;
    }
    
    /**
     * Randomize the network in the range min[inclusive] to max[exclusive]
     * @param min
     * @param max
     */
    public abstract T randomize(float min, float max);
    
    public abstract T xavier();
    
    public abstract T kaiming();
    
    public final T copy(boolean copyWeights) {
        return copy(copyWeights, true);
    }
    
    public abstract T copy(boolean copyWeights, boolean createWeights);
    
    public abstract void copyWeights(T n);

    public abstract double compare(T n0);
    
    public abstract float[] compute(NetworkInput input);
    
    /* CPU compute */
    public abstract float[] compute(float[] input);
    
    public abstract float[] compute(float[] input, int count);
    
    public abstract float[] compute(FloatBuffer input);
    
    public abstract float[] compute(FloatBuffer input, int count);

    public abstract void crossOverMutate(T n0, T n1, float f, float mutationAmount, double m);

    public abstract void clipWeights(float clipMin, float clipMax);

    public abstract void updateWeightsFromGPU();
    
    /* GPU Stuff */
    public abstract void crossOverMutateGPU(T n0, T n1, float min, float max, double mutation, boolean nocopy);
    
    public abstract void clipWeightsGPU(float clipMin, float clipMax);
    
    protected void lockMemory(long size, int deviceId) {
        Semaphore lock = CudaEngine.getMemLock(deviceId);
        
        if(lock == null)
            throw new RuntimeException("Must specify maximum memory usage for device: " + deviceId);

        int memoryKB = (int)Math.ceil(size / 1024.0);
        
        try {
            lock.tryAcquire(memoryKB, 10, TimeUnit.SECONDS);
        } catch (InterruptedException ex) {
            throw new RuntimeException(ex);
        }
    }
    
    protected void releaseMemory(long size) {
        Semaphore lock = CudaEngine.getMemLock(getGPUDeviceId());
        
        int memoryKB = (int)Math.ceil(size / 1024.0);
        
        lock.release(memoryKB);
    }
    
}
