/*
 * The MIT License
 *
 * Copyright 2024 ahmed.
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
package org.fjnn.normalizer;

import java.util.Map;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.base.ModelComponent;

/**
 *
 * @author ahmed
 */
public abstract class Normalizer extends ModelComponent {
    protected final float epsilon = 1e-5f;
    
    protected final int neurons;
    
    public Normalizer() {
        this(0);
    }
    
    protected Normalizer(int neurons) {
        this.neurons = neurons;
    }
    
    public abstract Normalizer withNeurons(int neurons);
        
    public abstract void compute(float[] input, int count);
    
    public abstract void computeGPU(CUdeviceptr ptr, long count, CUstream stream);

    public static Normalizer deserialize(Map serialized) {
        String type = (String)serialized.get("type");
        
        switch (type) {
            case "LayerNormalizer":
                return LayerNormalizer.deserialize(serialized);
                
            // Add other normalizer types here as they're implemented
            // case "BatchNormalizer":
            //     return BatchNormalizer.deserialize(serialized);
            // case "InstanceNormalizer":
            //     return InstanceNormalizer.deserialize(serialized);
            
            default:
                throw new RuntimeException("Unknown normalizer type: " + type);
        }
    }
    
    /**
     * Memory required for normalizer parameters
     * @return 
     */
    public abstract long getParameterCount();

    /**
     * Memory required during forward pass
     * @param batchSize
     * @return 
     */
    public abstract long getForwardMemoryRequired(int batchSize);

    /**
     * Memory required during backward pass
     * @param batchSize
     * @return 
     */
    public abstract long getBackwardMemoryRequired(int batchSize);
}
