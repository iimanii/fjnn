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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import org.fjnn.activation.Activation;
import org.fjnn.base.LayeredNetwork;
import org.fjnn.cuda.CudaEngine;

/**
 *
 * @author ahmed
 */
public class NetworkPool {
    MultiNetwork[] pool;

    int size;
    
    private static class NetworkIndex {
        int index;
        int parent;
        int subindex;               /* index of this network inside the multinetwork */
        LayeredNetwork network;

        public NetworkIndex(int index, int parent, int subindex, LayeredNetwork network) {
            this.index = index;
            this.parent = parent;
            this.subindex = subindex;
            this.network = network;
        }
    }
    
    Map<Integer, NetworkIndex> indexing;
            
    public NetworkPool(int size, boolean threadSafe) {
        this.size = size;
        this.indexing = new HashMap<>();
        
        int deviceCount = CudaEngine.getDeviceCount();
        boolean multiPerDevice = size / deviceCount > 16;
        
        ArrayList<MultiNetwork> list = new ArrayList<>(deviceCount * 2);
        
        int temp = size + deviceCount - 1;
        
        MultiNetwork n;
        
        for(int i=0; i < deviceCount; i++) {
            int count = (temp - i) / deviceCount;
            
            System.out.print("Device " + i);
            if(multiPerDevice) {
                n = new MultiNetwork((count + 1) / 2, threadSafe);
                n.deviceId = i;
                list.add(n);
                System.out.print(" " + (count + 1) / 2);
                
                n = new MultiNetwork(count / 2, threadSafe);
                n.deviceId = i;
                list.add(n);
                System.out.print(" " + count  / 2);
            } else {
                n = new MultiNetwork(count, threadSafe);
                n.deviceId = i;
                list.add(n);
                System.out.print(" " + count);
            }
            System.out.println();
        }
        
        pool = new MultiNetwork[list.size()];
        list.toArray(pool);
    }
    
    public NetworkPool addLayer(int neurons, Activation activation, boolean hasBias) {
        for(MultiNetwork n : pool)
            n.addLayer(neurons, activation, hasBias);
        
        return this;
    }
    
    public NetworkPool build() {
        int index = 0;
        
        for(int i=0; i < pool.length; i++) {
            MultiNetwork n = pool[i];
            n.build();
            
            for(int j=0; j < n.size; j++, index++)
                indexing.put(index, new NetworkIndex(index, i, j, n.getNetwork(j)));
        }
        
        if(index != size)
            throw new RuntimeException("Fix me");
        
        return this;
    }
//    
//    public MultiNetwork getMultiNetwork(int index) {
//        return pool[index];
//    }
    
    public LayeredNetwork getNetwork(int index) {
        return indexing.get(index).network;
    }
    
//    public int multiSize() {
//        return pool.length;
//    }
    
    
    /**
     * 
     * @return true on success
     */
    public boolean prepareGPU() {
        for(MultiNetwork m : pool)
            if(!m.prepareGPU())
                return false;
        
        return true;
    }
        
    public float[][] computeGPU(float[][] input) {
        float[][][] prepared = prepareInput(input);
        float[][][] output = new float[pool.length][][];
        
        for(int i=0; i < pool.length; i++)
            output[i] = pool[i].computeGPU(prepared[i]);
        
        return prepareOutput(output);
        
    }
    
    public int size() {
        return size;
    }
    
    private float[][][] prepareInput(float[][] input) {
        float[][][] result = new float[pool.length][][];
        
        for(int i=0; i < pool.length; i++)
            result[i] = new float[pool[i].size][];
        
        for(int i=0; i < input.length; i++) {
            NetworkIndex index = indexing.get(i);
            result[index.parent][index.subindex] = input[i];
        }
        
        return result;
    }

    private float[][] prepareOutput(float[][][] output) {
        float[][] result = new float[size][];
        
        for(int i=0; i < size; i++) {
            NetworkIndex index = indexing.get(i);
            result[i] = output[index.parent][index.subindex];
        }

        return result;
    }
}
