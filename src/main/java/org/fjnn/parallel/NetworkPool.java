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
    
    Map<Integer, LayeredNetwork> indexing;
            
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
        
        for(MultiNetwork n : pool) {
            n.build();
            
            for(int i=0; i < n.size; i++)
                indexing.put(index++, n.getNetwork(i));
        }
        
        if(index != size)
            throw new RuntimeException("Fix me");
        
        return this;
    }
    
    public MultiNetwork getMultiNetwork(int index) {
        return pool[index];
    }
    
    public LayeredNetwork getNetwork(int index) {
        return indexing.get(index);
    }
    
    public int multiSize() {
        return pool.length;
    }
    
    public int size() {
        return size;
    }
}
