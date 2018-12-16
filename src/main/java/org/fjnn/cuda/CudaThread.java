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
package org.fjnn.cuda;

import java.util.Stack;
import java.util.concurrent.ConcurrentHashMap;
import jcuda.driver.*;
import org.fjnn.util.Rng;

/**
 *
 * @author ahmed
 */
public class CudaThread {
    private final static ConcurrentHashMap<Long, Stack<Integer>> ThreadDeviceIdMap = new ConcurrentHashMap<>();
    
    public static int getThreadDeviceId() {
        long threadId = Thread.currentThread().getId();
        
        if(!ThreadDeviceIdMap.containsKey(threadId))
            return -1;
        
        Stack<Integer> s = ThreadDeviceIdMap.get(threadId);
        
        if(s.isEmpty())
            return -1;
        
        return s.peek();
    }
    
    /**
     * Prepares thread to run GPU code
     * Call finalizeThread() when done
     * @return 
     */
    public static int prepareThread() {
        int device = Rng.nextInt(CudaEngine.getDeviceCount());
        
        prepareThread(device);
        
        return device;
    }

    /**
     * Prepares thread to run GPU code on a specific device
     * Call finalizeThread() when done
     * @param deviceId
     * @return 
     */
    public static void prepareThread(int deviceId) {
        CUcontext context = CudaEngine.getContext(deviceId);

        long threadId = Thread.currentThread().getId();
        
        if(!ThreadDeviceIdMap.containsKey(threadId))
            ThreadDeviceIdMap.put(threadId, new Stack<>());
        
        Stack s = ThreadDeviceIdMap.get(threadId);
        s.push(deviceId);
        
        JCudaDriver.cuCtxPushCurrent(context);
    }

    /**
     * Cleans up the thread from GPU context
     * @param context 
     * @return  
     */
    public static void finalizeThread() {
        long threadId = Thread.currentThread().getId();
        
        if(!ThreadDeviceIdMap.containsKey(threadId))
            throw new RuntimeException("Thread does not have Context");

        Stack<Integer> s = ThreadDeviceIdMap.get(threadId);
        
        if(s.isEmpty())
            throw new RuntimeException("Thread does not have Context");

        CUcontext context = new CUcontext();
        JCudaDriver.cuCtxPopCurrent(context);

        if(!context.equals(CudaEngine.getContext(s.pop())))
            throw new RuntimeException("Invalid Context");
    }
}
