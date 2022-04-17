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

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import jcuda.driver.*;
import jcuda.runtime.JCuda;
import org.fjnn.util.Rng;

/**
 *
 * @author ahmed
 */
public class CudaEngine {
    private static int DeviceCount;
    private static Map <Integer, CudaDevice> DeviceList;

    private static boolean initialized;
    private static boolean usePinnedMemory = true;
    
    
    public static synchronized void init() {
        if(initialized)
            return;
        
        DeviceList = new HashMap<>();

        /* init driver */
        JCudaDriver.cuInit(0);
        
        /* throw exceptions on failure */
        JCudaDriver.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);
        
        /* get device count */
        int[] count = new int[1];
        JCudaDriver.cuDeviceGetCount(count);
        
        DeviceCount = count[0];
        
        int minThreadsPerBlock = DeviceCount == 0 ? 1024 : Integer.MAX_VALUE;
        
        ExecutorService threadPool = Executors.newCachedThreadPool();
        ArrayList<Future<CudaDevice>> futures = new ArrayList<>();

        for(int i=0; i < DeviceCount; i++) {
            int id = i;
            futures.add(threadPool.submit(() -> new CudaDevice(id)));
        }
        
        for(Future<CudaDevice> f : futures) {
            try {
                CudaDevice d = f.get();
                minThreadsPerBlock = Math.min(d.getMaxThreadsPerBlock(), minThreadsPerBlock);
                DeviceList.put(d.getId(), d);
            } catch (InterruptedException | ExecutionException ex) {
                throw new RuntimeException(ex);
            }
        }
        
        try {
            CudaModule.saveUtilFile(minThreadsPerBlock);
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
        
        threadPool.shutdown();
        initialized = true;
    }

    /**
     * 
     * @param deviceId
     * @param module
     * @param function
     * @return 
     */
    public static CUfunction getKernel(String module, String function, int deviceId) {
        return DeviceList.get(deviceId).getModule(module).getFunction(function);
    }

    /**
     * Get context for specific device
     * @param deviceId
     * @return 
     */
    static CUcontext getContext(int deviceId) {        
        return DeviceList.get(deviceId).getContext();
    }
    
    /**
     * 
     * @return Number of GPUs accessible
     */
    public static int getDeviceCount() {
        return DeviceCount;
    }
    
    /**
     * @param deviceId
     * @return Number of threads per block for the device attached to this thread
     */
    public static CUstream getStream(int deviceId) {
        return DeviceList.get(deviceId).getStream();
    }
    
    /**
     * @param deviceId
     * @return Number of threads per block for the device attached to this thread
     */
    public static int getMaxThreadsPerBlock(int deviceId) {
        return DeviceList.get(deviceId).getMaxThreadsPerBlock();
    }
    
    /**
     * @param deviceId
     * @return Number of threads per block for the device attached to this thread
     */
    public static int getMultiProcessorCount(int deviceId) {
        return DeviceList.get(deviceId).getMultiProcessorCount();
    }
    
    /**
     * 
     * @param state 
     */
    public static void setUsePinnedMemory(boolean state) {
        usePinnedMemory = state;
    }
    
    /**
     * 
     * @return 
     */
    public static boolean usePinnedMemory() {
        return usePinnedMemory;
    }
    
    private final static ThreadLocal<Stack<Integer>> THREAD_DEVICE_ID = ThreadLocal.withInitial(() -> {
        return new Stack<>();
    });
    
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

        THREAD_DEVICE_ID.get().push(deviceId);
        
        JCudaDriver.cuCtxPushCurrent(context);
    }

    /**
     * Cleans up the thread from GPU context
     * @param context 
     * @return  
     */
    public static void finalizeThread() {        
        Stack<Integer> s = THREAD_DEVICE_ID.get();
        
        if(s.isEmpty())
            throw new RuntimeException("Thread does not have context");
        
        CUcontext context = new CUcontext();
        JCudaDriver.cuCtxPopCurrent(context);

        if(!context.equals(CudaEngine.getContext(s.pop())))
            throw new RuntimeException("Invalid Context poped");
        
//        CUcontext context = new CUcontext();
//        do {
//            JCudaDriver.cuCtxGetCurrent(context);
//            
//            if(!context.equals(new CUcontext()))
//                JCudaDriver.cuCtxPopCurrent(context);
//            
//        } while(!context.equals(new CUcontext()));
    }
    
    public static int getThreadDeviceId() {
        Stack<Integer> s = THREAD_DEVICE_ID.get();
        
        if(s.isEmpty())
            return -1;
        
        return s.peek();
    }
}
