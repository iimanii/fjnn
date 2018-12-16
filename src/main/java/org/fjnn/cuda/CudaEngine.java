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

import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import jcuda.driver.*;
import jcuda.runtime.JCuda;

/**
 *
 * @author ahmed
 */
public class CudaEngine {
    private static int DeviceCount;
    private static Map <Integer, CudaDevice> DeviceList;
    private static boolean initialized;
    private static boolean usePinnedMemory = true;
    
    public static class CUdeviceptr2D {
        public CUdeviceptr ptr;
        public long pitch;

        public CUdeviceptr2D(CUdeviceptr ptr, long pitch_ptr) {
            this.ptr = ptr;
            this.pitch = pitch_ptr;
        }
    }
    
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
        
        ExecutorService threadPool = Executors.newCachedThreadPool();
        ArrayList<Future<CudaDevice>> futures = new ArrayList<>();

        for(int i=0; i < DeviceCount; i++) {
            int id = i;
            futures.add(threadPool.submit(new Callable<CudaDevice>() {
                @Override
                public CudaDevice call() throws Exception {
                    return new CudaDevice(id);
                }
            }));
        }
        
        for(Future<CudaDevice> f : futures) {
            try {
                CudaDevice d = f.get();
                DeviceList.put(d.getId(), d);
            } catch (InterruptedException | ExecutionException ex) {
                throw new RuntimeException(ex);
            }
        }
        
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
     * Stores array on all GPUs
     * @param name
     * @param array
     * @return 
     */
    public static void storeGlobal(String name, float[] array) {
        for(CudaDevice e : DeviceList.values()) {
            e.storeResource(name, array);
        }
    }

    /**
     * Remove from all devices
     * @param name 
     */
    public static void deleteGlobal(String name) {
        for(CudaDevice e : DeviceList.values()) {
            e.deleteResource(name);
        }
    }
    
    /**
     * Free all global memory
     */
    public static void clearGlobal() {
        for(CudaDevice e : DeviceList.values()) {
            e.clearResources();
        }        
    }

    /**
     * 
     * @param name
     * @param deviceId
     * @return 
     */
    public static CudaResource getResource(String name, int deviceId) {
        return DeviceList.get(deviceId).getResource(name);
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
    public static int getMaxThreadsPerBlock(int deviceId) {
        return DeviceList.get(deviceId).getMaxThreadsPerBlock();
    }
    
    /**
     * TODO: use CudaResource
     * @param size
     * @param deviceId
     * @return 
     */
    public static CUdeviceptr getSharedResource(int size, int deviceId) {
        return DeviceList.get(deviceId).allocSharedResource(size);
    }
    
    /**
     * TODO: use CudaResource
     * @param ptr
     * @param deviceId 
     */    
    public static void freeSharedResource(CUdeviceptr ptr, int deviceId) {
        DeviceList.get(deviceId).freeSharedResource(ptr);
    }
    
    /**
     * 
     * @param state 
     */
    public static void setUsePinnedMemory(boolean state) {
        usePinnedMemory = state;
    }
    
    public static boolean usePinnedMemory() {
        return usePinnedMemory;
    }
}
