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
import java.util.concurrent.Semaphore;
import jcuda.driver.*;
import jcuda.jcublas.cublasHandle;
import jcuda.jcurand.curandGenerator;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.fjnn.util.Rng;

/**
 *
 * @author ahmed
 */
public class CudaEngine {
    private static int DeviceCount;
    private static CudaDevice[] DeviceList;

    private static boolean initialized;
    
    private final static ThreadLocal<Stack<Integer>> THREAD_DEVICE_ID = ThreadLocal.withInitial(() -> {
        return new Stack<>();
    });
    
    public static synchronized void init() {
        if(initialized)
            return;
        
        /* throw exceptions on failure */
        JCudaDriver.setExceptionsEnabled(true);
        JCuda.setExceptionsEnabled(true);
        
        /* init driver */
        JCudaDriver.cuInit(0);
        
        /* get device count */
        int[] count = new int[1];
        JCudaDriver.cuDeviceGetCount(count);
        
        DeviceCount = count[0];        
        DeviceList = new CudaDevice[DeviceCount];

        int minThreadsPerBlock = DeviceCount == 0 ? 1024 : Integer.MAX_VALUE;
        int[] maxGridSize = {Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE};
        
        ExecutorService threadPool = Executors.newCachedThreadPool();
        ArrayList<Future<CudaDevice>> futures = new ArrayList<>();

        for(int i=0; i < DeviceCount; i++) {
            int id = i;
//            futures.add(threadPool.submit(() -> new CudaDevice(id)));
            DeviceList[i] = new CudaDevice(id);
            CudaDevice d = DeviceList[i];
            minThreadsPerBlock = Math.min(d.properties.maxThreadsPerBlock, minThreadsPerBlock);
            int[] deviceMaxGridSize = d.properties.maxGridSize;
            
            for(int j=0; j < maxGridSize.length; j++)
                maxGridSize[j] = Math.min(maxGridSize[j], deviceMaxGridSize[j]);            
        }
        
        for(Future<CudaDevice> f : futures) {
            CudaDevice d;
            try {
                d = f.get();
            } catch (InterruptedException | ExecutionException ex) {
                throw new RuntimeException(ex);
            }

            minThreadsPerBlock = Math.min(d.properties.maxThreadsPerBlock, minThreadsPerBlock);
            DeviceList[d.getId()] = d;
            int[] deviceMaxGridSize = d.properties.maxGridSize;
            
            for(int i=0; i < maxGridSize.length; i++)
                maxGridSize[i] = Math.min(maxGridSize[i], deviceMaxGridSize[i]);
        }
        try {
            CudaModule.saveUtilFile(minThreadsPerBlock, maxGridSize);
            CudaModule.saveMatrixFile(128, 128, 16, 128, 4, 2, 1, false);
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
        
        threadPool.shutdown();
        initialized = true;
    }

    public static synchronized void free() {
        for(CudaDevice cd : DeviceList)
            cd.free();
        
        DeviceList = null;
        initialized = false;
    }
    
    
    static CUcontext getContext(int deviceId) {        
        return DeviceList[deviceId].getContext();
    }
    
    public static CUfunction getKernel(String module, String function, int deviceId) {
        return DeviceList[deviceId].getModule(module).getFunction(function);
    }
    
    public static void loadModule(String module, int deviceId) {
        DeviceList[deviceId].loadModule(module);
    }
    
    public static void reloadModule(String module, int deviceId, boolean recompile) {
        DeviceList[deviceId].reloadModule(module, recompile);
    }
//    public static CUstream getStream(int deviceId) {
//        return DeviceList.get(deviceId).getStream();
//    }
//    

    public static cublasHandle getCublasHandle(int deviceId) {
        return DeviceList[deviceId].getCublasHandle();
    }
    
    public static curandGenerator getCurandGenerator(int deviceId) {
        return DeviceList[deviceId].getCurandGenerator();
    }

    public static cudaDeviceProp getDeviceProperties(int deviceId) {
        return DeviceList[deviceId].properties;
    }

    public static CUdeviceptr getMempool(long size) {
        return getMempool(getThreadDeviceId(), size);
    }
    
    public static CUdeviceptr getMempool(int deviceId, long size) {
        return DeviceList[deviceId].mempool.get(size);
    }
    
    public static CUdeviceptr getMempoolFloat(long size) {
        return getMempoolFloat(getThreadDeviceId(), size);
    }
    
    public static CUdeviceptr getMempoolFloat(int deviceId, long size) {
        return DeviceList[deviceId].mempool.getFloat(size);
    }
    
    public static void freeMempool(CUdeviceptr ptr) {
        freeMempool(getThreadDeviceId(), ptr);
    }
    
    public static void freeMempool(int deviceId, CUdeviceptr ptr) {
        DeviceList[deviceId].mempool.free(ptr);
    }
    
    public static Semaphore getMemLock(int deviceId) {
        return DeviceList[deviceId].memlock;
    }
    
    public static void setMaxUsedMemory(int deviceId, long bytes) {
        DeviceList[deviceId].setMaxUsedMemory(bytes);
    }
    
    public static long getFreeMemory(int deviceId) {
        long[] free = new long[1];
        long[] total = new long[1];
        
        CudaEngine.prepareThread(deviceId);
        JCudaDriver.cuMemGetInfo(free, total);
        CudaEngine.finalizeThread();

        return free[0];
    }
    
    public static void printMempoolStats(int deviceId) {
        DeviceList[deviceId].mempool.printMempoolStats();
    }
    
    /**
     * Prepares thread to run GPU code on a specific device
     * Call finalizeThread() when done
     * @param deviceId 
     */
    public static void prepareThread(int deviceId) {
        CUcontext context = CudaEngine.getContext(deviceId);

        THREAD_DEVICE_ID.get().push(deviceId);
        
        JCudaDriver.cuCtxPushCurrent(context);
    }

    /**
     * Cleans up the thread from GPU context  
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

    public static CUstream aquireStream(int deviceId) {
        return DeviceList[deviceId].aquireStream();
    }

    public static void releaseStream(int deviceId, CUstream stream) {
        DeviceList[deviceId].releaseStream(stream);        
    }
    
    /**
     * @param deviceId
     * @return Number of threads per block for the device attached to this thread
     */
    public static int getMaxThreadsPerBlock(int deviceId) {
        return DeviceList[deviceId].properties.maxThreadsPerBlock;
    }
    
    /**
    * @param deviceId
    * @return Number of threads per block for the device attached to this thread
    */
    public static int[] getMaxGridSize(int deviceId) {
        return DeviceList[deviceId].properties.maxGridSize;
    }
    
    /**
     * @param deviceId
     * @return Number of threads per block for the device attached to this thread
     */
    public static int getMultiProcessorCount(int deviceId) {
        return DeviceList[deviceId].properties.multiProcessorCount;
    }
    
    /**
     * @return Number of GPUs accessible
     */
    public static int getDeviceCount() {
        return DeviceCount;
    }
    
}
