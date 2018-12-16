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

import java.util.HashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUctx_flags;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.fjnn.cuda.CudaEngine.CUdeviceptr2D;

/**
 *
 * @author ahmed
 */
public class CudaDevice {
    private final int deviceId;
    private final CUdevice device;
    private final CUcontext context;

    private final int maxThreadsPerBlock;

    private final HashMap<String, CudaModule> modules;
    private final HashMap<String, CudaResource> resources;
    
    ConcurrentLinkedQueue<CUdeviceptr> freeResources;
    ConcurrentLinkedQueue<CUdeviceptr2D> freeResources2D;
    
    final long TEMP_RESOURCE_SIZE = 4092;
    
    CudaDevice(int deviceId) {
        this.deviceId = deviceId;
        
        this.device = new CUdevice();
        JCudaDriver.cuDeviceGet(this.device, deviceId);

        this.context = new CUcontext();
        JCudaDriver.cuCtxCreate(this.context, CUctx_flags.CU_CTX_SCHED_AUTO, device);
        
        modules = new HashMap<>();
        
        cudaDeviceProp prop = new cudaDeviceProp();
        JCuda.cudaGetDeviceProperties(prop, deviceId);

        maxThreadsPerBlock = prop.maxThreadsPerBlock;
        
        resources = new HashMap<>();
        freeResources = new ConcurrentLinkedQueue<>();
        
        JCudaDriver.cuCtxPopCurrent(context);
    }
        
    CudaModule getModule(String name) {
        if(!modules.containsKey(name))
            loadModule(name);
        
        return modules.get(name);        
    }
    
    synchronized void storeResource(String name, float[] array) {
        /* ensure we have the right context */
        JCudaDriver.cuCtxPushCurrent(context);
        
        CUdeviceptr ptr = CudaUtil.toGPU(array);
        
        if(resources.containsKey(name))
            JCudaDriver.cuMemFree(resources.get(name).ptr);
        
        CudaResource resource = new CudaResource(name, deviceId, ptr, array.length, Sizeof.FLOAT);
        
        resources.put(name, resource);

        JCudaDriver.cuCtxPopCurrent(context);
    }
    
    synchronized void deleteResource(String name) {
        if(resources.containsKey(name)) {
            JCudaDriver.cuMemFree(resources.get(name).ptr);
            resources.remove(name);
        }
    }

    synchronized void clearResources() {
        for(CudaResource resource : resources.values()) {
            JCudaDriver.cuMemFree(resource.ptr);
        }
        
        resources.clear();
    }
        
    CudaResource getResource(String name) {
        return resources.get(name);
    }
    
    CUcontext getContext() {
        return context;
    }
    
    int getMaxThreadsPerBlock() {
        return maxThreadsPerBlock;
    }

    private synchronized void loadModule(String name) {
        if(modules.containsKey(name))
            return;
        
        modules.put(name, new CudaModule(name));
    }
    
    CUdeviceptr allocSharedResource(int size) {
        if(size > TEMP_RESOURCE_SIZE)
            throw new RuntimeException("Resource required exceeds: " + TEMP_RESOURCE_SIZE);
       
        CUdeviceptr ptr = freeResources.poll();
        
        if(ptr == null) {
            ptr = new CUdeviceptr();
            JCudaDriver.cuMemAlloc(ptr, TEMP_RESOURCE_SIZE * Sizeof.FLOAT);
        }
        
        return ptr;
    }
    
    void freeSharedResource(CUdeviceptr ptr) {
        freeResources.add(ptr);
    }
        
    int getId() {
        return deviceId;
    }
}
