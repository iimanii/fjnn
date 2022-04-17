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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import jcuda.driver.CUcontext;
import jcuda.driver.CUctx_flags;
import jcuda.driver.CUdevice;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;

/**
 *
 * @author ahmed
 */
public class CudaDevice {
    private final int deviceId;
    private final CUdevice device;
    private final CUcontext context;
    private final List<CUstream> streams;
    private final cudaDeviceProp properties;

    private final HashMap<String, CudaModule> modules;
    
    CudaDevice(int deviceId) {
        this.deviceId = deviceId;
        
        this.device = new CUdevice();
        JCudaDriver.cuDeviceGet(this.device, deviceId);

        this.context = new CUcontext();
//        JCudaDriver.cuCtxCreate(this.context, CUctx_flags.CU_CTX_SCHED_AUTO, device);
//        JCudaDriver.cuCtxPopCurrent(context);
        JCudaDriver.cuDevicePrimaryCtxRetain (this.context, device);
        
        modules = new HashMap<>();
        
        properties = new cudaDeviceProp();
        JCuda.cudaGetDeviceProperties(properties, deviceId);
        
        streams = new ArrayList<>();
    }
        
    CudaModule getModule(String name) {
        if(!modules.containsKey(name))
            loadModule(name);
        
        return modules.get(name);        
    }
    
    CUcontext getContext() {
        return context;
    }

    CUstream getStream() {
        if(streams.size() < properties.multiProcessorCount)
            createStream();
        
        int r = (int)(Math.random() * streams.size());
        
        return streams.get(r);
    }
    
    int getId() {
        return deviceId;
    }
        
    public int getMaxThreadsPerBlock() {
        return properties.maxThreadsPerBlock;
    }
    
    public int getMultiProcessorCount() {
        return properties.multiProcessorCount;
    }
    
    private synchronized void loadModule(String name) {
        if(modules.containsKey(name))
            return;
        
        try {
            modules.put(name, new CudaModule(name));
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    private synchronized void createStream() {
        if(streams.size() >= properties.multiProcessorCount)
            return;
        
        CUstream stream = new CUstream();
        JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_NON_BLOCKING);
        streams.add(stream);
    }
}
