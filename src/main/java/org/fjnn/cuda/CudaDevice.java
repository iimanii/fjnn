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
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import static jcuda.jcublas.cublasMath.CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.jcurand.curandRngType;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaDeviceProp;
import org.fjnn.util.rng;

/**
 *
 * @author ahmed
 */
public class CudaDevice {
    private final int deviceId;
    private final CUdevice device;
    private final CUcontext context;
    private final HashMap<String, CudaModule> modules;
    
    private int streamSelect;
    private final Queue<CUstream> streams;
    private final cublasHandle cublasHandle;     
    private final curandGenerator curandGenerator;
    private final int cublasMemRequirement;
    
    protected CudaPool pool;
    protected Semaphore memlock;    
    protected final CudaMempool mempool;
    protected final cudaDeviceProp properties;
    
    CudaDevice(int deviceId) {
        this.deviceId = deviceId;
        
        this.device = new CUdevice();
        JCudaDriver.cuDeviceGet(this.device, deviceId);

        this.context = new CUcontext();
//        JCudaDriver.cuCtxCreate(this.context, CUctx_flags.CU_CTX_SCHED_AUTO, device);
//        JCudaDriver.cuCtxPopCurrent(context);
        JCudaDriver.cuDevicePrimaryCtxRetain (this.context, device);
        JCudaDriver.cuCtxPushCurrent(context);
        
        modules = new HashMap<>();
        
        properties = new cudaDeviceProp();
        JCuda.cudaGetDeviceProperties(properties, deviceId);
        
        streams = new ArrayDeque<>();
        
        curandGenerator = new curandGenerator();
        JCurand.curandCreateGenerator(curandGenerator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
        JCurand.curandSetPseudoRandomGeneratorSeed(curandGenerator, rng.nextLong(Long.MAX_VALUE));
    
        long[] free = new long[1];
        long[] total = new long[1];
        
        JCudaDriver.cuMemGetInfo(free, total);
        
        cublasHandle tempCublas = new cublasHandle();
        JCublas2.cublasCreate(tempCublas);
        
        mempool = new CudaMempool(free[0]);
        memlock = new Semaphore((int) Math.floor(free[0] / 1024.0), true);
        
        JCudaDriver.cuMemGetInfo(free, total);
        
        long[] free_after = new long[1];
        
        cublasHandle = new cublasHandle();
        JCublas2.cublasCreate(cublasHandle);
        JCudaDriver.cuMemGetInfo(free_after, total);
        
        cublasMemRequirement = (int) (free[0] - free_after[0]);
        
        System.out.println("CUBLAS memory usage: " +  (cublasMemRequirement / 1e6f)); 
        
        JCublas2.cublasDestroy(tempCublas);
        
//        JCublas2.cublasSetMathMode(cublasHandle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
        JCudaDriver.cuCtxPopCurrent(context);
    }
        
    CudaModule getModule(String name, boolean cubin) {
        if(!modules.containsKey(name))
            loadModule(name, cubin);
        
        return modules.get(name);        
    }
    
    CUcontext getContext() {
        return context;
    }

//    CUstream getStream() {
//        if(streams.size() < properties.multiProcessorCount)
//            createStream();
//        
//        streamSelect = ++streamSelect % streams.size();
//        
//        return streams.get(streamSelect);
//    }

    cublasHandle getCublasHandle() {
        return cublasHandle;
    }

    curandGenerator getCurandGenerator() {
        return curandGenerator;
    }
    
    synchronized void setMaxUsedMemory(long bytes) {
        memlock = new Semaphore((int) Math.floor(bytes / 1024.0), true);
        mempool.setMaxMemory(bytes);
    }
    
    int getId() {
        return deviceId;
    }
    
    protected synchronized void loadModule(String name, boolean cubin) {
        if(modules.containsKey(name))
            return;
        
        try {
            modules.put(name, new CudaModule(name, cubin, false));
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }
    
    protected synchronized void reloadModule(String name, boolean cubin, boolean recompile) {
        if(modules.containsKey(name)) {
           modules.get(name).unload();
           modules.remove(name);
        }
        
        try {
            modules.put(name, new CudaModule(name, cubin, recompile));
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }
    
    protected synchronized void unloadModule(String name) {
        if(modules.containsKey(name)) {
           modules.get(name).unload();
           modules.remove(name);
        }
    }
    
    private final AtomicInteger streamCounter = new AtomicInteger();
    
    private CUstream createStream() {
//        if(streams.size() >= properties.multiProcessorCount)
//            return;
        
//        System.out.println("Stream created: " + streamCounter.incrementAndGet());
        
        CUstream stream = new CUstream();
        JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_NON_BLOCKING);
        return stream;
//        streams.add(stream);
    }
    
    synchronized void free() {
        for(CUstream cu : streams)
            JCudaDriver.cuStreamDestroy(cu);
        
        JCublas2.cublasDestroy(cublasHandle);
        JCurand.curandDestroyGenerator(curandGenerator);
        mempool.free();
    }

    CUstream aquireStream() {
        synchronized(streams) {
            return streams.isEmpty() ? createStream() : streams.poll();
        }
    }

    void releaseStream(CUstream stream) {
        synchronized(streams) {
            streams.add(stream);
        }
    }
}
