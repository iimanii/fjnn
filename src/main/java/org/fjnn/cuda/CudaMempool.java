/*
 * The MIT License
 *
 * Copyright 2023 Ahmed Tarek.
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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.JCudaDriver;

/**
 *
 * @author ahmed
 */
public class CudaMempool {
    
    static long DEFAULT_EXPIRY_MS = 60 * 1000;

    void free() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    static class ExpiringCUdeviceptr {
        CUdeviceptr ptr;
        long expireAt;
        long size;

        public ExpiringCUdeviceptr(CUdeviceptr ptr, long size, long expireAt) {
            this.ptr = ptr;
            this.size = size;
            this.expireAt = expireAt;
        }
    }
    
    static class PoolStorage {
        final ConcurrentHashMap<Long, Queue<ExpiringCUdeviceptr>> pool = new ConcurrentHashMap<>();
        final ConcurrentHashMap<CUdeviceptr, Long> lookup = new ConcurrentHashMap<>();
        final AtomicLong cachedsize = new AtomicLong();
        
        synchronized void register(CUdeviceptr ptr, long size) {
            lookup.put(ptr, size);            
        }
        
        synchronized void put(CUdeviceptr ptr) {
            Long size = lookup.get(ptr);
            
            if(size == null)
                throw new RuntimeException("Unknown size for ptr: " + ptr);
            
            if(!pool.containsKey(size))
                pool.putIfAbsent(size, new LinkedBlockingQueue<>());
            
            ExpiringCUdeviceptr exPtr = new ExpiringCUdeviceptr(ptr, size, System.currentTimeMillis() + DEFAULT_EXPIRY_MS);
            pool.get(size).add(exPtr);
            cachedsize.addAndGet(size);
        }
        
        synchronized CUdeviceptr get(long size) {
            if(!pool.containsKey(size))
                return null;
            
            ExpiringCUdeviceptr exPtr = pool.get(size).poll();
            
            if(exPtr == null)
                return null;
            
            cachedsize.addAndGet(-exPtr.size);
            
            return exPtr.ptr;
        }
        
        synchronized long clean(long minClearSize) {
            long cleanCount = 0;
            
            long t = System.nanoTime();
            long currentTime = System.currentTimeMillis();
            
            for(Iterator<Map.Entry<Long, Queue<ExpiringCUdeviceptr>>> it = pool.entrySet().iterator(); it.hasNext();) {
                Map.Entry<Long, Queue<ExpiringCUdeviceptr>> e = it.next();
                Queue<ExpiringCUdeviceptr> q = e.getValue();
                
                while(!q.isEmpty()) {
                    ExpiringCUdeviceptr c = q.peek();

                    if(c.expireAt > currentTime)
                        break;
                    
                    q.poll();
                    
                    cachedsize.addAndGet(-c.size);
                    lookup.remove(c.ptr);
                    
                    JCudaDriver.cuMemFree(c.ptr);

                    cleanCount += c.size;
                }
                
                if(q.isEmpty())
                    it.remove();
            }
            
            int loops = 0;
            
            if(minClearSize > cleanCount) {
                List<Queue<ExpiringCUdeviceptr>> l = new ArrayList<>(pool.values());
                int i = 0;
                int len = l.size();
                
                while(cachedsize.get() > 0 && minClearSize > cleanCount) {
                    Queue<ExpiringCUdeviceptr> q = l.get(i);
                    
                    ExpiringCUdeviceptr c = q.poll();
                    
                    if(c != null) {
                        cachedsize.addAndGet(-c.size);
                        lookup.remove(c.ptr);

                        JCudaDriver.cuMemFree(c.ptr);

                        cleanCount += c.size;
                    }
                    
                    i = (i + 1) % len;
                    
                    if(loops++ > 5000)
                        System.out.println("problem");
                }
            }
            
            long t2 = System.nanoTime();
//            System.out.println("free memory: " + cleanCount / 1e6f + " " + cachedsize.get() / 1e6f + " " + (t2-t) / 1e6f);
            
            return cleanCount;
        }
    }
    
    final PoolStorage storage;
    final AtomicLong totalAllocations;
    long maxPoolMemory;
    final ReentrantLock createLock;
    
    public CudaMempool(long maxPoolMemory) {
        this.storage = new PoolStorage();
        this.maxPoolMemory = maxPoolMemory;
        this.totalAllocations = new AtomicLong();
        this.createLock = new ReentrantLock();
    }
    
    public CUdeviceptr getFloat(long size) {
        return get(size * CudaUtil.FLOAT_SIZE);
    }
    
    public CUdeviceptr get(long size) {
        CUdeviceptr ptr = storage.get(size);
        
        if(ptr != null)
            return ptr;
        
        createLock.lock();
        
        try {
            ptr = storage.get(size);

            if(ptr != null)
                return ptr;
            
            /* create new */
            long available = maxPoolMemory - totalAllocations.get();

            if(available < size) {
//                System.out.printf("creating: %.4f, %.4f %.4f %.4f\n", size / 1e6f, totalAllocations.get() / 1e6f, storage.cachedsize.get() / 1e6f, available / 1e6f);
                totalAllocations.addAndGet(-storage.clean(size - available));
            }
            
            ptr = CudaUtil.createByte(size);
//                System.out.println("creating: " + ptr);

            totalAllocations.addAndGet(size);
            storage.register(ptr, size);

            return ptr;
        } finally {
            createLock.unlock();
        }
    }
    
    public void free(CUdeviceptr ptr) {
        storage.put(ptr);
    }
    
    void setMaxMemory(long maxPoolMemory) {
        this.maxPoolMemory = maxPoolMemory;
    }
    
    public void printMempoolStats() {
        System.out.printf("Mempool: %.4f %.4f %.4f %.4f %.4f\n", 
                                               maxPoolMemory / 1e6f,
                                      totalAllocations.get() / 1e6f, 
                                    storage.cachedsize.get() / 1e6f, 
         (totalAllocations.get() - storage.cachedsize.get()) / 1e6f, 
                    (maxPoolMemory - totalAllocations.get()) / 1e6f);        
    }
}
