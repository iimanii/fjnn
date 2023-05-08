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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import jcuda.driver.CUdeviceptr;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class CudaPool {
    final long size;
    final CUdeviceptr pool;
    final AtomicLong available;
    final ReentrantLock memLock;
    final LinkedList<Object> threadQueue;
    
    final memoryBlock head;
    final HashMap<String, memoryBlock> blockMap;
    final TreeSet<memoryBlock> freeBlocks;

    static class memoryBlock {
        final long size;
        final long bytePtr;
        boolean used;
        
        memoryBlock nextBlock;
        memoryBlock prevBlock;

        public memoryBlock(long size, long bytePtr) {
            this.size = size;
            this.bytePtr = bytePtr;
        }
        
        public void setNext(memoryBlock el) {
            nextBlock = el;
            
            if(el != null)
                el.prevBlock = this;
        }
        
        public void setPrevious(memoryBlock el) {
            prevBlock = el;
            
            if(el != null)
                el.nextBlock = this;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (obj == null) {
                return false;
            }
            if (getClass() != obj.getClass()) {
                return false;
            }
            
            final memoryBlock other = (memoryBlock) obj;
            
            return this.bytePtr == other.bytePtr;
        }

        @Override
        public int hashCode() {
            int hash = 3;
            hash = 53 * hash + (int) (this.bytePtr ^ (this.bytePtr >>> 32));
            return hash;
        }
    }
    
    public CudaPool(long poolSize) {        
        size = poolSize;
        pool = CudaUtil.createByte(poolSize);
        available = new AtomicLong(poolSize);
        memLock = new ReentrantLock(true);
        threadQueue = new LinkedList();
        freeBlocks = new TreeSet<>((o1, o2) -> {
            return Long.compare(o1.size, o2.size);
        });
        blockMap = new HashMap<>();
        
        head = new memoryBlock(0, -1);
        head.used = true;
        
        memoryBlock first = new memoryBlock(size, 0);
        head.setNext(first);
        
        freeBlocks.add(first);
    }
    
    public CUdeviceptr getFloat(long size) {
        return get(size * CudaUtil.FLOAT_SIZE);
    }
    
    public CUdeviceptr tryGet(long size) {
        long aligned = CudaUtil.alignLength(size, 256);
        
        return aquire(aligned);
    }
    
    public CUdeviceptr get(long size) {
        long aligned = CudaUtil.alignLength(size, 256);
        
        CUdeviceptr ptr;
        
        while((ptr = aquire(aligned)) == null)
            queueThread();
        
        return ptr;
    }
    
    public void free(CUdeviceptr ptr) {
        memoryBlock block = blockMap.get(ptr.toString());
        
        if(block == null)
            throw new RuntimeException("invalid device pointer");
        
        memLock.lock();
        try {
            /* merge everything before and after */
            List<memoryBlock> toMerge = new ArrayList<>();
            toMerge.add(block);
            
            memoryBlock c = block;
            while(c.prevBlock != null && !c.prevBlock.used) {
                c = c.prevBlock;
                toMerge.add(c);
            }
            memoryBlock prev = c.prevBlock;
            
            c = block;
            while(c.nextBlock != null && !c.nextBlock.used) {
                c = c.nextBlock;
                toMerge.add(c);
            }
            memoryBlock next = c.nextBlock;
            
            memoryBlock merged = mergeBlocks(toMerge, block.bytePtr);
            
            merged.setPrevious(prev);
            merged.setNext(next);

            /* add new block to free blocks */
            for(memoryBlock mb : toMerge)
                freeBlocks.remove(mb);
            
            freeBlocks.add(merged);
            blockMap.remove(ptr.toString());
//            
//            synchronized(threadQueue) {
//                System.out.println("freeing threads: " + threadQueue.size());
//                while(!threadQueue.isEmpty()) {
//                    Object o = threadQueue.poll();
//                    synchronized(o) {
//                        o.notify();
//                    }
//                }
//            }
            
            available.addAndGet(block.size);
        } finally {
            memLock.unlock();
        }
    }
    
    private CUdeviceptr aquire(long size) {
        memLock.lock();
        
        System.out.println("available: " + available.get() + " " + size);
        try {
            memoryBlock selected = freeBlocks.ceiling(new memoryBlock(size, -1));
            
            if(selected == null) {
                System.out.println(CudaEngine.getThreadDeviceId() + " null " + freeBlocks.floor(new memoryBlock(size, -1)).size);
                return null;
            }
            
            freeBlocks.remove(selected);
            
            if(selected.size > size) {
                memoryBlock used = new memoryBlock(size, selected.bytePtr);
                memoryBlock excess = new memoryBlock(selected.size - size, selected.bytePtr + size);
                used.setNext(excess);
                used.setPrevious(selected.prevBlock);
                excess.setNext(selected.nextBlock);
                
                selected = used;
                freeBlocks.add(excess);
            }
            
            selected.used = true;
            
            CUdeviceptr ptr = pool.withByteOffset(selected.bytePtr);
            blockMap.put(ptr.toString(), selected);
            available.addAndGet(-size);
            
            return ptr;
        } finally {
            memLock.unlock();
        }
    }

    private void queueThread() {
//        Object lock = new Object();
//        
//        synchronized(threadQueue) {
//            threadQueue.add(lock);
//        }
        
        try {
            Thread.sleep(50);
        } catch (InterruptedException ex) {
            throw new RuntimeException(ex);
        }
    }
    
    private memoryBlock mergeBlocks(List<memoryBlock> list, long currentPtr) {
        long s = 0;
        long p = currentPtr;
        
        for(memoryBlock mb : list) {
            s += mb.size;
            
            if(mb.bytePtr < p)
                p = mb.bytePtr;
        }
        
        return new memoryBlock(s, p);
    }
//    
//    public void free(CUdeviceptr ptr) {
//        storage.put(ptr);
//    }
//    
//    void setMaxMemory(long maxPoolMemory) {
//        this.maxPoolMemory = maxPoolMemory;
//    }
//    
    public void printMempoolStats() {
        System.out.printf("Pool: %d %d\n", 
                            size,
                            available.get());
        
        memoryBlock b = head;
        
        while(b.nextBlock != null) {
            memoryBlock b0 = b;
            b = b.nextBlock;
            System.out.printf("%s %d %d\n", b.used ? "#" : " ", b.bytePtr, b.size);
            
            if(b0 != b.prevBlock)
                throw new RuntimeException();
        }
        
        System.out.println("#############################");
        
        for(memoryBlock mb : freeBlocks.descendingSet()) {
            System.out.printf("%d %d\n", mb.bytePtr, mb.size);
        }
        
        System.out.println("#############################");
    }
}
