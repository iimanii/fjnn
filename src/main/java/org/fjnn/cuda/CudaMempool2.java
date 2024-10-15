/*
 * The MIT License
 *
 * Copyright 2024 ahmed.
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

import jcuda.driver.*;

/**
 *
 * @author ahmed
 */

public class CudaMempool2 {

    private CUmemoryPool mempool;
    public final long maxPoolMemory;

    public CudaMempool2(long maxPoolMemory, int deviceId) {
        this.maxPoolMemory = maxPoolMemory;
        this.mempool = new CUmemoryPool();
        
        CUmemPoolProps poolProps = new CUmemPoolProps();
        poolProps.allocType = CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED;
        poolProps.handleTypes = CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE;
        poolProps.location = new CUmemLocation();
        poolProps.location.type = CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE;
        poolProps.location.id = deviceId;

        JCudaDriver.cuMemPoolCreate(mempool, poolProps);  // Creates a new memory pool
    }

    // Allocate float memory from the pool
    public CUdeviceptr getFloat(long size, CUstream stream) {
        return get(size * CudaUtil.FLOAT_SIZE, stream);
    }

    // Allocate memory from the pool asynchronously
    public CUdeviceptr get(long size, CUstream stream) {
        CUdeviceptr ptr = new CUdeviceptr();
        JCudaDriver.cuMemAllocFromPoolAsync(ptr, size, mempool, stream);
        
        return ptr;
    }

    // Destroy the pool when done
    public void destroy() {
        JCudaDriver.cuMemPoolDestroy(mempool);
        mempool = null;
    }

//    // Print current memory pool statistics (for debugging purposes)
//    public void printMempoolStats() {
//        System.out.printf("Mempool Stats: %.4f MB allocated\n", totalAllocations.get() / 1e6);
//    }
}

