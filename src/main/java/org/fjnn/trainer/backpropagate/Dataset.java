/*
 * The MIT License
 *
 * Copyright 2025 ahmed.
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
package org.fjnn.trainer.backpropagate;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 * 
 * Training dataset class
 */
public class Dataset {
    private final float[] input;
    private final float[] targets;
    private final float[] targetWeights;    /* per-sample weights */

    public final int inputDim;
    public final int targetDim;
    public final int batchSize;             /* input samples per batch */
    public final int batchCount;            /* total batches */
    public final int inputCount;            /* total input samples */
    
    // Batched data
    private float[][] inputBatches;
    private float[][] targetBatches;
    private float[][] targetWeightBatches;      /* batched weights */

    // GPU data
    private CUdeviceptr[] inputBatchesGPU;
    private CUdeviceptr[] targetBatchesGPU;
    private CUdeviceptr[] targetWeightBatchesGPU;   /* GPU weight batches */
    private int gpuDevice = -1;
    
    // State flags
    private boolean cpuReady = false;
    private boolean gpuReady = false;
    
    public Dataset(float[] data, float[] targets, int inputDim, int targetDim, int batchSize) {
        this(data, targets, null, inputDim, targetDim, batchSize);
    }
    
    public Dataset(float[] data, float[] targets, float[] targetWeights, int inputDim, int targetDim, int batchSize) {
        if (data == null || targets == null)
            throw new IllegalArgumentException("Data and targets cannot be null");
        
        if (inputDim <= 0 || targetDim <= 0 || batchSize <= 0)
            throw new IllegalArgumentException("Dimensions and batch size must be positive");
        
        if (data.length % inputDim != 0)
            throw new IllegalArgumentException("Data length must be divisible by inputDim");
        
        if (targets.length % targetDim != 0)
            throw new IllegalArgumentException("Targets length must be divisible by targetDim");
        
        int dataSamples = data.length / inputDim;
        int targetSamples = targets.length / targetDim;
        
        if (dataSamples != targetSamples)
            throw new IllegalArgumentException("Number of data samples (" + dataSamples + ") must equal number of target samples (" + targetSamples + ")");
        
        if (dataSamples % batchSize != 0)
            throw new IllegalArgumentException("Sample count (" + dataSamples + ") must be divisible by batch size (" + batchSize + ")");
        
        // Validate weights if provided
        if (targetWeights != null && targetWeights.length != targets.length)
            throw new IllegalArgumentException("Weights length (" + targetWeights.length + ") must equal number of targets (" + targets.length + ")");
        
        this.input = Arrays.copyOf(data, data.length);
        this.targets = Arrays.copyOf(targets, targets.length);
        this.inputDim = inputDim;
        this.targetDim = targetDim;
        this.batchSize = batchSize;
        this.inputCount = dataSamples;
        this.batchCount = inputCount / batchSize;
        this.targetWeights = targetWeights;
    }
    
    public void prepareBatches() {
        inputBatches = new float[batchCount][];
        targetBatches = new float[batchCount][];
        targetWeightBatches = targetWeights != null ? new float[batchCount][] : null;


        for (int b = 0; b < batchCount; b++) {
            int start = b * batchSize;
            
            inputBatches[b] = new float[batchSize * inputDim];
            targetBatches[b] = new float[batchSize * targetDim];
            targetWeightBatches[b] = new float[batchSize * targetDim];

            System.arraycopy(input, start * inputDim, inputBatches[b], 0, batchSize * inputDim);
            System.arraycopy(targets, start * targetDim, targetBatches[b], 0, batchSize * targetDim);
            
            if (targetWeights != null) {
                targetWeightBatches[b] = new float[batchSize * targetDim];
                System.arraycopy(targetWeights, start * targetDim, targetWeightBatches[b], 0, batchSize * targetDim);
            }
        }
        
        cpuReady = true;
    }
    
    public void prepareBatchesGPU(int device) {
        // Free existing GPU data if on different device
        if (gpuReady && gpuDevice != device)
            freeGPU();
        
        if (!gpuReady) {
            CudaEngine.prepareThread(device);
            
            inputBatchesGPU = new CUdeviceptr[batchCount];
            targetBatchesGPU = new CUdeviceptr[batchCount];
            targetWeightBatchesGPU = targetWeights != null ? new CUdeviceptr[batchCount] : null;
                
            // Create temporary batch arrays
            float[] inputBatch = new float[batchSize * inputDim];
            float[] targetBatch = new float[batchSize * targetDim];
            float[] targetWeightBatch = new float[batchSize * targetDim];
            
            for (int b = 0; b < batchCount; b++) {
                int start = b * batchSize;
                
                System.arraycopy(input, start * inputDim, inputBatch, 0, batchSize * inputDim);
                System.arraycopy(targets, start * targetDim, targetBatch, 0, batchSize * targetDim);
                
                inputBatchesGPU[b] = CudaUtil.toGPU(inputBatch);
                targetBatchesGPU[b] = CudaUtil.toGPU(targetBatch);
                
                if(targetWeights != null) {
                    System.arraycopy(targetWeights, start * targetDim, targetWeightBatch, 0, batchSize * targetDim);  
                    targetWeightBatchesGPU[b] = CudaUtil.toGPU(targetWeightBatch);                  
                }
            }
            
            CudaEngine.finalizeThread();
            
            gpuDevice = device;
            gpuReady = true;
        }
    }
    
    public void shuffle() {
        // Store current state
        boolean wasCpuReady = cpuReady;
        boolean wasGpuReady = gpuReady;
        int prevGpuDevice = gpuDevice;
        
        // Free current prepared data
        if (gpuReady) {
            freeGPU();
        }
        cpuReady = false;
        
        // Create list of sample indices
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < inputCount; i++) {
            indices.add(i);
        }        
        Collections.shuffle(indices);
        
        // Parallel shuffle using ThreadPoolExecutor
        float[] shuffledData = new float[input.length];
        float[] shuffledTargets = new float[targets.length];
        float[] shuffledWeights = targetWeights != null ? new float[targetWeights.length] : null;
        
        int numThreads = Math.min(Runtime.getRuntime().availableProcessors(), inputCount);
        int samplesPerThread = inputCount / numThreads;
        
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        Future<?>[] futures = new Future[numThreads];
        
        for (int t = 0; t < numThreads; t++) {
            final int threadIdx = t;
            futures[t] = executor.submit(() -> {
                int start = threadIdx * samplesPerThread;
                int end = (threadIdx == numThreads - 1) ? inputCount : start + samplesPerThread;
                
                for (int i = start; i < end; i++) {
                    int originalIdx = indices.get(i);
                    System.arraycopy(input, originalIdx * inputDim, shuffledData, i * inputDim, inputDim);
                    System.arraycopy(targets, originalIdx * targetDim, shuffledTargets, i * targetDim, targetDim);
                    if (targetWeights != null) {
                        System.arraycopy(targetWeights, originalIdx * targetDim, shuffledWeights, i * targetDim, targetDim);
                    }
                }
            });
        }
        
        try {
            for (Future<?> future : futures)
                future.get();
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException("Shuffle failed", e);
        } finally {
            executor.shutdown();
        }
        
        // Replace original arrays
        System.arraycopy(shuffledData, 0, input, 0, input.length);
        System.arraycopy(shuffledTargets, 0, targets, 0, targets.length);

        if (targetWeights != null) {
            System.arraycopy(shuffledWeights, 0, targetWeights, 0, targetWeights.length);
        }
        
        // Restore previous state
        if (wasCpuReady)
            prepareBatches();
        
        if (wasGpuReady)
            prepareBatchesGPU(prevGpuDevice);
    }
    
    public float[] getInput(int batchIndex) {
        if (!cpuReady)
            throw new IllegalStateException("Must call prepareBatches() first");
        
        if (batchIndex < 0 || batchIndex >= batchCount)
            throw new IndexOutOfBoundsException("Batch index out of range");
        
        return inputBatches[batchIndex];
    }
    
    public CUdeviceptr getInputGPU(int batchIndex) {
        if (!gpuReady)
            throw new IllegalStateException("Must call prepareBatchesGPU() first");
        
        if (batchIndex < 0 || batchIndex >= batchCount)
            throw new IndexOutOfBoundsException("Batch index out of range");
        
        return inputBatchesGPU[batchIndex];
    }
    
    public float[] getTarget(int batchIndex) {
        if (!cpuReady)
            throw new IllegalStateException("Must call prepareBatches() first");
        
        if (batchIndex < 0 || batchIndex >= batchCount)
            throw new IndexOutOfBoundsException("Batch index out of range");
        
        return targetBatches[batchIndex];
    }
    
    public float[] getTargetWeights(int batchIndex) {
        if (!cpuReady) 
            throw new IllegalStateException("CPU batches not prepared");
        
        if (targetWeights == null) 
            throw new IllegalStateException("No target weights provided");
        
        return targetWeightBatches[batchIndex];
    }
    
    public CUdeviceptr getTargetGPU(int batchIndex) {
        if (!gpuReady)
            throw new IllegalStateException("Must call prepareBatchesGPU() first");
        
        if (batchIndex < 0 || batchIndex >= batchCount)
            throw new IndexOutOfBoundsException("Batch index out of range");
        
        return targetBatchesGPU[batchIndex];
    }
    
    public CUdeviceptr getTargetWeightsGPU(int batchIndex) {
        if (!gpuReady) 
            throw new IllegalStateException("GPU batches not prepared");
        
        if (targetWeights == null) 
            throw new IllegalStateException("No target weights provided");
        
        return targetWeightBatchesGPU[batchIndex];
    }
    
    public void freeGPU() {
        if (gpuReady) {
            CudaEngine.prepareThread(gpuDevice);
            for (int b = 0; b < batchCount; b++) {
                CudaUtil.free(inputBatchesGPU[b]);
                CudaUtil.free(targetBatchesGPU[b]);
                if (targetWeightBatchesGPU != null)
                    CudaUtil.free(targetWeightBatchesGPU[b]);
            }
            CudaEngine.finalizeThread();
            
            inputBatchesGPU = null;
            targetBatchesGPU = null;
            targetWeightBatchesGPU = null;
            gpuReady = false;
            gpuDevice = -1;
        }
    }
    
    public boolean cpuReady() { 
        return cpuReady; 
    }
    
    public boolean gpuReady() { 
        return gpuReady; 
    }
    
    public boolean hasTargetWeights() {
        return targetWeights != null;
    }
}