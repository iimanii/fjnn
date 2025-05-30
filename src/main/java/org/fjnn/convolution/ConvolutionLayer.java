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
package org.fjnn.convolution;

import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.cublasHandle;
import org.fjnn.base.ModelComponent;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.base.output.FeedForwardOutput;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.convolution.output.layer.ConvolutionLayerBackpropagateOutput;
import org.fjnn.convolution.output.layer.ConvolutionLayerBackpropagateOutputGPU;
import org.fjnn.convolution.output.layer.ConvolutionLayerForwardOutput;
import org.fjnn.convolution.output.layer.ConvolutionLayerForwardOutputGPU;
import org.fjnn.convolution.output.unit.ConvolutionUnitBackpropagateOutput;
import org.fjnn.convolution.output.unit.ConvolutionUnitBackpropagateOutputGPU;
import org.fjnn.convolution.output.unit.ConvolutionUnitForwardOutput;
import org.fjnn.convolution.output.unit.ConvolutionUnitForwardOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;

/**
 * A convolution layer that contains multiple ConvolutionUnit objects (Kernel, KernelGroup)
 * and processes them in parallel, concatenating their outputs.
 * 
 * @author ahmed
 */
public class ConvolutionLayer {
    private final ConvolutionUnit[] units;
    
    public final int strideSize;
    
    // Thread pool for parallel CPU processing
    private static final ExecutorService CPU_EXECUTOR = Executors.newFixedThreadPool(
        Math.min(Runtime.getRuntime().availableProcessors(), 8),
        r -> {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        }
    );
    
    /**
     * Creates a convolution layer with multiple units that all process the same input
     * @param units Array of ConvolutionUnit objects (Kernel, KernelGroup, etc.)
     */
    public ConvolutionLayer(ConvolutionUnit... units) {
        if (units.length == 0)
            throw new IllegalArgumentException("ConvolutionLayer requires at least one unit");
        
        this.units = Arrays.copyOf(units, units.length);
        
        strideSize = units[0].getStrideSize();
        
        for (int i = 1; i < units.length; i++) {
            if (units[i].getStrideSize() != strideSize) {
                throw new IllegalArgumentException(
                    "All units must have same input stride. Expected: " + strideSize + 
                    ", but unit " + i + " has: " + units[i].getStrideSize());
            }
        }
    }
    
    public ConvolutionLayerForwardOutput feedForward(float[] input, int batchSize) {
        if (input.length % strideSize != 0)
            throw new IllegalArgumentException("Sequence length must be multiple of total kernel width");
            
        // Process all units in parallel
        Future<ConvolutionUnitForwardOutput>[] futures = new Future[units.length];

        for (int i = 0; i < units.length; i++) {
            final int unitIndex = i;
            futures[i] = CPU_EXECUTOR.submit(() -> units[unitIndex].feedForward(input, batchSize));
        }
        
        ConvolutionUnitForwardOutput[] unitOutputs = new ConvolutionUnitForwardOutput[units.length];
        
        try {
            for (int i = 0; i < units.length; i++)
                unitOutputs[i] = futures[i].get();
        } catch (ExecutionException | InterruptedException ex) {
            throw new RuntimeException("Error in parallel forward pass", ex);
        }
        
        return new ConvolutionLayerForwardOutput(unitOutputs, input, batchSize);
    }
    
    public ConvolutionLayerForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchSize, CUstream stream, cublasHandle handle) {
        ConvolutionUnitForwardOutputGPU[] unitOutputs = new ConvolutionUnitForwardOutputGPU[units.length];
        
        for (int i = 0; i < units.length; i++) {
            unitOutputs[i] = units[i].feedForwardGPU(input, batchSize, stream, handle);
        }
        
        return new ConvolutionLayerForwardOutputGPU(unitOutputs, input, batchSize);
    }
    
    public ConvolutionLayerBackpropagateOutput backpropagate(ConvolutionLayerForwardOutput forwardOutput, float[][] unitDeltaLoss) {
        int inputSize = forwardOutput.input.length / forwardOutput.batchSize;
        
        // Process units in parallel
        Future<ConvolutionUnitBackpropagateOutput>[] futures = new Future[units.length];
        
        for (int i = 0; i < units.length; i++) {
            final int unitIndex = i;
            futures[i] = CPU_EXECUTOR.submit(() -> 
                units[unitIndex].backpropagate(forwardOutput.unitOutputs[unitIndex], unitDeltaLoss[unitIndex]));
        }
        
        // Collect results and aggregate input gradients
        float[] aggregatedInputGradients = new float[inputSize * forwardOutput.batchSize];
        ConvolutionUnitBackpropagateOutput[] unitBackprops = new ConvolutionUnitBackpropagateOutput[units.length];
        
        try {
            for (int i = 0; i < units.length; i++) {
                unitBackprops[i] = futures[i].get();
                
                // Accumulate input gradients (all units process same input)
                for (int j = 0; j < aggregatedInputGradients.length; j++) {
                    aggregatedInputGradients[j] += unitBackprops[i].inputGradients[j];
                }
            }
        } catch (ExecutionException | InterruptedException ex) {
            throw new RuntimeException("Error in parallel backward pass", ex);
        }
        
        return new ConvolutionLayerBackpropagateOutput(aggregatedInputGradients, unitBackprops, inputSize, forwardOutput.batchSize);
    }
    
    public ConvolutionLayerBackpropagateOutputGPU backpropagateGPU(ConvolutionLayerForwardOutputGPU forwardOutput, 
                                                                   CUdeviceptr[] unitDeltaLoss, 
                                                                   CUstream stream, 
                                                                   cublasHandle handle) {
        long totalInputLength = CudaUtil.length(forwardOutput.input) / CudaUtil.FLOAT_SIZE;
        int inputSize = (int)(totalInputLength / forwardOutput.batchSize);

        // Process units sequentially on GPU
        ConvolutionUnitBackpropagateOutputGPU[] unitBackprops = new ConvolutionUnitBackpropagateOutputGPU[units.length];

        for (int i = 0; i < units.length; i++) {
            unitBackprops[i] = units[i].backpropagateGPU(forwardOutput.unitOutputs[i], unitDeltaLoss[i], stream, handle);
        }

        // Aggregate input gradients
        CUdeviceptr aggregatedInputGradients = CudaUtil.createFloatAsync(inputSize * forwardOutput.batchSize, stream);
        JCudaDriver.cuMemsetD32Async(aggregatedInputGradients, 0, inputSize * forwardOutput.batchSize, stream);

        for (int i = 0; i < units.length; i++) {
            CudaFunctions.vector.add(aggregatedInputGradients, unitBackprops[i].inputGradients, aggregatedInputGradients, inputSize * forwardOutput.batchSize, stream);
        }

        return new ConvolutionLayerBackpropagateOutputGPU(aggregatedInputGradients, unitBackprops, inputSize, forwardOutput.batchSize);
    }

    public void updateWeights(ConvolutionLayerBackpropagateOutput layerGradients, float learningRate) {
        // Apply gradients to each unit
        for (int i = 0; i < units.length; i++) {
            if (units[i] instanceof Kernel) {
                ((Kernel) units[i]).updateWeights(
                    layerGradients.unitBackprops[i].weightGradients[0],
                    layerGradients.unitBackprops[i].biasGradients[0],
                    learningRate);
            } else if (units[i] instanceof KernelGroup) {
                ((KernelGroup) units[i]).updateWeights(layerGradients.unitBackprops[i], learningRate);
            }
        }
    }
    
    public void updateWeightsGPU(ConvolutionLayerBackpropagateOutputGPU layerGradients, float learningRate, CUstream stream) {        
        // Apply gradients to each unit
        for (int i = 0; i < units.length; i++) {
            if (units[i] instanceof Kernel) {
                ((Kernel) units[i]).updateWeightsGPU(
                    layerGradients.unitBackprops[i].weightGradients[0],
                    layerGradients.unitBackprops[i].biasGradients[0],
                    learningRate, stream);
            } else if (units[i] instanceof KernelGroup) {
                ((KernelGroup) units[i]).updateWeightsGPU(layerGradients.unitBackprops[i], learningRate, stream);
            }
        }
    }
    
    public void prepareGPU(CUstream stream) {
        for (ConvolutionUnit unit : units) {
            unit.prepareGPU(stream);
        }
    }

    public void freeGPU(CUstream stream) {
        for (ConvolutionUnit unit : units) {
            unit.freeGPU(stream);
        }
    }

    public boolean gpuReady() {
        for (ConvolutionUnit unit : units) {
            if (!unit.gpuReady())
                return false;
        }
        return true;
    }
    
    public void syncWeightsFromGPU() {
        for (ConvolutionUnit unit : units) {
            unit.updateWeightsFromGPU();
        }
    }
}
