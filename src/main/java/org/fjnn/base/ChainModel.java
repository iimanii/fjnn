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
package org.fjnn.base;

import org.fjnn.base.output.FeedForwardOutput;
import org.fjnn.base.output.FeedForwardOutputMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.adapter.BatchSizeAdapter;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.base.output.BackpropagateOutputMap;
import org.fjnn.base.output.BackpropagateOutputMapGPU;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.base.output.FeedForwardOutputMapGPU;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.loss.Loss;

/**
 *
 * @author ahmed
 */
public class ChainModel {
    private final List<ModelComponent> components;
    private int currentBatchSize;    // Tracks the current batch size
    private int currentMultiplier;   // Tracks the relative change in batch size
    
    // Constructor
    public ChainModel(int inputSize) {
        this.components = new ArrayList<>();
        this.currentBatchSize = inputSize;
        this.currentMultiplier = 1;                     // Initially, no change in size        
    }

    // Add a component to the chain
    public void addComponent(ModelComponent component) {
        if (component.getInputSize() != currentBatchSize)
            throw new IllegalStateException("Input size mismatch. Expected: " + currentBatchSize + ", but got: " + component.getInputSize());
        
        if (component instanceof BatchSizeAdapter) {
            BatchSizeAdapter adapter = (BatchSizeAdapter) component;
            adjustMultiplier(currentBatchSize, adapter.getOutputSize());
        }
        
        components.add(component);
        currentBatchSize = component.getOutputSize();   // Update the current batch size
    }
    
    public void addBatchSizeAdapter(int targetBatchSize) {
        if (currentBatchSize == targetBatchSize)
            throw new IllegalArgumentException("No adapter needed. Current and target batch sizes are the same.");

        adjustMultiplier(currentBatchSize, targetBatchSize);
        
        components.add(new BatchSizeAdapter(currentBatchSize, targetBatchSize));
        currentBatchSize = targetBatchSize;
    }

    public void restoreBatchCount() {
        if (currentMultiplier == 1)
            throw new IllegalArgumentException("No restore needed.");

        int targetBatchSize = currentBatchSize * currentMultiplier;

        components.add(new BatchSizeAdapter(currentBatchSize, targetBatchSize));
        currentBatchSize = targetBatchSize;
        currentMultiplier = 1; // Reset multiplier
    }
    
    private void adjustMultiplier(int currentBatchSize, int targetBatchSize) {
        if(targetBatchSize <= 0)
            throw new RuntimeException("Invalid targetBatchSize: " + targetBatchSize);
        
        if (targetBatchSize > currentBatchSize) {
            // Increasing batch size
            int adjustmentFactor = targetBatchSize / currentBatchSize;
            
            if (targetBatchSize % currentBatchSize != 0)
                throw new IllegalArgumentException("Invalid batch size transition: " + targetBatchSize + " is not a multiple of " + currentBatchSize);

            if (currentMultiplier % adjustmentFactor != 0)
                throw new IllegalArgumentException("Invalid batch size transition: " + currentMultiplier + " is not divisible by " + adjustmentFactor);

            currentMultiplier /= adjustmentFactor; // Decrease multiplier
        } else {
            // Decreasing batch size
            int adjustmentFactor = currentBatchSize / targetBatchSize;
            
            if (currentBatchSize % targetBatchSize != 0)
                throw new IllegalArgumentException("Invalid batch size transition: " + currentBatchSize + " is not a multiple of " + targetBatchSize);

            currentMultiplier *= adjustmentFactor; // Increase multiplier
        }

        if (currentMultiplier < 1)
            throw new IllegalStateException("Multiplier cannot be less than 1");
    }
    
    public int getCurrentBatchSize() {
        return currentBatchSize;
    }

    public int getCurrentBatchMultiplier() {
        return currentMultiplier;
    }
    
    // Perform a forward pass through all components
    public FeedForwardOutputMap feedForward(float[] inputs, int inputBatchSize, int inputBatchCount) {
        FeedForwardOutputMap context = new FeedForwardOutputMap();
        float[] currentOutput = inputs;
        int batchSize = inputBatchSize;
        int batchCount = inputBatchCount;

        for (int i=0; i < components.size(); i++) {
            FeedForwardOutput result = components.get(i).feedForward(currentOutput, batchSize, batchCount);
            context.addOutput(i, result);          // Store the result
            currentOutput = result.output();    // Update output for the next component
            batchSize = result.batchSize;
            batchCount = result.batchCount;
        }

        return context;
    }
    
    // Perform a forward pass through all components
    public FeedForwardOutputMapGPU feedForwardGPU(CUdeviceptr input, int inputBatchSize, int inputBatchCount, CUstream stream) {
        FeedForwardOutputMapGPU context = new FeedForwardOutputMapGPU();
        CUdeviceptr currentOutput = input;
        int batchSize = inputBatchSize;
        int batchCount = inputBatchCount;
        
        for (int i=0; i < components.size(); i++) {                    
            FeedForwardOutputGPU result = components.get(i).feedForwardGPU(currentOutput, batchSize, batchCount, stream);
            context.addOutput(i, result);           // Store the result
            currentOutput = result.output();        // Update output for the next component
            batchSize = result.batchSize;
            batchCount = result.batchCount;
        }

        return context;
    }
        
    // Perform a backward pass through all components
    public BackpropagateOutputMap backpropagate(FeedForwardOutputMap context, float[] truth, int batchSize, int batchCount, Loss lossFunction, float learningRate) {
        float[] deltaLoss = lossFunction.derivative(context.output(), truth);
        int deltaLossSize = batchSize;
        int deltaLossCount = batchCount;
        
        BackpropagateOutputMap result = new BackpropagateOutputMap(deltaLoss);
        
        /* make a copy */
        deltaLoss = Arrays.copyOf(deltaLoss, deltaLoss.length);
        
        for (int i = components.size() - 1; i >= 0; i--) {
            BackpropagateOutput output = components.get(i).backpropagate(context.getOutput(i), deltaLoss, deltaLossSize, deltaLossCount, learningRate);
            deltaLoss = Arrays.copyOf(output.deltaLoss(), output.deltaLoss().length);
            deltaLossSize = output.batchSize;
            deltaLossCount = output.batchCount;
            
            result.addOutput(i, output);
        }
        
        return result;
    }
        
    // Perform a backward pass through all components
    public BackpropagateOutputMapGPU backpropagateGPU(FeedForwardOutputMapGPU context, CUdeviceptr truth, int batchSize, int batchCount, Loss lossFunction, float learningRate, CUstream stream) {
        // Step 1: Get result and calculate delta loss on GPU
        long totalOutputSize = batchSize * batchCount;
        CUdeviceptr outputPtr = context.output();
        CUdeviceptr deltaLoss = CudaUtil.createFloatAsync(totalOutputSize, stream);
        
        lossFunction.derivativeGPU(outputPtr, truth, deltaLoss, totalOutputSize, stream);
        int deltaLossSize = batchSize;
        int deltaLossCount = batchCount;
        
        BackpropagateOutputMapGPU result = new BackpropagateOutputMapGPU(deltaLoss);
        
        /* make a copy */
        deltaLoss = CudaUtil.copyFloatAsync(deltaLoss, totalOutputSize, stream);
        
        for (int i = components.size() - 1; i >= 0; i--) {
            BackpropagateOutputGPU output = components.get(i).backpropagateGPU(context.getOutput(i), deltaLoss, deltaLossSize, deltaLossCount, learningRate, stream);
            deltaLoss = CudaUtil.copyFloatAsync(output.deltaLoss(), output.batchSize * output.batchCount, stream);
            deltaLossSize = output.batchSize;
            deltaLossCount = output.batchCount;
            
            result.addOutput(i, output);
        }
        
        return result;
    }
    
    public void prepareGPU() {
        prepareGPU(null);
    }
    
    public void prepareGPU(CUstream stream) {
        for (ModelComponent component : components)
            component.prepareGPU(stream);
    }
    
    public void freeGPU() {
        freeGPU(null);
    }
    
    public void freeGPU(CUstream stream) {
        for (ModelComponent component : components)
            component.freeGPU(stream);
    }

    // Retrieve the list of components
    public List<ModelComponent> getComponents() {
        return new ArrayList<>(components);
    }
    
    // Retrieve the list of components
    public ModelComponent getComponent(int index) {
        return components.get(index);
    }
    
    public ChainModel copy() {
        ChainModel chainCopy = new ChainModel(currentBatchSize);

        // Copy the component list
        for(ModelComponent component : components) {
            chainCopy.components.add(component.copy());
        }

        // Copy the current state
        chainCopy.currentBatchSize = this.currentBatchSize;
        chainCopy.currentMultiplier = this.currentMultiplier;

        return chainCopy;
    }
}
