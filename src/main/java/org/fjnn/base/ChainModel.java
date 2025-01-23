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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.adapter.InputDimAdapter;
import org.fjnn.adapter.PositionalEncoderAdapter;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.base.output.BackpropagateOutputMap;
import org.fjnn.base.output.BackpropagateOutputMapGPU;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.base.output.FeedForwardOutputMapGPU;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.loss.Loss;
import org.fjnn.network.NeuralNetwork;
import org.fjnn.network.outputs.NeuralNetworkForwardOutput;
import org.fjnn.network.outputs.NeuralNetworkForwardOutputGPU;

/**
 *
 * @author ahmed
 */
public class ChainModel {
    public final int inputDim;
    
    private final List<ModelComponent> components;
    private int outputDim;               // Tracks the current output size
    private int splitRatio;              // Tracks the relative change in inputDim splits .. starts at 1
    
    /**
     * @param inputDim Raw input size [in floats] to the chain
     */
    public ChainModel(int inputDim) {
        this.components = new ArrayList<>();
        this.inputDim = inputDim;
        this.outputDim = inputDim;
        this.splitRatio = 1;
    }

    // Add a component to the chain
    public void addComponent(ModelComponent component) {
        if (component.getInputSize() != outputDim) {
            addInputDimAdapter(component.getInputSize());
        } else if (component instanceof InputDimAdapter) {
            InputDimAdapter adapter = (InputDimAdapter) component;
            adjustSplit(outputDim, adapter.getOutputSize());
        }
        
        components.add(component);
        outputDim = component.getOutputSize();   // Update the current output size
    }
    
    public void addInputDimAdapter(int targetInputDim) {
        if (outputDim == targetInputDim)
            throw new IllegalArgumentException("No adapter needed. Current and target batch sizes are the same.");

        adjustSplit(outputDim, targetInputDim);
        
        components.add(new InputDimAdapter(outputDim, targetInputDim));
        outputDim = targetInputDim;
    }

    public void restoreBatchSplit() {
        if (splitRatio == 1)
            throw new IllegalArgumentException("No restore needed.");

        int targetBatchSize = outputDim * splitRatio;

        components.add(new InputDimAdapter(outputDim, targetBatchSize));
        outputDim = targetBatchSize;
        splitRatio = 1; // Reset multiplier
    }
    
    private void adjustSplit(int currentDim, int targetDim) {
        if(targetDim <= 0)
            throw new RuntimeException("Invalid targetBatchSize: " + targetDim);
        
        if (targetDim > currentDim) {
            // Decrease split
            int adjustmentFactor = targetDim / currentDim;
            
            if (targetDim % currentDim != 0)
                throw new IllegalArgumentException("Invalid input dim transition: " + targetDim + " is not a multiple of " + currentDim);

            if (splitRatio % adjustmentFactor != 0)
                throw new IllegalArgumentException("Invalid iniput dim transition: " + splitRatio + " is not divisible by " + adjustmentFactor);

            splitRatio /= adjustmentFactor; // Decrease multiplier
        } else {
            // Increase split
            int adjustmentFactor = currentDim / targetDim;
            
            if (currentDim % targetDim != 0)
                throw new IllegalArgumentException("Invalid input dim transition: " + currentDim + " is not a multiple of " + targetDim);

            splitRatio *= adjustmentFactor;
        }

        if (splitRatio < 1)
            throw new IllegalStateException("Multiplier cannot be less than 1");
    }
    
    public int getOutputDim() {
        return outputDim;
    }

    public int getSplitRatio() {
        return splitRatio;
    }
    
    /**
     * Perform a forward pass through all components
     *
     * @param input     A flat array representing the input data for the forward pass.
     *                  The length of this array should be equal to `inputDim * batchSize`,
     *                  where `inputDim` is the number of features per input sample and 
     *                  `batchSize` is the number of samples in the batch.
     * @param inputDim  The number of features (or input dimensions) for each individual input sample.
     * @param batchSize The number of input samples in the batch to be processed together.
     * 
     * @return A `FeedForwardOutputMap` containing the output of the forward pass for each component
     *         required by backpropagate method
     *
     * @throws IllegalArgumentException
     */
    public FeedForwardOutputMap feedForward(float[] input, int inputDim, int batchSize) {
        FeedForwardOutputMap context = new FeedForwardOutputMap();
        float[] currentOutput = input;
        int currentInputDim = inputDim;
        int currentBatchSize = batchSize;

        for (int i=0; i < components.size(); i++) {
            FeedForwardOutput result = components.get(i).feedForward(currentOutput, currentInputDim, currentBatchSize);
            context.addOutput(i, result);           // Store the result
            currentOutput = result.output();        // Update output for the next component
            currentInputDim = result.outputDim;
            currentBatchSize = result.batchSize;
        }

        return context;
    }
    
    // Perform a forward pass through all components
    public FeedForwardOutputMapGPU feedForwardGPU(CUdeviceptr input, int inputDim, int batchSize, CUstream stream) {
        FeedForwardOutputMapGPU context = new FeedForwardOutputMapGPU();
        CUdeviceptr currentOutput = input;
        int currentInputDim = inputDim;
        int currentBatchCount = batchSize;
        
        for (int i=0; i < components.size(); i++) {                    
            FeedForwardOutputGPU result = components.get(i).feedForwardGPU(currentOutput, currentInputDim, currentBatchCount, stream);
            context.addOutput(i, result);           // Store the result
            currentOutput = result.output();        // Update output for the next component
            currentInputDim = result.outputDim;
            currentBatchCount = result.batchSize;
        }

        return context;
    }
        
    // Perform a backward pass through all components
    public BackpropagateOutputMap backpropagate(FeedForwardOutputMap context, float[] truth, int truthDim, int batchSize, Loss lossFunction) {
        float[] deltaLoss = null;
        int deltaLossDim = truthDim;
        int deltaLossBatchSize = batchSize;
        boolean lossCalculated = false;
        
        BackpropagateOutputMap result = new BackpropagateOutputMap();
        
        BackpropagateOutput output;
        
        for (int i = components.size() - 1; i >= 0; i--) {
            ModelComponent component = components.get(i);

            if (!lossCalculated) {
                if (component instanceof NeuralNetwork) {
                    // Case 1: Neural Network - use loss function
                    output = ((NeuralNetwork)component).backpropagate((NeuralNetworkForwardOutput)context.getOutput(i), truth, lossFunction);
                    lossCalculated = true;
                } else if (component instanceof InputDimAdapter) {
                    // Case 2: Size Adapter - copy and adjust truth                 
                    output = component.backpropagate(context.getOutput(i), truth, deltaLossDim, deltaLossBatchSize);
                } else {
                    // Case 3: Different type - calculate deltaLoss
                    deltaLoss = lossFunction.derivative(context.output(), truth);
                    lossCalculated = true;
                    deltaLoss = Arrays.copyOf(deltaLoss, deltaLossDim * deltaLossBatchSize);
                    output = component.backpropagate(context.getOutput(i), deltaLoss, deltaLossDim, deltaLossBatchSize);
                }
            } else {
                // Normal backprop after loss is used
                deltaLoss = Arrays.copyOf(deltaLoss, deltaLossDim * deltaLossBatchSize);
                output = component.backpropagate(context.getOutput(i), deltaLoss, deltaLossDim, deltaLossBatchSize);
            }

            result.addOutput(i, output);
            deltaLossDim = output.deltaLossDim;
            deltaLossBatchSize = output.batchSize;
            deltaLoss = output.deltaLoss();
        }
        
        return result;
    }
        
    // Perform a backward pass through all components
    public BackpropagateOutputMapGPU backpropagateGPU(FeedForwardOutputMapGPU context, CUdeviceptr truth, int truthDim, int batchSize, Loss lossFunction, CUstream stream) {
        CUdeviceptr deltaLoss = null;
        int deltaLossDim = truthDim;
        int deltaLossBatchSize = batchSize; 
        boolean lossCalculated = false;

        CUdeviceptr currentTruth = truth;
        BackpropagateOutputMapGPU result = new BackpropagateOutputMapGPU();
        BackpropagateOutputGPU output;

        for (int i = components.size() - 1; i >= 0; i--) {
            ModelComponent component = components.get(i);
            if (!lossCalculated) {
                if (component instanceof NeuralNetwork) {
                    // Case 1: Neural Network - use loss function
                    output = ((NeuralNetwork)component).backpropagateGPU((NeuralNetworkForwardOutputGPU)context.getOutput(i), currentTruth, lossFunction, stream);
                    lossCalculated = true;
                } else if (component instanceof InputDimAdapter) {
                    // Case 2: Size Adapter - use truth directly
                    CUdeviceptr truthCopy = CudaUtil.copyFloatAsync(currentTruth, deltaLossDim * deltaLossBatchSize, stream);   
                    output = component.backpropagateGPU(context.getOutput(i), truthCopy, deltaLossDim, deltaLossBatchSize, stream);
                    currentTruth = output.deltaLoss();  // Update truth pointer to adapter's output
                } else {
                    // Case 3: Different type - calculate deltaLoss
                    deltaLoss = CudaUtil.createFloatAsync(deltaLossDim * deltaLossBatchSize, stream);
                    lossFunction.derivativeGPU(context.output(), currentTruth, deltaLoss, deltaLossDim * deltaLossBatchSize, stream);
                    lossCalculated = true;
                    
                    deltaLoss = CudaUtil.copyFloatAsync(deltaLoss, deltaLossDim * deltaLossBatchSize, stream);
                    output = component.backpropagateGPU(context.getOutput(i), deltaLoss, deltaLossDim, deltaLossBatchSize, stream);
                }
            } else {
                // Normal backprop after loss is used
                deltaLoss = CudaUtil.copyFloatAsync(deltaLoss, deltaLossDim * deltaLossBatchSize, stream);
                output = component.backpropagateGPU(context.getOutput(i), deltaLoss, deltaLossDim, deltaLossBatchSize, stream);
            }

            result.addOutput(i, output);
            deltaLossDim = output.deltaLossDim;
            deltaLossBatchSize = output.batchSize;
            deltaLoss = output.deltaLoss();
        }

        return result;        
    }
    
    public void applyGradients(BackpropagateOutputMap gradients, float learningRate) {
        // Apply gradients to each component in reverse order (same as backprop)
        for (int i = components.size() - 1; i >= 0; i--) {
            ModelComponent component = components.get(i);
            BackpropagateOutput componentGradients = gradients.getOutput(i);

            // Apply gradients to this component
            component.applyGradients(componentGradients, learningRate);
        }
    }

    public void applyGradientsGPU(BackpropagateOutputMapGPU gradients, float learningRate, CUstream stream) {
        // Apply gradients to each component in reverse order (same as backprop)
        for (int i = components.size() - 1; i >= 0; i--) {
            ModelComponent component = components.get(i);
            BackpropagateOutputGPU componentGradients = gradients.getOutput(i);

            // Apply gradients to this component
            component.applyGradientsGPU(componentGradients, learningRate, stream);
        }
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

    public boolean gpuReady() {
        for (ModelComponent component : components) {
            if(!component.gpuReady())
                return false;
        }
        
        return true;
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
        ChainModel chainCopy = new ChainModel(outputDim);

        // Copy the component list
        for(ModelComponent component : components) {
            chainCopy.components.add(component.copy());
        }

        // Copy the current state
        chainCopy.outputDim = this.outputDim;
        chainCopy.splitRatio = this.splitRatio;

        return chainCopy;
    }
    
    public HashMap serialize() {
        HashMap obj = new HashMap();

        // Save state info
        obj.put("inputDim", inputDim);
        obj.put("outputDim", outputDim);
        obj.put("splitRatio", splitRatio);

        // Serialize components
        List<HashMap> componentArray = new ArrayList<>();
        obj.put("components", componentArray);

        for (ModelComponent component : components) {
            // Each component handles its own serialization
            componentArray.add(component.serialize());
        }

        return obj;
    }

    public static ChainModel deserialize(Map serialized) {
        int inputDim = (Integer) serialized.get("inputDim");
        
        // Create chain model with initial batch size
        ChainModel chain = new ChainModel(inputDim);

        // Restore state
        chain.outputDim = (Integer) serialized.get("outputDim");
        chain.splitRatio = (Integer) serialized.get("splitRatio");

        // Deserialize components
        List<Map> componentArray = (List) serialized.get("components");

        for (Map componentObj : componentArray) {
            ModelComponent component = ModelComponent.deserialize(componentObj);
            chain.components.add(component);
        }

        return chain;
    }

    public void updateWeightsFromGPU() {
        for (ModelComponent component : components) {
            component.updateWeightsFromGPU();
        }
    }
    
    public long getParametersCount() {
        long parametersCount = 0;
        
        for (ModelComponent component : components) {
            parametersCount += component.getParametersCount();
        }
        
        return parametersCount;
    }
    
    public long getBackpropagateMemoryRequired(int batchCount) {
        long parametersCount = 0;
        
        for (ModelComponent component : components) {
            parametersCount += component.getBackpropagateMemoryRequired(batchCount);
        }
        
        return parametersCount;        
    }
}
