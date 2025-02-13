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
package org.fjnn.adapter;

import java.util.HashMap;
import java.util.Map;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.adapter.output.AdapterBackpropagateOutput;
import org.fjnn.adapter.output.AdapterBackpropagateOutputGPU;
import org.fjnn.adapter.output.AdapterForwardOutput;
import org.fjnn.adapter.output.AdapterForwardOutputGPU;
import org.fjnn.base.ModelComponent;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.base.output.FeedForwardOutput;
import org.fjnn.base.output.FeedForwardOutputGPU;

/**
 *
 * @author ahmed
 */
public class InputDimAdapter extends ModelComponent {
    private final int inputDim;       // Expected input batch size
    private final int targetDim;      // Target batch size for the next component
    private boolean gpuReady;

    public InputDimAdapter(int inputDim, int targetDim) {
        this.inputDim = inputDim;
        this.targetDim = targetDim;
    }

    @Override
    public AdapterForwardOutput feedForward(float[] input, int batchSize) {
        // Calculate the total number of samples
        long totalSamples = batchSize * inputDim;

        // Calculate the target batch size
        long targetBatchSize = totalSamples / targetDim;

        // Validate divisibility
        if (totalSamples % targetDim != 0) {
            throw new IllegalArgumentException(
                "Total samples " + totalSamples + " is not divisible by target batch size " + targetDim
            );
        }
        
        // Pass inputs directly with the new batch size
        return new AdapterForwardOutput(targetDim, (int)targetBatchSize, input);
    }

    @Override
    public AdapterForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchSize, CUstream stream) {
        // Calculate the total number of samples
        long totalSamples = batchSize * inputDim;

        // Calculate the target batch size
        long targetBatchSize = totalSamples / targetDim;

        // Validate divisibility
        if (totalSamples % targetDim != 0) {
            throw new IllegalArgumentException(
                "Total samples " + totalSamples + " is not divisible by target batch size " + targetDim
            );
        }
        
        // Pass inputs directly with the new batch size
        return new AdapterForwardOutputGPU(targetDim, (int)targetBatchSize, input, true);
    }

    @Override
    public boolean gpuReady() {
        return gpuReady;
    }

    @Override
    protected void prepareGPU0(CUstream stream) {
        gpuReady = true;
    }

    @Override
    protected void freeGPU0(CUstream stream) {
        gpuReady = false;
    }

    @Override
    public int getInputSize() {
        return inputDim;
    }

    @Override
    public int getOutputSize() {
        return targetDim;
    }

    @Override
    public BackpropagateOutput backpropagate(FeedForwardOutput output, float[] deltaLoss) {
        // The deltaLoss is passed back unchanged since this adapter doesn't transform gradients.
        long totalSamples = output.totalSize;

        // Check that the size matches the input batch size.
        if (totalSamples % inputDim != 0) {
            throw new IllegalArgumentException("Backpropagation size mismatch.");
        }

        long sourceBatchSize = totalSamples / inputDim;
        
        return new AdapterBackpropagateOutput(inputDim, (int)sourceBatchSize, deltaLoss);
    }

    @Override
    public BackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU output, CUdeviceptr deltaLoss, CUstream stream) {
        // Similar to the CPU version, pass back the deltaLoss unchanged.
        long totalSamples = output.totalSize;

        // Check that the size matches the input batch size.
        if (totalSamples % inputDim != 0) {
            throw new IllegalArgumentException("Backpropagation size mismatch.");
        }

        long sourceBatchSize = totalSamples / inputDim;
        
        return new AdapterBackpropagateOutputGPU(inputDim, (int)sourceBatchSize, deltaLoss, false);
    }

    @Override
    public ModelComponent copy() {
        return new InputDimAdapter(inputDim, targetDim);
    }
    
    @Override
    public HashMap serialize() {
       HashMap obj = new HashMap();

       // Store component type and main properties
       obj.put("type", "BatchSizeAdapter"); 
       obj.put("inputDim", getInputSize());
       obj.put("targetDim", getOutputSize());

       return obj;
    }

    public static InputDimAdapter deserialize(Map serialized) {
       int inputDim = (Integer)serialized.get("inputDim");
       int targetDim = (Integer)serialized.get("targetDim");
       
       return new InputDimAdapter(inputDim, targetDim);
    }

    @Override
    public void updateWeightsFromGPU() {

    }

    @Override
    public long getParametersCount() {
        return 0;
    }

    @Override
    public long getBackpropagateMemoryRequired(int batchSize) {
        return 0;
    }

    @Override
    public void applyGradients(BackpropagateOutput gradients, float learningRate, float weightDecay) {
        // do nothing
    }

    @Override
    public void applyGradientsGPU(BackpropagateOutputGPU gradients, float learningRate, float weightDecay, CUstream stream) {
        // do nothing        
    }
}
