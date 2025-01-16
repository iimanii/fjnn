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
import org.fjnn.loss.Loss;

/**
 *
 * @author ahmed
 */
public class BatchSizeAdapter extends ModelComponent {
    private final int inputBatchSize;       // Expected input batch size
    private final int targetBatchSize;      // Target batch size for the next component
    private boolean gpuReady;

    public BatchSizeAdapter(int inputBatchSize, int targetBatchSize) {
        this.inputBatchSize = inputBatchSize;
        this.targetBatchSize = targetBatchSize;
    }

    @Override
    public AdapterForwardOutput feedForward(float[] input, int batchCount) {
        // Calculate the total number of samples
        int totalSamples = batchCount * inputBatchSize;

        // Calculate the target batch count
        int targetBatchCount = totalSamples / targetBatchSize;

        // Validate divisibility
        if (totalSamples % targetBatchSize != 0) {
            throw new IllegalArgumentException(
                "Total samples " + totalSamples + " is not divisible by target batch size " + targetBatchSize
            );
        }
        
        // Pass inputs directly with the new batch size
        return new AdapterForwardOutput(targetBatchSize, targetBatchCount, input);
    }

    @Override
    public AdapterForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchCount, CUstream stream) {
        // Calculate the total number of samples
        int totalSamples = batchCount * inputBatchSize;

        // Calculate the target batch count
        int targetBatchCount = totalSamples / targetBatchSize;

        // Validate divisibility
        if (totalSamples % targetBatchSize != 0) {
            throw new IllegalArgumentException(
                "Total samples " + totalSamples + " is not divisible by target batch size " + targetBatchSize
            );
        }
        
        // Pass inputs directly with the new batch size
        return new AdapterForwardOutputGPU(targetBatchSize, targetBatchCount, input, true);
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
        return inputBatchSize;
    }

    @Override
    public int getOutputSize() {
        return targetBatchSize;
    }

    @Override
    public BackpropagateOutput backpropagate(FeedForwardOutput output, float[] deltaLoss, float learningRate) {
        // The deltaLoss is passed back unchanged since this adapter doesn't transform gradients.
        int totalSamples = output.totalSize;

        // Check that the size matches the input batch size.
        if (totalSamples % inputBatchSize != 0) {
            throw new IllegalArgumentException("Backpropagation size mismatch.");
        }

        int sourceBatchCount = totalSamples / inputBatchSize;
        return new AdapterBackpropagateOutput(inputBatchSize, sourceBatchCount, deltaLoss);
    }

    @Override
    public BackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU output, CUdeviceptr deltaLoss, float learningRate, CUstream stream) {
        // Similar to the CPU version, pass back the deltaLoss unchanged.
        int totalSamples = output.totalSize;

        // Check that the size matches the input batch size.
        if (totalSamples % inputBatchSize != 0) {
            throw new IllegalArgumentException("Backpropagation size mismatch.");
        }

        int sourceBatchCount = totalSamples / inputBatchSize;
        return new AdapterBackpropagateOutputGPU(inputBatchSize, sourceBatchCount, deltaLoss, false);
    }

    @Override
    public ModelComponent copy() {
        return new BatchSizeAdapter(inputBatchSize, targetBatchSize);
    }
}
