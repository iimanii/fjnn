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
package org.fjnn.normalizer;

import java.util.HashMap;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.runtime.cudaStream_t;
import org.fjnn.base.ModelComponent;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.base.output.FeedForwardOutput;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.normalizer.outputs.DropoutBackpropagateOutput;
import org.fjnn.normalizer.outputs.DropoutBackpropagateOutputGPU;
import org.fjnn.normalizer.outputs.DropoutForwardOutput;
import org.fjnn.normalizer.outputs.DropoutForwardOutputGPU;
import org.fjnn.util.Rng;

/**
 *
 * @author ahmed
 */
public class Dropout extends ModelComponent {
    private final int neurons;
    public final float rate;
    
    public Dropout(int neurons, float rate) {
        if (rate < 0 || rate >= 1) {
            throw new IllegalArgumentException("Dropout rate must be between 0 and 1");
        }
        
        this.neurons = neurons;
        this.rate = rate;
    }

    @Override
    public DropoutForwardOutput feedForward(float[] input, int batchSize) {
        DropoutForwardOutput output = new DropoutForwardOutput(neurons, batchSize);
        
        float scale = 1.0f / (1.0f - rate);
           
        // Generate single mask for all batches
        for(int n = 0; n < neurons; n++) {
            output.mask[n] = Rng.nextFloat() > rate ? 1.0f : 0.0f;
        }
    
        for(int b = 0; b < batchSize; b++) {
            int offset = b * neurons;
            for(int n = 0; n < neurons; n++) {
                int idx = offset + n;
                output.output[idx] = input[idx] * output.mask[n] * scale;
            }
        }
        
        return output;
    }

    @Override
    public DropoutForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchSize, CUstream stream) {
        DropoutForwardOutputGPU output = new DropoutForwardOutputGPU(input, neurons, batchSize, stream);
        float scale = 1.0f / (1.0f - rate);

        // Generate random mask using cuRAND
        curandGenerator generator = CudaEngine.getCurandGenerator(deviceId);
        synchronized(generator) {
            JCurand.curandSetStream(generator, new cudaStream_t(stream));
            JCurand.curandGenerateUniform(generator, output.mask, neurons);
        }
        
        // Apply threshold to create binary mask
        CudaFunctions.vector.threshold(output.mask, rate, neurons, stream);
        
        // multiply
        CudaFunctions.vector.multiplyStride(input, output.mask, output.output, scale, neurons, batchSize, stream);

        return output;        
    }

    @Override
    public BackpropagateOutput backpropagate(FeedForwardOutput output, float[] deltaLoss) {
        if(!(output instanceof DropoutForwardOutput))
            throw new RuntimeException("DropoutForwardOutput required for backpropagation");
        
        int batchSize = output.batchSize;
        DropoutForwardOutput dropoutOutput = (DropoutForwardOutput)output;
        DropoutBackpropagateOutput backOutput = new DropoutBackpropagateOutput(neurons, batchSize);
        float scale = 1.0f / (1.0f - rate);
    
        // Process batch by batch
        for(int b = 0; b < batchSize; b++) {
            int offset = b * neurons;
            for(int i = 0; i < neurons; i++) {
                backOutput.deltaLoss[offset + i] = deltaLoss[offset + i] * dropoutOutput.mask[i] * scale;
            }
        }
        
        return backOutput;
    }

    @Override
    public BackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU output, CUdeviceptr deltaLoss, CUstream stream) {
        if(!(output instanceof DropoutForwardOutputGPU))
            throw new RuntimeException("DropoutForwardOutputGPU required for GPU backpropagation");

        DropoutForwardOutputGPU dropoutOutput = (DropoutForwardOutputGPU)output;
        DropoutBackpropagateOutputGPU backOutput = new DropoutBackpropagateOutputGPU(deltaLoss, neurons, output.batchSize, stream);
        float scale = 1.0f / (1.0f - rate);

        CudaFunctions.vector.multiplyStride(deltaLoss, dropoutOutput.mask, backOutput.deltaLoss, scale, neurons, output.batchSize, stream);

        return backOutput;
    }

    @Override
    public void applyGradients(BackpropagateOutput gradients, float learningRate, float weightDecay) {
        /* no weights */
    }

    @Override
    public void applyGradientsGPU(BackpropagateOutputGPU gradients, float learningRate, float weightDecay, CUstream stream) {
        /* no weights */
    }

    @Override
    public boolean gpuReady() {
        return deviceId != -1;
    }

    @Override
    protected void prepareGPU0(CUstream stream) {
        /* no weights */
    }

    @Override
    protected void freeGPU0(CUstream stream) {
        /* no weights */
    }

    @Override
    public int getInputSize() {
        return neurons;
    }

    @Override
    public int getOutputSize() {
        return neurons;
    }

    @Override
    public ModelComponent copy() {
        return new Dropout(neurons, rate);
    }

    @Override
    public HashMap serialize() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void updateWeightsFromGPU() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public long getParametersCount() {
        return 0;
    }

    @Override
    public long getBackpropagateMemoryRequired(int batchSize) {
        return neurons * batchSize;
    }
}
