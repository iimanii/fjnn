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

import java.util.HashMap;
import java.util.Map;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.base.output.FeedForwardOutput;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.adapter.InputDimAdapter;
import org.fjnn.adapter.PositionalEncoderAdapter;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.loss.Loss;
import org.fjnn.network.NeuralNetwork;
import org.fjnn.normalizer.Normalizer;

/**
 *
 * @author ahmed
 */
public abstract class ModelComponent {
    /* device id for gpu to use */
    protected int deviceId = -1;
    
    public final int getGPUDeviceId() {
        return deviceId;
    }
    
    public FeedForwardOutput feedForward(float[] input, int inputSize, int batchSize) {
        if(inputSize != getInputSize())
            throw new RuntimeException("Invalid batch size " + batchSize + " != " + getInputSize());
        
        return feedForward(input, batchSize);
    }
    public abstract FeedForwardOutput feedForward(float[] input, int batchSize);
    
    public final FeedForwardOutputGPU feedForwardGPU(CUdeviceptr input, int inputSize, int batchSize, CUstream stream) {
        checkGPUContext();
        
        if(inputSize != getInputSize())
            throw new RuntimeException("Invalid batch size " + inputSize + " != " + getInputSize());
        
        return feedForwardGPU(input, batchSize, stream);
    }
    public abstract FeedForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchSize, CUstream stream);
    
    public BackpropagateOutput backpropagate(FeedForwardOutput output, float[] deltaLoss, int deltaLossSize, int deltaLossCount) {
        if(output.outputDim != deltaLossSize || getOutputSize() != deltaLossSize)
            throw new RuntimeException("Invalid batch size " + deltaLossSize + " != " + getInputSize());
        
        if(output.batchSize != deltaLossCount)
            throw new RuntimeException("Invalid batch count " + deltaLossCount + " != " + output.batchSize);
        
        return backpropagate(output, deltaLoss);
    }
    protected abstract BackpropagateOutput backpropagate(FeedForwardOutput output, float[] deltaLoss);
    
    public BackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU output, CUdeviceptr deltaLoss, int deltaLossDim, int batchSize, CUstream stream) {
        if(output.outputDim != deltaLossDim || getOutputSize() != deltaLossDim)
            throw new RuntimeException("Invalid batch size " + deltaLossDim + " != " + getInputSize());
        
        if(output.batchSize != batchSize)
            throw new RuntimeException("Invalid batch count " + batchSize + " != " + output.batchSize);
        
        return backpropagateGPU(output, deltaLoss, stream);
    }
    protected abstract BackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU output, CUdeviceptr deltaLoss, CUstream stream);
    
    public abstract void applyGradients(BackpropagateOutput gradients, float learningRate, float weightDecay);
    public abstract void applyGradientsGPU(BackpropagateOutputGPU gradients, float learningRate, float weightDecay, CUstream stream);

    public abstract boolean gpuReady();
    
    public final void prepareGPU() {
        prepareGPU(null);
    }
    public final void prepareGPU(CUstream stream) {
        if(gpuReady())
            throw new RuntimeException("GPU already initialized for this component");        
        
        boolean invalidThreadId = CudaEngine.getThreadDeviceId() == -1;
        
        if (invalidThreadId)
            throw new RuntimeException("Invalid cuda context, must call CudaEngine.prepareThread(...)");
        
        if(stream == null) {
            stream = CudaUtil.createStream();
            
            prepareGPU0(stream);

            JCudaDriver.cuStreamSynchronize(stream);
            
            CudaUtil.freeStream(stream);
        } else {            
            prepareGPU0(stream);
        }
        
        deviceId = CudaEngine.getThreadDeviceId();
    }    
    protected abstract void prepareGPU0(CUstream stream);

    public final void freeGPU() {
        freeGPU(null);
    }
    public final void freeGPU(CUstream stream) {
        if(!gpuReady())
            throw new RuntimeException("Calling freeGPU on a compoenet that is not loaded to GPU");

        if(stream == null) {
            stream = CudaUtil.createStream();
            
            freeGPU0(stream);

            JCudaDriver.cuStreamSynchronize(stream);
            
            CudaUtil.freeStream(stream);
        } else {            
            freeGPU0(stream);
        }
        
        deviceId = -1;
    }
    protected abstract void freeGPU0(CUstream stream);
    
    protected void checkGPUContext() {
        if (!gpuReady())
            throw new RuntimeException("Component is not loaded to the GPU, please call prepareGPU first");

        boolean invalidThreadId = CudaEngine.getThreadDeviceId() != deviceId;

        if (invalidThreadId)
            throw new RuntimeException("Invalid cuda context for device: " + deviceId + " must call CudaEngine.prepareThread(...)");
    }
    
    public abstract int getInputSize();
    public abstract int getOutputSize();
    
    public abstract ModelComponent copy();
    
    public abstract HashMap serialize();

    public static ModelComponent deserialize(Map serialized) {
        String type = (String)serialized.get("type");
        
        switch (type) {
            case "NeuralNetwork":
                return NeuralNetwork.deserialize(serialized);
            case "BatchSizeAdapter":
                return InputDimAdapter.deserialize(serialized);
            case "PositionalEncoderAdapter":
                return PositionalEncoderAdapter.deserialize(serialized);
            case "LayerNormalizer":
                return Normalizer.deserialize(serialized);
                
            default:
                throw new RuntimeException("Unknown component type: " + type);
        }
    }
    
    public abstract void updateWeightsFromGPU();
    
    public abstract long getParametersCount();
    
    public abstract long getBackpropagateMemoryRequired(int batchSize);
}
