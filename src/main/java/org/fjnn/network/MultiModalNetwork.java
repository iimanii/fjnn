/*
 * The MIT License
 *
 * Copyright 2023 ahmed.
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
package org.fjnn.network;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.activation.Activation;
import org.fjnn.base.ModelComponent;
import org.fjnn.base.Network;
import org.fjnn.base.NetworkInput;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.base.output.FeedForwardOutput;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class MultiModalNetwork extends Network<MultiModalNetwork> {
    NeuralNetwork[] heads;
    NeuralNetwork base;
    
    public MultiModalNetwork(NeuralNetwork[] inputs, int output, Activation outputActivation) {
        super(getInputSize(inputs), output, outputActivation);
        
        for(Network n : inputs) {
            if(!n.isFinalized())
                throw new RuntimeException("All input networks must be finalized");
        }
        
        int intermediateInput = 0;
        for(Network n : inputs)
            intermediateInput += n.getOutputSize();
        
        this.heads = inputs;
        this.base = new NeuralNetwork(intermediateInput, output, outputActivation);
        this.finalized = false;
    }
    
    MultiModalNetwork(NeuralNetwork[] inputs, NeuralNetwork base) {
        super(getInputSize(inputs), base.getOutputSize(), base.getOutputLayer().activation);
        
        for(Network n : inputs) {
            if(!n.isFinalized())
                throw new RuntimeException("All input networks must be finalized");
        }
        
        if(!base.isFinalized())
            throw new RuntimeException("base network must be finalized");
        
        this.heads = inputs;
        this.base = base;
        this.finalized = true;
    }
    
    private static int getInputSize(Network[] inputs) {
        int size = 0;
        
        for(Network n : inputs)
            size += n.getInputSize();
        
        return size;
    }

    @Override
    public void copyWeights(MultiModalNetwork n) {
        for(int i=0; i < heads.length; i++) {
            heads[i].copyWeights(n.heads[i]);
        }
        
        base.copyWeights(n.base);
    }
    
    public void copyWeights(NeuralNetwork n) {        
        base.copyWeights(n);
    }
    
    /* for building the network */
    public MultiModalNetwork addHiddenLayer(int neurons, Activation activation) {
        base.addHiddenLayer(neurons, activation);

        return this;
    }
    
    public MultiModalNetwork addConnection(int fromLayer, int toLayer) {
        base.addConnection(fromLayer, toLayer);
        
        return this;
    }
    
    public final MultiModalNetwork build() {
        base.build();
        
        this.finalized = true;
        
        return this;
    }
    
    public int getLayerCount() {
        return base.getLayerCount();
    }

    @Override
    public void randomize(float min, float max) {
        base.randomize(min, max);
    }

    @Override
    public MultiModalNetwork copy(boolean copyWeights, boolean createWeights) {
        NeuralNetwork[] inputsCopy = new NeuralNetwork[heads.length];
        
        for(int i=0; i < heads.length; i++) {
            inputsCopy[i] = heads[i].copy(copyWeights, createWeights);
        }
        
        NeuralNetwork baseCopy = base.copy(copyWeights, createWeights);
        
        return new MultiModalNetwork(inputsCopy, baseCopy);
    }

    @Override
    public double compare(MultiModalNetwork n0) {
        double score = 0;
        
        for(int i=0; i < heads.length; i++) {
            score += heads[i].compare(n0.heads[i]);
        }
            
        score += base.compare(n0.base);
        
        return score;
    }

    @Override
    public float[] compute(float[] input) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public float[] compute(float[] input, int count) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public float[] compute(FloatBuffer input) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public float[] compute(FloatBuffer input, int count) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void crossOverMutate(MultiModalNetwork a, MultiModalNetwork b, float min, float max, double mutation) {
        for(int i=0; i < heads.length; i++) {
            heads[i].crossOverMutate(a.heads[i], b.heads[i], min, max, mutation);
        }
        
        base.crossOverMutate(a.base, b.base, min, max, mutation);
    }

    @Override
    public void clipWeights(float clipMin, float clipMax) {
        for(int i=0; i < heads.length; i++) {
            heads[i].clipWeights(clipMin, clipMax);
        }
        
        base.clipWeights(clipMin, clipMax);
    }

    @Override
    public void ensureCPU() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean gpuReady() {
        for(int i=0; i < heads.length; i++) {
            if(!heads[i].gpuReady())
                return false;
        }
        
        return base.gpuReady();
    }

//    @Override
//    public void prepareGPU(int deviceId) {
//        for(int i=0; i < heads.length; i++) {
//            heads[i].prepareGPU(deviceId);
//        }
//        
//        base.prepareGPU(deviceId);
//        this.deviceId = deviceId;
//    }

    
    /**
     * each input has the following format
     * [n0_0][n0_1] ... [n0_count-1] ... [nx_0][nx_1] ...]
     * @param input
     * @param count
     * @return 
     */
    public float[] computeGPU(CUdeviceptr[] input, int count) {
        long memory = getGPUInternalComputeMemoryRequired(count);
        memory += getGPUComputeMemoryRequired(count);
        lockMemory(memory, getGPUDeviceId());
        
        boolean prepareThread = CudaEngine.getThreadDeviceId() != getGPUDeviceId();
        
        if(prepareThread)
           CudaEngine.prepareThread(getGPUDeviceId());
        
        CUdeviceptr intermediate = CudaEngine.getMempoolFloat(base.getInputSize() * count);
        
        try {
            int interOffset = 0;
            CUstream stream = CudaUtil.createStream();
            List<CUdeviceptr> outputs = new ArrayList<>();

            for(int i=0; i < heads.length; i++) {
                NeuralNetwork n = heads[i];
                CUdeviceptr currentInput = input[i];
                CUdeviceptr currentOutput = CudaEngine.getMempoolFloat(n.getOutputSize() * count);
                
//                System.out.println("xyz " + currentInput + " " + currentOutput + " " + n.getInputSize() + " " + n.getInputSize() * count);
                n.computeGPU(currentInput, currentOutput, count, stream);
                
                CUdeviceptr interDst = intermediate.withByteOffset(interOffset);
//                System.out.println("xyz " + interDst + " " + n.getOutputSize() + " " + base.getInputSize() + " " + count);
                CudaFunctions.vector.copyStride(currentOutput, interDst, n.getOutputSize(), base.getInputSize(), count, stream);
                
                outputs.add(currentOutput);
            
//                util.printArray(CudaUtil.fromGPUFloat(currentInput, n.getInputSize() * count));
//                util.printArray(CudaUtil.fromGPUFloat(currentOutput, n.getOutputSize() * count));
//                util.printArray(CudaUtil.fromGPUFloat(intermediate, base.getInputSize() * count));
//                System.out.println(inputOffset / CudaUtil.FLOAT_SIZE + " " + interDst / CudaUtil.FLOAT_SIZE);
//                System.out.println("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
                interOffset += n.getOutputSize() * CudaUtil.FLOAT_SIZE;
            }
            
            JCudaDriver.cuStreamSynchronize(stream);
            CudaUtil.freeStream(stream);
            
            for(CUdeviceptr ptr : outputs)
                CudaEngine.freeMempool(getGPUDeviceId(), ptr);
//            System.out.println("hags");
//            util.printArray(CudaUtil.fromGPUFloat(intermediate, base.getInputSize() * count));
            
            float[] result = base.computeGPU(intermediate, count);
            CudaEngine.freeMempool(getGPUDeviceId(), intermediate);
            
            if(prepareThread)
               CudaEngine.finalizeThread();

            return result;
        } finally {
            releaseMemory(memory);
        }
    }

    @Override
    public void crossOverMutateGPU(MultiModalNetwork a, MultiModalNetwork b, float min, float max, double mutation, boolean nocopy) {
        for(int i=0; i < heads.length; i++) {
            heads[i].crossOverMutateGPU(a.heads[i], b.heads[i], min, max, mutation, nocopy);
        }
        
        base.crossOverMutateGPU(a.base, b.base, min, max, mutation, nocopy);
    }

    @Override
    public void clipWeightsGPU(float clipMin, float clipMax) {
        for(int i=0; i < heads.length; i++) {
            heads[i].clipWeightsGPU(clipMin, clipMax);
        }
        
        base.clipWeightsGPU(clipMin, clipMax);
    }

//    @Override
//    public void freeGPU() {
//        for(int i=0; i < heads.length; i++) {
//            heads[i].freeGPU();
//        }
//        
//        base.freeGPU();
//    }

    private long getGPUInternalComputeMemoryRequired(int count) {
        return 2 * base.getInputSize() * count * CudaUtil.FLOAT_SIZE;
    }
    
    @Override
    public long getGPUComputeMemoryRequired(int count) {
        long memory = 0;
        
        for(int i=0; i < heads.length; i++) {
            memory += heads[i].getGPUComputeMemoryRequired(count);
        }
        
        memory += base.getGPUComputeMemoryRequired(count);
        
        return memory;
    }

    @Override
    public long getGPUPrepareMemoryRequired() {
        long memory = 0;
        
        for(int i=0; i < heads.length; i++) {
            memory += heads[i].getGPUPrepareMemoryRequired();
        }
        
        memory += base.getGPUPrepareMemoryRequired();
        
        return memory;
    }

    @Override
    public long getWeightsCount() {
        long weights = 0;
        
        for(int i=0; i < heads.length; i++) {
            weights += heads[i].getWeightsCount();
        }
        
        weights += base.getWeightsCount();
        
        return weights;
    }

    @Override
    public void kaiming() {
        for(int i=0; i < heads.length; i++) {
            heads[i].kaiming();
        }
        
        base.kaiming();
    }

    @Override
    public HashMap serialize(Set<String> ignoreProperties) {
        HashMap obj = super.serialize(ignoreProperties);
        
        List arr = new ArrayList();
        obj.put("inputs", arr);
        
        for(NeuralNetwork n : heads)
            arr.add(n.serialize(ignoreProperties));
        
        obj.put("base", base.serialize(ignoreProperties));
        
        return obj;
    }
    
    public static MultiModalNetwork deserialize(HashMap serialized) {
        List<HashMap> arr = (List)serialized.get("inputs");
        
        NeuralNetwork[] inputs = new NeuralNetwork[arr.size()];
        for(int i=0; i < arr.size(); i++)
            inputs[i] = NeuralNetwork.deserialize(arr.get(i));
        
        NeuralNetwork base = NeuralNetwork.deserialize((HashMap)serialized.get("base"));
        
        MultiModalNetwork result = new MultiModalNetwork(inputs, base);
        
        result.properties.putAll((HashMap)serialized.get("properties"));
        return result;
    }

    @Override
    public float[] compute(NetworkInput input) {
        if(input.type != NetworkInput.InputType.deviceptr_list)
            throw new RuntimeException("Invalid input for multimodal: " + input.type);
        
        return computeGPU(input.getInputDevicePtrList(), input.count);
    }
    
    public NeuralNetwork[] getHeads() {
        return heads;
    }
    
    public NeuralNetwork getBase() {
        return base;
    }

    @Override
    public FeedForwardOutput feedForward(float[] input, int batchCount) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public FeedForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchCount, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    protected void prepareGPU0(CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    protected void freeGPU0(CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public BackpropagateOutput backpropagate(FeedForwardOutput output, float[] deltaLoss, float learningRate) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public BackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU output, CUdeviceptr deltaLoss, float learningRate, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public ModelComponent copy() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public HashMap serialize() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void updateWeightsFromGPU() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
