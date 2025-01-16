/*
 * The MIT License
 *
 * Copyright 2018 Ahmed Tarek.
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

import org.fjnn.network.outputs.NeuralNetworkBackpropagateOutput;
import org.fjnn.network.outputs.NeuralNetworkForwardOutput;
import org.fjnn.network.outputs.NeuralNetworkForwardOutputGPU;
import org.fjnn.network.outputs.NeuralNetworkBackpropagateOutputGPU;
import org.fjnn.base.Network;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.cublasHandle;
import jcuda.jcurand.curandGenerator;
import org.fjnn.activation.Activation;
import org.fjnn.base.output.FeedForwardOutput;
import org.fjnn.base.NetworkInput;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.loss.Loss;
import org.fjnn.network.Layer.crossOverMutateResult;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 */
public class NeuralNetwork extends Network<NeuralNetwork> {
    Layer[] layers;
    
    /* index of last layer */
    int last;

    /* memory requirement for prepare gpu */
    long weightsMemReq;
    
    /* memory requirement a compute iteration */
    long computeMemReq;
    
    /* true if CPU memory was emptied */
    boolean cpuFree;
    
    /* for building the network */
    List<LayerPlan> plan;
    
    /* I:size->H:size[activation].....->O:size[activation] */
    String signature;
    
    public NeuralNetwork(int input, int output, Activation outputActivation) {
        super(input, output, outputActivation);
        
        this.plan = new ArrayList<>();
    }
        
    NeuralNetwork(Layer[] layers) {
        super(layers[0].neurons, layers[layers.length-1].neurons, layers[layers.length-1].activation);
        
        this.layers = layers;
        this.finalized = true;
        this.last = layers.length - 1;

        computeMemoryRequirements();
    }
    
    @Override
    public NeuralNetwork copy() {
        return copy(true, true);
    }
    
    @Override
    public NeuralNetwork copy(boolean copyWeights, boolean createWeights) {
        Layer[] copied = new Layer[layers.length];
        
        for(int i=0; i < layers.length; i++) {
            copied[i] = layers[i].copy(copyWeights, createWeights);
        }
        
        return new NeuralNetwork(copied);
    }
    
    @Override
    public void copyWeights(NeuralNetwork n) {
        for(int i=0; i < layers.length; i++) {
            layers[i].copyWeights(n.layers[i]);
        }
    }
    
    /* for building the network */
    public NeuralNetwork addHiddenLayer(int neurons, Activation activation) {
        if(finalized)
            throw new RuntimeException("network already finalized");
        
        plan.add(new LayerPlan(neurons, activation));
        return this;
    }
    
    public NeuralNetwork addConnection(int fromLayer, int toLayer) {
        if(!finalized)
            throw new RuntimeException("network layer structure is not finalized yet");
        
        Layer sourceLayer = layers[fromLayer];
        Layer targetLayer = layers[toLayer];
    
        int targetNeurons = targetLayer.neurons;
        
        // Add forward connection from source to target
        sourceLayer.addConnection(toLayer, targetNeurons);
        
        // Add reverse connection from target to source (incoming)
        targetLayer.addReverseConnection(fromLayer);//, sourceLayer.getConnection(toLayer));
    
        computeMemoryRequirements();
        return this;
    }
    
    public final NeuralNetwork build() {
        if(finalized)
            throw new RuntimeException("network already finalized");

        /* input + output + hidden */
        layers = new Layer[plan.size()+2];

        int i=0;

        /* input layer */
        layers[0] = new Layer(i++, inputSize, null);

        Layer prev = layers[0];

        for(LayerPlan p : plan) {
            Layer layer = new Layer(i++, p.neurons, p.activation);

            // Add connection from previous layer to the current layer
            prev.addConnection(layer.index, layer.neurons);

            // Also add the reverse connection from the current layer to the previous layer
            layer.addReverseConnection(prev.index);//, prev.getConnection(layer.index));

            layers[layer.index] = layer;            
            prev = layer;
        }

        /* output layer */
        layers[i] = new Layer(i, outputSize, outputActivation);

        // Add connection from the last hidden layer to the output layer
        prev.addConnection(i, outputSize);

        // Also add the reverse connection from output layer to the previous layer
        layers[i].addReverseConnection(prev.index);//, prev.getConnection(i));

        last = i;

        computeMemoryRequirements();

        finalized = true;

        return this;
    }
    
    /* get functions */
    public Layer getInputLayer() {
        return layers[0];
    }
    
    public Layer getOutputLayer() {
        return layers[last];
    }
    
    public int getHiddenCount() {
        return layers.length - 2;
    }
    
    public Layer getLayer(int layer) {
        return layers[layer];
    }
        
    public int getLayerCount() {
        return layers.length;
    }
    
    /* total number of weights including bias */
    @Override
    public long getWeightsCount() {
        long count = 0;
        
        for(Layer l : layers)
            count += l.getWeightsCount();
        
        return count;
    }
    
    /* initialization functions */
    @Override
    public void randomize(float min, float max) {
        for(Layer l : layers)
            l.initUniform(min, max);
    }
    
    public void xavier() {
        xavier(1);
    }
    public void xavier(int scalar) {
        for(Layer l : layers)
            l.xavier(scalar);
    }
    
    @Override
    public void kaiming() {
        kaiming(1);
    }
    public void kaiming(int scalar) {
        for(Layer l : layers)
            l.kaiming(scalar);
    }

    /*
     * Compute CPU
     */
    @Override
    public float[] compute(float[] input) {
        return compute(input, 1);
    }
    
    @Override
    public float[] compute(float[] input, int count) {
        float[][] results = new float[getLayerCount()][];
        results[0] = util.copyArray(input);
        
        for(int i=0; i < layers.length; i++)
            layers[i].feedForward(results[i], count, results);
        
        return results[last];
    }
    
    @Override
    public float[] compute(FloatBuffer input) {
        return compute(input, 1);
    }
    
    @Override
    public float[] compute(FloatBuffer input, int count) {
        if(!nativeReady())
            prepareCPU();
        
        FloatBuffer[] results = new FloatBuffer[getLayerCount()];
        results[0] = input;
        
        for(int i=0; i < layers.length; i++)
            layers[i].feedForward(results[i], count, results);
        
        float[] result = new float[outputSize * count];
        results[last].rewind();
        results[last].get(result);

        return result;
    }
    
    
    @Override
    public NeuralNetworkForwardOutput feedForward(float[] input, int count) {
        NeuralNetworkForwardOutput activations = new NeuralNetworkForwardOutput(getOutputSize(), count, getLayerCount());
        activations.preActivations[0] = input;
        
        for (int i = 0; i < layers.length; i++)
            layers[i].feedForward(activations);

        return activations;
    }

    /*
     * https://www.mldawn.com/deriving-the-gradient-descent-rule-part-2/
     *
     *  1. Compute delta for the output layer using the derivative of the loss function with respect to the output.
     *  2. Calculate weight and bias gradients for the output layer using the delta and activations from the previous layer.
     *  3. Backpropagate the delta to the previous layer (hidden layers) using the weights of the current layer and the derivative of the activation function.
     *  4. Repeat steps 2-3 for each hidden layer, moving backward through the network.
     *  5. Accumulate gradients across all samples in the batch.
     *  6. After processing the entire batch, update weights and biases using the accumulated gradients and the learning rate.
     *  7. Clear the accumulated gradients after the update.
     */
    public NeuralNetworkBackpropagateOutput backpropagate(NeuralNetworkForwardOutput output, float[] target, float learningRate, Loss lossFunction) {
        float[] result = output.output();
        float[] deltaLoss = lossFunction.derivative(result, target);
        
        return backpropagate(output, deltaLoss, learningRate);
    }
    
    @Override
    public NeuralNetworkBackpropagateOutput backpropagate(FeedForwardOutput feedforward, float[] deltaLoss, float learningRate) {
        if(!(feedforward instanceof NeuralNetworkForwardOutput))
            throw new RuntimeException("Invalid feedforward output");
        
        NeuralNetworkForwardOutput output = (NeuralNetworkForwardOutput)feedforward;
        
        float[][] preActivationDeltas = new float[layers.length][];

        // Step 3: Initialize the ConnectionGradient Map to store gradients for all connections
        List<Map<Integer, ConnectionGradient>> layerConnectionGradients = new ArrayList<>();
        for (int i = 0; i < layers.length; i++) {
            layerConnectionGradients.add(new HashMap<>());
        }

        // Step 1: Initialize the activation deltas for the output layer
        float[] currentActivationDeltas = deltaLoss;

        // Step 4: Backpropagate through all hidden layers
        for (int i = layers.length - 1; i >= 0; i--) {
            Layer currentLayer = layers[i];
            currentLayer.backpropagate(output, currentActivationDeltas, preActivationDeltas, layerConnectionGradients.get(i));
            currentActivationDeltas = null;
        }

        // Step 5: Update weights and biases using the accumulated gradients for each layer
        for (int i = 0; i < layers.length - 1; i++) {  // Iterate through all layers except the last (output layer)
            Layer currentLayer = layers[i];
            currentLayer.updateWeights(layerConnectionGradients.get(i), learningRate);
        }
        
        return new NeuralNetworkBackpropagateOutput(preActivationDeltas, layerConnectionGradients, getInputSize(), output.batchCount);
    }
    
    /*
     * Compute GPU 
     */
    public float[] computeGPU(CUdeviceptr input) {
        return computeGPU(input, 1);
    }
    
    public float[] computeGPU(CUdeviceptr input, int count) {
        return computeGPU(input, count, true);
    }
    
    public float[] computeGPU(CUdeviceptr input, int count, boolean memlock) {
        long memory = getGPUComputeMemoryRequired(count);
        
        int deviceId = getGPUDeviceId();
        
        if(memlock)
            lockMemory(memory, deviceId);
        
        try {
            checkGPUContext();

            cublasHandle handle = CudaEngine.getCublasHandle(deviceId);

            CUdeviceptr[] layerResults = new CUdeviceptr[getLayerCount()];
            layerResults[0] = input;

            for(int i=0; i < layers.length; i++)
                layers[i].feedForwardGPU(layerResults[i], count, layerResults, null, handle);

            float[] result = CudaUtil.fromGPUFloat(layerResults[last], count * outputSize);

            for(int i=1; i < layerResults.length; i++)
                CudaUtil.free(layerResults[i]);

            return result;
        } finally {
            if(memlock)
                releaseMemory(memory);
        }
    }
    
    public void computeGPU(CUdeviceptr input, CUdeviceptr output, int count, CUstream stream) {
        checkGPUContext();
        
        cublasHandle handle = CudaEngine.getCublasHandle(deviceId);

        CUdeviceptr[] layerResults = new CUdeviceptr[getLayerCount()];
        layerResults[0] = input;
        layerResults[last] = output;
        
        JCudaDriver.cuMemsetD32Async(layerResults[last], 0, outputSize * count, stream);
        
        for(int i=0; i < layers.length; i++)
            layers[i].feedForwardGPU(layerResults[i], count, layerResults, stream, handle);
        
        for(int i=1; i < layerResults.length-1; i++)
            CudaUtil.freeAsync(layerResults[i], stream);
    }
    
    @Override
    public NeuralNetworkForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchCount, CUstream stream) {
        cublasHandle handle = CudaEngine.getCublasHandle(deviceId);
        
        NeuralNetworkForwardOutputGPU activations = new NeuralNetworkForwardOutputGPU(getOutputSize(), batchCount, getLayerCount());
        activations.preActivations[0] = input;

        for(int i=0; i < layers.length; i++) {
            layers[i].feedForwardGPU(activations, stream, handle);
        }
        
        return activations;
    }
    
    public void backpropagateGPU(NeuralNetworkForwardOutputGPU output, CUdeviceptr truth, float learningRate, Loss lossFunction, CUstream stream) {
        backpropagateGPUDebug(output, truth, learningRate, lossFunction, stream).freeAsync(stream);
    }
    
    public NeuralNetworkBackpropagateOutputGPU backpropagateGPUDebug(NeuralNetworkForwardOutputGPU output, CUdeviceptr truth, float learningRate, Loss lossFunction, CUstream stream) {
        checkGPUContext();
        
        // Step 1: Get result and calculate delta loss on GPU
        long totalOutputSize = outputSize * output.batchCount;
        CUdeviceptr outputPtr = output.output();
        CUdeviceptr deltaLoss = CudaUtil.createFloatAsync(totalOutputSize, stream);
        
        lossFunction.derivativeGPU(outputPtr, truth, deltaLoss, totalOutputSize, stream);
        
        return backpropagateGPU(output, deltaLoss, learningRate, stream);
    }
    
    @Override
    public NeuralNetworkBackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU feedforward, CUdeviceptr deltaLoss, float learningRate, CUstream stream) {
        if(!(feedforward instanceof NeuralNetworkForwardOutputGPU))
            throw new RuntimeException("Invalid feedforward output");
        
        NeuralNetworkForwardOutputGPU output = (NeuralNetworkForwardOutputGPU) feedforward;
        
        cublasHandle handle = CudaEngine.getCublasHandle(deviceId);
        
        CUdeviceptr[] preActivationDeltas = new CUdeviceptr[layers.length];
        
        // Step 3: Store gradients for all connections in GPU memory
        List<Map<Integer, ConnectionGradientGPU>> layerConnectionGradients = new ArrayList<>();
        for (int i = 0; i < layers.length; i++) {
            layerConnectionGradients.add(new HashMap<>());
        }
        
        CUdeviceptr currentActivationDeltas = deltaLoss;
                
        // Step 4: Backpropagate through layers
        for (int i = layers.length - 1; i >= 0; i--) {
            Layer currentLayer = layers[i];
            currentLayer.backpropagateGPU(output, currentActivationDeltas, preActivationDeltas, layerConnectionGradients.get(i), stream, handle);
            currentActivationDeltas = null;
        }

        // Step 5: Update weights and biases using accumulated gradients
        for (int i = 0; i < layers.length - 1; i++) {
            Layer currentLayer = layers[i];
            currentLayer.updateWeightsGPU(layerConnectionGradients.get(i), learningRate, stream, handle);
        }
        
        return new NeuralNetworkBackpropagateOutputGPU(preActivationDeltas, layerConnectionGradients, getInputSize(), output.batchCount);
    }
    
    @Override
    public long getGPUComputeMemoryRequired(int inputcount) {
        return computeMemReq * inputcount;
    }
    
    public long getGPUBackpropagateMemoryRequired(int inputcount) {
        /* pre and post activations stored */
        long feedForward = 2 * computeMemReq * inputcount;
        
        /* activation deltas + gradients */
        long backpropagate = computeMemReq * inputcount + weightsMemReq;
        
        return feedForward + backpropagate;
    }
    
    @Override
    public long getGPUPrepareMemoryRequired() {
        return weightsMemReq;
    }
//    
//    public double sumAbsWeightsGPU() {
//        if(!gpuReady())
//            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
//        
//        boolean prepareThread = CudaEngine.getThreadDeviceId() != deviceId;
//        
//        if(prepareThread)
//           CudaEngine.prepareThread(deviceId);
//        
//        CUstream stream = CudaEngine.aquireStream(deviceId);
//        float[][] results = new float[getLayerCount()][];
//        
//        for(int i=0; i < layers.length; i++) {
//            results[i] = layers[i].sumAbsWeightsGPU(stream);
//        }
//        
//        JCudaDriver.cuStreamSynchronize(stream);
//        
//        CudaEngine.releaseStream(deviceId, stream);
//
//        double result = 0;
//        for(float[] a : results)
//            for(float f : a)
//                result += f;
//        
//        for(int i=0; i < layers.length; i++)
//            CudaEngine.freeMempool(deviceId, layerResults[i]);
//        
//        if(prepareThread)
//           CudaEngine.finalizeThread();
//        
//        return result;
//    }
    
    /* add neurons to layer */
    public void expandLayer(int layer, int neurons) {
        expandLayer(layer, neurons, 0);
    }
    
    public void expandLayer(int layer, int neurons, float initialWeight) {
        if(deviceId != -1)
            freeGPU();
        
        layers[layer].addNeurons(neurons, initialWeight);
        
        if(layer != 0)
            layers[layer-1].addLinks(neurons, initialWeight);
        
        signature = null;
    }
    
    @Override
    public void crossOverMutate(NeuralNetwork a, NeuralNetwork b, float min, float max, double mutation) {
        for(int i=0; i < layers.length; i++)
            layers[i].crossOverMutate(a.getLayer(i), b.getLayer(i), min, max, mutation);
    }
    
    @Override
    public void crossOverMutateGPU(NeuralNetwork a, NeuralNetwork b, float min, float max, double mutation, boolean nocopy) {
        if(!a.gpuReady() || !b.gpuReady())
            throw new RuntimeException("Parent networks are not loaded to the GPU, please call prepareGPU first");
        
        if(a.deviceId != b.deviceId)
            throw new RuntimeException("Parent networks are on different GPUs");
        
        if(gpuReady())
            throw new RuntimeException("Network is gpu ready, please call freeGPU first");
        
        boolean invalidThreadId = CudaEngine.getThreadDeviceId() != deviceId;

        if(invalidThreadId)
            throw new RuntimeException("Invalid cuda context for device: " + deviceId + " must call CudaEngine.prepareThread(...)");
            
        this.deviceId = a.deviceId;
            
        /* 1x prepare gpu + 3x rng arrays */
        lockMemory(weightsMemReq * 4, deviceId);
        
        try {
            CUstream stream = CudaUtil.createStream();
            curandGenerator generator = CudaEngine.getCurandGenerator(deviceId);

            for(int i=0; i < layers.length; i++)
                layers[i].crossOverMutateGPU(a.getLayer(i), b.getLayer(i), min, max, mutation, nocopy, stream, generator);

            JCudaDriver.cuStreamSynchronize(stream);
            CudaUtil.freeStream(stream);

            for(int i=0; i < layers.length; i++)
                layers[i].freeGPURng();
        } finally {
            releaseMemory(weightsMemReq * 3);
        }
    }
    
    @Override
    public void clipWeights(float min, float max) {
        for(int i=0; i < layers.length; i++)
            layers[i].clipWeights(min, max);
    }

    @Override
    public void clipWeightsGPU(float min, float max) {
        if(gpuReady())
            throw new RuntimeException("Network is gpu ready, please call freeGPU first");
        
        CudaEngine.prepareThread(deviceId);
        CUstream stream = CudaUtil.createStream();

        for(int i=0; i < layers.length; i++)
            layers[i].clipWeightsGPU(min, max, stream);

        JCudaDriver.cuStreamSynchronize(stream);
        CudaUtil.freeStream(stream);

        CudaEngine.finalizeThread();
    }    
    
    public void crossOverMutate(NeuralNetwork a, NeuralNetwork b, NeuralNetwork min, NeuralNetwork max, double mutation) {
        for(int i=0; i < layers.length; i++)
            layers[i].crossOverMutate(a.getLayer(i), b.getLayer(i), min.getLayer(i), max.getLayer(i), mutation);
    }
    
    public crossOverMutateResult crossOverMutate(NeuralNetwork a, NeuralNetwork b, 
                                NeuralNetwork minA, NeuralNetwork maxA, 
                                NeuralNetwork minB, NeuralNetwork maxB, 
                                double mutation) {
        
        crossOverMutateResult result = new crossOverMutateResult();
        
        for(int i=0; i < layers.length; i++)
            layers[i].crossOverMutate(a.getLayer(i), b.getLayer(i), 
                                      minA.getLayer(i), maxA.getLayer(i), 
                                      minB.getLayer(i), maxB.getLayer(i), 
                                      mutation, result);
        
        return result;
    }
    
    @Override
    public double compare(NeuralNetwork a) {
        double score = 0;
        
        for(int i=0; i < a.getLayerCount(); i++)
            score += layers[i].compare(a.getLayer(i));
        
        return score;
    }
    
//    public float compareGPU(NeuralNetwork a) {        
//        if(!gpuReady() || !a.gpuReady())
//            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
//        
//        CudaEngine.prepareThread(deviceId);
//        
//        float result = compareGPU(a, CudaEngine.getStream(deviceId));
//        
//        CudaEngine.finalizeThread();
//        
//        return result;
//    }
    
    public float compareGPU(NeuralNetwork a, CUstream stream) {
        if(!gpuReady() || !a.gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        float score = 0;
        
        for(int i=0; i < a.getLayerCount(); i++)
            score += layers[i].compareGPU(a.getLayer(i), stream);
        
        
        return score;
    }
    
    @Override
    protected void freeGPU0(CUstream stream) {
        for(Layer l : layers)
            l.freeGPU(stream);
        
        releaseMemory(weightsMemReq);
    }
    
    /**
     * Average of all weights
     * @return 
     */
    public float mean() {
        float sum = 0;
        int count = 0;
        
        for (Layer l : layers) {
            for(Connection c : l.connections.values()) {
                float[] weights = c.weights;

                for(float w : weights)
                    sum += w;

                count += weights.length;

                float[] bias = c.biases;

                for(float b : bias)
                    sum += b;

                count += bias.length;
            }
        }
        
        return sum / count;
    }
    
    /**
     * Standard deviation of all weights
     * @param mean
     * @return 
     */
    public float sd(double mean) {
        float sd = 0;
        int count = 0;
        
        for (Layer l : layers) {
            for(Connection c : l.connections.values()) {
                float[] weights = c.weights;

                for(float w : weights)
                    sd += Math.pow(w - mean, 2);

                count += weights.length;

                float[] bias = c.biases;

                for(float b : bias)
                    sd += Math.pow(b - mean, 2);

                count += bias.length;
            }
        }
        
        return (float) Math.sqrt(sd / count);
    }
    
    public String getSignature() {
        if(signature != null)
            return signature;
        
        StringBuilder b = new StringBuilder();
        b.append("I:").append(layers[0].neurons).append(",");
        
        for(int i=1; i < layers.length-1; i++) {
            b.append("H:").append(layers[i].neurons);
            if(layers[i].activation != null)
                b.append("[").append(layers[i].activation.toName()).append("]");
            b.append(",");
        }
        
        b.append("O:").append(layers[last].neurons);
        if(layers[last].activation != null)
            b.append("[").append(layers[last].activation.toName()).append("]");
        
        signature = b.toString();
        
        return signature;
    }

    private CUdeviceptr[] createComputeLayerGPUMemory(int count, CUstream stream) {
        long totalSize = getGPUComputeMemoryRequired(count);
        CUdeviceptr pool = CudaUtil.create(totalSize);
        JCudaDriver.cuMemsetD32Async(pool, 0, totalSize/4, stream);
        
        CUdeviceptr[] result = new CUdeviceptr[getLayerCount()];
        long ptr = 0;
        
        for(int j=1; j < layers.length; j++) {
            result[j] = pool.withByteOffset(ptr);
            ptr += CudaUtil.alignLength(layers[j].neurons * count * CudaUtil.FLOAT_SIZE, CudaUtil.DEFAULT_MEM_ALIGN);
        }
        
        return result;
    }

    private void computeMemoryRequirements() {
        computeMemReq = 0;
        
        for(int j=1; j < layers.length; j++)
            computeMemReq += layers[j].neurons;
        
        computeMemReq *= Float.BYTES;
        
        weightsMemReq = 0;
        for(Layer l : layers)
            weightsMemReq += l.getMemoryRequirements();
    }

    @Override
    public float[] compute(NetworkInput input) {
        switch(input.type) {
            case float_array:
                return compute(input.getInputArray(), input.count);
            case floatbuffer_array:
                return compute(input.getInputFloatBuffer(), input.count);
            case deviceptr:
                return computeGPU(input.getInputDevicePtr(), input.count);
            default:
                throw new RuntimeException("invalid input type for neuralnetwork: " + input.type);
        }
    }
    
    
    public boolean nativeReady() {
        boolean nativeReady = true;
        for(Layer l : layers)
            nativeReady &= l.nativeReady();
        
        return nativeReady;
    }
    
    public void prepareCPU() {
        for(Layer l : layers)
            l.prepareCPU();
    }
    
    public void freeCPU() {
        if(cpuFree)
            return;
        
        for(Layer l : layers)
            l.freeCPU();
        
        cpuFree = true;
    }
    
    @Override
    public void ensureCPU() {
        checkGPUContext();
        
        CUstream stream = CudaUtil.createStream();
        
        for(Layer l : layers)
            l.ensureCPU(stream);
        
        JCudaDriver.cuStreamSynchronize(stream);        
        CudaUtil.freeStream(stream);
    }
    
    public void updateWeightsFromGPU() {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU");
        
        CudaEngine.prepareThread(deviceId);
        
        CUstream stream = CudaUtil.createStream();
        
        for(Layer l : layers)
            l.updateWeightsFromGPU(stream);
        
        JCudaDriver.cuStreamSynchronize(stream);
        
        CudaUtil.freeStream(stream);
        
        CudaEngine.finalizeThread();
    }

    public HashMap serialize() {
        return serialize(new HashSet<>());
    }

    @Override
    public HashMap serialize(Set<String> ignoreProperties) {
        HashMap obj = super.serialize(ignoreProperties);
        
        List<HashMap> array = new ArrayList<>();
        
        obj.put("layers", array);
        
        for(Layer l: layers)
            array.add(l.serialize());
        
        return obj;
    }
    
    public static NeuralNetwork deserialize(Map serialized) {
        List<HashMap> array = (List)serialized.get("layers");
        
        Layer[] layers = new Layer[array.size()];
        
        for(int i=0; i < array.size(); i++)
            layers[i] = Layer.deserialize(array.get(i));
        
        /* check reverse connections */
        for(int i=0; i < layers.length; i++) {
            Layer sourceLayer = layers[i];
            for(int j : sourceLayer.connections.keySet()) {
                Layer targetLayer = layers[j];
                
                if(!targetLayer.reverseConnections.contains(i)) {
                    System.out.printf("adding reverse connection: %d <- %d\n", j, i);
                    targetLayer.addReverseConnection(i);
                }
            }
        }
        
        NeuralNetwork result = new NeuralNetwork(layers);
        
        result.properties.putAll((HashMap)serialized.get("properties"));
        return result;
    }
    
    @Override
    public boolean gpuReady() {
        boolean gpuReady = true;
        for(Layer l : layers)
            gpuReady &= l.gpuReady();
        
        return gpuReady;
    }

    @Override
    public void prepareGPU0(CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        lockMemory(weightsMemReq, device);

        for(Layer l : layers)
            l.prepareGPU(stream);
    }
    
    public void moveGPU(int newDevice) {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        if(deviceId == newDevice)
            throw new RuntimeException("Moving GPU to same device");
            
        freeGPU();
        
        CudaEngine.prepareThread(newDevice);
        
        prepareGPU();
        
        CudaEngine.finalizeThread();
    }    
}
