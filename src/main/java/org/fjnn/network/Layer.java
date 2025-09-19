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

import org.fjnn.network.gradient.ConnectionGradient;
import org.fjnn.network.gradient.ConnectionGradientGPU;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.cublasHandle;
import jcuda.jcurand.curandGenerator;
import org.fjnn.activation.Activation;
import org.fjnn.activation.output.ActivationForwardOutput;
import org.fjnn.activation.output.ActivationForwardOutputGPU;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.base.output.FeedForwardOutput;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.network.outputs.NeuralNetworkForwardOutput;
import org.fjnn.network.outputs.NeuralNetworkForwardOutputGPU;
import org.fjnn.normalizer.Dropout;
import org.fjnn.normalizer.Normalizer;
import org.fjnn.normalizer.outputs.DropoutForwardOutput;
import org.fjnn.normalizer.outputs.DropoutForwardOutputGPU;
import org.fjnn.util.rng;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 * 
 */
public class Layer {
    /* index of the layer in the network */
    public final int index;
    
    /* number of neurons for this layer */
    public final int neurons;
    
    public final Activation activation;
    public final Normalizer normalizer;    
    public final Dropout dropout;
    
    /* connections from this layer to next layers */
    final Map<Integer, Connection> connections;
    
    public Layer(int index, int neurons, Activation activation, Normalizer normalizer, float dropout) {
        this.index = index;
        this.neurons = neurons;
        this.activation = activation;
        this.normalizer = normalizer;
        this.dropout = new Dropout(neurons, dropout);

        this.connections = new HashMap<>();
    }
    
    protected Layer copy(boolean copyWeights, boolean createWeights) {
        Layer result = new Layer(index, neurons, activation, normalizer == null ? null : (Normalizer)normalizer.copy(), dropout.rate);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            result.addConnection(e.getKey(), e.getValue().copy(copyWeights, createWeights));
        }
        
//        for(Integer e : reverseConnections)
//            result.addReverseConnection(e);
        
        return result;
    }

    protected void copyWeights(Layer layer) {
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            e.getValue().copyWeights(layer.getConnection(e.getKey()));
        }
    }
    
    protected void addConnection(int toLayer, int links) {
        if(toLayer == index)
            throw new RuntimeException("Adding connection to the same layer");
        
        if(connections.containsKey(toLayer))
            throw new RuntimeException("Already connected to layer: " + toLayer);
        
        connections.put(toLayer, new Connection(neurons, links));
    }
    
    protected void addConnection(int toLayer, Connection connection) {
        if(connections.containsKey(toLayer))
            throw new RuntimeException("Already connected to layer: " + toLayer);
        
        connections.put(toLayer, connection);
    }
    
    /* returns connection to the next layer */
    public Connection getConnection() {
        return getConnection(index+1);
    }
    
    public Connection getConnection(int toLayer) {
        return connections.get(toLayer);
    }
    
    public Map<Integer, Connection> getConnections() {
        return new HashMap<>(connections);
    }
    
    public void initUniform(float min, float max) {
        for(Connection c : connections.values())
            c.initUniform(min, max);
    }
    
    public void initGaussian(float mean, float sd) {
        for(Connection c : connections.values())
            c.initGaussian(mean, sd);
    }

    public void xavier(float scalar) {
        for(Connection c : connections.values()) {
            float variance = 2.0f / (c.neurons + c.links);
            float sd = (float) Math.sqrt(variance) * scalar;
            c.initGaussian(0, sd);
        }
    }

    public void kaiming(int scalar) {
        for(Connection c : connections.values()) {
            float variance = 2.0f / c.neurons;
            float sd = (float) Math.sqrt(variance) * scalar;
            c.initGaussian(0, sd);
        }
    }
    
    protected void feedForward(float[] input, int batchSize, float[][] result) {
        if (normalizer != null)
            normalizer.compute(input, batchSize);

        if(activation != null)
            activation.compute(input, input, neurons, batchSize);
        
        if(dropout.rate != 0)
            dropout.feedForward(input, batchSize);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int toLayer = e.getKey();
            Connection c = e.getValue();
            
            if(result[toLayer] == null)
                result[toLayer] = new float[c.links * batchSize];
                
            c.feedForward(input, batchSize, result[toLayer]);
        }
    }
    
    protected void feedForward(NeuralNetworkForwardOutput activations, boolean disableDropout) {
        int batchSize = activations.batchSize;
        
        float[] processedInput = activations.layerInputs[this.index];
        
        if (normalizer != null) {
            FeedForwardOutput normOutput = normalizer.feedForward(processedInput, batchSize);
            activations.normalizerOutputs[this.index] = normOutput;
            processedInput = normOutput.output();
        }
        
        if (activation != null) {
            ActivationForwardOutput activationOutput = activation.feedForward(processedInput, neurons, batchSize);
            activations.activationOutputs[this.index] = activationOutput;
            processedInput = activationOutput.output();
        }
        
        if (dropout.rate != 0 && !disableDropout) {
            DropoutForwardOutput dropoutOutput = dropout.feedForward(processedInput, batchSize);
            activations.dropoutOutputs[this.index] = dropoutOutput;
            processedInput = dropoutOutput.output();
        }

        activations.layerOutputs[this.index] = processedInput;
        
        for (Map.Entry<Integer, Connection> e : connections.entrySet()) {
            int toLayer = e.getKey();
            Connection connection = e.getValue();

            // Initialize space for the next layer's pre-activations if it hasn't been set yet
            if (activations.layerInputs[toLayer] == null)
                activations.layerInputs[toLayer] = new float[connection.links * batchSize];

            // Perform feedForward to the next layer's pre-activations
            connection.feedForward(processedInput, batchSize, activations.layerInputs[toLayer]);
        }
    }

    
    public void backpropagate(NeuralNetworkForwardOutput activations, 
                              float[][] preActivationDeltas, 
                              Map<Integer, ConnectionGradient> connectionGradients,
                              BackpropagateOutput[] normalizerGradients) {
        if(preActivationDeltas[this.index] != null)
            throw new RuntimeException("issue with preActivation");
        
        // Step 1: Initialize the activation deltas for the current layer
        float[] currentActivationDeltas = new float[this.neurons * activations.batchSize];

        // Step 2: Loop through each connection to the next layers
        for (Map.Entry<Integer, Connection> entry : connections.entrySet()) {
            int nextLayerIndex = entry.getKey();
            Connection connection = entry.getValue();
            float[] nextPreActivationDeltas = preActivationDeltas[nextLayerIndex];  // delta^{(l+1)}

            // Step 3: Compute Weight and Bias Gradients and accumulate pre-activation deltas for the current layer
            ConnectionGradient gradient = connection.backpropagate(currentActivationDeltas, nextPreActivationDeltas, activations.layerOutputs[this.index], activations.batchSize);
            connectionGradients.put(nextLayerIndex, gradient);
        }
        
        // Dropout backprop
        if(activations.dropoutOutputs[this.index] != null) {
            BackpropagateOutput dropoutOutput = dropout.backpropagate(activations.dropoutOutputs[this.index], currentActivationDeltas);
            currentActivationDeltas = dropoutOutput.deltaLoss();
        }

        // Step 7: Convert Post-Activation Delta to Pre-Activation Delta for the current layer (l)
        // Equation: delta^{(l)}_j = Delta a^{(l)}_j * sigma'(z^{(l)}_j)
        if(activation != null) {
            activation.gradient(activations.activationOutputs[this.index].preActivation, 
                                activations.activationOutputs[this.index].postActivation, 
                                currentActivationDeltas, neurons, activations.batchSize);
        }
        
        if(normalizer != null) {
            BackpropagateOutput normOutput = normalizer.backpropagate(activations.normalizerOutputs[this.index], currentActivationDeltas, neurons, activations.batchSize);
            currentActivationDeltas = normOutput.deltaLoss();            
            normalizerGradients[this.index] = normOutput;
        }
        
        preActivationDeltas[this.index] = currentActivationDeltas;
    }

    public void updateWeights(Map<Integer, ConnectionGradient> connectionGradients, 
                              BackpropagateOutput normalizerGradients, 
                              float learningRate, 
                              float normalizerLearningRate,
                              float weightDecay) {
        if(normalizer != null)
            normalizer.applyGradients(normalizerGradients, normalizerLearningRate, weightDecay);
        
        // Loop through each connection of the current layer
        for (Map.Entry<Integer, Connection> entry : connections.entrySet()) {
            int nextLayerIndex = entry.getKey();
            Connection connection = entry.getValue();

            // Retrieve the gradient for this connection from the map
            ConnectionGradient gradient = connectionGradients.get(nextLayerIndex);
            if (gradient == null)
                throw new RuntimeException("no gradient for connection " + index + " -> " + nextLayerIndex);

            connection.updateWeights(gradient, learningRate, weightDecay);
        }
    }

    public boolean nativeReady() {
        boolean nativeReady = true;
        for(Connection c : connections.values())
            nativeReady &= c.nativeReady;
        
        return nativeReady;
    }
    
    protected void prepareCPU() {
        for(Connection c : connections.values())
            c.prepareCPU();
    }
    
    void updateWeightsFromGPU(CUstream stream) {
        for(Connection c : connections.values())
            c.syncWeightsFromGPU(stream);
        
        if(normalizer != null)
            normalizer.updateWeightsFromGPU();
    }
    
    protected void freeCPU() {
        for(Connection c : connections.values())
            c.freeCPU();
        
        connections.clear();
    }
    
    protected void crossOverMutate(Layer a, Layer b, float min, float max, double mutation) {
        /* crossover mutate all connections */
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int layer = e.getKey();
            Connection c = e.getValue();
            
            c.crossOverMutate(a.getConnection(layer), b.getConnection(layer), min, max, mutation);
        }
    }
    
    /**
    * Memory required for layer parameters (connections + normalizer)
     * @return 
    */
    protected long getParameterCount() {
        long parameters = 0;

        // Connection parameters (weights, biases, optimizer state)
        for(Connection c : connections.values())
            parameters += c.getParameterCount();

        // Normalizer parameters (gamma, beta)
        if(normalizer != null)
            parameters += normalizer.getParameterCount();

        // Dropout has no parameters

        return parameters;
    }
    
    /**
    * Memory required during forward pass
     * @param batchSize
     * @return 
    */
    protected long getForwardMemoryRequired(int batchSize) {
        long memory = 0;

        // Layer inputs for NEXT layers - allocated by THIS layer in feedForwardGPU()
        for(Connection c : connections.values()) {
            memory += c.links * (batchSize * CudaUtil.FLOAT_SIZE);
        }

        // Component allocations (each component allocates its own memory)
        if(activation != null)
            // ActivationForwardOutputGPU allocates postActivation
            memory += neurons * batchSize * CudaUtil.FLOAT_SIZE;

        if(normalizer != null)
            memory += normalizer.getForwardMemoryRequired(batchSize);

        if(dropout.rate != 0)
            memory += dropout.getForwardMemoryRequired(batchSize);

        return memory;
    }

    /**
     * Memory required during backward pass
     * @param batchSize
     * @return 
     */
    protected long getBackwardMemoryRequired(int batchSize) {
        long memory = 0;

        // Activation deltas for THIS layer - allocated in backpropagateGPU()
        // Only allocated for non-input layers (input layer doesn't call backpropagateGPU)
        if(index > 0) {
            memory += neurons * batchSize * CudaUtil.FLOAT_SIZE;
        }

        // Connection gradients - allocated by Connection.backpropagateGPU()
        for(Connection c : connections.values()) {
            memory += c.getBackwardMemoryRequired(batchSize);
        }

        // Normalizer gradients - allocated by normalizer.backpropagateGPU()
        if(normalizer != null) {
            memory += normalizer.getBackwardMemoryRequired(batchSize);
        }

        // Dropout has no backward memory requirements

        return memory;
    }
    
    HashMap serialize() {
        HashMap result = new HashMap();
        result.put("index", index);
        result.put("neurons", neurons);
        result.put("activation", activation == null ? null : activation.serialize());
        result.put("normalizer", normalizer == null ? null : normalizer.serialize());
        result.put("dropout", dropout.rate);
        
        List<HashMap> array = new ArrayList<>();
        result.put("connections", array);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            HashMap connection = e.getValue().serialize();
            connection.put("toLayer", e.getKey());
            array.add(connection);
        }
        
//        List<Integer> reverse = new ArrayList<>(reverseConnections);
//        result.put("reverseConnections", reverse);
        
        return result;
    }
    
    static Layer deserialize(HashMap serialized) {
        int index = (Integer)serialized.get("index");
        int neurons = (Integer)serialized.get("neurons");
        
        Map activationMap = (Map)serialized.get("activation");
        Activation activation = activationMap == null ? null : Activation.deserialize(activationMap);
        
        Map normalizerMap = (Map)serialized.get("normalizer");
        Normalizer normalizer = normalizerMap == null ? null : Normalizer.deserialize(normalizerMap);
        
        float dropout = (Float)serialized.get("dropout");
        
        Layer layer = new Layer(index, neurons, activation, normalizer, dropout);
        
        List<HashMap> connections = (List)serialized.get("connections");
        
        for(int i=0; i < connections.size(); i++) {
            HashMap connection = connections.get(i);
            int toLayer = (Integer)connection.get("toLayer");
            layer.addConnection(toLayer, Connection.deserialize(connection));
        }
        
//        List<Integer> reverse = (List) object.get("reverseConnections");

//        for(int i : reverse) {
//            layer.addReverseConnection(i);
//        }
        
        return layer;
    }
    
    public boolean gpuReady() {
        boolean gpuReady = true;
        for(Connection c : connections.values())
            gpuReady &= c.gpuReady;
        
        if(normalizer != null)
            gpuReady &= normalizer.gpuReady();
        
        return gpuReady;
    }
    
    protected void prepareGPU(CUstream stream) {
        for(Connection c : connections.values())
            c.prepareGPU(stream);
        
        if(normalizer != null)
            normalizer.prepareGPU(stream);
        
        dropout.prepareGPU(stream);
    }
//    
//    protected long getMemoryRequirements() {
//        long total = 0;
//        
//        for(Connection c : connections.values()) {
//            total += c.getMemoryRequirements();
//        }
//        
//        return total;
//    }
    
    protected void computeGPU(CUdeviceptr input, int batchSize, CUdeviceptr[] result, CUstream stream, cublasHandle handle) {
        if(normalizer != null)
            normalizer.computeGPU(input, batchSize, stream);
        
        if(activation != null)
            activation.computeGPU(input, input, neurons, batchSize, stream);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int toLayer = e.getKey();
            Connection c = e.getValue();
            
            if(result[toLayer] == null) {
               if(stream == null) {
                    result[toLayer] = CudaUtil.createFloat(c.links * batchSize);
                    JCudaDriver.cuMemsetD32(result[toLayer], 0, c.links * batchSize);                   
               } else {
                    result[toLayer] = CudaUtil.createFloatAsync(c.links * batchSize, stream);
                    JCudaDriver.cuMemsetD32Async(result[toLayer], 0, c.links * batchSize, stream);
               }
            }
            
            c.feedForwardGPU(input, batchSize, result[toLayer], stream, handle);
        }
    }
    
    protected void feedForwardGPU(NeuralNetworkForwardOutputGPU activations, boolean disableDropout, CUstream stream, cublasHandle handle) {
        int batchSize = activations.batchSize;
        
        CUdeviceptr processedInput = activations.layerInputs[this.index];
        
        if (normalizer != null) {
            FeedForwardOutputGPU normOutput = normalizer.feedForwardGPU(processedInput, batchSize, stream);
            activations.normalizerOutputs[this.index] = normOutput;
            processedInput = normOutput.output();
        }
        
        if (activation != null) {
            ActivationForwardOutputGPU activationOutput = activation.feedForwardGPU(processedInput, neurons, batchSize, stream);
            activations.activationOutputs[this.index] = activationOutput;
            processedInput = activationOutput.output();
        }
        
        if (dropout.rate != 0 && !disableDropout) {
            DropoutForwardOutputGPU dropoutOutput = dropout.feedForwardGPU(processedInput, batchSize, stream);
            activations.dropoutOutputs[this.index] = dropoutOutput;
            processedInput = dropoutOutput.output();
        }
        
        activations.layerOutputs[this.index] = processedInput;

        for (Map.Entry<Integer, Connection> e : connections.entrySet()) {
            int toLayer = e.getKey();
            Connection connection = e.getValue();

            // Initialize space for the next layer's pre-activations if it hasn't been set yet
            long size = connection.links * batchSize;
            
            if (activations.layerInputs[toLayer] == null) {
                activations.layerInputs[toLayer] = CudaUtil.createFloatAsync(size, stream);
                JCudaDriver.cuMemsetD32Async(activations.layerInputs[toLayer], 0, size, stream);
            }

            // Perform feedForward to the next layer's pre-activations
            connection.feedForwardGPU(processedInput, batchSize, activations.layerInputs[toLayer], stream, handle);
        }
    }

    protected void backpropagateGPU(NeuralNetworkForwardOutputGPU activations, 
                                    CUdeviceptr[] preActivationDeltas, 
                                    Map<Integer, ConnectionGradientGPU> connectionGradients,
                                    BackpropagateOutputGPU[] normalizerGradients,
                                    CUstream stream, cublasHandle handle) {
        if(preActivationDeltas[this.index] != null)
            throw new RuntimeException("issue with preActivation");
        
        long size = (long)this.neurons * activations.batchSize;
        
        // Step 1: Initialize the activation deltas for the current layer only if it has incoming connections
        CUdeviceptr currentActivationDeltas = CudaUtil.createFloatAsync(size, stream);
//            JCudaDriver.cuMemsetD32Async(currentActivationDeltas, 0, size, stream);
//        }

        // initially set to false to initalize the array with deltas
        // so we avoid using memset
        boolean accumulateDeltas = false;
        
        // Step 2: Loop through each connection to the next layers        
        for (Map.Entry<Integer, Connection> entry : connections.entrySet()) {
            int nextLayerIndex = entry.getKey();
            Connection connection = entry.getValue();
            CUdeviceptr nextPreActivationDeltas = preActivationDeltas[nextLayerIndex];

            // Step 3: Compute Weight and Bias Gradients and accumulate pre-activation deltas for the current layer
            ConnectionGradientGPU gradient = connection.backpropagateGPU(currentActivationDeltas, 
                                                                         nextPreActivationDeltas, 
                                                                         activations.layerOutputs[this.index], 
                                                                         activations.batchSize,
                                                                         accumulateDeltas,
                                                                         stream,
                                                                         handle);
            connectionGradients.put(nextLayerIndex, gradient);
            
            /* start accumulating after first iteration */
            accumulateDeltas = true;
        }

        // Dropout backprop
        if(activations.dropoutOutputs[this.index] != null) {
            BackpropagateOutputGPU dropoutOutput = dropout.backpropagateGPU(activations.dropoutOutputs[this.index], currentActivationDeltas, stream);
            currentActivationDeltas = dropoutOutput.deltaLoss();
        }
        
        // Step 7: Convert Post-Activation Delta to Pre-Activation Delta for the current layer (l)
        // Equation: delta[l] = delta[l+1] * activation_derivative(...)
        if(activation != null) {
            activation.gradientGPU(activations.activationOutputs[this.index].preActivation, 
                                   activations.activationOutputs[this.index].postActivation, 
                                   currentActivationDeltas, neurons, activations.batchSize, stream);
        }
        
        if(normalizer != null) {
            BackpropagateOutputGPU normOutput = normalizer.backpropagateGPU(activations.normalizerOutputs[this.index], currentActivationDeltas, neurons, activations.batchSize, stream);
            normalizerGradients[this.index] = normOutput;
        }
        
        preActivationDeltas[this.index] = currentActivationDeltas;
    }

    public void updateWeightsGPU(Map<Integer, ConnectionGradientGPU> connectionGradients, 
                                 BackpropagateOutputGPU normalizerGradients, 
                                 float learningRate, float normalizerLearningRate, float weightDecay,
                                 CUstream stream, cublasHandle handle) {
        if(normalizer != null)
            normalizer.applyGradientsGPU(normalizerGradients, normalizerLearningRate, weightDecay, stream);
        
        // Loop through each connection of the current layer
        for (Map.Entry<Integer, Connection> entry : connections.entrySet()) {
            int nextLayerIndex = entry.getKey();
            Connection connection = entry.getValue();

            // Retrieve the gradient for this connection from the map
            ConnectionGradientGPU gradient = connectionGradients.get(nextLayerIndex);
            
            if (gradient == null)
                throw new RuntimeException("no gradient for connection " + index + " -> " + nextLayerIndex);
            
            connection.updateWeightsGPU(gradient, learningRate, weightDecay, stream, handle);
        }
    }

    protected void freeGPU(CUstream stream) {
        for(Connection c : connections.values())
            c.freeGPU(stream);
        
        if(normalizer != null)
            normalizer.freeGPU(stream);
        
        dropout.freeGPU(stream);
    }

    protected void freeGPURng() {
        for(Connection c : connections.values())
            c.freeGPURng();
    }
    
    protected void crossOverMutate(Layer a, Layer b, Layer min, Layer max, double mutation) {
        throw new RuntimeException("unsupported");
//        float[] wa = a.getWeights();
//        float[] wb = b.getWeights();
//        float[] wmin = min.getWeights();
//        float[] wmax = max.getWeights();
//
//        for(int j=0; j < wa.length; j++) {
//            float w = Rng.nextBoolean() ? wa[j] : wb[j];
//
//            if(Rng.nextDouble() < mutation)
//                w = w + (float) Rng.nextDoubleGaussian(wmin[j], wmax[j]);
//
//            weights[j] = w;
//        }
//        
//        if(!hasBias)
//            return;
//        
//        float[] ba = a.getBias();
//        float[] bb = b.getBias();
//        float[] bmin = min.getBias();
//        float[] bmax = max.getBias();
//
//        for(int j=0; j < ba.length; j++) {
//            float w = Rng.nextBoolean() ? ba[j] : bb[j];
//
//            if(Rng.nextDouble() < mutation)
//                w = w + (float) Rng.nextDoubleGaussian(bmin[j], bmax[j]);
//
//            bias[j] = w;
//        }
//        
//        gpuReady = false;
    }

    public static class crossOverMutateResult {
        public int forcePick_A;
        public int forcePick_B;
        public int randomPick_A;
        public int randomPick_B;
    }
    
    protected void crossOverMutate(Layer a, Layer b, Layer minA, Layer maxA, Layer minB, Layer maxB, double mutation, crossOverMutateResult r) {
        throw new RuntimeException("unsupported");
//        float[] wa = a.getWeights();
//        float[] wb = b.getWeights();
//        float[] wminA = minA.getWeights();
//        float[] wmaxA = maxA.getWeights();
//        float[] wminB = minB.getWeights();
//        float[] wmaxB = maxB.getWeights();
//
//        for(int j=0; j < wa.length; j++)
//            weights[j] = crossOverMutateCalculateWeight(wa[j], wb[j], wminA[j], wmaxA[j], wminB[j], wmaxB[j], mutation, r);
//        
//        if(!hasBias)
//            return;
//        
//        float[] ba = a.getBias();
//        float[] bb = b.getBias();
//        float[] bminA = minA.getBias();
//        float[] bmaxA = maxA.getBias();
//        float[] bminB = minB.getBias();
//        float[] bmaxB = maxB.getBias();
//
//        for(int j=0; j < ba.length; j++)
//            bias[j] = crossOverMutateCalculateWeight(ba[j], bb[j], bminA[j], bmaxA[j], bminB[j], bmaxB[j], mutation, r);
//        
//        gpuReady = false;
    }
    
    protected void crossOverMutateGPU(Layer a, Layer b, float min, float max, double mutation, boolean nocopy, CUstream stream, curandGenerator generator) {
        /* crossover mutate all connections */
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int layer = e.getKey();
            Connection c = e.getValue();
            
            c.crossOverMutateGPU(a.getConnection(layer), b.getConnection(layer), min, max, mutation, nocopy, stream, generator);
        }
    }

    void clipWeights(float min, float max) {
        /* crossover mutate all connections */
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            Connection c = e.getValue();
            
            c.clipWeights(min, max);
        }
    }
    
    void clipWeightsGPU(float min, float max, CUstream stream) {
        /* crossover mutate all connections */
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            Connection c = e.getValue();
            
            c.clipWeightsGPU(min, max, stream);
        }
    }

    protected double compare(Layer l) {
        double score = 0;
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            score += e.getValue().compare(l.getConnection(e.getKey()));
        }
        
        return score;
    }
    
    /*
    * if any of the nodes are really sensitive to change .. pick them 
    * otherwise pick random
    */
    private float crossOverMutateCalculateWeight(float wa, float wb, float minA, float maxA, float minB, float maxB, double mutation, crossOverMutateResult r) {
        double rangeA = maxA - minA;
        double rangeB = maxB - minB;
        
        float w, m = 0;
        
        if(rangeA < rangeB/2) {
            if(rng.nextDouble() < mutation)
                m = (float) rng.nextDoubleGaussian(minA, maxA);
            r.forcePick_A++;
            w = wa;
        } else if(rangeB < rangeA/2) {
            if(rng.nextDouble() < mutation)
                m = (float) rng.nextDoubleGaussian(minB, maxB);
            r.forcePick_B++;
            w = wb;
        } else if(rng.nextBoolean()) {
            if(rng.nextDouble() < mutation)
                m = (float) rng.nextDoubleGaussian(minA, maxA);
            r.randomPick_A++;
            w = wa;
        } else {
            if(rng.nextDouble() < mutation)
                m = (float) rng.nextDoubleGaussian(minB, maxB);
            r.randomPick_B++;
            w = wb;
        }
        
        return w + m;
    }

    protected float compareGPU(Layer l, CUstream stream) {
        throw new RuntimeException("unsupported");
//        
//        int size = neurons * links;
//        
//        if(size == 0)
//            return 0;
//
//        float sum_weights = CudaUtil.sum_abs_differenceGPU(weightsGPU, l.weightsGPU, size, stream);
//        float sum_biases = 0;
//        
//        if(hasBias)
//           sum_biases = CudaUtil.sum_abs_differenceGPU(biasesGPU, l.biasesGPU, links, stream);
//        
//        return sum_weights + sum_biases;
    }

    synchronized void addNeurons(int amount, float weight) {
        throw new RuntimeException("unsupported");
//        neurons += amount;
//        
//        weights = Arrays.copyOf(weights, neurons * links);
//        if(condition != null)
//            condition = Arrays.copyOf(condition, neurons);
//        
//        if(weight != 0) {
//            int start = (neurons - amount) * links;
//            for(int i=start; i < weights.length; i++) {
//                weights[i] = weight;
//            }
//        }
    }

    synchronized void addLinks(int amount, float weight) {
        throw new RuntimeException("unsupported");
//        int prev = links;
//        
//        links += amount;
//        
//        float[] newWeights = new float[neurons * links];
//        if(weight != 0)
//            Arrays.fill(newWeights, weight);
//        
//        for(int i=0, j=0; j < newWeights.length; i+=prev, j+=links)
//            System.arraycopy(weights, i, newWeights, j, prev);
//        
//        weights = newWeights;
//        
//        float[] newBias = new float[links];
//        if(weight != 0)
//            Arrays.fill(newBias, weight);
//        
//        System.arraycopy(bias, 0, newBias, 0, links-amount);
//        bias = newBias;
    }
    
    synchronized void removeLinks(int amount) {
        throw new RuntimeException("unsupported");
//        int prev = links;
//        
//        links -= amount;
//        
//        float[] newWeights = new float[neurons * links];
//        
//        for(int i=0, j=0; j < newWeights.length; i+=prev, j+=links)
//            System.arraycopy(weights, i, newWeights, j, links);
//        
//        weights = newWeights;
//        bias = Arrays.copyOf(bias, links);
    }
    
    protected void feedForward(FloatBuffer input, int batchSize, FloatBuffer[] result) {
        if(activation != null)
            activation.compute(input, input, neurons, batchSize);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int layer = e.getKey();
            Connection c = e.getValue();
            
            if(result[layer] == null)
                result[layer] = ByteBuffer.allocateDirect(c.links * batchSize * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
            
            c.feedForward(input, batchSize, result[layer]);
        }
    }
}
