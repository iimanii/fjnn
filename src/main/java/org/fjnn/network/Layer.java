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
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.network.NeuralNetwork.NeuralNetworkFeedForwardResult;
import org.fjnn.network.NeuralNetwork.FeedForwardResultsGPU;
import org.fjnn.util.Rng;
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
    
    /* connections from this layer to next layers */
    final Map<Integer, Connection> connections;
   
    /* List of incoming connections .. for backpropagation */
    final List<Integer> reverseConnections;
    
    public Layer(int index, int neurons, Activation activation) {
        this.index = index;
        this.neurons = neurons;
        this.activation = activation;
        this.connections = new HashMap<>();
        this.reverseConnections = new ArrayList<>();
    }

    Layer copy(boolean copyWeights, boolean createWeights) {
        Layer result = new Layer(index, neurons, activation);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            result.addConnection(e.getKey(), e.getValue().copy(copyWeights, createWeights));
        }
        
        for(Integer e : reverseConnections)
            result.addReverseConnection(e);
        
        return result;
    }

    void copyWeights(Layer layer) {
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
    
    protected void addReverseConnection(int fromLayer) {
        if (reverseConnections.contains(fromLayer))
            throw new RuntimeException("Already reverse connected to layer: " + fromLayer);
        
        reverseConnections.add(fromLayer);
    }
        
    /* returns connection to the next layer */
    public Connection getConnection() {
        return getConnection(index+1);
    }
    
    public Connection getConnection(int toLayer) {
        return connections.get(toLayer);
    }
    
//    public Connection getReverseConnection(int fromLayer) {
//        return reverseConnections.get(fromLayer);
//    }
    
    public Map<Integer, Connection> getConnections() {
        return new HashMap<>(connections);
    }
    
//    public List<Connection> getReverseConnections() {
//        return new ArrayList<>(reverseConnections.values());
//    }
    
    public void initUniform(float min, float max) {
        for(Connection c : connections.values())
            c.initUniform(min, max);
    }
    
    public void initGaussian(float mean, float sd) {
        for(Connection c : connections.values())
            c.initGaussian(mean, sd);
    }

    public void xavier(int scalar) {
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
    
    protected void feedForward(float[] input, int count, float[][] result) {
        if(activation != null)
            activation.compute(input, neurons, count);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int toLayer = e.getKey();
            Connection c = e.getValue();
            
            if(result[toLayer] == null)
                result[toLayer] = new float[c.links * count];
                
            c.feedForward(input, count, result[toLayer]);
        }
    }
    
    protected void feedForward(NeuralNetworkFeedForwardResult activations) {
        /* make a copy to perform activation on the array while keeping the original preActivation */
        activations.postActivations[this.index] = util.copyArray(activations.preActivations[index]);
        
        int count = activations.count;
        
        if (activation != null)
            activation.compute(activations.postActivations[this.index], neurons, count);

        for (Map.Entry<Integer, Connection> e : connections.entrySet()) {
            int toLayer = e.getKey();
            Connection connection = e.getValue();

            // Initialize space for the next layer's pre-activations if it hasn't been set yet
            if (activations.preActivations[toLayer] == null)
                activations.preActivations[toLayer] = new float[connection.links * count];

            // Perform feedForward to the next layer's pre-activations
            connection.feedForward(activations.postActivations[this.index], count, activations.preActivations[toLayer]);
        }
    }

    protected void feedForward(FloatBuffer input, int count, FloatBuffer[] result) {
        if(activation != null)
            activation.compute(input, neurons, count);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int layer = e.getKey();
            Connection c = e.getValue();
            
            if(result[layer] == null)
                result[layer] = ByteBuffer.allocateDirect(c.links * count * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
            
            c.feedForward(input, count, result[layer]);
        }
    }
    
    public void backpropagate(NeuralNetworkFeedForwardResult activations, float[][] preActivationDeltas, Map<Integer, ConnectionGradient> connectionGradients) {
        if(preActivationDeltas[this.index] != null)
            throw new RuntimeException("issue with preActivation");
        
        // Step 1: Initialize the activation deltas for the current layer
        float[] currentActivationDeltas = new float[this.neurons * activations.count];

        // Step 2: Loop through each connection to the next layers
        for (Map.Entry<Integer, Connection> entry : connections.entrySet()) {
            int nextLayerIndex = entry.getKey();
            Connection connection = entry.getValue();
            float[] nextPreActivationDeltas = preActivationDeltas[nextLayerIndex];  // delta^{(l+1)}


            // Step 3: Compute Weight and Bias Gradients and accumulate pre-activation deltas for the current layer
            ConnectionGradient gradient = connection.backpropagate(currentActivationDeltas, nextPreActivationDeltas, activations.postActivations[this.index], activations.count);
            connectionGradients.put(nextLayerIndex, gradient);
        }

        // Step 7: Convert Post-Activation Delta to Pre-Activation Delta for the current layer (l)
        // Equation: delta^{(l)}_j = Delta a^{(l)}_j * sigma'(z^{(l)}_j)
        if(activation != null) {
            for (int j = 0; j < currentActivationDeltas.length; j++) {
                currentActivationDeltas[j] *= activation.derivative(activations.preActivations[this.index][j]);
            }
        }
        
        preActivationDeltas[this.index] = currentActivationDeltas;
    }

    public void updateWeights(Map<Integer, ConnectionGradient> connectionGradients, float learningRate) {
        // Loop through each connection of the current layer
        for (Map.Entry<Integer, Connection> entry : connections.entrySet()) {
            int nextLayerIndex = entry.getKey();
            Connection connection = entry.getValue();

            // Retrieve the gradient for this connection from the map
            ConnectionGradient gradient = connectionGradients.get(nextLayerIndex);
            if (gradient == null)
                throw new RuntimeException("no gradient for connection " + index + " -> " + nextLayerIndex);

            // Update weights
            for (int i = 0; i < connection.weights.length; i++) {
                connection.weights[i] -= learningRate * gradient.weightGradients[i];
            }

            // Update biases
            for (int i = 0; i < connection.biases.length; i++) {
                connection.biases[i] -= learningRate * gradient.biasGradients[i];
            }
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
    

    void ensureCPU(CUstream stream) {
        for(Connection c : connections.values())
            c.ensureCPU(stream);
    }
    
    void updateWeightsFromGPU(CUstream stream) {
        for(Connection c : connections.values())
            c.updateWeightsFromGPU(stream);
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
    
    protected int getWeightsCount() {
        int count = 0;
        
        for(Connection c : connections.values()) {
            count += c.getWeights().length;
            if(!c.disableBias)
                count += c.getBias().length;
        }
        
        return count;
    }
    
    HashMap serialize() {
        HashMap result = new HashMap();
        result.put("index", index);
        result.put("neurons", neurons);
        result.put("activation", activation == null ? null : activation.toName());
        
        List<HashMap> array = new ArrayList<>();
        result.put("connections", array);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            HashMap connection = e.getValue().serialize();
            connection.put("toLayer", e.getKey());
            array.add(connection);
        }
        
        List<Integer> reverse = new ArrayList<>(reverseConnections);
        result.put("reverseConnections", reverse);
        
        return result;
    }
    
    static Layer deserialize(HashMap object) {
        int index = (Integer)object.get("index");
        int neurons = (Integer)object.get("neurons");
        Activation activation = Activation.fromName((String)object.get("activation"));
        
        Layer layer = new Layer(index, neurons, activation);
        
        List<HashMap> connections = (List)object.get("connections");
        
        for(int i=0; i < connections.size(); i++) {
            HashMap connection = connections.get(i);
            int toLayer = (Integer)connection.get("toLayer");
            layer.addConnection(toLayer, Connection.deserialize(connection));
        }
        
        List<Integer> reverse = (List) object.get("reverseConnections");

        for(int i : reverse) {
            layer.addReverseConnection(i);
        }
        
        return layer;
    }
    
    /**
     * deprecated stuff
     * @return 
     */
    
    public boolean gpuReady() {
        boolean gpuReady = true;
        for(Connection c : connections.values())
            gpuReady &= c.gpuReady;
        
        return gpuReady;
    }
    
    protected void prepareGPU(CUstream stream) {
        for(Connection c : connections.values())
            c.prepareGPU(stream);
    }
    
    protected long getMemoryRequirements() {
        long total = 0;
        
        for(Connection c : connections.values()) {
            total += c.getMemoryRequirements();
        }
        
        return total;
    }
    
    protected void feedForwardGPU(CUdeviceptr input, int count, CUdeviceptr[] result, CUstream stream, cublasHandle handle) {
        if(activation != null)
            activation.computeGPU(input, neurons, count, stream);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int toLayer = e.getKey();
            Connection c = e.getValue();
            
            if(result[toLayer] == null) {
               if(stream == null) {
                    result[toLayer] = CudaUtil.createFloat(c.links * count);
                    JCudaDriver.cuMemsetD32(result[toLayer], 0, c.links * count);                   
               } else {
                    result[toLayer] = CudaUtil.createFloatAsync(c.links * count, stream);
                    JCudaDriver.cuMemsetD32Async(result[toLayer], 0, c.links * count, stream);
               }
            }
            
            c.feedForwardGPU(input, count, result[toLayer], stream, handle);
        }
    }
    
    protected void feedForwardGPU(FeedForwardResultsGPU activations, CUstream stream, cublasHandle handle) {
        int count = activations.count;
        
        if (activation != null) {
            /* make a copy to perform activation on the array while keeping the original preActivation */
            activations.postActivations[this.index] = CudaUtil.copyFloatAsync(activations.preActivations[this.index], neurons * count, stream);
            
            /* compute activation values */
            activation.computeGPU(activations.postActivations[this.index], neurons, count, stream);
        } else {
            activations.postActivations[this.index] = activations.preActivations[this.index];
        }

        for (Map.Entry<Integer, Connection> e : connections.entrySet()) {
            int toLayer = e.getKey();
            Connection connection = e.getValue();

            // Initialize space for the next layer's pre-activations if it hasn't been set yet
            long size = connection.links * count;
            
            if (activations.preActivations[toLayer] == null) {
                activations.preActivations[toLayer] = CudaUtil.createFloatAsync(size, stream);
                JCudaDriver.cuMemsetD32Async(activations.preActivations[toLayer], 0, size, stream);
            }

            // Perform feedForward to the next layer's pre-activations
            connection.feedForwardGPU(activations.postActivations[this.index], count, activations.preActivations[toLayer], stream, handle);
        }
    }

    protected void backpropagateGPU(FeedForwardResultsGPU activations, CUdeviceptr[] preActivationDeltas, Map<Integer, ConnectionGradientGPU> connectionGradients, CUstream stream, cublasHandle handle) {
        if(preActivationDeltas[this.index] != null)
            throw new RuntimeException("issue with preActivation");
        
        long size = (long)this.neurons * activations.count;
        
        // Step 1: Initialize the activation deltas for the current layer only if it has incomming connections
        CUdeviceptr currentActivationDeltas = null;
        
        if(!reverseConnections.isEmpty()) {
            currentActivationDeltas = CudaUtil.createFloatAsync(size, stream);
//            JCudaDriver.cuMemsetD32Async(currentActivationDeltas, 0, size, stream);
        }

        // initially set to false to initalize the array with deltas
        // so we avoid using memset
        boolean accumulateDeltas = false;
        
        // Step 2: Loop through each connection to the next layers        
        for (Map.Entry<Integer, Connection> entry : connections.entrySet()) {
            int nextLayerIndex = entry.getKey();
            Connection connection = entry.getValue();
            CUdeviceptr nextPreActivationDeltas = preActivationDeltas[nextLayerIndex];  // delta^{(l+1)}

            // Step 3: Compute Weight and Bias Gradients and accumulate pre-activation deltas for the current layer
            ConnectionGradientGPU gradient = connection.backpropagateGPU(currentActivationDeltas, 
                                                                         nextPreActivationDeltas, 
                                                                         activations.postActivations[this.index], 
                                                                         activations.count,
                                                                         accumulateDeltas,
                                                                         stream,
                                                                         handle);
            connectionGradients.put(nextLayerIndex, gradient);
            
            /* start accumulating after first iteration */
            accumulateDeltas = true;
        }

        // Step 7: Convert Post-Activation Delta to Pre-Activation Delta for the current layer (l)
        // Equation: delta^{(l)}_j = Delta a^{(l)}_j * sigma'(z^{(l)}_j)
        if(activation != null) {
            CUdeviceptr derivative = CudaUtil.copyFloatAsync(activations.preActivations[this.index], size, stream);
            activation.derivativeGPU(derivative, this.neurons, activations.count, stream);
            
            CudaFunctions.vector.multiply(currentActivationDeltas, derivative, currentActivationDeltas, size, stream);
            CudaUtil.freeAsync(derivative, stream);
            
            /* can use preActivation[this.index] right away .. not used anywhere else */
//            activation.derivativeGPU(activations.preActivations[this.index], this.neurons, activations.count, stream);
//            CudaFunctions.vector.multiply(currentActivationDeltas, activations.preActivations[this.index], currentActivationDeltas, size, stream);
        }
        
        preActivationDeltas[this.index] = currentActivationDeltas;

    }

    public void updateWeightsGPU(Map<Integer, ConnectionGradientGPU> connectionGradients, float learningRate, CUstream stream, cublasHandle handle) {
        // Loop through each connection of the current layer
        for (Map.Entry<Integer, Connection> entry : connections.entrySet()) {
            int nextLayerIndex = entry.getKey();
            Connection connection = entry.getValue();

            // Retrieve the gradient for this connection from the map
            ConnectionGradientGPU gradient = connectionGradients.get(nextLayerIndex);
            
            if (gradient == null)
                throw new RuntimeException("no gradient for connection " + index + " -> " + nextLayerIndex);
            
            connection.updateWeightsGPU(gradient, learningRate, stream, handle);
        }
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

    protected void freeGPU() {
        for(Connection c : connections.values())
            c.freeGPU();
    }
    
    protected void freeGPU(CUstream stream) {
        for(Connection c : connections.values())
            c.freeGPU(stream);
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

    public boolean hasBias() {
        return getConnection().hasBias();
    }

//    void sumAbsWeightsGPU(CUdeviceptr ptr, CUstream stream) {
//        result[toLayer] = CudaEngine.getMempoolFloat(connections.size());
//        JCudaDriver.cuMemsetD32Async(result[toLayer], 0, c.links * count, stream);   
//        = CudaUtil.fromGPUFloat(layerResults[i], layers[i].getConnections().size(), stream);
//    }

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
            if(Rng.nextDouble() < mutation)
                m = (float) Rng.nextDoubleGaussian(minA, maxA);
            r.forcePick_A++;
            w = wa;
        } else if(rangeB < rangeA/2) {
            if(Rng.nextDouble() < mutation)
                m = (float) Rng.nextDoubleGaussian(minB, maxB);
            r.forcePick_B++;
            w = wb;
        } else if(Rng.nextBoolean()) {
            if(Rng.nextDouble() < mutation)
                m = (float) Rng.nextDoubleGaussian(minA, maxA);
            r.randomPick_A++;
            w = wa;
        } else {
            if(Rng.nextDouble() < mutation)
                m = (float) Rng.nextDoubleGaussian(minB, maxB);
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
}
