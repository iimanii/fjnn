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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import org.fjnn.activation.Activation;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaModule;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.util.Rng;
import org.fjnn.util.intrinsic;
import org.fjnn.util.util;
import org.json.JSONArray;
import org.json.JSONObject;

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
    
    final Map<Integer, Connection> connections;
    
    public Layer(int index, int neurons, Activation activation) {
        this.index = index;
        this.neurons = neurons;
        this.activation = activation;
        this.connections = new HashMap<>();
    }

    Layer copy(boolean copyWeights, boolean createWeights) {
        Layer result = new Layer(index, neurons, activation);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            result.addConnection(e.getKey(), e.getValue().copy(copyWeights, createWeights));
        }
        
        return result;
    }

    void copyWeights(Layer layer) {
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            e.getValue().copyWeights(layer.getConnection(e.getKey()));
        }
    }
    
    public void addConnection(int toLayer, int links) {
        if(toLayer == index)
            throw new RuntimeException("Adding connection to the same layer");
        
        if(connections.containsKey(toLayer))
            throw new RuntimeException("Already connected to layer: " + toLayer);
        
        connections.put(toLayer, new Connection(neurons, links));
    }
    
    public void addConnection(int toLayer, Connection connection) {
        if(connections.containsKey(toLayer))
            throw new RuntimeException("Already connected to layer: " + toLayer);
        
        connections.put(toLayer, connection);
    }
    
    public Connection getConnection() {
        return getConnection(index+1);
    }
    
    public Connection getConnection(int toLayer) {
        return connections.get(toLayer);
    }
    
    public List<Connection> getConnections() {
        return new ArrayList<>(connections.values());
    }

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
    
    protected void feedForward(float[] input, float[][] result) {
        if(activation != null)
            activation.compute(input);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int toLayer = e.getKey();
            Connection c = e.getValue();
            
            if(result[toLayer] == null)
                result[toLayer] = new float[c.links];
                
            c.feedForward(input, result[toLayer]);
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
    
    protected void feedForward(FloatBuffer input, FloatBuffer[] result) {
        if(activation != null)
            activation.compute(input);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int layer = e.getKey();
            Connection c = e.getValue();
            
            if(result[layer] == null)
                result[layer] = ByteBuffer.allocateDirect(c.links * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
        
            c.feedForward(input, result[layer]);
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
    
    JSONObject serialize() {
        JSONObject result = new JSONObject();
        result.put("index", index);
        result.put("neurons", neurons);
        result.put("activation", activation == null ? JSONObject.NULL : activation.toName());
        
        JSONArray array = new JSONArray();
        result.put("connections", array);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            JSONObject connection = e.getValue().serialize();
            connection.put("toLayer", e.getKey());
            array.put(connection);
        }
        
        return result;
    }
    
    static Layer deserialize(JSONObject object) {
        int index = object.getInt("index");
        int neurons = object.getInt("neurons");
        Activation activation = Activation.fromName(object.optString("activation"));
        
        Layer layer = new Layer(index, neurons, activation);
        
        JSONArray connections = object.getJSONArray("connections");
        
        for(int i=0; i < connections.length(); i++) {
            JSONObject connection = connections.getJSONObject(i);
            int toLayer = connection.getInt("toLayer");
            layer.addConnection(toLayer, Connection.deserialize(connection));
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
    
    protected void feedForwardGPU(CUdeviceptr input, CUdeviceptr[] result, CUstream stream, cublasHandle handle) {
        if(activation != null)
            activation.computeGPU(input, neurons, 1, stream);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int toLayer = e.getKey();
            Connection c = e.getValue();
            
            if(result[toLayer] == null) {
               result[toLayer] = CudaEngine.getMempoolFloat(c.links);
               JCudaDriver.cuMemsetD32Async(result[toLayer], 0, c.links, stream);
            }
        
            c.feedForwardGPU(input, result[toLayer], stream, handle);
        }
    }
    
    protected void feedForwardGPU(CUdeviceptr input, int count, CUdeviceptr[] result, CUstream stream, cublasHandle handle) {
        if(activation != null)
            activation.computeGPU(input, neurons, count, stream);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int toLayer = e.getKey();
            Connection c = e.getValue();
            
            if(result[toLayer] == null) {
               result[toLayer] = CudaEngine.getMempoolFloat(c.links * count);
               JCudaDriver.cuMemsetD32Async(result[toLayer], 0, c.links * count, stream);           
            }
            
            c.feedForwardGPU(input, count, result[toLayer], stream, handle);
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

    protected void freeGPU(int deviceId) {
        for(Connection c : connections.values())
            c.freeGPU(deviceId);
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
    
    protected float compare(Layer l) {
        float score = 0;
        
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
