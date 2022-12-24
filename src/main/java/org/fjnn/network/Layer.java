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
import org.fjnn.base.BaseLayer;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaModule;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.util.Rng;
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

    /* can we call computeGPU */
    protected boolean gpuReady;
    
    public Layer(int index, int neurons, Activation activation) {
        this.index = index;
        this.neurons = neurons;
        this.activation = activation;
        this.connections = new HashMap<>();

        gpuReady = false;
    }

    Layer copy(boolean withoutWeights) {
        Layer result = new Layer(index, neurons, activation);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            result.addConnection(e.getKey(), e.getValue().copy(withoutWeights));
        }
        
        return result;
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

    public void randomize(float min, float max) {
        for(Connection c : connections.values())
            c.randomize(min, max);
    }

    protected void feedForward(float[] input, float[][] result) {
        if(activation != null)
            activation.compute(input);
        
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int layer = e.getKey();
            Connection c = e.getValue();
            
            float[] computed = c.feedForward(input);
            
            if(result[layer] == null)
                result[layer] = computed;
            else
                util.addArray(result[layer], computed);
        }
    }
    
    protected void freeCPU() {
        for(Connection c : connections.values())
            c.free();
        
        connections.clear();
    }
    
    protected void crossOverMutate(Layer a, Layer b, double min, double max, double mutation) {
        /* crossover mutate all connections */
        for(Entry<Integer, Connection> e : connections.entrySet()) {
            int layer = e.getKey();
            Connection c = e.getValue();
            
            c.crossOverMutate(a.getConnection(layer), b.getConnection(layer), min, max, mutation);
        }
        
        gpuReady = false;
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
        result.put("activation", activation.toName());
        
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
     */
    
    protected CUdeviceptr feedForwardGPU(CUdeviceptr ptr, CUstream stream, cublasHandle handle) {
        if(activation != null)
            activation.computeGPU(ptr, neurons, stream);
        
        throw new RuntimeException("unsupported");

//        /* output layer */
//        if(isOutput)
//            return ptr;
//        
//        long biasSize = links * (long) Sizeof.FLOAT;
//        CUdeviceptr resultGPU = new CUdeviceptr();
//        JCudaDriver.cuMemAlloc(resultGPU, biasSize);
//        
//        if(hasBias)
//            JCudaDriver.cuMemcpyDtoDAsync(resultGPU, biasesGPU, biasSize, stream);
//
//        /* NOTE: cublas uses column-major format */
//        int row_a = links;
//        int col_a = neurons;
//        CUdeviceptr d_A = weightsGPU;
//
//        CUdeviceptr d_B = ptr;        
//        CUdeviceptr d_C = resultGPU;
//        
//        Pointer pAlpha = Pointer.to(new float[]{1.0f});
//        Pointer pBeta = Pointer.to(new float[]{hasBias ? 1.0f : 0.0f});
//
//        /* Compute Vector Matrix Multiplication */
//        JCublas2.cublasSgemv(handle, cublasOperation.CUBLAS_OP_N,
//                            row_a, col_a,
//                            pAlpha, d_A, row_a, 
//                            d_B, 1, 
//                            pBeta, d_C, 1);
//        
//        return resultGPU;   
    }
    
    protected CUdeviceptr feedForwardGPU(CUdeviceptr ptr, CUstream stream, cublasHandle handle, int count) {
        if(activation != null)
            activation.computeGPU(ptr, neurons * count, stream);
        
        throw new RuntimeException("unsupported");
        
        /* output layer */
//        if(isOutput)
//            return ptr;
//        
//        long biasSize = links * (long) Sizeof.FLOAT;
//        CUdeviceptr resultGPU = new CUdeviceptr();
//        JCudaDriver.cuMemAlloc(resultGPU, count * biasSize);
//        
//        if(hasBias)
//            for(int i=0; i < count; i++)
//                JCudaDriver.cuMemcpyDtoDAsync(resultGPU.withByteOffset(i * biasSize), biasesGPU, biasSize, stream);
//
//        /* NOTE: cublas uses column-major format */
//        int row_a = links;
//        int col_a = neurons;
//        CUdeviceptr d_A = weightsGPU;
//
//        int row_b = neurons;
//        int col_b = count;
//        CUdeviceptr d_B = ptr;
//        
//        int row_c = links;
//        CUdeviceptr d_C = resultGPU;
//        
//        Pointer pAlpha = Pointer.to(new float[]{1.0f});
//        Pointer pBeta = Pointer.to(new float[]{hasBias ? 1.0f : 0.0f});
//
//        /* Compute Matrix Multiplication */
//        JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 
//                            row_a, col_b, col_a, 
//                            pAlpha, d_A, row_a, 
//                            d_B, row_b, 
//                            pBeta, d_C, row_c);
//        
//        return resultGPU;
    }
    
    protected void prepareGPU(CUstream stream) {
        if(gpuReady)
            return;
        
        throw new RuntimeException("unsupported");
        
//        if(!isOutput) {
//            if(weightsGPU == null)
//               weightsGPU = CudaUtil.create(weights.length);
//
//            if(biasesGPU == null && hasBias)
//               biasesGPU = CudaUtil.create(bias.length);
//
//            JCudaDriver.cuMemcpyHtoDAsync(weightsGPU, Pointer.to(weights), weights.length * (long) Sizeof.FLOAT, stream);
//
//            if(hasBias)
//                JCudaDriver.cuMemcpyHtoDAsync(biasesGPU, Pointer.to(bias), bias.length * (long) Sizeof.FLOAT, stream);
//        }
//
//        if(condition != null) {
//            if(conditionGPU == null)
//               conditionGPU = CudaUtil.createBytes(condition.length);
//
//            byte[] temp = new byte[condition.length];
//
//            for(int i=0; i < condition.length; i++)
//                temp[i] = condition[i] ? (byte)1 : 0;
//            
//            JCudaDriver.cuMemcpyHtoDAsync(conditionGPU, Pointer.to(temp), condition.length, stream);
//        }
//
//        gpuReady = true;
    }
    
    protected void freeGPU() {
        throw new RuntimeException("unsupported");
//        if(isOutput)
//            return;
//        
//        if(weightsGPU != null)
//            JCudaDriver.cuMemFree(weightsGPU);
//        
//        if(conditionGPU != null)
//            JCudaDriver.cuMemFree(conditionGPU);    
//        
//        if(biasesGPU != null)
//            JCudaDriver.cuMemFree(biasesGPU);
//        
//        weightsGPU = null;
//        conditionGPU = null;
//        gpuReady = false;
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

    protected void crossOverMutateGPU(Layer a, Layer b, double min, double max, double mutation, CUstream stream, curandGenerator generator) {
        throw new RuntimeException("unsupported");
//        
//        int deviceId = CudaEngine.getThreadDeviceId();
//        int size = neurons * links;
//        int bias = hasBias ? links : 0;
//        float mean = (float) (max + min) / 2.0f;
//        float stdDev = (float) (max - min) / 10.0f;
//        
//        if(size == 0)
//            return;
//        
//        CUdeviceptr uniform = CudaUtil.create((size + bias) * 2);
//        CUdeviceptr gaussian = CudaUtil.create((size + bias));
//
//        JCurand.curandGenerateUniform(generator, uniform, (size + bias) * 2);
//        JCurand.curandGenerateNormal(generator, gaussian, size + bias, mean, stdDev);
//            
//        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_GENETIC, "crossOverMutate", deviceId);
//
//        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(deviceId), size);
//        int gridSizeX = (size - 1) / blockSizeX + 1;
//
//        Pointer kernelParameters = Pointer.to(
//            Pointer.to(a.weightsGPU),
//            Pointer.to(b.weightsGPU),
//            Pointer.to(weightsGPU),
//            Pointer.to(new long[]{size}),
//            Pointer.to(new double[]{mutation}),
//            Pointer.to(uniform),
//            Pointer.to(uniform.withByteOffset(size * (long) Sizeof.FLOAT)),
//            Pointer.to(gaussian)
//        );        
//                
//        JCudaDriver.cuLaunchKernel(function,
//            gridSizeX, 1, 1,       // Grid dimension
//            blockSizeX, 1, 1,      // Block dimension
//            0, stream,             // Shared memory size and stream
//            kernelParameters, null // Kernel- and extra parameters
//        );
//        
//        weights = CudaUtil.fromGPU(weightsGPU, size, stream);
//        
//        if(hasBias) {
//            kernelParameters = Pointer.to(
//                Pointer.to(a.biasesGPU),
//                Pointer.to(b.biasesGPU),
//                Pointer.to(biasesGPU),
//                Pointer.to(new long[]{bias}),
//                Pointer.to(new double[]{mutation}),
//                Pointer.to(uniform.withByteOffset((size * 2) * (long) Sizeof.FLOAT)),
//                Pointer.to(uniform.withByteOffset((size * 2 + bias) * (long) Sizeof.FLOAT)),
//                Pointer.to(gaussian.withByteOffset(size * (long) Sizeof.FLOAT))
//            );
//
//            JCudaDriver.cuLaunchKernel(function,
//                gridSizeX, 1, 1,       // Grid dimension
//                blockSizeX, 1, 1,      // Block dimension
//                0, stream,             // Shared memory size and stream
//                kernelParameters, null // Kernel- and extra parameters
//            );
//
//            this.bias = CudaUtil.fromGPU(biasesGPU, bias, stream);
//        }
//        
//        JCudaDriver.cuMemFree(uniform);
//        JCudaDriver.cuMemFree(gaussian);
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
