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
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicLong;
import javax.naming.OperationNotSupportedException;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.jcurand.curandRngType;
import jcuda.runtime.cudaStream_t;
import org.fjnn.activation.Activation;
import org.fjnn.base.Network;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.network.Layer.crossOverMutateResult;
import org.fjnn.util.Rng;
import org.fjnn.util.util;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 *
 * @author ahmed
 */
public class NeuralNetwork extends Network {
    Layer[] layers;
    
    /* index of last layer */
    int last;
    
    /* a network can only be built once */
    boolean finalized;

    /* device id for gpu to use */
    int deviceId;

    /* memory requirement for prepare gpu */
    long prepareMemReq;
    
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
        
        this.deviceId = -1;
        this.plan = new ArrayList<>();
    }
        
    NeuralNetwork(Layer[] layers) {
        super(layers[0].neurons, layers[layers.length-1].neurons, layers[layers.length-1].activation);
        
        this.layers = layers;
        this.finalized = true;
        this.deviceId = -1;
        this.last = layers.length - 1;

        computeMemoryRequirements();
    }
    
    public NeuralNetwork copy(boolean copyWeights) {
        return copy(copyWeights, true);
    }
    
    public NeuralNetwork copy(boolean copyWeights, boolean createWeights) {
        Layer[] copied = new Layer[layers.length];
        
        for(int i=0; i < layers.length; i++) {
            copied[i] = layers[i].copy(copyWeights, createWeights);
        }
        
        return new NeuralNetwork(copied);
    }
    
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
        
        int targetNeurons = layers[toLayer].neurons;
        
        layers[fromLayer].addConnection(toLayer, targetNeurons);
        
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
            prev.addConnection(layer.index, layer.neurons);
            
            layers[layer.index] = layer;            
            prev = layer;
        }
        
        /* output layer */
        layers[i] = new Layer(i, outputSize, outputActivation);
        prev.addConnection(i, outputSize);
        last = i;
        
        computeMemoryRequirements();
        
        finalized = true;
        
        return this;
    }
    
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
    public int getWeightsCount() {
        int count = 0;
        
        for(Layer l : layers)
            count += l.getWeightsCount();
        
        return count;
    }
    
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
    
    public void kaiming() {
        kaiming(1);
    }
    public void kaiming(int scalar) {
        for(Layer l : layers)
            l.kaiming(scalar);
    }
    
    /**
     * @param input
     * @return Compute the output of the neural network on given input
     */
    public float[] compute(float[] input) {
        float[][] results = new float[getLayerCount()][];
        results[0] = util.copyArray(input);
        
        for(int i=0; i < layers.length; i++)
            layers[i].feedForward(results[i], results);
        
        return results[last];
    }
    
    public float[] compute(float[] input, int count) {
        float[][] results = new float[getLayerCount()][];
        results[0] = util.copyArray(input);
        
        for(int i=0; i < layers.length; i++)
            layers[i].feedForward(results[i], count, results);
        
        return results[last];
    }
    
    public float[] compute(FloatBuffer input) {
        if(!nativeReady())
            prepareCPU();
        
        FloatBuffer[] results = new FloatBuffer[getLayerCount()];
        results[0] = input;
        
        for(int i=0; i < layers.length; i++)
            layers[i].feedForward(results[i], results);
            
        float[] result = new float[outputSize];
        results[last].rewind();
        results[last].get(result);
        
        return result;
    }
    
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
    
    /*
     * https://www.mldawn.com/deriving-the-gradient-descent-rule-part-2/
     */
    public void backpropagate(float[][] loss, int threads) {
        throw new RuntimeException("unsupported");
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
    
    public void ensureCPU() {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU");
        
        CudaEngine.prepareThread(deviceId);
        
        CUstream stream = CudaEngine.aquireStream(deviceId);
        
        for(Layer l : layers)
            l.ensureCPU(stream);
        
        JCudaDriver.cuStreamSynchronize(stream);
        
        CudaEngine.releaseStream(deviceId, stream);
        
        CudaEngine.finalizeThread();
    }

    public JSONObject serialize() {
        return serialize(new HashSet<>());
    }

    public JSONObject serialize(Set<String> ignoreProperties) {
        JSONObject obj = new JSONObject();
        
        Map<String, Object> map = new HashMap<>();
        
        for(Map.Entry<String, Object> e : properties.entrySet())
            if(!ignoreProperties.contains(e.getKey()))
                map.put(e.getKey(), e.getValue());
        
        obj.put("properties", map);
        
        JSONArray array = new JSONArray();
        
        obj.put("layers", array);
        
        for(Layer l: layers)
            array.put(l.serialize());
        
        return obj;
    }
    
    public static NeuralNetwork deserialize(JSONObject serialized) {
        JSONArray array = serialized.getJSONArray("layers");
        
        Layer[] layers = new Layer[array.length()];
        
        for(int i=0; i < array.length(); i++)
            layers[i] = Layer.deserialize(array.getJSONObject(i));
        
        NeuralNetwork result = new NeuralNetwork(layers);
        
        result.properties.putAll(serialized.getJSONObject("properties").toMap());
        return result;
    }
    
    public boolean gpuReady() {
        boolean gpuReady = true;
        for(Layer l : layers)
            gpuReady &= l.gpuReady();
        
        return gpuReady;
    }
    
    static AtomicLong COUNTER = new AtomicLong();
    
    public void prepareGPU() {
        if(gpuReady())
            throw new RuntimeException("GPU already initialized for connection");
        
        int device = (int) (COUNTER.getAndIncrement() % CudaEngine.getDeviceCount());
        prepareGPU(device);
    }
    
    public void prepareGPU(int device) {
        if(gpuReady())
            throw new RuntimeException("GPU already initialized for connection");
        
        lockMemory(prepareMemReq, device);
        
        this.deviceId = device;
        
        CudaEngine.prepareThread(device);

        CUstream stream = CudaEngine.aquireStream(device);

        for(Layer l : layers)
            l.prepareGPU(stream);

        JCudaDriver.cuStreamSynchronize(stream);

        CudaEngine.releaseStream(device, stream);

        CudaEngine.finalizeThread();
    }
    
    public void moveGPU(int newDevice) {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        if(deviceId == newDevice)
            throw new RuntimeException("Moving GPU to same device");
            
        freeGPU();
        prepareGPU(newDevice);
    }
    
    public float[] computeGPU(CUdeviceptr input) {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        CudaEngine.prepareThread(deviceId);
        
        CUstream stream = CudaEngine.aquireStream(deviceId);
        cublasHandle handle = CudaEngine.getCublasHandle(deviceId);
        
        CUdeviceptr[] layerResults = new CUdeviceptr[getLayerCount()];
        layerResults[0] = input;
        
        for(int i=0; i < layers.length; i++)
            layers[i].feedForwardGPU(layerResults[i], layerResults, stream, handle);
            
        float[] result = CudaUtil.fromGPUFloat(layerResults[last], outputSize, stream);
        
        JCudaDriver.cuStreamSynchronize(stream);
        
        CudaEngine.releaseStream(deviceId, stream);
        
        for(int i=1; i < layerResults.length; i++)
            CudaEngine.freeMempool(deviceId, layerResults[i]);
        
        CudaEngine.finalizeThread();
        
        return result;
    }

    public float[] computeGPUUnsafe(CUdeviceptr input, int count) {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        CudaEngine.prepareThread(deviceId);
        
        CUstream stream = CudaEngine.aquireStream(deviceId);
        cublasHandle handle = CudaEngine.getCublasHandle(deviceId);
//        JCublas2.cublasCreate(handle);

        CUdeviceptr[] layerResults = new CUdeviceptr[getLayerCount()];
        layerResults[0] = input;
        
        for(int i=0; i < layers.length; i++)
            layers[i].feedForwardGPU(layerResults[i], count, layerResults, stream, handle);
        
        float[] result = CudaUtil.fromGPUFloat(layerResults[last], count * outputSize, stream);
        
        JCudaDriver.cuStreamSynchronize(stream);
        
        CudaEngine.releaseStream(deviceId, stream);

        for(int i=1; i < layerResults.length; i++)
            CudaEngine.freeMempool(deviceId, layerResults[i]);
        
//        JCublas2.cublasDestroy(handle);
        CudaEngine.finalizeThread();
        
        return result;
    }
    
    public float[] computeGPU(CUdeviceptr input, int count) {
        long memory = getGPUComputeMemoryRequired(count);
        lockMemory(memory, deviceId);
        
        try {
            return computeGPUUnsafe(input, count);
        } finally {
            releaseMemory(memory);
        }
    }
    
    public long getGPUComputeMemoryRequired(int inputcount) {
        return computeMemReq * inputcount;
    }
    
    public long getGPUPrepareMemoryRequired() {
        return prepareMemReq;
    }
    
    public int getGPUDeviceId() {
        return deviceId;
    }

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
    
    public void crossOverMutate(NeuralNetwork a, NeuralNetwork b, float min, float max, double mutation) {
        for(int i=0; i < layers.length; i++)
            layers[i].crossOverMutate(a.getLayer(i), b.getLayer(i), min, max, mutation);
    }
    
    public void crossOverMutateGPU(NeuralNetwork a, NeuralNetwork b, float min, float max, double mutation, boolean nocopy) {
        if(!a.gpuReady() || !b.gpuReady())
            throw new RuntimeException("Parent networks are not loaded to the GPU, please call prepareGPU first");
        
        if(a.deviceId != b.deviceId)
            throw new RuntimeException("Parent networks are on different GPUs");
        
        if(gpuReady())
            throw new RuntimeException("Network is gpu ready, please call freeGPU first");
        
        this.deviceId = a.deviceId;
            
        /* 1x prepare gpu + 3x rng arrays */
        lockMemory(prepareMemReq * 4, deviceId);
        
        try {
            CudaEngine.prepareThread(deviceId);        
            CUstream stream = CudaEngine.aquireStream(deviceId);
            curandGenerator generator = CudaEngine.getCurandGenerator(deviceId);

            for(int i=0; i < layers.length; i++)
                layers[i].crossOverMutateGPU(a.getLayer(i), b.getLayer(i), min, max, mutation, nocopy, stream, generator);

            JCudaDriver.cuStreamSynchronize(stream);
            CudaEngine.releaseStream(deviceId, stream);

            for(int i=0; i < layers.length; i++)
                layers[i].freeGPURng();

            CudaEngine.finalizeThread();
        } finally {
            releaseMemory(prepareMemReq * 3);
        }
    }
    
    public void clipWeights(float min, float max) {
        for(int i=0; i < layers.length; i++)
            layers[i].clipWeights(min, max);
    }

    public void clipWeightsGPU(float min, float max) {
        if(gpuReady())
            throw new RuntimeException("Network is gpu ready, please call freeGPU first");
        
        CudaEngine.prepareThread(deviceId);
        CUstream stream = CudaEngine.aquireStream(deviceId);

        for(int i=0; i < layers.length; i++)
            layers[i].clipWeightsGPU(min, max, stream);

        JCudaDriver.cuStreamSynchronize(stream);
        CudaEngine.releaseStream(deviceId, stream);

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
    
    public float compare(NeuralNetwork a) {
        float score = 0;
        
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
    
    public void freeGPU() {
        for(Layer l : layers)
            l.freeGPU(deviceId);
        
        releaseMemory(prepareMemReq);
        
        deviceId = -1;
    }
    
    private void lockMemory(long size, int deviceId) {
        Semaphore lock = CudaEngine.getMemLock(deviceId);
        
        if(lock == null)
            throw new RuntimeException("Must specify maximum memory usage for device: " + deviceId);

        int memoryKB = (int)Math.ceil(size / 1024.0);
        
        lock.acquireUninterruptibly(memoryKB);        
    }
    
    private void releaseMemory(long size) {
        Semaphore lock = CudaEngine.getMemLock(deviceId);
        
        int memoryKB = (int)Math.ceil(size / 1024.0);
        
        lock.release(memoryKB);
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
        CUdeviceptr pool = CudaUtil.createByte(totalSize);
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
        
        prepareMemReq = 0;
        for(Layer l : layers)
            prepareMemReq += l.getMemoryRequirements();
    }
}
