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

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
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

    /* true if CPU memory was emptied */
    boolean cpuFree;
    
    /* handle for cublas */
    cublasHandle handle;
    
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
    }
    
    public NeuralNetwork copy(boolean withoutWeights) {
        Layer[] copied = new Layer[layers.length];
        
        for(int i=0; i < layers.length; i++) {
            copied[i] = layers[i].copy(withoutWeights);
        }
        
        return new NeuralNetwork(copied);
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
            l.randomize(min, max);
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
    
    public void backpropagate(float[][] loss, int threads) {
        throw new RuntimeException("unsupported");
    }
    
    public void freeCPU() {
        if(cpuFree)
            return;
        
        for(Layer l : layers)
            l.freeCPU();
        
        cpuFree = true;
    }
    
    public JSONObject serialize() {
        JSONObject obj = new JSONObject();
        
        obj.put("properties", properties);
        
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
    
    /**
     * Select random GPU and initialize weights
     */
    static final AtomicInteger COUNTER = new AtomicInteger();
    
    public boolean gpuReady() {
        boolean gpuReady = true;
        for(Layer l : layers)
            gpuReady &= l.gpuReady;
        
        return gpuReady;
    }
    
    public boolean prepareGPU() {        
        int device = deviceId == -1 ? COUNTER.getAndIncrement() % CudaEngine.getDeviceCount() : deviceId;

        CudaEngine.prepareThread(device);
        
        boolean result = prepareGPU(device, CudaEngine.getStream(device));
        
        CudaEngine.finalizeThread();
        
        return result;
    }
    
    public boolean prepareGPU(int device, CUstream stream) {
        if(this.deviceId != device && this.deviceId != -1) {
            CudaEngine.prepareThread(this.deviceId);
            freeGPU();
            CudaEngine.finalizeThread();
        }
        
        this.deviceId = device;
        
        CudaEngine.prepareThread(device);
        
        if(handle == null) {
            handle = new cublasHandle();
            JCublas2.cublasCreate(handle);
        }
        
        for(Layer l : layers)
            l.prepareGPU(stream);
        
        JCudaDriver.cuStreamSynchronize(stream);
        CudaEngine.finalizeThread();
        
        return true;
    }
    
    public float[] computeGPU(float[] input) {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        CudaEngine.prepareThread(deviceId);
        
        float[] result = computeGPU(input, CudaEngine.getStream(deviceId));

        CudaEngine.finalizeThread();

        return result;
    }
    
    public float[] computeGPU(float[] input, CUstream stream) {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        cudaStream_t stream0 = new cudaStream_t(stream);
        JCublas2.cublasSetStream(handle, stream0);
        
        Pointer devicePtr;
        boolean usePinned = CudaEngine.usePinnedMemory();

        if(usePinned) {
           devicePtr = new Pointer();
            
           JCudaDriver.cuMemAllocHost(devicePtr, input.length * (long) Sizeof.FLOAT);
          
           FloatBuffer b = devicePtr.getByteBuffer().asFloatBuffer();
           b.put(input);
        } else {
            devicePtr = Pointer.to(input);
        }
        
        CUdeviceptr ptr = CudaUtil.toGPU(devicePtr, input.length, stream);
        
        for(Layer l : layers) {
            CUdeviceptr temp = l.feedForwardGPU(ptr, stream, handle);
            if(!temp.equals(ptr))
                JCudaDriver.cuMemFree(ptr);
            ptr = temp;
        }

        float[] result = CudaUtil.fromGPU(ptr, getOutputSize(), stream);
        
        if(usePinned)
            JCudaDriver.cuMemFreeHost(devicePtr);

        JCudaDriver.cuMemFree(ptr);
        
        return result;
    }

    public float[][] computeGPU(float[][] inputs) {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        CudaEngine.prepareThread(deviceId);

        float[][] result = computeGPU(inputs, CudaEngine.getStream(deviceId));

        CudaEngine.finalizeThread();

        return result;
    }
    
    public float[][] computeGPU(float[][] inputs, CUstream stream) {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        cudaStream_t stream0 = new cudaStream_t(stream);
        JCublas2.cublasSetStream(handle, stream0);
        
        Pointer devicePtr;
        boolean usePinned = CudaEngine.usePinnedMemory();
        
        int count = inputs.length;
        float[] input = util.to1D(inputs, getInputSize());
        
        if(usePinned) {
           devicePtr = new Pointer();
            
           JCudaDriver.cuMemAllocHost(devicePtr, input.length * (long) Sizeof.FLOAT);
          
           FloatBuffer b = devicePtr.getByteBuffer().asFloatBuffer();
           b.put(input);
        } else {
            devicePtr = Pointer.to(input);
        }
        
        CUdeviceptr ptr = CudaUtil.toGPU(devicePtr, input.length, stream);
        
        for(Layer l : layers) {
            CUdeviceptr temp = l.feedForwardGPU(ptr, stream, handle, count);
            if(!temp.equals(ptr))
                JCudaDriver.cuMemFree(ptr);
            ptr = temp;
        }

        float[] result = CudaUtil.fromGPU(ptr, count * getOutputSize(), stream);
        
        if(usePinned)
            JCudaDriver.cuMemFreeHost(devicePtr);

        JCudaDriver.cuMemFree(ptr);
        
        return util.to2D(result, count, getOutputSize());
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
    
    public void crossOverMutate(NeuralNetwork a, NeuralNetwork b, double min, double max, double mutation) {
        for(int i=0; i < layers.length; i++)
            layers[i].crossOverMutate(a.getLayer(i), b.getLayer(i), min, max, mutation);
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
    
    public void crossOverMutateGPU(NeuralNetwork a, NeuralNetwork b, double min, double max, double mutation) {
        if(!gpuReady() || !a.gpuReady() || !b.gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        CudaEngine.prepareThread(deviceId);

        crossOverMutateGPU(a, b, min, max, mutation, CudaEngine.getStream(deviceId));

        CudaEngine.finalizeThread();
    }
    
    public void crossOverMutateGPU(NeuralNetwork a, NeuralNetwork b, double min, double max, double mutation, CUstream stream) {
        if(!gpuReady() || !a.gpuReady() || !b.gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        curandGenerator generator = new curandGenerator();
        JCurand.curandCreateGenerator(generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
        JCurand.curandSetStream(generator, new cudaStream_t(stream));
        JCurand.curandSetPseudoRandomGeneratorSeed(generator, Rng.nextLong(Long.MAX_VALUE));
        
        for(int i=0; i < layers.length; i++)
            layers[i].crossOverMutateGPU(a.getLayer(i), b.getLayer(i), min, max, mutation, stream, generator);
        
        JCurand.curandDestroyGenerator(generator);
    }

    public float compare(NeuralNetwork a) {
        float score = 0;
        
        for(int i=0; i < a.getLayerCount(); i++)
            score += layers[i].compare(a.getLayer(i));
        
        return score;
    }
    
    public float compareGPU(NeuralNetwork a) {        
        if(!gpuReady() || !a.gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        CudaEngine.prepareThread(deviceId);
        
        float result = compareGPU(a, CudaEngine.getStream(deviceId));
        
        CudaEngine.finalizeThread();
        
        return result;
    }
    
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
            l.freeGPU();

        deviceId = -1;
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
}
