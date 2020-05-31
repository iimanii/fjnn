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
import org.fjnn.base.BaseLayer;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicInteger;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import org.fjnn.activation.Activation;
import org.fjnn.base.LayeredNetwork;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaThread;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.serializer.LayerStub;
import org.fjnn.serializer.NetworkStub;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 */
public class NeuralNetwork extends LayeredNetwork {
    Layer[] layers;
    
    Layer input;
    Layer output;
    
    NetworkBuilder builder;
    
    /* a network can only be built once */
    boolean finalized;

    /* device id for gpu to use */
    int deviceId;
    
    /* Weights loaded to GPU */
    boolean gpuReady;
    
    /* true if CPU memory was emptied */
    boolean cpuFree;
    
    boolean threadSafe;
    
    CUstream mainstream;
    
    /* use this to transfer faster */
    Pointer pinned;
    FloatBuffer buffer;
    
    public NeuralNetwork(boolean threadSafe) {
        this.threadSafe = threadSafe;
        this.properties = new HashMap<>();
        
        this.cpuFree = false;
        this.gpuReady = false;
        this.finalized = false;
        
        this.deviceId = -1;
        this.builder = new NetworkBuilder(threadSafe);
    }
    
    public NeuralNetwork(NetworkStub stub) {
        this.threadSafe = stub.threadSafe;
        this.properties = new HashMap<>(stub.properties);
        
        this.cpuFree = false;
        this.gpuReady = false;
        this.finalized = false;
        
        this.deviceId = -1;
        
        this.builder = new NetworkBuilder(threadSafe);
        
        for(LayerStub lstub : stub.layers)
            builder.addLayer(lstub);

        build();
    }
    
    public NeuralNetwork addLayer(int neurons, Activation activation, boolean hasBias) {
        return addLayer(neurons, activation, hasBias, null);
    }
    
    public NeuralNetwork addLayer(int neurons, Activation activation, boolean hasBias, boolean[] condition) {
        builder.addLayer(neurons, activation, hasBias, condition);
        return this;
    }
    
    public final NeuralNetwork build() {
        if(finalized)
            return this;
        
        layers = builder.buildLayers();
        input = layers[0];
        output = layers[layers.length - 1];

        /* no need to keep builder in memory */
        builder = null;
        
        return this;
    }
    
    @Override
    public NeuralNetwork randomize(float min, float max) {
        for(Layer l : layers)
            l.randomize(-1, 1);
        
        return this;
    }
    
    /**
     * @param layer
     * @param from
     * @param to
     * @return 
     */
    @Override
    public float getWeight(int layer, int from, int to) {
        return layers[layer].getWeight(from, to);
    }
    
    /**
     * 
     * @return 
     */
    @Override
    public float[][][] getWeights() {
        float[][][] result = new float[layers.length][][];
        
        for(int i=0; i < result.length; i++)
            result[i] = layers[i].getWeights();
        
        return result;
    }
    
    /**
     * 
     * @param layer
     * @param from
     * @param to
     * @param value 
     */
    @Override
    public void setWeight(int layer, int from, int to, float value) {
        layers[layer].setWeight(from, to, value);
        gpuReady = false;
    }
    
    /**
     * 
     * @param values 
     */
    @Override
    public void setWeights(float[][][] values) {
        for(int i=0; i < values.length; i++)
            layers[i].setWeights(values[i]);
        
        gpuReady = false;
    }
    
    /**
     * 
     * @param layer
     * @param to
     * @return 
     */
    @Override
    public float getBias(int layer, int to) {
        return layers[layer].getBias(to);
    }
    
    /**
     * 
     * @param layer
     * @param to
     * @param value 
     */
    @Override
    public void setBias(int layer, int to, float value) {
        layers[layer].setBias(to, value);
        gpuReady = false;
    }
    
    @Override
    public void setBiases(float[][] values) {
        for(int i=0; i < values.length; i++)
            layers[i].setBiases(values[i]);
        
        gpuReady = false;
    }
    
    
    @Override
    public float[][] getBiases() {
        float[][] result = new float[layers.length][];
        
        for(int i=0; i < result.length; i++)
            result[i] = layers[i].getBiases();
        
        return result;
    }

    @Override
    public boolean hasBias(int layer) {
        return layers[layer].hasBias();
    }
    
    /**
     * 
     * @param layer
     * @return 
     */
    @Override
    public Activation getActivation(int layer) {
        return layers[layer].getActivation();
    }
    
    /**
     * 
     * @param layer
     * @return 
     */
    @Override
    public boolean[] getCondition(int layer) {
        return layers[layer].getCondition();
    }
    
    /**
     * 
     * @return Number of inputs
     */
    @Override
    public int getInputSize() {
        return input.size();
    }
    
    /**
     * 
     * @return Number of outputs
     */
    @Override
    public int getOutputSize() {
        return output.size();
    }

    /**
     * 
     * @param layer
     * @return Neuron count for a specific layer
     */
    @Override
    public int getLayerSize(int layer) {
        return layers[layer].size();
    }
    
    public int getLayerTotalSize(int layer) {
        return layers[layer].totalSize();
    }
    
    /**
     * @return Number of hidden layers
     */
    @Override
    public int getHiddenLayerCount() {
        return layers.length - 2;
    }
    
    /**
     * @return Total number of layers
     */
    @Override
    public int getLayerCount() {
        return layers.length;
    }
    
    
    @Override
    public Object getProperty(String name) {
        return properties.get(name);
    }
    
    @Override
    public void setProperty(String name, Object object) {
        properties.put(name, object);
    }
    
    @Override
    public boolean hasProperty(String name) {
        return properties.containsKey(name);
    }
    
    /**
     * @param input
     * @return Compute the output of the neural network on given input
     */
    public float[] compute(float[] input) {
        float[] result = util.copyArray(input);
        
        for(Layer l : layers) {
            result = l.feedForward(result);
        }
        
        return result;
    }

    public void freeCPU() {
        if(cpuFree)
            return;
        
        for(BaseLayer l : layers)
            l.freeCPU();
        
        cpuFree = true;
    }
    
    /**
     * Select random GPU and initialize weights
     */
    static AtomicInteger counter = new AtomicInteger();

    public boolean prepareGPU() {
        int device = deviceId == -1 ? counter.getAndIncrement() % CudaEngine.getDeviceCount() : deviceId;
        return prepareGPU(device);
    }
    
    public boolean prepareGPU(int device) {
        if(this.deviceId != device && this.deviceId != -1) {
            CudaThread.prepareThread(this.deviceId);
            freeGPU();
        }
        
        this.deviceId = device;
        
        CudaThread.prepareThread(device);
        
        if(mainstream == null) {
            mainstream = new CUstream();
            JCudaDriver.cuStreamCreate(mainstream, CUstream_flags.CU_STREAM_DEFAULT);
        }

                
        if(pinned == null && CudaEngine.usePinnedMemory()) {        
            pinned = new Pointer();
            JCudaDriver.cuMemAllocHost(pinned, (input.size() + 1) * (long) Sizeof.FLOAT);
            buffer = pinned.getByteBuffer().asFloatBuffer();    
        }

        for(Layer l : layers)
            l.prepareGPU(mainstream);
        
        gpuReady = true;
        
        JCudaDriver.cuStreamSynchronize(mainstream);
        CudaThread.finalizeThread();
        
        return true;
    }
    
    public float[] computeGPU(float[] input) {
        if(!gpuReady)
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        if(threadSafe)
            return computeGPUThreadSafe(input);
        
        CudaThread.prepareThread(deviceId);
    
        Pointer devicePtr = null;
        int length = this.input.size() + 1;
        
        if(pinned != null) {
           buffer.clear();
           buffer.put(input);
           buffer.put(1);
           
           devicePtr = pinned;
        } else {
            float[] inputWithBias = util.copyArray(input, length);
            inputWithBias[length - 1] = 1;
            
            devicePtr = Pointer.to(inputWithBias);
        }

        CUdeviceptr temp = CudaEngine.getSharedResource(length, deviceId);
        CUdeviceptr ptr = temp;

        JCudaDriver.cuMemcpyHtoDAsync(ptr, devicePtr, length * (long)Sizeof.FLOAT, mainstream);
        
        for(Layer l : layers)
            ptr = l.feedForwardGPU(ptr, mainstream);
        
        float[] result = CudaUtil.fromGPU(ptr, output.size(), mainstream);
        
        CudaEngine.freeSharedResource(temp, deviceId);
        JCudaDriver.cuStreamSynchronize(mainstream);

        CudaThread.finalizeThread();
        
        return result;
    }

    private float[] computeGPUThreadSafe(float[] input) {
        CudaThread.prepareThread(deviceId);
        
        CUstream stream = new CUstream();
        JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_DEFAULT);

        Pointer devicePtr;
        int length = this.input.size();
        boolean usePinned = CudaEngine.usePinnedMemory();

        if(usePinned) {
           devicePtr = new Pointer();
            
           JCudaDriver.cuMemAllocHost(devicePtr, length * (long) Sizeof.FLOAT);
          
           FloatBuffer b = devicePtr.getByteBuffer().asFloatBuffer();
           b.put(input);
        } else {
            float[] inputWithBias = util.copyArray(input, this.input.size());
            
            devicePtr = Pointer.to(inputWithBias);
        }
        
        CUdeviceptr ptr = CudaUtil.toGPU(devicePtr, length, stream);
        
        for(Layer l : layers) {
            CUdeviceptr temp = l.feedForwardGPU(ptr, stream);
            if(!temp.equals(ptr))
                JCudaDriver.cuMemFree(ptr);
            ptr = temp;
        }

        float[] result = CudaUtil.fromGPU(ptr, output.size(), stream);
    
        if(usePinned)
            JCudaDriver.cuMemFreeHost(devicePtr);

        JCudaDriver.cuMemFree(ptr);
    
        JCudaDriver.cuStreamSynchronize(stream);
        JCudaDriver.cuStreamDestroy(stream);

        CudaThread.finalizeThread();
        
        return result;
    }
        
    public void freeGPU() {
        for(BaseLayer l : layers)
            l.freeGPU();

        if(mainstream != null)
            JCudaDriver.cuStreamDestroy(mainstream);
    
        if(pinned != null)
            JCudaDriver.cuMemFreeHost(pinned);
        
        deviceId = -1;
        buffer = null;
        pinned = null;
        mainstream = null;
        gpuReady = false;
    }

    @Override
    public NetworkStub getStub() {
        LayerStub[] stubs = new LayerStub[layers.length];
        for(int i=0; i < layers.length; i++)
            stubs[i] = layers[i].getStub();
        
        return new NetworkStub(stubs, properties, threadSafe);
    }
    
    @Override
    public void fromNetwork(LayeredNetwork n, boolean copyProperties) {
        super.fromNetwork(n, copyProperties);
        gpuReady = false;
    }
    
    public boolean isThreadSafe() {
        return threadSafe;
    }
}
