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
package org.fjnn.parallel;

import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUmemorytype;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import org.fjnn.activation.Activation;
import org.fjnn.base.BaseNetwork;
import org.fjnn.base.LayeredNetwork;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.network_old.LayerStub;
import org.fjnn.network_old.NetworkStub;
import org.fjnn.parallel.ParallelUtil.CUdeviceptr2D;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 */
public class MultiNetwork extends BaseNetwork {
    ParallelLayer[] layers;
    
    ParallelLayer input;
    ParallelLayer output;
    
    MultiNetworkBuilder builder;
    
    ShadowNetwork[] networks;

    /* a network can only be built once */
    boolean finalized;

    /* device id for gpu to use */
    int deviceId;
    
    /* Weights loaded to GPU */
    boolean gpuReady;
    
    /* number of neural networks */
    int size;
    
    /* true if CPU memory was emptied */
    boolean cpuFree;
    
    boolean threadSafe;
    
    CUstream mainstream;
    
    /* use this to transfer faster */
    Pointer pinned;
    FloatBuffer buffer;
    CUdeviceptr2D input2D;
    
    /**
     * 
     * @param size
     * @param threadSafe 
     */
    public MultiNetwork(int size, boolean threadSafe) {
        this(size, threadSafe, null);
    }
    
    /**
     * 
     * @param size
     * @param threadSafe
     * @param base
     */
    public MultiNetwork(int size, boolean threadSafe, LayeredNetwork base) {
        this.size = size;
        this.threadSafe = threadSafe;
        this.properties = new HashMap<>();
        
        this.cpuFree = false;
        this.gpuReady = false;
        this.finalized = false;
        
        this.deviceId = -1;
        this.builder = new MultiNetworkBuilder(size, threadSafe);
        
        this.networks = new ShadowNetwork[size];
        
        if(base == null)
            return;
        
        for(int i=0; i < base.getLayerCount(); i++)
            addLayer(base.getLayerSize(i), base.getActivation(i), base.hasBias(i), base.getCondition(i));
    }
    
    /**
     * 
     * @param neurons
     * @param activation
     * @param hasBias
     * @return 
     */
    public final MultiNetwork addLayer(int neurons, Activation activation, boolean hasBias) {
        return addLayer(neurons, activation, hasBias, null);
    }
    
    /**
     * 
     * @param neurons
     * @param activation
     * @param hasBias
     * @param condition
     * @return 
     */
    public final MultiNetwork addLayer(int neurons, Activation activation, boolean hasBias, boolean[] condition) {
        builder.addLayer(neurons, activation, hasBias, condition);
        return this;
        
    }
    
    /**
     * 
     * @return 
     */
    public final MultiNetwork build() {
        if(finalized)
            return this;
        
        layers = builder.buildLayers();
        input = layers[0];
        output = layers[layers.length - 1];

        /* no need to keep builder in memory */
        builder = null;

        for(int i=0; i < this.networks.length; i++)
            this.networks[i] = new ShadowNetwork(this, i);
        
        return this;
    }
    
    /**
     * 
     * @return 
     */
    @Override
    public void randomize() {
        for(ParallelLayer l : layers)
            l.randomize();
    }
    
    /**
     * @param index
     * @param layer
     * @param from
     * @param to
     * @return 
     */
    public float getWeight(int index, int layer, int from, int to) {
        return layers[layer].getWeight(index, from, to);
    }
    
    /**
     * 
     * @param index
     * @param layer
     * @param from
     * @param to
     * @param value 
     */
    public void setWeight(int index, int layer, int from, int to, float value) {
        layers[layer].setWeight(index, from, to, value);
        gpuReady = false;
    }
    

    /**
     * 
     * @param index
     * @param values 
     */
    public void setWeights(int index, float[][][] values) {
        for(int i=0; i < values.length; i++)
            layers[i].setWeights(index, values[i]);
        
        gpuReady = false;
    }
    
    /**
     * 
     * @param index
     * @return 
     */
    public float[][][] getWeights(int index) {
        float[][][] result = new float[layers.length][][];
        
        for(int i=0; i < result.length; i++)
            result[i] = layers[i].getWeights(index);
        
        return result;        
    }
    
    /**
     * 
     * @param index
     * @param layer
     * @param to
     * @return 
     */
    public float getBias(int index, int layer, int to) {
        return layers[layer].getBias(index, to);
    }
    
    /**
     * 
     * @param index
     * @param layer
     * @param to
     * @param value 
     */
    public void setBias(int index, int layer, int to, float value) {
        layers[layer].setBias(index, to, value);
        gpuReady = false;
    }
    
    /**
     * 
     * @param index
     * @param values 
     */
    public void setBiases(int index, float[][] values) {
        for(int i=0; i < values.length; i++)
            layers[i].setBiases(index, values[i]);
        
        gpuReady = false;
    }
    
    /**
     * 
     * @param index
     * @return 
     */
    public float[][] getBiases(int index) {
        float[][] result = new float[layers.length][];
        
        for(int i=0; i < result.length; i++)
            result[i] = layers[i].getBiases(index);
        
        return result;
    }
    
    /**
     * 
     * @param layer
     * @return 
     */
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
     * @return Number of inputs
     */
    @Override
    public int getInputSize() {
        return input.neurons();
    }
    
    /**
     * 
     * @return Number of outputs
     */
    @Override
    public int getOutputSize() {
        return output.neurons();
    }

    /**
     * 
     * @param layer
     * @return Neuron count for a specific layer
     */
    @Override
    public int getLayerSize(int layer) {
        return layers[layer].neurons();
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
    
    /**
     * 
     */
    public int getSize() {
        return size;
    }
    
    public LayeredNetwork getNetwork(int index) {
        return networks[index];
    }
    
    public List<LayeredNetwork> getAll() {
        List<LayeredNetwork> list = new ArrayList<>();
        
        for(LayeredNetwork n : networks)
            list.add(n);
        
        return list;
    }
    
    /**
     * 
     */
    public void freeCPU() {
        if(cpuFree)
            return;
        
        for(ParallelLayer l : layers)
            l.freeCPU();
        
        cpuFree = true;
    }
    
    /**
     * Select random GPU and initialize weights
     */
    static AtomicInteger counter = new AtomicInteger();

    /**
     * 
     * @return 
     */
    public boolean prepareGPU() {
        int device = deviceId == -1 ? counter.getAndIncrement() % CudaEngine.getDeviceCount() : deviceId;
        return prepareGPU(device);
    }
    
    /**
     * 
     * @param device
     * @return 
     */
    public boolean prepareGPU(int device) {
        if(this.deviceId != device && this.deviceId != -1) {
            CudaEngine.prepareThread(this.deviceId);
            freeGPU();
        }
        
        this.deviceId = device;
        
        CudaEngine.prepareThread(device);
        
        if(mainstream == null) {
            mainstream = new CUstream();
            JCudaDriver.cuStreamCreate(mainstream, CUstream_flags.CU_STREAM_DEFAULT);
        }

        if(pinned == null && CudaEngine.usePinnedMemory()) {        
            pinned = new Pointer();
            JCudaDriver.cuMemAllocHost(pinned, size * (input.neurons() + 1) * (long) Sizeof.FLOAT);
            buffer = pinned.getByteBuffer().asFloatBuffer();    
        }
        
        if(input2D == null)
            input2D = ParallelUtil.createPitch((input.neurons() + 1), size);

        for(ParallelLayer l : layers)
            l.prepareGPU(mainstream);
        
        gpuReady = true;
        
        JCudaDriver.cuStreamSynchronize(mainstream);
        CudaEngine.finalizeThread();
        
        return true;
    }
    

    /**
     * @param input
     * @return Compute the output of the neural network on given input
     */
    public float[][] compute(float[][] input) {
        float[][] result = util.copyArray(input);
        
        for(ParallelLayer l : layers) {
            result = l.feedForward(result);
        }
        
        return result;
    }
    

    public float[][] computeGPU(float[][] input) {
        if(!gpuReady)
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        if(threadSafe)
            throw new UnsupportedOperationException("Thread safe MultiNetworks not supported yet.");
        
        CudaEngine.prepareThread(deviceId);
    
        Pointer devicePtr;
        int width = (this.input.neurons() + 1);
        int length = this.size * width;
        
        if(pinned != null) {
           buffer.clear();
           
           for(int i=0; i < size; i++) {
               buffer.put(input[i]);
               buffer.put(1);
           }
           
           devicePtr = pinned;
        } else {
            float[] inputWithBias = new float[length];
            
            for(int i=0; i < size; i++) {
                System.arraycopy(input, 0, inputWithBias, i * length, length - 1);
                inputWithBias[(i+1) * length - 1] = 1;
            }
            
            devicePtr = Pointer.to(inputWithBias);
        }
        
        CUDA_MEMCPY2D memcpy = new CUDA_MEMCPY2D();
        memcpy.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
        memcpy.srcHost = devicePtr;
        memcpy.srcPitch = width * (long) Sizeof.FLOAT;

        memcpy.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
        memcpy.dstDevice = input2D.ptr;
        memcpy.dstPitch = input2D.pitch * (long) Sizeof.FLOAT;

        memcpy.WidthInBytes = width * (long) Sizeof.FLOAT;
        memcpy.Height = size;

        memcpy.srcXInBytes = memcpy.srcY = memcpy.dstXInBytes = memcpy.dstY = 0;

        JCudaDriver.cuMemcpy2DAsync(memcpy, mainstream);
        
        CUdeviceptr2D ptr = new CUdeviceptr2D(input2D.ptr, input2D.pitch);
        
        for(ParallelLayer l : layers)
            ptr = l.feedForwardGPU(ptr, mainstream);
        
        float[][] result = ParallelUtil.fromGPU(ptr, output.neurons(), size, mainstream);
        
        JCudaDriver.cuStreamSynchronize(mainstream);

        CudaEngine.finalizeThread();
        
        return result;
    }

//    private float[] computeGPUThreadSafe(float[] input) {
//        CudaThread.prepareThread(deviceId);
//        
//        CUstream stream = new CUstream();
//        JCudaDriver.cuStreamCreate(stream, CUstream_flags.CU_STREAM_DEFAULT);
//        
//        float[] inputWithBias = util.copyArray(input, this.input.neurons);
//        
//        CUdeviceptr ptr = CudaUtil.toGPU(inputWithBias, stream);
//        
//        for(MultiLayer l : layers) {
//            CUdeviceptr temp = l.feedForwardGPU(ptr, stream);
//            if(!temp.equals(ptr))
//                JCudaDriver.cuMemFree(ptr);
//            ptr = temp;
//        }
//
//        float[] result = CudaUtil.fromGPU(ptr, output.size(), stream);
//
//        JCudaDriver.cuMemFree(ptr);
//
//        JCudaDriver.cuStreamSynchronize(stream);
//        JCudaDriver.cuStreamDestroy(stream);
//
//        CudaThread.finalizeThread();
//        
//        return result;
//    }
//        
    public void freeGPU() {
        for(ParallelLayer l : layers)
            l.freeGPU();

        JCudaDriver.cuStreamDestroy(mainstream);
        JCudaDriver.cuMemFreeHost(pinned);
        JCudaDriver.cuMemFree(input2D.ptr);
        
        deviceId = -1;
        buffer = null;
        input2D = null;
        mainstream = null;
        gpuReady = false;
    }

    @Override
    public void randomize(float min, float max) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public MultiNetwork randomize(int index, float min, float max) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.        
    }
    
    public NetworkStub getStub(int index) {
        LayerStub[] stubs = new LayerStub[layers.length];
        for(int i=0; i < layers.length; i++)
            stubs[i] = layers[i].getStub(index);
        
        return new NetworkStub(stubs, networks[index].getProperties(), threadSafe);
    }

    @Override
    public boolean[] getCondition(int layer) {
        return layers[layer].getCondition();
    }
}
