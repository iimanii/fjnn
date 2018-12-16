/*
 * The MIT License
 *
 * Copyright 2018 ahmed.
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

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.activation.Activation;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaModule;
import org.fjnn.cuda.CudaThread;
import org.fjnn.serializer.LayerStub;
import org.fjnn.util.Rng;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 * 
 * Not thread safe, but 10% faster
 */
public class FastLayer extends Layer {
    /**
     * Store everything in 1D array
     * [N(0-0), N(1-0), ... N(B-0) N(1-0) N(1-1) ....
     */
    float[] weights;

    /* weights stored on GPU */
    CUdeviceptr weightsGPU;
    
    /* activation conditions on GPU */
    CUdeviceptr conditionGPU;

    /* results are calculated here */
    CUdeviceptr resultGPU;
    
    /**
     * Number of neurons + 1 (bias)
     */
    final int totalNeurons;
        
    /* unconnected layer / output layer */
    public FastLayer(Activation activation, int neurons) {
        this(activation, neurons, 0, false, null);
    }
    
    public FastLayer(Activation activation, int neurons, int links, boolean hasBias, boolean[] condition) {
        super(activation, neurons, links, hasBias, condition);
        
        this.totalNeurons = neurons + 1;

        this.weights = new float[totalNeurons * links];
    }
    
    /**
     * 
     * @param from
     * @param to
     * @return 
     */
    @Override
    protected float getWeight(int from, int to) {
        if(from < 0 || from >= neurons)
            throw new ArrayIndexOutOfBoundsException();
        
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            
        
        return weights[from + totalNeurons * to];        
    }

    /**
     * 
     * @param from
     * @param to
     * @param value 
     */
    @Override
    protected void setWeight(int from, int to, float value) {
        if(from < 0 || from >= neurons)
            throw new ArrayIndexOutOfBoundsException();
        
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();
        
        weights[from + totalNeurons * to] = value;

        gpuReady = false;
    }

    @Override
    protected float[][] getWeights() {
        float[][] result = new float[links][neurons];
        
        for(int i=0; i < links; i++)
            System.arraycopy(this.weights, i * totalNeurons, result[i], 0, neurons);
        
        return result;
    }
    
    @Override
    public void setWeights(float[][] values) {
        for(int i=0; i < links; i++)
            for(int j=0; j < neurons; j++)
//                this.weights[] = values[]
//            System.arraycopy(values[i], 0, this.weights, i*totalNeurons, neurons);
        
        gpuReady = false;
    }

    @Override
    protected float getBias(int to) {
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();  
        
        return weights[totalNeurons * to + neurons];
    }

    @Override
    protected void setBias(int to, float value) {
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            

        if(hasBias)
            weights[totalNeurons * to + neurons] = value;
        
        gpuReady = false;
    }
    
    @Override
    protected float[] getBiases() {
        float[] result = new float[links];
        
        for(int i=0; i < links; i++)
            result[i] = weights[totalNeurons * i + neurons];
        
        return result;
    }

    @Override
    public void setBiases(float[] values) {
        gpuReady = false;
        
        if(!hasBias)
            return;

        for(int i=0; i < links; i++)
            weights[totalNeurons * i + neurons] = values[i];
    }

    @Override
    public void randomize(float min, float max) {
        int len = totalNeurons * links;
        
        for(int i=0; i < len; i++)
            weights[i] = Rng.nextFloat(min, max);
        
        if(!hasBias)
            for(int i=neurons; i < len; i+=totalNeurons)
                weights[i] = 0;
    }

    @Override
    protected float[] feedForward(float[] input) {
        if(activation != null) {
            if(condition == null)
                activation.compute(input);
            else
                activation.computeConditional(input, condition);
        }
        
        /* output layer */
        if(isOutput)
            return input;
                
        float[] output = new float[links];
        
        for(int i=0; i < links; i++) {
            double sum = 0;
            int k = i * totalNeurons;
            
            /* neurons */
            for(int j=0; j < neurons; j++)
                sum += input[j] * weights[k + j];
            
            /* bias */
            sum += weights[k + neurons];
            
            output[i] = (float) sum;
        }
        
        return output;
    }
    
    @Override
    public void prepareGPU(CUstream stream) {
        if(isOutput) {
            gpuReady = true;
            return;
        }
        
        if(weightsGPU == null) {
            weightsGPU = new CUdeviceptr();
            JCudaDriver.cuMemAlloc(weightsGPU, weights.length * (long) Sizeof.FLOAT);
        }
        
        if(resultGPU == null) {
            resultGPU = new CUdeviceptr();
            JCudaDriver.cuMemAlloc(resultGPU, (links + 1) * (long) Sizeof.FLOAT);
        }
        
        if(condition != null && conditionGPU == null) {
            conditionGPU = new CUdeviceptr();
            JCudaDriver.cuMemAlloc(conditionGPU, condition.length * (long) Sizeof.BYTE);

            byte[] temp = new byte[condition.length];
            
            for(int i=0; i < condition.length; i++)
                temp[i] = condition[i] ? (byte)1 : 0;
            
            JCudaDriver.cuMemcpyHtoDAsync(conditionGPU, Pointer.to(temp), condition.length * (long)Sizeof.BYTE, stream);
        }

        JCudaDriver.cuMemcpyHtoDAsync(weightsGPU, Pointer.to(weights), weights.length * (long)Sizeof.FLOAT, stream);

        /**
         * Make sure we initialize the array to 1
         * important for including bias nodes
         */
        JCudaDriver.cuMemsetD32Async(resultGPU, Float.floatToIntBits(1.0f), links + 1, stream);

        gpuReady = true;
    }
    
    /**
     * ptr size must be equal to totalNeurons
     * @param ptr
     * @param stream
     * @return 
     */
    @Override
    protected CUdeviceptr feedForwardGPU(CUdeviceptr ptr, CUstream stream) {
        if(activation != null) {
            if(condition == null)
                activation.computeGPU(ptr, neurons, stream);
            else
                activation.computeGPUConditional(ptr, conditionGPU, neurons, stream);
        }
        
        /* output layer */
        if(isOutput)
            return ptr;
        
        int deviceId = CudaThread.getThreadDeviceId();
        
        /* Compute Matrix Multiplication */
        CUfunction matrixMulVector = CudaEngine.getKernel(CudaModule.MODULE_MATRIX, "matrix_mul_vector", deviceId);
                        
        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(deviceId), totalNeurons);
        int gridSizeX = links;

        Pointer kernelParameters = Pointer.to(
            Pointer.to(weightsGPU),
            Pointer.to(ptr),
            Pointer.to(resultGPU),
            Pointer.to(new long[]{totalNeurons})
        );

        JCudaDriver.cuLaunchKernel(matrixMulVector,
            gridSizeX, 1, 1,        // Grid dimension
            blockSizeX, 1, 1,       // Block dimension
            0, stream,              // Shared memory size and stream
            kernelParameters, null  // Kernel- and extra parameters
        );
        
        return resultGPU;        
    }
    
    @Override
    public void freeCPU() {
        weights = null;
    }
    
    @Override
    public void freeGPU() {
        if(isOutput)
            return;
        
        if(weightsGPU != null)
            JCudaDriver.cuMemFree(weightsGPU);
        
        if(conditionGPU != null)
            JCudaDriver.cuMemFree(conditionGPU);    
    
        if(resultGPU != null)
            JCudaDriver.cuMemFree(resultGPU);
        
        weightsGPU = null;
        conditionGPU = null;
        resultGPU = null;
        gpuReady = false;
    }

    @Override
    protected LayerStub getStub() {
        float[][] copy = new float[links][neurons];
        
        for(int i=0; i < links; i++)
            System.arraycopy(this.weights, i*totalNeurons, copy[i], 0, neurons);
        
        float[] biases = new float[links];
        
        for(int i=0; i < links; i++)
            biases[i] = this.weights[totalNeurons * i + neurons];
        
        return new LayerStub(neurons, copy, activation, hasBias, biases);
    }
}
