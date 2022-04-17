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
package org.fjnn.network_old;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.activation.Activation;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaModule;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.util.Rng;

/**
 *
 * @author ahmed
 */
public class ConcurrentLayer extends Layer {

    /**
     * Store everything in 1D array
     * [N(0-0), N(1-0), ... N(1-0) N(1-1) ....
     */
    float[] weights;

    /* separate biases in a separate array */
    float[] biases;
    
    /* weights stored on GPU */
    CUdeviceptr weightsGPU;
    CUdeviceptr biasesGPU;    
        
    /* activation conditions on GPU */
    CUdeviceptr conditionGPU;

    /* can we call computeGPU */
    protected boolean gpuReady;
    
    /* unconnected layer / output layer */
    public ConcurrentLayer(Activation activation, int neurons) {
        this(activation, neurons, 0, false, null);
    }
    
    public ConcurrentLayer(Activation activation, int neurons, int links, boolean hasBias, boolean[] condition) {
        super(activation, neurons, links, hasBias, condition);

        this.weights = new float[neurons * links];
        this.biases  = new float[links];
    }
    
    @Override
    protected float getWeight(int from, int to) {
        if(from < 0 || from >= neurons)
            throw new ArrayIndexOutOfBoundsException();
        
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            
        
        return weights[from + neurons * to];        
    }

    @Override
    protected void setWeight(int from, int to, float value) {
        if(from < 0 || from >= neurons)
            throw new ArrayIndexOutOfBoundsException();
        
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            
        
        weights[from + neurons * to] = value;
        
        gpuReady = false;
    }

    @Override
    protected float getBias(int to) {
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            

        return biases[to];
    }

    @Override
    protected void setBias(int to, float value) {
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            

        if(hasBias)
            biases[to] = value;
        
        gpuReady = false;
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
            int k = i * neurons;
            
            /* neurons */
            for(int j=0; j < neurons; j++)
                sum += input[j] * weights[k + j];
            
            /* bias */
            sum += biases[i];
            
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
        
        if(weightsGPU != null)
            JCudaDriver.cuMemFree(weightsGPU);
        
        if(biasesGPU != null)
            JCudaDriver.cuMemFree(biasesGPU);

        if(condition != null && conditionGPU == null) {
            conditionGPU = new CUdeviceptr();
            JCudaDriver.cuMemAlloc(conditionGPU, condition.length * (long) Sizeof.BYTE);

            byte[] temp = new byte[condition.length];
            
            for(int i=0; i < condition.length; i++)
                temp[i] = condition[i] ? (byte)1 : 0;
            
            JCudaDriver.cuMemcpyHtoDAsync(conditionGPU, Pointer.to(temp), condition.length * (long)Sizeof.BYTE, stream);
        }
                
        weightsGPU = CudaUtil.toGPU(weights, stream);
        biasesGPU = CudaUtil.toGPU(biases, stream);
        
        gpuReady = true;
    }
    

    @Override
    protected CUdeviceptr feedForwardGPU(CUdeviceptr ptr, CUstream stream) {
        if(activation != null) {
            if(condition == null)
                activation.computeGPU(ptr, neurons, stream);
            else
                activation.computeGPUConditional(ptr, conditionGPU, neurons, stream, 1);
        }

        
        /* output layer */
        if(isOutput)
            return ptr;
        
        int deviceId = CudaEngine.getThreadDeviceId();

        CUdeviceptr result = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(result, links * (long)Sizeof.FLOAT);
        
        /* Compute Matrix Multiplication */
        CUfunction matrixMulVector = CudaEngine.getKernel(CudaModule.MODULE_MATRIX, "matrix_mul_vector", deviceId);
                        
        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(deviceId), neurons);
        int gridSizeX = links;

        Pointer kernelParameters = Pointer.to(
            Pointer.to(weightsGPU),
            Pointer.to(ptr),
            Pointer.to(result),
            Pointer.to(new long[]{neurons})
        );

        JCudaDriver.cuLaunchKernel(matrixMulVector,
            gridSizeX, 1, 1,        // Grid dimension
            blockSizeX, 1, 1,       // Block dimension
            0, stream,              // Shared memory size and stream
            kernelParameters, null  // Kernel- and extra parameters
        );
        
        /* Add Bias */
        CUfunction Accumulator = CudaEngine.getKernel(CudaModule.MODULE_MATRIX, "accumulate_vector", deviceId);
        
        blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(deviceId), links);
        gridSizeX = (links - 1) / blockSizeX + 1;

        kernelParameters = Pointer.to(
            Pointer.to(result),
            Pointer.to(biasesGPU),
            Pointer.to(new long[]{links})
        );

        JCudaDriver.cuLaunchKernel(Accumulator,
            gridSizeX, 1, 1,        // Grid dimension
            blockSizeX, 1, 1,       // Block dimension
            0, stream,              // Shared memory size and stream
            kernelParameters, null  // Kernel- and extra parameters
        );
        
        return result;        
    }
 
    @Override
    public void freeCPU() {
        weights = null;
        biases = null;
    }
    
    @Override
    public void freeGPU() {
        if(isOutput)
            return;
        
        JCudaDriver.cuMemFree(weightsGPU);
        JCudaDriver.cuMemFree(biasesGPU);
        
        weightsGPU = null;
        biasesGPU = null;
        gpuReady = false;
    }    

    @Override
    public void setWeights(float[][] values) {
        for(int i=0; i < links; i++)
            System.arraycopy(values[i], 0, this.weights, i*neurons, neurons);

        gpuReady = false;
    }

    @Override
    public void setBiases(float[] values) {
        gpuReady = false;
        
        if(!hasBias)
            return;
        
        System.arraycopy(values, 0, biases, 0, biases.length);
    }

    @Override
    protected LayerStub getStub() {
        float[][] weights = new float[links][neurons];
        
        for(int i=0; i < links; i++)
            System.arraycopy(this.weights, i*neurons, weights[i], 0, neurons);
        
        return new LayerStub(neurons, weights, activation, hasBias, biases, condition);
    }

    @Override
    public void randomize(float min, float max) {
        for(int i=0; i < weights.length; i++)
            weights[i] = Rng.nextFloat(min, max);
        
        if(hasBias)
            for(int i=0; i < links; i++)
                biases[i] = Rng.nextFloat(min, max);
    }

    @Override
    protected float[][] getWeights() {
        float[][] result = new float[links][neurons];
        
        for(int i=0; i < links; i++)
            System.arraycopy(weights, i*neurons, result[i], 0, neurons);

        return result;
    }

    @Override
    protected float[] getBiases() {
        float[] result = new float[links];
        
        System.arraycopy(biases, 0, result, 0, links);
        
        return result;
    }
}
