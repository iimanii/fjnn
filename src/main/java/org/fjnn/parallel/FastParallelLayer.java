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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.cudaStream_t;
import org.fjnn.activation.Activation;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaEngine.CUdeviceptr2D;
import org.fjnn.cuda.CudaModule;
import org.fjnn.cuda.CudaThread;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.serializer.LayerStub;

/**
 *
 * @author ahmed
 * 
 * Not thread safe, but 10% faster
 */
public class FastParallelLayer extends ParallelLayer {
    /**
     * Each [layer] array
     * [N(0-0), N(1-0), ... N(B-0) N(1-0) N(1-1) ....
     */
    float[][] weights;

    /* weights stored on GPU */
    CUdeviceptr2D weightsGPU;

    /* condition activation */
    CUdeviceptr conditionGPU;
    
    /* results are calculated here */
    CUdeviceptr2D resultGPU;
    
    /* Number of neurons + 1 (bias) */
    final int totalNeurons;
    
    /* unconnected layer / output layer */
    public FastParallelLayer(int count, Activation activation, int neurons) {
        this(count, activation, neurons, 0, false);
    }
    
    public FastParallelLayer(int count, Activation activation, int neurons, int links, boolean hasBias) {
        this(count, activation, neurons, links, hasBias, null);
    }

    public FastParallelLayer(int count, Activation activation, int neurons, int links, boolean hasBias, boolean[] condition) {
        super(count, activation, neurons, links, hasBias, condition);
        
        this.totalNeurons = neurons + 1;

        this.weights = new float[count][totalNeurons * links];
    }
    
    @Override
    protected float getWeight(int index, int from, int to) {
        if(from < 0 || from >= neurons)
            throw new ArrayIndexOutOfBoundsException();
        
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            
        
        return weights[index][from + totalNeurons * to];        

    }

    @Override
    protected void setWeight(int index, int from, int to, float value) {
        if(from < 0 || from >= neurons)
            throw new ArrayIndexOutOfBoundsException();
        
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            
        
        weights[index][from + totalNeurons * to] = value;

        gpuReady = false;
    }

    @Override
    protected void setWeights(int index, float[][] values) {
        for(int i=0; i < links; i++)
            System.arraycopy(values[i], 0, this.weights[index], i * totalNeurons, neurons);
        
        gpuReady = false;
    }

    @Override
    protected  float[][] getWeights(int index) {
        float[][] result = new float[links][neurons];
        
        for(int i=0; i < links; i++)
            System.arraycopy(this.weights[index], i * totalNeurons, result[i], 0, neurons);
        
        return result;
    }
    
    @Override
    protected float getBias(int index, int to) {
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();  
        
        return weights[index][totalNeurons * to + neurons];
    }

    @Override
    protected void setBias(int index, int to, float value) {
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            

        if(hasBias)
            weights[index][totalNeurons * to + neurons] = value;
        
        gpuReady = false;
    }

    @Override
    protected void setBiases(int index, float[] values) {
        if(!hasBias)
            return;

        for(int i=0; i < links; i++)
            weights[index][totalNeurons * i + neurons] = values[i];

        gpuReady = false;
    }
    
    @Override
    public float[] getBiases(int index) {
        float[] result = new float[links];
        
        for(int i=0; i < links; i++)
            result[i] = weights[index][totalNeurons * i + neurons];
        
        return result;
    }
    

    @Override
    protected float[][] feedForward(float[][] input) {
        List<Future<float[]>> list = new ArrayList<>();
        
        for(int i=0; i < layersCount; i++) {
            int index = i;
            
            list.add(THREAD_POOL.submit(new Callable<float[]>() {
                @Override
                public float[] call() throws Exception {
                    if(activation != null) {
                        if(condition == null)
                            activation.compute(input[index]);
                        else
                            activation.computeConditional(input[index], condition);
                    }

                    /* output layer */
                    if(isOutput)
                        return input[index];

                    float[] output = new float[links];

                    for(int i=0; i < links; i++) {
                        double sum = 0;
                        int k = i * totalNeurons;

                        /* neurons */
                        for(int j=0; j < neurons; j++)
                            sum += input[index][j] * weights[index][k + j];

                        /* bias */
                        sum += weights[index][k + neurons];

                        output[i] = (float) sum;
                    }

                    return output;
                }
            }));
        }
        
        float[][] output = new float[layersCount][];
        
        for(int i=0; i < list.size(); i++) {
            try {
                output[i] = list.get(i).get();
            } catch (InterruptedException | ExecutionException ex) {
                throw new RuntimeException(ex);
            }
        }
        
        return output;
    }

    @Override
    protected CUdeviceptr2D feedForwardGPU(CUdeviceptr2D ptr, CUstream stream) {
        if(activation != null) {
            if(condition == null)
                activation.computeMultiGPU(ptr, neurons, layersCount, stream);
            else
                activation.computeMultiGPUConditional(ptr, conditionGPU, neurons, layersCount, stream);
        }
        
        /* output layer */
        if(isOutput)
            return ptr;
        
        int deviceId = CudaThread.getThreadDeviceId();
        
        /* Compute Matrix Multiplication */
        CUfunction multiMatrixMulVector = CudaEngine.getKernel(CudaModule.MODULE_MATRIX, "multi_matrix_mul_vector", deviceId);
                        
        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(deviceId), totalNeurons);
        int gridSizeX = links;
        int gridSizeY = layersCount;
        
        Pointer kernelParameters = Pointer.to(
            Pointer.to(weightsGPU.ptr),
            Pointer.to(new long[]{weightsGPU.pitch}),
            Pointer.to(ptr.ptr),
            Pointer.to(new long[]{ptr.pitch}),
            Pointer.to(resultGPU.ptr),
            Pointer.to(new long[]{resultGPU.pitch}),
            Pointer.to(new int[]{totalNeurons})
        );


        JCudaDriver.cuLaunchKernel(multiMatrixMulVector,
                gridSizeX, gridSizeY, 1,    // Grid dimension
                blockSizeX, 1, 1,           // Block dimension
                0, stream,                  // Shared memory size and stream
                kernelParameters, null      // Kernel- and extra parameters
        );
        
        return resultGPU;
    }
//
//    @Override
//    protected void randomize(float min, float max) {
//        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
//    }
//
    @Override
    public void prepareGPU(CUstream stream) {
        if(isOutput) {
            gpuReady = true;
            return;
        }
        
        int width = totalNeurons * links;
        int height = layersCount;
        
        if(weightsGPU == null)
            weightsGPU = CudaUtil.createPitch(width, height);
        
        if(condition != null && conditionGPU == null) {
            conditionGPU = new CUdeviceptr();
            JCudaDriver.cuMemAlloc(conditionGPU, condition.length * (long) Sizeof.BYTE);

            byte[] temp = new byte[condition.length];
            
            for(int i=0; i < condition.length; i++)
                temp[i] = condition[i] ? (byte)1 : 0;
            
            JCudaDriver.cuMemcpyHtoDAsync(conditionGPU, Pointer.to(temp), condition.length * (long)Sizeof.BYTE, stream);
        }

        if(resultGPU == null)
            resultGPU = CudaUtil.createPitch(links + 1, height);
        
        float[] temp = new float[height * width];

        for(int i=0; i < weights.length; i++)
            System.arraycopy(weights[i], 0, temp, i * width, width);
        
        JCuda.cudaMemcpy2DAsync(weightsGPU.ptr, weightsGPU.pitch * (long)Sizeof.FLOAT, 
                                Pointer.to(temp), width * (long) Sizeof.FLOAT,
                                width * (long)Sizeof.FLOAT, height, 
                                cudaMemcpyKind.cudaMemcpyHostToDevice, new cudaStream_t(stream));

        /**
         * Make sure we initialize the array to 1
         * important for including bias nodes
         */
        JCudaDriver.cuMemsetD2D32Async(resultGPU.ptr, resultGPU.pitch * (long)Sizeof.FLOAT, Float.floatToIntBits(1.0f), links + 1, height, stream);

        gpuReady = true;
    }
//
//    @Override
//    protected void freeCPU() {
//        weights = null;
//    }
//
//    @Override
//    protected void freeGPU() {
//        if(isOutput)
//            return;
//        
//        JCudaDriver.cuMemFree(weightsGPU.ptr);
//        JCudaDriver.cuMemFree(resultGPU.ptr);
//        
//        weightsGPU = null;
//        resultGPU = null;
//        gpuReady = false;
//    }
//
//    @Override
//    protected CudaEngine.CUdeviceptr2D feedForwardGPU(CudaEngine.CUdeviceptr2D ptr, CUstream stream) {
//        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
//    }

    @Override
    public void randomize(float min, float max) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void freeCPU() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void freeGPU() {
        if(isOutput)
            return;
        
        if(weightsGPU != null)
            JCudaDriver.cuMemFree(weightsGPU.ptr);

        if(conditionGPU != null)
            JCudaDriver.cuMemFree(conditionGPU);

        if(resultGPU != null)
            JCudaDriver.cuMemFree(resultGPU.ptr);
        
        weightsGPU = null;
        conditionGPU = null;
        resultGPU = null;
        gpuReady = false;
    }

    @Override
    protected LayerStub getStub(int index) {
        float[][] copy = new float[links][neurons];
        
        for(int i=0; i < links; i++)
            System.arraycopy(this.weights[index], i*totalNeurons, copy[i], 0, neurons);
        
        float[] biases = new float[links];
        
        for(int i=0; i < links; i++)
            biases[i] = this.weights[index][totalNeurons * i + neurons];
        
        return new LayerStub(neurons, copy, activation, hasBias, biases);
    }
}
