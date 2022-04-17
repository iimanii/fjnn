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
import org.fjnn.base.BaseLayer;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaModule;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.util.Rng;

/**
 *
 * @author ahmed
 * 
 * Not thread safe, but 10% faster
 */
public class Layer extends BaseLayer {
    /**
     * Store everything in 1D array
     * [N(0-0) N(0-1) N(0-2) ... N(1-0) N(1-1) ....
     */
    float[] weights;

    /* separate bias in a separate array */
    float[] bias;
    
    /* weights stored on GPU */
    CUdeviceptr weightsGPU;

    /* bias stored on GPU */
    CUdeviceptr biasesGPU;

    /* activation conditions on GPU */
    CUdeviceptr conditionGPU;
    
    /* can we call computeGPU */
    protected boolean gpuReady;
        
    public Layer(LayerStub stub, int links) {
        super(stub.activation, stub.neurons, links, stub.hasBias, stub.condition);
        
        if(stub.weights != null)
            this.weights = stub.weights;
        else
            this.weights = new float[neurons * links];

        
        if(stub.biases != null)
            this.bias = stub.biases;
        else
            this.bias = new float[links];
        
        gpuReady = false;
    }
    
    /**
     * 
     * @param from
     * @param to
     * @return 
     */
    public float getWeight(int from, int to) {
        if(from < 0 || from >= neurons)
            throw new ArrayIndexOutOfBoundsException();
        
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            
        
        return weights[from * links + to];
    }
    
    public float[] getWeights() {
        return weights;
    }
    
    public float[][] getWeights2D() {
        float[][] result = new float[neurons][links];
        
        for(int i=0; i < neurons; i++)
            for(int j=0; j < links; j++)
                result[i][j] = weights[i * neurons + j];
        
        return result;
    }

    /**
     * 
     * @param from
     * @param to
     * @param value 
     */
    public void setWeight(int from, int to, float value) {
        if(from < 0 || from >= neurons)
            throw new ArrayIndexOutOfBoundsException();
        
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();
        
        weights[from * links + to] = value;

        gpuReady = false;
    }
    
    public void setWeights(float[] values) {
        if(values.length != weights.length)
            throw new RuntimeException();
        
        weights = Arrays.copyOf(values, values.length);
        
        gpuReady = false;
    }
    
    public void setWeights2D(float[][] values) {
        for(int i=0; i < neurons; i++)
            for(int j=0; j < links; i++)
                weights[i * links + j] = values[i][j];
        
        gpuReady = false;
    }

    public float getBias(int to) {
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();  
        
        return bias[to];
    }
    
    public float[] getBias() {
        return bias;
    }

    public void setBias(int to, float value) {
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            

        if(hasBias)
            bias[to] = value;
        
        gpuReady = false;
    }
    
    public void setBias(float[] values) {
        if(hasBias) {
            if(values.length != links)
                throw new RuntimeException();
            
            System.arraycopy(values, 0, bias, 0, links);
        }

        gpuReady = false;
    } 

    @Override
    public void randomize(float min, float max) {
        int len = neurons * links;
        
        for(int i=0; i < len; i++)
            weights[i] = Rng.nextFloat(min, max);
        
        if(hasBias)
            for(int i=0; i < links; i++)
                bias[i] = Rng.nextFloat(min, max);
    }

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

        for(int i=0; i < neurons; i++) {
            int k = i * links;
            
            /* neurons */
            for(int j=0; j < links; j++)
                output[j] += input[i] * weights[k + j];
        }
        
        if(hasBias)
            for(int i=0; i < links; i++)
                output[i] += bias[i];

//        for(int i=0; i < output.length; i++) {
//            System.out.print(String.format("%.3f ", output[i]));
//        }
//        
//        System.out.println();
        
        return output;
    }
    
    @Override
    protected void prepareGPU(CUstream stream) {
        if(gpuReady)
            return;
        
        if(!isOutput) {
            if(weightsGPU == null)
               weightsGPU = CudaUtil.create(weights.length);

            if(biasesGPU == null && hasBias)
               biasesGPU = CudaUtil.create(bias.length);

            JCudaDriver.cuMemcpyHtoDAsync(weightsGPU, Pointer.to(weights), weights.length * (long) Sizeof.FLOAT, stream);

            if(hasBias)
                JCudaDriver.cuMemcpyHtoDAsync(biasesGPU, Pointer.to(bias), bias.length * (long) Sizeof.FLOAT, stream);
        }

        if(condition != null) {
            if(conditionGPU == null)
               conditionGPU = CudaUtil.createBytes(condition.length);

            byte[] temp = new byte[condition.length];

            for(int i=0; i < condition.length; i++)
                temp[i] = condition[i] ? (byte)1 : 0;
            
            JCudaDriver.cuMemcpyHtoDAsync(conditionGPU, Pointer.to(temp), condition.length, stream);
        }

        gpuReady = true;
    }
    
    protected CUdeviceptr feedForwardGPU(CUdeviceptr ptr, CUstream stream, cublasHandle handle) {
        if(activation != null) {
            if(condition == null)
                activation.computeGPU(ptr, neurons, stream);
            else
                activation.computeGPUConditional(ptr, conditionGPU, neurons, stream, 1);
        }
        
        /* output layer */
        if(isOutput)
            return ptr;
        
        long biasSize = links * (long) Sizeof.FLOAT;
        CUdeviceptr resultGPU = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(resultGPU, biasSize);
        
        if(hasBias)
            JCudaDriver.cuMemcpyDtoDAsync(resultGPU, biasesGPU, biasSize, stream);

        /* NOTE: cublas uses column-major format */
        int row_a = links;
        int col_a = neurons;
        CUdeviceptr d_A = weightsGPU;

        CUdeviceptr d_B = ptr;        
        CUdeviceptr d_C = resultGPU;
        
        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{hasBias ? 1.0f : 0.0f});

        /* Compute Vector Matrix Multiplication */
        JCublas2.cublasSgemv(handle, cublasOperation.CUBLAS_OP_N,
                            row_a, col_a,
                            pAlpha, d_A, row_a, 
                            d_B, 1, 
                            pBeta, d_C, 1);
        
        return resultGPU;   
    }
    
    protected CUdeviceptr feedForwardGPU(CUdeviceptr ptr, CUstream stream, cublasHandle handle, int count) {
        if(activation != null) {
            if(condition == null)
                activation.computeGPU(ptr, neurons * count, stream);
            else
                activation.computeGPUConditional(ptr, conditionGPU, neurons, stream, count);
        }
        
        /* output layer */
        if(isOutput)
            return ptr;
        
        long biasSize = links * (long) Sizeof.FLOAT;
        CUdeviceptr resultGPU = new CUdeviceptr();
        JCudaDriver.cuMemAlloc(resultGPU, count * biasSize);
        
        if(hasBias)
            for(int i=0; i < count; i++)
                JCudaDriver.cuMemcpyDtoDAsync(resultGPU.withByteOffset(i * biasSize), biasesGPU, biasSize, stream);

        /* NOTE: cublas uses column-major format */
        int row_a = links;
        int col_a = neurons;
        CUdeviceptr d_A = weightsGPU;

        int row_b = neurons;
        int col_b = count;
        CUdeviceptr d_B = ptr;
        
        int row_c = links;
        CUdeviceptr d_C = resultGPU;
        
        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{hasBias ? 1.0f : 0.0f});

        /* Compute Matrix Multiplication */
        JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, 
                            row_a, col_b, col_a, 
                            pAlpha, d_A, row_a, 
                            d_B, row_b, 
                            pBeta, d_C, row_c);
        
        return resultGPU;
    }
    
    @Override
    protected void freeCPU() {
        weights = null;
    }
    
    @Override
    protected void freeGPU() {
        if(isOutput)
            return;
        
        if(weightsGPU != null)
            JCudaDriver.cuMemFree(weightsGPU);
        
        if(conditionGPU != null)
            JCudaDriver.cuMemFree(conditionGPU);    
        
        if(biasesGPU != null)
            JCudaDriver.cuMemFree(biasesGPU);
        
        weightsGPU = null;
        conditionGPU = null;
        gpuReady = false;
    }

    protected LayerStub getStub(boolean withWeights) {
        float[] w = weights != null && withWeights ? Arrays.copyOf(weights, weights.length) : null;
        float[] b = bias != null && withWeights ? Arrays.copyOf(bias, bias.length) : null;
        boolean[] c = condition != null ? Arrays.copyOf(condition, condition.length) : null;
        
        return new LayerStub(neurons, w, activation, hasBias, b, c);
    }

    protected void crossOverMutate(Layer a, Layer b, double min, double max, double mutation) {
        float[] wa = a.getWeights();
        float[] wb = b.getWeights();

        for(int j=0; j < wa.length; j++) {
            float w = Rng.nextBoolean() ? wa[j] : wb[j];

            if(Rng.nextDouble() < mutation)
                w = w + (float) Rng.nextDoubleGaussian(min, max);

            weights[j] = w;
        }
        
        if(!hasBias)
            return;
        
        float[] ba = a.getBias();
        float[] bb = b.getBias();

        for(int j=0; j < ba.length; j++) {
            float w = Rng.nextBoolean() ? ba[j] : bb[j];

            if(Rng.nextDouble() < mutation)
                w = w + (float) Rng.nextDouble(min, max);

            bias[j] = w;
        }
        
        gpuReady = false;
    }
    
    protected void crossOverMutate(Layer a, Layer b, Layer min, Layer max, double mutation) {
        float[] wa = a.getWeights();
        float[] wb = b.getWeights();
        float[] wmin = min.getWeights();
        float[] wmax = max.getWeights();

        for(int j=0; j < wa.length; j++) {
            float w = Rng.nextBoolean() ? wa[j] : wb[j];

            if(Rng.nextDouble() < mutation)
                w = w + (float) Rng.nextDoubleGaussian(wmin[j], wmax[j]);

            weights[j] = w;
        }
        
        if(!hasBias)
            return;
        
        float[] ba = a.getBias();
        float[] bb = b.getBias();
        float[] bmin = min.getBias();
        float[] bmax = max.getBias();

        for(int j=0; j < ba.length; j++) {
            float w = Rng.nextBoolean() ? ba[j] : bb[j];

            if(Rng.nextDouble() < mutation)
                w = w + (float) Rng.nextDoubleGaussian(bmin[j], bmax[j]);

            bias[j] = w;
        }
        
        gpuReady = false;
    }
    
    public static class crossOverMutateResult {
        public int forcePick_A;
        public int forcePick_B;
        public int randomPick_A;
        public int randomPick_B;
    }
    
    protected void crossOverMutate(Layer a, Layer b, Layer minA, Layer maxA, Layer minB, Layer maxB, double mutation, crossOverMutateResult r) {
        float[] wa = a.getWeights();
        float[] wb = b.getWeights();
        float[] wminA = minA.getWeights();
        float[] wmaxA = maxA.getWeights();
        float[] wminB = minB.getWeights();
        float[] wmaxB = maxB.getWeights();

        for(int j=0; j < wa.length; j++)
            weights[j] = crossOverMutateCalculateWeight(wa[j], wb[j], wminA[j], wmaxA[j], wminB[j], wmaxB[j], mutation, r);
        
        if(!hasBias)
            return;
        
        float[] ba = a.getBias();
        float[] bb = b.getBias();
        float[] bminA = minA.getBias();
        float[] bmaxA = maxA.getBias();
        float[] bminB = minB.getBias();
        float[] bmaxB = maxB.getBias();

        for(int j=0; j < ba.length; j++)
            bias[j] = crossOverMutateCalculateWeight(ba[j], bb[j], bminA[j], bmaxA[j], bminB[j], bmaxB[j], mutation, r);
        
        gpuReady = false;
    }
    
    protected void crossOverMutateGPU(Layer a, Layer b, double min, double max, double mutation, CUstream stream, curandGenerator generator) {
        int deviceId = CudaEngine.getThreadDeviceId();
        int size = neurons * links;
        int bias = hasBias ? links : 0;
        float mean = (float) (max + min) / 2.0f;
        float stdDev = (float) (max - min) / 10.0f;
        
        if(size == 0)
            return;
        
        CUdeviceptr uniform = CudaUtil.create((size + bias) * 2);
        CUdeviceptr gaussian = CudaUtil.create((size + bias));

        JCurand.curandGenerateUniform(generator, uniform, (size + bias) * 2);
        JCurand.curandGenerateNormal(generator, gaussian, size + bias, mean, stdDev);
            
        CUfunction function = CudaEngine.getKernel(CudaModule.MODULE_GENETIC, "crossOverMutate", deviceId);

        int blockSizeX = Math.min(CudaEngine.getMaxThreadsPerBlock(deviceId), size);
        int gridSizeX = (size - 1) / blockSizeX + 1;

        Pointer kernelParameters = Pointer.to(
            Pointer.to(a.weightsGPU),
            Pointer.to(b.weightsGPU),
            Pointer.to(weightsGPU),
            Pointer.to(new long[]{size}),
            Pointer.to(new double[]{mutation}),
            Pointer.to(uniform),
            Pointer.to(uniform.withByteOffset(size * (long) Sizeof.FLOAT)),
            Pointer.to(gaussian)
        );        
                
        JCudaDriver.cuLaunchKernel(function,
            gridSizeX, 1, 1,       // Grid dimension
            blockSizeX, 1, 1,      // Block dimension
            0, stream,             // Shared memory size and stream
            kernelParameters, null // Kernel- and extra parameters
        );
        
        weights = CudaUtil.fromGPU(weightsGPU, size, stream);
        
        if(hasBias) {
            kernelParameters = Pointer.to(
                Pointer.to(a.biasesGPU),
                Pointer.to(b.biasesGPU),
                Pointer.to(biasesGPU),
                Pointer.to(new long[]{bias}),
                Pointer.to(new double[]{mutation}),
                Pointer.to(uniform.withByteOffset((size * 2) * (long) Sizeof.FLOAT)),
                Pointer.to(uniform.withByteOffset((size * 2 + bias) * (long) Sizeof.FLOAT)),
                Pointer.to(gaussian.withByteOffset(size * (long) Sizeof.FLOAT))
            );

            JCudaDriver.cuLaunchKernel(function,
                gridSizeX, 1, 1,       // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, stream,             // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
            );

            this.bias = CudaUtil.fromGPU(biasesGPU, bias, stream);
        }
        
        JCudaDriver.cuMemFree(uniform);
        JCudaDriver.cuMemFree(gaussian);
    }

    protected float compare(Layer l) {
        float score = 0;
        float[] wa = l.getWeights();

        for(int j=0; j < wa.length; j++)            
            score += Math.abs(wa[j] - weights[j]);

        float[] ba = l.getBias();

        for(int j=0; j < ba.length; j++)
            score += Math.abs(ba[j] - bias[j]);            
        
        return score;
    }
    
    protected float compareGPU(Layer l, CUstream stream) {        
        int size = neurons * links;
        
        if(size == 0)
            return 0;

        float sum_weights = CudaUtil.sum_abs_differenceGPU(weightsGPU, l.weightsGPU, size, stream);
        float sum_biases = 0;
        
        if(hasBias)
           sum_biases = CudaUtil.sum_abs_differenceGPU(biasesGPU, l.biasesGPU, links, stream);
        
        return sum_weights + sum_biases;
    }

    synchronized void addNeurons(int amount, float weight) {
        neurons += amount;
        
        weights = Arrays.copyOf(weights, neurons * links);
        if(condition != null)
            condition = Arrays.copyOf(condition, neurons);
        
        if(weight != 0) {
            int start = (neurons - amount) * links;
            for(int i=start; i < weights.length; i++) {
                weights[i] = weight;
            }
        }
    }

    synchronized void addLinks(int amount, float weight) {
        int prev = links;
        
        links += amount;
        
        float[] newWeights = new float[neurons * links];
        if(weight != 0)
            Arrays.fill(newWeights, weight);
        
        for(int i=0, j=0; j < newWeights.length; i+=prev, j+=links)
            System.arraycopy(weights, i, newWeights, j, prev);
        
        weights = newWeights;
        
        float[] newBias = new float[links];
        if(weight != 0)
            Arrays.fill(newBias, weight);
        
        System.arraycopy(bias, 0, newBias, 0, links-amount);
        bias = newBias;
    }
    
    synchronized void removeLinks(int amount) {
        int prev = links;
        
        links -= amount;
        
        float[] newWeights = new float[neurons * links];
        
        for(int i=0, j=0; j < newWeights.length; i+=prev, j+=links)
            System.arraycopy(weights, i, newWeights, j, links);
        
        weights = newWeights;
        bias = Arrays.copyOf(bias, links);
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
}
