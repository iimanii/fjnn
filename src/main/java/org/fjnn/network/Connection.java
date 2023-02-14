/*
 * The MIT License
 *
 * Copyright 2022 ahmed.
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
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.runtime.cudaStream_t;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;
import static org.fjnn.cuda.CudaUtil.FLOAT_SIZE;
import org.fjnn.util.Rng;
import org.fjnn.util.intrinsic;
import org.fjnn.util.util;
import org.json.JSONObject;
import run.timer;

/**
 *
 * @author ahmed
 */
public class Connection {
    final public int neurons;
    final public int links;
    
    /**
     * Store everything in 1D array
     * [N(0-0) N(0-1) N(0-2) ... N(1-0) N(1-1) ....
     */
    float[] weights;

    /* bias in a separate array */
    float[] biases;
    
    /* weights to be used with native */
    FloatBuffer weightsCPU;
    
    /* bias to be used with native */
    FloatBuffer biasCPU;
            
    /* weights stored on GPU */
    CUdeviceptr weightsGPU;

    /* bias stored on GPU */
    CUdeviceptr biasesGPU;
    
    /* for mutation */
    CUdeviceptr rngPool;
    
    /* true if weights were loaded to native memory */
    boolean nativeReady;
    
    boolean gpuReady;
    
    boolean disableBias;

    Connection(int neurons, int links) {
        this(neurons, links, true);
    }
    
    Connection(int neurons, int links, boolean creatWeights) {
        this.neurons = neurons;
        this.links = links;
        
        if(creatWeights) {
            this.weights = new float[neurons * links];
            this.biases = new float[links];
        }
    }
    
    private Connection(int neurons, int links, float[] weights, float[] biases) {
        this.neurons = neurons;
        this.links = links;
        this.weights = weights;
        this.biases = biases;
        
        if(this.biases.length != links || this.weights.length != links * neurons)
            throw new RuntimeException("Inconsistent connection");
    }
    
    Connection copy(boolean copyWeights, boolean creatWeights) {
        if(!copyWeights)
            return new Connection(neurons, links, creatWeights);
        
        float[] wc = Arrays.copyOf(weights, weights.length);
        float[] bc = Arrays.copyOf(biases, biases.length);
        
        return new Connection(neurons, links, wc, bc);
    }
    
    void feedForward(float[] input, float[] result) {
        for(int i=0; i < neurons; i++) {
            int k = i * links;
            
            /* neurons */
            for(int j=0; j < links; j++)
                result[j] += input[i] * weights[k + j];
        }
        
        for(int i=0; i < links; i++) {
            result[i] += biases[i];
        }
    }
    
    void feedForward(float[] input, int count, float[] result) {
        for(int c=0; c < count; c++) {
            int x = c * neurons;
            int y = c * links;
            
            for(int i=0; i < neurons; i++) {
                int k = i * links;

                /* neurons */
                for(int j=0; j < links; j++)
                    result[y + j] += input[x + i] * weights[k + j];
            }

            for(int i=0; i < links; i++) {
                result[y + i] += biases[i];
            }
        }
    }
    
    void feedForward(FloatBuffer input, FloatBuffer result) {
        intrinsic.sgemv(input, result, weightsCPU, biasCPU, neurons, links);
    }
    
    void feedForward(FloatBuffer input, int count, FloatBuffer result) {
        intrinsic.sgemm(input, count, result, weightsCPU, biasCPU, neurons, links);
    }
    
    void feedForwardGPU(CUdeviceptr input, CUdeviceptr result, CUstream stream, cublasHandle handle) {
        /* add bias to current result accumulator */
        CudaFunctions.addStride(result, biasesGPU, links, 1, stream);
        
        Pointer p = Pointer.to(new float[]{1.0f});
        
        /* NOTE: cublas uses column-major format */
        int m = links;
        int n = neurons;
        
        CUdeviceptr a = weightsGPU;
        CUdeviceptr x = input;        
        CUdeviceptr y = result;

        cudaStream_t stream0 = new cudaStream_t(stream);
        synchronized(handle) {
            JCublas2.cublasSetStream(handle, stream0);
            /* Compute Vector Matrix Multiplication */
            JCublas2.cublasSgemv(handle, cublasOperation.CUBLAS_OP_N, m, n, p, a, m, x, 1, p, y, 1);
        }
    }
    
    void feedForwardGPU(CUdeviceptr input, int count, CUdeviceptr result, CUstream stream, cublasHandle handle) {
        /* add bias to current result accumulator */
        CudaFunctions.addStride(result, biasesGPU, links, count, stream);
        
        Pointer p = Pointer.to(new float[]{1.0f});
        
        /* NOTE: cublas uses column-major format */
        int m = links;
        int n = count;
        int k = neurons;
        
        CUdeviceptr a = weightsGPU;
        CUdeviceptr b = input;        
        CUdeviceptr c = result;

        synchronized(handle) {
            JCublas2.cublasSetStream(handle, new cudaStream_t(stream));
            
            /* Compute Matrix Multiplication */
            JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, m, n, k, p, a, m, b, k, p, c, m);
        }
    }
    
    void prepareCPU() {
        weightsCPU = ByteBuffer.allocateDirect(weights.length * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
        weightsCPU.put(weights);
        
        biasCPU = ByteBuffer.allocateDirect(biases.length * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
        biasCPU.put(biases);
        
        nativeReady = true;
    }

    void ensureCPU(CUstream stream) {
        if(weights == null && weightsGPU != null)
            weights = CudaUtil.fromGPUFloat(weightsGPU, neurons * links, stream);
        
        if(biases == null && biasesGPU != null)
            biases = CudaUtil.fromGPUFloat(biasesGPU, links, stream);
    }
    
    long getMemoryRequirements() {
        return (neurons * links + links) * FLOAT_SIZE;
    }
    
    void prepareGPU(CUstream stream) {                
        initGPUWeights();
        
        JCudaDriver.cuMemcpyHtoDAsync(weightsGPU, Pointer.to(weights), weights.length * FLOAT_SIZE, stream);
        JCudaDriver.cuMemcpyHtoDAsync(biasesGPU, Pointer.to(biases), biases.length * FLOAT_SIZE, stream);

        gpuReady = true;
    }

    void freeGPU(int deviceId) {
        if(weightsGPU != null)
            CudaEngine.freeMempool(deviceId, weightsGPU);
        
//        if(biasesGPU != null)
//            JCudaDriver.cuMemFree(biasesGPU);
        
        weightsGPU = null;
        biasesGPU = null;
        
        freeGPURng();
        
        gpuReady = false;
    }
    
    void freeGPURng() {
        if(rngPool != null)
            CudaEngine.freeMempool(rngPool);
        
        rngPool = null;
    }
    
    private void initGPUWeights() {
        if(weightsGPU != null || biasesGPU != null)
            throw new RuntimeException("GPU already initialized for connection");
        
        long lengthWeights = CudaUtil.alignLength(neurons * links, CudaUtil.DEFAULT_MEM_ALIGN);
        long lengthBias = links;
        long lengthTotal = lengthWeights + lengthBias;
        
        weightsGPU = CudaEngine.getMempoolFloat(lengthTotal);
        biasesGPU = weightsGPU.withByteOffset(lengthWeights * CudaUtil.FLOAT_SIZE);
//        biasesGPU = //CudaUtil.createFloat(lengthBias);        
    }
    
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

    public void setWeight(int from, int to, float value) {
        if(from < 0 || from >= neurons)
            throw new ArrayIndexOutOfBoundsException();
        
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();
        
        weights[from * links + to] = value;

        nativeReady = false;
        gpuReady = false;
    }
    
    public void setWeights(float[] values) {
        if(values.length != weights.length)
            throw new RuntimeException();
        
        weights = Arrays.copyOf(values, values.length);
        
        nativeReady = false;
        gpuReady = false;
    }
    
    public void setWeights2D(float[][] values) {
        for(int i=0; i < neurons; i++)
            for(int j=0; j < links; i++)
                weights[i * links + j] = values[i][j];
        
        nativeReady = false;
        gpuReady = false;
    }

    public float getBias(int to) {
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();  
        
        return biases[to];
    }
    
    public float[] getBias() {
        return biases;
    }

    public void setBias(int to, float value) {
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            

        biases[to] = value;
        
        nativeReady = false;
        gpuReady = false;
    }
    
    public void setBias(float[] values) {
        if(values.length != links)
            throw new RuntimeException();

        System.arraycopy(values, 0, biases, 0, links);

        nativeReady = false;
        gpuReady = false;
    } 

    public void randomize(float min, float max) {
        int len = neurons * links;
        
        for(int i=0; i < len; i++)
            weights[i] = Rng.nextFloat(min, max);
        
        for(int i=0; i < links; i++)
            biases[i] = Rng.nextFloat(min, max);
        
        nativeReady = false;
        gpuReady = false;
    }

    void freeCPU() {
        weights = null;
        biases = null;
        weightsCPU = null;
        biasCPU = null;
    }

    void crossOverMutate(Connection a, Connection b, float min, float max, double mutation) {
        float[] wa = a.weights;
        float[] wb = b.weights;

        for(int j=0; j < wa.length; j++) {
            float w = Rng.nextBoolean() ? wa[j] : wb[j];

            if(Rng.nextDouble() < mutation)
                w = w + Rng.nextFloat(min, max);

            weights[j] = w;
        }
        
        float[] ba = a.biases;
        float[] bb = b.biases;

        for(int j=0; j < ba.length; j++) {
            float w = Rng.nextBoolean() ? ba[j] : bb[j];

            if(Rng.nextDouble() < mutation)
                w = w + Rng.nextFloat(min, max);

            biases[j] = w;
        }

        nativeReady = false;
        gpuReady = false;
    }

    void crossOverMutateGPU(Connection a, Connection b, float min, float max, double mutation, boolean nocopy, CUstream stream, curandGenerator generator) {
        /* mutate weights */
        initGPUWeights();
        
        int lengthWeights = (int) CudaUtil.alignLength(neurons * links, CudaUtil.DEFAULT_MEM_ALIGN);
        int lengthBias = links;
        int lengthTotal = lengthWeights + lengthBias;

        rngPool = CudaEngine.getMempoolFloat(lengthTotal * 3);
        CUdeviceptr rngMutate = rngPool.withByteOffset(lengthTotal * CudaUtil.FLOAT_SIZE);
        CUdeviceptr rngCrossover = rngPool.withByteOffset(2 * lengthTotal * CudaUtil.FLOAT_SIZE);
        
        synchronized(generator) {
            JCurand.curandSetStream(generator, new cudaStream_t(stream));
            JCurand.curandGenerateUniform(generator, rngPool, lengthTotal * 3);
        }
        
        /* mutate weights and biases in one kernel launch */
        CudaFunctions.crossoverMutate(a.weightsGPU, b.weightsGPU, weightsGPU, lengthTotal, min, max, mutation, 
                                      rngCrossover, rngMutate, rngPool, CudaUtil.PREFERRED_BLOCK_SIZE, stream);
        
        
//        CudaFunctions.crossoverMutate(a.biasesGPU, b.biasesGPU, biasesGPU, links, min, max, mutation, 
//                                      rngCrossover.withByteOffset(lengthWeights * CudaUtil.FLOAT_SIZE), 
//                                      rngMutate.withByteOffset(lengthWeights * CudaUtil.FLOAT_SIZE), 
//                                      rngPool.withByteOffset(lengthWeights * CudaUtil.FLOAT_SIZE), 
//                                      stream);
        if(!nocopy) {
            weights = CudaUtil.fromGPUFloat(weightsGPU, neurons * links, stream);
            biases = CudaUtil.fromGPUFloat(biasesGPU, links, stream);
        }
        
        nativeReady = false;
        gpuReady = true;
    }
    
    JSONObject serialize() {
        JSONObject result = new JSONObject();

        result.put("neurons", neurons);
        result.put("links", links);
        
        result.put("weights", util.base64encode(util.compress(util.toByteArray(weights))));
        result.put("biases", util.base64encode(util.compress(util.toByteArray(biases))));
        
        return result;
    }
    
    static Connection deserialize(JSONObject obj) {
        int neurons = obj.getInt("neurons");
        int links = obj.getInt("links");
        
        float[] weights = util.toFloatArray(util.decompress(util.base64decode(obj.getString("weights"))));
        float[] biases = util.toFloatArray(util.decompress(util.base64decode(obj.getString("biases"))));
        
        return new Connection(neurons, links, weights, biases);
    }
    
    float compare(Connection c) {
        float score = 0;
        
        float[] wa = c.getWeights();

        for(int j=0; j < wa.length; j++)            
            score += Math.abs(wa[j] - weights[j]);

        float[] ba = c.getBias();

        for(int j=0; j < ba.length; j++)
            score += Math.abs(ba[j] - biases[j]);            
        
        return score;
    }

    public boolean hasBias() {
        return !disableBias;
    }

    void copyWeights(Connection connection) {
        weights = Arrays.copyOf(connection.weights, connection.weights.length);
        biases = Arrays.copyOf(connection.biases, connection.biases.length);
        
        this.nativeReady = false;
        this.gpuReady = false;
    }
}
