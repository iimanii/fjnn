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

import org.fjnn.network.gradient.ConnectionGradient;
import org.fjnn.network.gradient.ConnectionGradientGPU;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.runtime.cudaStream_t;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.util.rng;
import org.fjnn.util.intrinsic;
import org.fjnn.util.util;

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
    
    // Adam optimizer state
    private boolean useAdam = false;
    public float[] weightMomentum;         // m for weights [neurons * links]
    public float[] weightVelocity;         // v for weights [neurons * links] 
    public float[] biasMomentum;           // m for biases [links]
    public float[] biasVelocity;           // v for biases [links]
    public int adamTimeStep = 0;           // t for bias correction

    // Adam hyperparameters
    protected float beta1 = 0.9f;
    protected float beta2 = 0.999f; 
    protected float adamEpsilon = 1e-8f;

    // GPU Adam state
    public CUdeviceptr weightMomentumGPU;
    public CUdeviceptr weightVelocityGPU;
    public CUdeviceptr biasMomentumGPU;
    public CUdeviceptr biasVelocityGPU;
    
    public Connection(int neurons, int links) {
        this(neurons, links, true);
    }
    
    public Connection(int neurons, int links, boolean creatWeights) {
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
    
    void feedForward(float[] input, int batchSize, float[] result) {
        for(int c=0; c < batchSize; c++) {
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
    
    
    void feedForwardGPU(CUdeviceptr input, int batchSize, CUdeviceptr output, CUstream stream, cublasHandle handle) {
        Pointer alpha = Pointer.to(new float[]{1.0f});
        
        /* output might contain results from other connections .. make sure we accumulate */
        Pointer beta = Pointer.to(new float[]{1.0f});
        
        /* NOTE: cublas uses column-major format */
        int m = links;
        int n = batchSize;
        int k = neurons;
        
        CUdeviceptr a = weightsGPU;
        CUdeviceptr b = input;        
        CUdeviceptr c = output;

        synchronized(handle) {
            cudaStream_t cudaStream = stream == null ? null : new cudaStream_t(stream);
            JCublas2.cublasSetStream(handle, cudaStream);
            
            /* Compute Matrix Multiplication */
            if(batchSize == 1)
                JCublas2.cublasSgemv(handle, cublasOperation.CUBLAS_OP_N, m, k, alpha, a, m, b, 1, beta, c, 1);
            else {
                int status = JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, m, n, k, alpha, a, m, b, k, beta, c, m);
//                System.err.println("cublasSgemm status: " + status + " " + JCublas2.cublasGetStatusName(status));
            }
        }
        
        /* add bias */
        CudaFunctions.vector.addStride(output, biasesGPU, links, batchSize, stream);
    }
    
    ConnectionGradient backpropagate(float[] currentActivationDeltas, float[] nextPreActivationDeltas, float[] currentPostActivations, int batchSize) {
        ConnectionGradient gradient = new ConnectionGradient(neurons, links);
                
        for(int c=0; c < batchSize; c++) {
            int x = c * neurons;
            int y = c * links;
            
            // Loop through each neuron in the current layer (l) to calculate weight gradient and update delta
            for (int i = 0; i < neurons; i++) {
                int k = i * links;

                // Loop through each neuron in the next layer (l+1)
                for (int j = 0; j < links; j++) {
                    // Step 4: Weight Gradient
                    // Equation: grad_W^{(l)}[i,j] = delta^{(l+1)}_j * a^{(l)}_i
                    gradient.weightGradients[k + j] += nextPreActivationDeltas[y + j] * currentPostActivations[x + i];

                    // Step 5: Accumulate Pre-Activation Delta for the current layer (l)
                    // Equation: delta^{(l)}_i += delta^{(l+1)}_j * W^{(l)}[i,j]
                    currentActivationDeltas[x + i] += nextPreActivationDeltas[y + j] * weights[k + j];
                }
            }

            // Step 6: Bias Gradient (Separate loop, since biases are indexed by next layer neurons)
            // Equation: grad_b^{(l+1)}_j = delta^{(l+1)}_j
            for (int j = 0; j < links; j++)
                gradient.biasGradients[j] += nextPreActivationDeltas[y + j];
        }
        
        if(batchSize > 1) {
            for (int i = 0; i < gradient.weightGradients.length; i++)
                gradient.weightGradients[i] /= batchSize;

            for (int i = 0; i < gradient.biasGradients.length; i++)
                gradient.biasGradients[i] /= batchSize;
        }

        return gradient;
    }
    
    ConnectionGradientGPU backpropagateGPU(CUdeviceptr currentActivationDeltas, 
                                           CUdeviceptr nextPreActivationDeltas, 
                                           CUdeviceptr postActivation, 
                                           int batchSize,
                                           boolean accumulateDeltas,
                                           CUstream stream, 
                                           cublasHandle handle) {        
        ConnectionGradientGPU gradient = new ConnectionGradientGPU(neurons, links, stream);
        
        float scale = 1.0f / batchSize;
        Pointer pAlpha = Pointer.to(new float[]{1.0f});
        Pointer pBeta = Pointer.to(new float[]{0.0f});

        // cuBLAS uses column-major format. Make sure the dimensions match accordingly
        int m = links;         // Output size (links in current layer, which is nextPreActivationDeltas size)
        int n = neurons;       // Batch size
        int k = batchSize;         // Batch size 

        CUdeviceptr a = nextPreActivationDeltas;    // delta^{(l+1)}, size [links x batchSize]
        CUdeviceptr b = postActivation;             // a^{(l)},       size [neurons x batchSize]
        CUdeviceptr c = gradient.weightGradients;   // grad_W^{(l)},  size [links x neurons]

        synchronized(handle) {
            JCublas2.cublasSetStream(handle, new cudaStream_t(stream));

            // Step 1: Compute the weight gradient using cuBLAS
            // gradW[l] = delta[l+1] * activations[l].transpose() 
            // Include averaging in the alpha
            JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_T, 
                                 m, n, k, 
                                 Pointer.to(new float[]{scale}), a, m, b, n, 
                                 pBeta, c, m);

            // Step 2: Compute pre-activation deltas for current layer
            // postActivationDeltas[l] += weights[l].transpose() * delta[l+1]   
            if(currentActivationDeltas != null) {
                if(accumulateDeltas)
                    pBeta = Pointer.to(new float[]{1.0f});
                
                CUdeviceptr d = currentActivationDeltas;  // activation delta, size [neurons x batchSize]
                JCublas2.cublasSgemm(handle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N, 
                                     n, k, m, 
                                     pAlpha, weightsGPU, m, a, m, 
                                     pBeta, d, n);
            }

            // Step 3: Compute bias gradients
            // biasGradients = sum(delta[l+1], across batch)
            CudaFunctions.vector.reduceSum(gradient.biasGradients, nextPreActivationDeltas, links, batchSize, stream);

            // Step 4: If batchSize > 1, average the gradients
            if (batchSize > 1) {
                Pointer alpha = Pointer.to(new float[]{1.0f / batchSize});
//                JCublas2.cublasSscal(handle, neurons * links, alpha, gradient.weightGradients, 1);
                JCublas2.cublasSscal(handle, links, alpha, gradient.biasGradients, 1);
            }
        }
        
        return gradient;
    }
    
    public void updateWeights(ConnectionGradient gradient, float learningRate, float weightDecay) {
        if (useAdam) {
            updateWeightsAdam(gradient, learningRate, weightDecay);
        } else {
            // Update weights
            for (int i = 0; i < weights.length; i++) {
                float gradientWithDecay = gradient.weightGradients[i] + weightDecay * weights[i];
                weights[i] -= learningRate * gradientWithDecay;
            }

            // Update biases  
            for (int i = 0; i < biases.length; i++) {
                biases[i] -= learningRate * gradient.biasGradients[i];
            }
        }
    }
    
    private void updateWeightsAdam(ConnectionGradient gradient, float learningRate, float weightDecay) {
        adamTimeStep++;

        // Update weights using Adam
        for (int i = 0; i < weights.length; i++) {
            float grad = gradient.weightGradients[i];

            weightMomentum[i] = beta1 * weightMomentum[i] + (1 - beta1) * grad;
            weightVelocity[i] = beta2 * weightVelocity[i] + (1 - beta2) * grad * grad;

            float mHat = weightMomentum[i] / (1 - (float)Math.pow(beta1, adamTimeStep));
            float vHat = weightVelocity[i] / (1 - (float)Math.pow(beta2, adamTimeStep));

            weights[i] -= learningRate * mHat / ((float)Math.sqrt(vHat) + adamEpsilon);

            // Apply weight decay after Adam update
            if (weightDecay > 0) {
                weights[i] -= learningRate * weightDecay * weights[i];
            }
        }

        // Update biases using Adam
        for (int i = 0; i < biases.length; i++) {
            float grad = gradient.biasGradients[i];

            biasMomentum[i] = beta1 * biasMomentum[i] + (1 - beta1) * grad;
            biasVelocity[i] = beta2 * biasVelocity[i] + (1 - beta2) * grad * grad;

            float mHat = biasMomentum[i] / (1 - (float)Math.pow(beta1, adamTimeStep));
            float vHat = biasVelocity[i] / (1 - (float)Math.pow(beta2, adamTimeStep));

            biases[i] -= learningRate * mHat / ((float)Math.sqrt(vHat) + adamEpsilon);
        }
    }
    
    public void updateWeightsGPU(ConnectionGradientGPU gradient, float learningRate, float weightDecay, CUstream stream, cublasHandle handle) {
        if (useAdam) {
            updateWeightsAdamGPU(gradient, learningRate, weightDecay, stream);
        } else {
            float alpha = -learningRate;

            synchronized(handle) {
                JCublas2.cublasSetStream(handle, new cudaStream_t(stream));

                if(weightDecay > 0)
                    // Calculate new gradient = weightGradients + weightDecay * weights;
                    CudaFunctions.connection.updateWeightsWithDecay(weightsGPU, gradient.weightGradients, learningRate, weightDecay, neurons * links, stream);
                else
                    // Update weights using cuBLAS: weights = weights - learningRate * weightGradients
                   JCublas2.cublasSaxpy(handle, neurons * links, Pointer.to(new float[]{alpha}), gradient.weightGradients, 1, weightsGPU, 1);

                // Update biases using cuBLAS: biases = biases - learningRate * biasGradients
                JCublas2.cublasSaxpy(handle, links, Pointer.to(new float[]{alpha}), gradient.biasGradients, 1, biasesGPU, 1);
            }
        }
    }
    
    private void updateWeightsAdamGPU(ConnectionGradientGPU gradient, float learningRate, float weightDecay, CUstream stream) {
        adamTimeStep++;

        float beta1Power = (float)Math.pow(beta1, adamTimeStep);
        float beta2Power = (float)Math.pow(beta2, adamTimeStep);

        // Update weights on GPU
        CudaFunctions.connection.adamUpdateConnectionWeights(weightsGPU, gradient.weightGradients, 
                                                weightMomentumGPU, weightVelocityGPU,
                                                learningRate, beta1, beta2, adamEpsilon, 
                                                beta1Power, beta2Power, weightDecay,
                                                neurons * links, stream);

        // Update biases on GPU  
        CudaFunctions.connection.adamUpdateConnectionBiases(biasesGPU, gradient.biasGradients,
                                               biasMomentumGPU, biasVelocityGPU,
                                               learningRate, beta1, beta2, adamEpsilon,
                                               beta1Power, beta2Power,
                                               links, stream);
    }
    
    void prepareCPU() {
        weightsCPU = ByteBuffer.allocateDirect(weights.length * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
        weightsCPU.put(weights);
        
        biasCPU = ByteBuffer.allocateDirect(biases.length * Float.BYTES).order(ByteOrder.nativeOrder()).asFloatBuffer();
        biasCPU.put(biases);
        
        nativeReady = true;
    }
    
    public void syncWeightsFromGPU(CUstream stream) {
        weights = CudaUtil.fromGPUFloatAsync(weightsGPU, neurons * links, stream);        
        biases = CudaUtil.fromGPUFloatAsync(biasesGPU, links, stream);
        
        // If using Adam, also update momentum and velocity
        if (useAdam && weightMomentumGPU != null) {
            weightMomentum = CudaUtil.fromGPUFloatAsync(weightMomentumGPU, neurons * links, stream);
            weightVelocity = CudaUtil.fromGPUFloatAsync(weightVelocityGPU, neurons * links, stream);
            biasMomentum = CudaUtil.fromGPUFloatAsync(biasMomentumGPU, links, stream);
            biasVelocity = CudaUtil.fromGPUFloatAsync(biasVelocityGPU, links, stream);
        }
    }
    
    /**
    * Total number of parameters (weights, biases, optimizer state)
    * @return 
    */
    public long getParameterCount() {
        long parameters = 0;

        // Weights memory
        parameters += neurons * links;

        // Biases memory  
        parameters += links;

        // Adam optimizer state (if enabled)
        if(useAdam) {
            // Weight momentum + velocity
            parameters += 2 * neurons * links;
            // Bias momentum + velocity
            parameters += 2 * links;
        }

        return parameters;
    }
    
    /**
    * Memory required during backward pass (ConnectionGradientGPU created in backpropagateGPU)
    * @param batchSize
    * @return 
    */
    public long getBackwardMemoryRequired(int batchSize) {
        // ConnectionGradientGPU constructor allocates:
        // - weightGradients: neurons * links * FLOAT_SIZE  
        // - biasGradients: links * FLOAT_SIZE
        return (neurons * links + links) * CudaUtil.FLOAT_SIZE;
    }
    
    public void prepareGPU(CUstream stream) {
        if(weightsGPU != null || biasesGPU != null)
            throw new RuntimeException("GPU already initialized for connection");
        
        weightsGPU = CudaUtil.toGPUAsync(weights, stream);
        biasesGPU = CudaUtil.toGPUAsync(biases, stream);
        
        // Initialize Adam state if needed
        if (useAdam) {
            weightMomentumGPU = CudaUtil.toGPUAsync(weightMomentum, stream);
            weightVelocityGPU = CudaUtil.toGPUAsync(weightVelocity, stream);
            biasMomentumGPU = CudaUtil.toGPUAsync(biasMomentum, stream);
            biasVelocityGPU = CudaUtil.toGPUAsync(biasVelocity, stream);
        }
        
        gpuReady = true;
    }
    
    public void freeGPU(CUstream stream) {
        CudaUtil.freeAsync(weightsGPU, stream);
        CudaUtil.freeAsync(biasesGPU, stream);
        
        // Free Adam state if allocated
        if (useAdam) {
            if (weightMomentumGPU != null) CudaUtil.freeAsync(weightMomentumGPU, stream);
            if (weightVelocityGPU != null) CudaUtil.freeAsync(weightVelocityGPU, stream);
            if (biasMomentumGPU != null) CudaUtil.freeAsync(biasMomentumGPU, stream);
            if (biasVelocityGPU != null) CudaUtil.freeAsync(biasVelocityGPU, stream);
        }
    
        weightsGPU = null;
        biasesGPU = null;
        weightMomentumGPU = null;
        weightVelocityGPU = null;
        biasMomentumGPU = null;
        biasVelocityGPU = null;
        
        freeGPURng();
        
        gpuReady = false;
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
            throw new RuntimeException("new weights " + values.length + " != " + weights.length);
        
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

    public void initUniform(float min, float max) {
        int len = neurons * links;
        
        for(int i=0; i < len; i++)
            weights[i] = rng.nextFloat(min, max);
        
        for(int i=0; i < links; i++)
            biases[i] = rng.nextFloat(min, max);
        
        nativeReady = false;
        gpuReady = false;
    }
    
    public void initGaussian(float mean, float sd) {
        int len = neurons * links;
        
        for(int i=0; i < len; i++)
            weights[i] = (float) rng.gaussian(mean, sd);
        
        for(int i=0; i < links; i++)
            biases[i] = (float) rng.gaussian(mean, sd);
        
        nativeReady = false;
        gpuReady = false;        
    }
    
    /**
     * Enable/disable Adam optimizer
     * @param useAdam
     */
    public void setUseAdam(boolean useAdam) {
        this.useAdam = useAdam;
        if (useAdam && weightMomentum == null) {
            initAdamState();
        }
    }
    
    /**
     * Initialize Adam optimizer state
     */
    private void initAdamState() {
        weightMomentum = new float[neurons * links];
        weightVelocity = new float[neurons * links];
        biasMomentum = new float[links];
        biasVelocity = new float[links];

        Arrays.fill(weightMomentum, 0.0f);
        Arrays.fill(weightVelocity, 0.0f);
        Arrays.fill(biasMomentum, 0.0f);
        Arrays.fill(biasVelocity, 0.0f);

        adamTimeStep = 0;
    }
    
    void freeCPU() {
        weights = null;
        biases = null;
        weightsCPU = null;
        biasCPU = null;
    }

    void freeGPURng() {
        if(rngPool != null)
            CudaEngine.freeMempool(rngPool);
        
        rngPool = null;
    }
//    
//    private void initGPU(CUstream stream) {
//        if(weightsGPU != null || biasesGPU != null)
//            throw new RuntimeException("GPU already initialized for connection");
//        
//        long lengthWeights = CudaUtil.alignLength(neurons * links, CudaUtil.DEFAULT_MEM_ALIGN_FLOAT);
//        long lengthBias = links;
//        long lengthTotal = lengthWeights + lengthBias;
//        
//        weightsGPU = CudaUtil.createFloatAsync(lengthTotal, stream);
//        biasesGPU = weightsGPU.withByteOffset(lengthWeights * CudaUtil.FLOAT_SIZE);
//    }
    
//    private void initGPUWeights() {
//        if(weightsGPU != null || biasesGPU != null)
//            throw new RuntimeException("GPU already initialized for connection");
//        
//        long lengthWeights = CudaUtil.alignLength(neurons * links, CudaUtil.DEFAULT_MEM_ALIGN_FLOAT);
//        long lengthBias = links;
//        long lengthTotal = lengthWeights + lengthBias;
//        
//        weightsGPU = CudaEngine.getMempoolFloat(lengthTotal);
//        biasesGPU = weightsGPU.withByteOffset(lengthWeights * CudaUtil.FLOAT_SIZE);
////        biasesGPU = //CudaUtil.createFloat(lengthBias);
//    }
    void crossOverMutate(Connection a, Connection b, float min, float max, double mutation) {
        float[] wa = a.weights;
        float[] wb = b.weights;

        for(int j=0; j < wa.length; j++) {
            float w = rng.nextBoolean() ? wa[j] : wb[j];

            if(rng.nextDouble() < mutation)
                w = w + rng.nextFloat(min, max);

            weights[j] = w;
        }
        
        float[] ba = a.biases;
        float[] bb = b.biases;

        for(int j=0; j < ba.length; j++) {
            float w = rng.nextBoolean() ? ba[j] : bb[j];

            if(rng.nextDouble() < mutation)
                w = w + rng.nextFloat(min, max);

            biases[j] = w;
        }

        nativeReady = false;
        gpuReady = false;
    }

    void crossOverMutateGPU(Connection a, Connection b, float min, float max, double mutation, boolean nocopy, CUstream stream, curandGenerator generator) {
        /* mutate weights */
        if(weightsGPU != null || biasesGPU != null)
            throw new RuntimeException("GPU already initialized for connection");
        
        long weightsLength = neurons * links + links;
        
        weightsGPU = CudaUtil.createFloatAsync(neurons * links, stream);
        biasesGPU = CudaUtil.createFloatAsync(links, stream);

        rngPool = CudaEngine.getMempoolFloat(weightsLength * 3);
        CUdeviceptr rngMutate = rngPool.withByteOffset(weightsLength * CudaUtil.FLOAT_SIZE);
        CUdeviceptr rngCrossover = rngPool.withByteOffset(2 * weightsLength * CudaUtil.FLOAT_SIZE);
        
        synchronized(generator) {
            JCurand.curandSetStream(generator, new cudaStream_t(stream));
            JCurand.curandGenerateUniform(generator, rngPool, weightsLength * 3);
        }
        
        /* mutate weights and biases in one kernel launch */
        CudaFunctions.CrossoverMutate(a.weightsGPU, b.weightsGPU, weightsGPU, weightsLength, min, max, mutation, 
                                      rngCrossover, rngMutate, rngPool, CudaUtil.PREFERRED_BLOCK_SIZE, stream);
        
        
//        CudaFunctions.crossoverMutate(a.biasesGPU, b.biasesGPU, biasesGPU, links, min, max, mutation, 
//                                      rngCrossover.withByteOffset(lengthWeights * CudaUtil.FLOAT_SIZE), 
//                                      rngMutate.withByteOffset(lengthWeights * CudaUtil.FLOAT_SIZE), 
//                                      rngPool.withByteOffset(lengthWeights * CudaUtil.FLOAT_SIZE), 
//                                      stream);
        if(!nocopy) {
            weights = CudaUtil.fromGPUFloatAsync(weightsGPU, neurons * links, stream);
            biases = CudaUtil.fromGPUFloatAsync(biasesGPU, links, stream);
        }
        
        nativeReady = false;
        gpuReady = true;
    }
    
    void clipWeights(float min, float max) {
        for(int i=0; i < weights.length; i++) {
            float w = weights[i];
            
            if(w > max)
                weights[i] = max;
            else if(w < min)
                weights[i] = min;
        }
        
        for(int i=0; i < biases.length; i++) {
            float w = biases[i];
            
            if(w > max)
                biases[i] = max;
            else if(w < min)
                biases[i] = min;
        }

        nativeReady = false;
        gpuReady = false;
    }
    
    void clipWeightsGPU(float min, float max, CUstream stream) {
        long weightsLength = neurons * links + links;
        CudaFunctions.ClipWeights(weightsGPU, weightsLength, min, max, stream);
    }
    
    HashMap serialize() {
        HashMap result = new HashMap();

        result.put("neurons", neurons);
        result.put("links", links);
        
        result.put("weights", util.base64encode(util.compress(util.toByteArray(weights))));
        result.put("biases", util.base64encode(util.compress(util.toByteArray(biases))));
        
        return result;
    }
    
    static Connection deserialize(HashMap obj) {
        int neurons = (Integer)obj.get("neurons");
        int links = (Integer)obj.get("links");
        
        float[] weights = util.toFloatArray(util.decompress(util.base64decode((String)obj.get("weights"))));
        float[] biases = util.toFloatArray(util.decompress(util.base64decode((String)obj.get("biases"))));
        
        return new Connection(neurons, links, weights, biases);
    }
    
    double compare(Connection c) {
        double score = 0;
        
        float[] wa = c.getWeights();

        for(int j=0; j < wa.length; j++)            
            score += Math.abs(wa[j] - weights[j]);

        float[] ba = c.getBias();

        for(int j=0; j < ba.length; j++)
            score += Math.abs(ba[j] - biases[j]);            
        
        return score;
    }
//
//    public boolean hasBias() {
//        return !disableBias;
//    }

    void copyWeights(Connection connection) {
        if(neurons != connection.neurons || links != connection.links)
            throw new RuntimeException(
                String.format("incompatible copy weights: %d:%d %d:%d",
                            neurons, connection.neurons, links, connection.links));
        
        weights = Arrays.copyOf(connection.weights, connection.weights.length);
        biases = Arrays.copyOf(connection.biases, connection.biases.length);
        
        this.nativeReady = false;
        this.gpuReady = false;
    }
    
    void feedForward(FloatBuffer input, int batchSize, FloatBuffer result) {
        if(batchSize == 1)
            intrinsic.sgemv(input, result, weightsCPU, biasCPU, neurons, links);
        else
            intrinsic.sgemm(input, batchSize, result, weightsCPU, biasCPU, neurons, links);
    }
}
