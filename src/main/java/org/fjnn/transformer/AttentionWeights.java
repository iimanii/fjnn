/*
 * The MIT License
 *
 * Copyright 2025 ahmed.
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
package org.fjnn.transformer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.cudaStream_t;
import org.fjnn.base.ModelComponent;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.base.output.FeedForwardOutput;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.transformer.output.AttentionWeightsBackpropOutput;
import org.fjnn.transformer.output.AttentionWeightsBackpropOutputGPU;
import org.fjnn.transformer.output.AttentionWeightsOutput;
import org.fjnn.transformer.output.AttentionWeightsOutputGPU;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 */
public class AttentionWeights extends ModelComponent {
    public final int vectorDim;         // Dimension of each vector in sequence
    public final int sequenceLen;       // Fixed sequence length
    public final int inputDim;          // total length of input
    
    private float[] weights;     // [vectorDim × vectorDim]
    private CUdeviceptr weightsGPU;    // GPU version of weights
    
    private boolean gpuReady;

    public AttentionWeights(int vectorDim, int sequenceLen) {
        this.vectorDim = vectorDim;
        this.sequenceLen = sequenceLen;
        this.inputDim = vectorDim * sequenceLen;
        this.weights = new float[vectorDim * vectorDim];
    }
    
    public void setWeights(float[] values) {
        if(values.length != weights.length)
            throw new RuntimeException("new weights " + values.length + " != " + weights.length);
        
        weights = Arrays.copyOf(values, values.length);
        gpuReady = false;
    }
    
    @Override
    public FeedForwardOutput feedForward(float[] input, int batchSize) {
        AttentionWeightsOutput output = new AttentionWeightsOutput(input, inputDim, batchSize);

        // Total number of vectors to process
        int totalVectors = batchSize * sequenceLen;

        // Process each vector
        for(int v = 0; v < totalVectors; v++) {
            int inputStart = v * vectorDim;
            int outputStart = v * vectorDim;

            // Transform vector using weights
            for(int i = 0; i < vectorDim; i++) {
                int weightStart = i * vectorDim;
                for(int j = 0; j < vectorDim; j++) {
                    output.output[outputStart + j] += input[inputStart + i] * weights[weightStart + j];
                }
            }
        }

        return output;
    }

    @Override
    public FeedForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchSize, CUstream stream) {
        AttentionWeightsOutputGPU output = new AttentionWeightsOutputGPU(input, inputDim, batchSize, stream);

        int totalVectors = batchSize * sequenceLen;
        cublasHandle handle = CudaEngine.getCublasHandle(deviceId);
        Pointer alpha = Pointer.to(new float[]{1.0f});
        Pointer beta = Pointer.to(new float[]{0.0f});

        synchronized(handle) {
            JCublas2.cublasSetStream(handle, new cudaStream_t(stream));

            // Transform all vectors in one operation
            JCublas2.cublasSgemm(handle, 
                cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N,
                vectorDim, totalVectors, vectorDim,   // Transform totalVectors vectors at once
                alpha,
                weightsGPU, vectorDim,    
                input, vectorDim,
                beta,
                output.output, vectorDim
            );
        }

        return output;
    }

    @Override
    public BackpropagateOutput backpropagate(FeedForwardOutput output, float[] deltaLoss) {
        if (!(output instanceof AttentionWeightsOutput))
            throw new IllegalArgumentException("Expected AttentionWeightsOutput");
        
        AttentionWeightsOutput attnOutput = (AttentionWeightsOutput)output;
        int totalVectors = output.batchSize * sequenceLen;

        AttentionWeightsBackpropOutput backOutput = new AttentionWeightsBackpropOutput(deltaLoss, vectorDim, sequenceLen, output.batchSize);

        // Process each vector
        for(int v = 0; v < totalVectors; v++) {
            int inputStart = v * vectorDim;
            int outputStart = v * vectorDim;

            // Compute gradients for this vector
            for(int i = 0; i < vectorDim; i++) {
                int weightStart = i * vectorDim;
                /* 
                 * Weight gradients dL/dW:
                 *    For each vector:
                 *    dL/dW[i,j] += x[i] * dL/dy[j]
                 */
                for(int j = 0; j < vectorDim; j++)
                    backOutput.weightGradients[weightStart + j] += attnOutput.input[inputStart + i] * deltaLoss[outputStart + j];

                // dL/dx[i] = Σ_j W[j,i] * dL/dy[j]
                for(int j = 0; j < vectorDim; j++)
                    backOutput.inputGradients[inputStart + i] += weights[j * vectorDim + i] * deltaLoss[outputStart + j];
            }
        }
        
        // Average weight gradients
        float scale = 1.0f / totalVectors;
        for(int i = 0; i < backOutput.weightGradients.length; i++) {
            backOutput.weightGradients[i] *= scale;
        }
        
        return backOutput;
    }

    @Override
    public BackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU output, CUdeviceptr deltaLoss, CUstream stream) {
        if (!(output instanceof AttentionWeightsOutputGPU))
               throw new IllegalArgumentException("Expected AttentionWeightsOutputGPU");

           AttentionWeightsOutputGPU attnOutput = (AttentionWeightsOutputGPU)output;
           int totalVectors = output.batchSize * sequenceLen;

           // Create output structure with GPU arrays
           AttentionWeightsBackpropOutputGPU backOutput = new AttentionWeightsBackpropOutputGPU(deltaLoss, vectorDim, sequenceLen, output.batchSize, stream);

           cublasHandle handle = CudaEngine.getCublasHandle(deviceId);

           // Scale factor for averaging gradients
           float scale = 1.0f / totalVectors;
           Pointer alpha = Pointer.to(new float[]{scale});  // Include averaging in alpha
           Pointer beta = Pointer.to(new float[]{0.0f});

           synchronized(handle) {
               JCublas2.cublasSetStream(handle, new cudaStream_t(stream));

               // Step 1: Weight gradients - outer products with averaging
               // dL/dW = (1/totalVectors) * input^T × deltaLoss  
               JCublas2.cublasSgemm(handle,
                   cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N,  // input transposed
                   vectorDim, vectorDim, totalVectors,                        // Matrix dimensions
                   alpha,                                                     // 1/totalVectors
                   attnOutput.input, vectorDim,                               // [totalVectors × vectorDim]
                   deltaLoss, vectorDim,                                      // [totalVectors × vectorDim]
                   beta,                                                      // beta = 0
                   backOutput.weightGradients, vectorDim                      // [vectorDim × vectorDim]
               );

               // Step 2: Input gradients - matrix multiplication
               // dL/dx = weights^T × deltaLoss
               alpha = Pointer.to(new float[]{1.0f});                         // No averaging needed for input gradients
               JCublas2.cublasSgemm(handle,
                   cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_N,  // weights transposed
                   vectorDim, totalVectors, vectorDim,                        // Matrix dimensions
                   alpha,                                                     // alpha = 1.0
                   weightsGPU, vectorDim,                                     // [vectorDim × vectorDim]
                   deltaLoss, vectorDim,                                      // [totalVectors × vectorDim]
                   beta,                                                      // beta = 0
                   backOutput.inputGradients, vectorDim                       // [totalVectors × vectorDim]
               );
           }

           return backOutput;
    }

    @Override
    public void applyGradients(BackpropagateOutput gradients, float learningRate, float weightDecay) {
        if (!(gradients instanceof AttentionWeightsBackpropOutput)) {
            throw new IllegalArgumentException("Expected AttentionWeightsBackpropOutput");
        }

        AttentionWeightsBackpropOutput backOutput = (AttentionWeightsBackpropOutput)gradients;
        
        // Update weights with gradients
        for(int i = 0; i < weights.length; i++) {
            float gradient = backOutput.weightGradients[i] + weightDecay * weights[i];
            weights[i] -= learningRate * gradient;
        }
        
        gpuReady = false;
    }

    @Override
    public void applyGradientsGPU(BackpropagateOutputGPU gradients, float learningRate, float weightDecay, CUstream stream) {
        if (!(gradients instanceof AttentionWeightsBackpropOutputGPU)) {
            throw new IllegalArgumentException("Expected AttentionWeightsBackpropOutputGPU");
        }

        AttentionWeightsBackpropOutputGPU backOutput = (AttentionWeightsBackpropOutputGPU)gradients;
        
        CudaFunctions.updateWeightsWithDecay(
            weightsGPU,
            backOutput.weightGradients,
            learningRate,
            weightDecay,
            weights.length,
            stream
        );
    }

    @Override
    public boolean gpuReady() {
        return gpuReady;
    }

    @Override
    public void prepareGPU0(CUstream stream) {
        weightsGPU = CudaUtil.toGPUAsync(weights, stream);
        gpuReady = true;
    }

    @Override
    public void freeGPU0(CUstream stream) {
        CudaUtil.freeAsync(weightsGPU, stream);
        gpuReady = false;
    }

    @Override
    public int getInputSize() {
        return inputDim;
    }

    @Override
    public int getOutputSize() {
        return sequenceLen * vectorDim;
    }

    @Override
    public ModelComponent copy() {
        AttentionWeights copy = new AttentionWeights(vectorDim, sequenceLen);
        System.arraycopy(weights, 0, copy.weights, 0, weights.length);
        return copy;
    }

    @Override
    public HashMap serialize() {
        HashMap obj = new HashMap();
        obj.put("type", "AttentionWeights");
        obj.put("vectorDim", vectorDim);
        obj.put("sequenceLen", sequenceLen);
        obj.put("weights", util.base64encode(util.compress(util.toByteArray(weights))));
        return obj;
    }
    
    public static AttentionWeights deserialize(Map serialized) {
        int vectorDim = (Integer)serialized.get("vectorDim");
        int sequenceLen = (Integer)serialized.get("sequenceLen");
        float[] weights = util.toFloatArray(util.decompress(util.base64decode((String)serialized.get("weights"))));
        
        AttentionWeights result = new AttentionWeights(vectorDim, sequenceLen);
        System.arraycopy(weights, 0, result.weights, 0, weights.length);
        return result;
    }

    @Override
    public void updateWeightsFromGPU() {
        if (!gpuReady) {
            throw new RuntimeException("GPU weights not initialized");
        }
        
        weights = CudaUtil.fromGPUFloat(weightsGPU, weights.length);
    }

    @Override
    public long getParametersCount() {
        return weights.length;
    }

    @Override
    public long getBackpropagateMemoryRequired(int batchSize) {
        // Memory needed for:
        // 1. transformedOutput [batchSize * sequenceLen * vectorDim]
        // 2. weightGradients [vectorDim * vectorDim] 
        // 3. inputGradients [batchSize * sequenceLen * vectorDim]
        return (2 * batchSize * sequenceLen * vectorDim + vectorDim * vectorDim) * CudaUtil.FLOAT_SIZE;
    }
}
