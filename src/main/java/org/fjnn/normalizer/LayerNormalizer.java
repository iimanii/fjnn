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
package org.fjnn.normalizer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.cudaStream_t;
import org.fjnn.base.ModelComponent;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.base.output.FeedForwardOutput;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.normalizer.outputs.LayerNormalizerBackpropagateOutput;
import org.fjnn.normalizer.outputs.LayerNormalizerBackpropagateOutputGPU;
import org.fjnn.normalizer.outputs.LayerNormalizerForwardOutput;
import org.fjnn.normalizer.outputs.LayerNormalizerForwardOutputGPU;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 * 
 * x̂[i] = (x[i] - μ) / √(σ² + ε)
 * y[i] = γ * x̂[i] + β
 */
public class LayerNormalizer extends Normalizer {
    private final float[] gamma;
    private final float[] beta;
    
    private CUdeviceptr gammaGPU;
    private CUdeviceptr betaGPU;
    
    public LayerNormalizer() {
        super(0);
        
        this.gamma = new float[neurons];
        this.beta = new float[neurons];
    }
    
    private LayerNormalizer(int neurons) {
        super(neurons);
        
        this.gamma = new float[neurons];
        this.beta = new float[neurons];
        Arrays.fill(gamma, 1.0f);
        Arrays.fill(beta, 0.0f);
    }
    
     /*
      * Requires at least 2 neurons to function properly since single-neuron normalization 
      * would result in zero output.
      */
    @Override
    public LayerNormalizer withNeurons(int neurons) {
        if(neurons < 2)
            throw new RuntimeException("LayerNormalizer requires at least 2 neurons to function properly");
        
        return new LayerNormalizer(neurons);
    }
    
    @Override
    public void compute(float[] input, int count) {
        for(int i = 0; i < count; i++) {
            float mean = 0;
            for(int j = 0; j < neurons; j++) {
                mean += input[i * neurons + j];
            }
            mean /= neurons;
            
            float variance = 0;
            for(int j = 0; j < neurons; j++) {
                float diff = input[i * neurons + j] - mean;
                variance += diff * diff;
            }
            variance = variance / neurons + epsilon;
            
            float std = (float) Math.sqrt(variance);
            for(int j = 0; j < neurons; j++) {
                int idx = i * neurons + j;
                input[idx] = gamma[j] * ((input[idx] - mean) / std) + beta[j];
            }
        }
    }

    @Override
    public void computeGPU(CUdeviceptr ptr, long count, CUstream stream) {
        // For in-place computation, we'll use same pointer for input and output
        // We need temporary storage for means and variances
        CUdeviceptr normalized = CudaUtil.createFloatAsync(neurons * count, stream);
        CUdeviceptr variances = CudaUtil.createFloatAsync(count, stream);

        // Call normalizer function
        CudaFunctions.normalization.LayerNormalizer(
            ptr,    // input
            normalized,
            ptr,    // output (same as input for in-place)
            variances,
            gammaGPU,
            betaGPU,
            neurons, 
            count, 
            stream
        );

        // Clean up temporary storage
        CudaUtil.freeAsync(normalized, stream);
        CudaUtil.freeAsync(variances, stream);
    }

    @Override
    public FeedForwardOutput feedForward(float[] input, int count) {
        LayerNormalizerForwardOutput output = new LayerNormalizerForwardOutput(input, neurons, count);

        // For each example in batch
        for(int i = 0; i < count; i++) {
            int from = i * neurons;
            int to = from + neurons;
            
            // Calculate mean
            float mean = 0;
            for(int j = from; j < to; j++) {
                mean += output.preNormalization[j];
            }
            mean /= neurons;
//            output.means[i] = mean;

            // Calculate variance
            float variance = 0;
            for(int j = from; j < to; j++) {
                float diff = output.preNormalization[j] - mean;
                variance += diff * diff;
            }
            variance = variance / neurons + epsilon;
            float std = (float) Math.sqrt(variance);
            output.stds[i] = std;

            // Normalize, scale and shift
            for(int j = 0; j < neurons; j++) {
                int idx = i * neurons + j;
                output.normalized[idx] = (output.preNormalization[idx] - mean) / std;
                output.postNormalization[idx] = gamma[j] * output.normalized[idx] + beta[j];
            }
        }

        return output;
    }
    
    @Override
    public FeedForwardOutputGPU feedForwardGPU(CUdeviceptr ptr, int count, CUstream stream) {
        LayerNormalizerForwardOutputGPU output = new LayerNormalizerForwardOutputGPU(ptr, neurons, count, stream);

        CudaFunctions.normalization.LayerNormalizer(output.preNormalization, 
                                                    output.normalized, 
                                                    output.postNormalization, 
                                                    output.stds,
                                                    gammaGPU,
                                                    betaGPU,
                                                    neurons, count, stream);

        return output;
    }

    @Override
    public BackpropagateOutput backpropagate(FeedForwardOutput output, float[] gradients) {
        if(!(output instanceof LayerNormalizerForwardOutput))
            throw new RuntimeException("LayerNormalizerForwardOutput object required for backpropagation");
        
        int batchSize = output.batchSize;
        
        LayerNormalizerForwardOutput lnOutput = (LayerNormalizerForwardOutput)output;
        LayerNormalizerBackpropagateOutput backOutput = new LayerNormalizerBackpropagateOutput(neurons, batchSize);

        // For each batch
        for(int b = 0; b < batchSize; b++) {
            int offset = b * neurons;
            float std = lnOutput.stds[b];           // Note: this already includes epsilon from forward pass
            
            // Temporary arrays for the batch
            float[] dxHat = new float[neurons];     // Gradients with respect to normalized values
            float sumDx = 0;                        // Sum of dxHat
            float sumDxXHat = 0;                    // Sum of dxHat * normalized values
            
            // For each neuron
            //  ∂L/∂x = (1/σ) * (∂L/∂x̂ - mean(∂L/∂x̂) - x̂ * mean(∂L/∂x̂ * x̂))
            //  ∂L/∂x̂ = ∂L/∂y * γ
            //  x̂ = (x - μ)/σ
            for(int i = 0; i < neurons; i++) {
                // Calculate dxHat = dy * gamma
                dxHat[i] = gradients[offset + i] * gamma[i];

                // Accumulate sums for later use
                sumDx += dxHat[i];
                sumDxXHat += dxHat[i] * lnOutput.normalized[offset + i];
                
                // Accumulate beta gradients for this batch .. (∂L/∂βᵢ = ∂L/∂yᵢ)
                backOutput.betaGradients[i] += gradients[offset + i];
                
                // Accumulate gamma gradients for this batch .. (∂L/∂γᵢ = ∂L/∂yᵢ * x̂ᵢ)
                backOutput.gammaGradients[i] += gradients[offset + i] * lnOutput.normalized[offset + i];;
            }
            
            // Second pass: calculate final input gradients
            float meanDx = sumDx / neurons;
            float meanDxHat = sumDxXHat / neurons;
            float invStd = 1.0f / std;
            
            // Average the gradients across batches
            for(int i = 0; i < neurons; i++) {
                // Final gradient computation combining all terms
                // dx = (1/std) * (dxHat - (sumDx/N) - xHat * (sumDxXHat/N))
                float xHat = lnOutput.normalized[offset + i];
                backOutput.deltaLoss[offset + i] = invStd * (dxHat[i] - meanDx - xHat * meanDxHat);
            }
        }
        
        // Average the gradients across batches
        float batchScale = 1.0f / batchSize;
        for(int i = 0; i < neurons; i++) {
            backOutput.betaGradients[i] *= batchScale;
            backOutput.gammaGradients[i] *= batchScale;
        }

        return backOutput;
    }

    /*
     * Notes: updates to the gradients are done in place
     */
    @Override
    public BackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU output, CUdeviceptr gradients, CUstream stream) {
        if(!(output instanceof LayerNormalizerForwardOutputGPU))
            throw new RuntimeException("LayerNormalizerForwardOutputGPU object required for GPU backpropagation");
    
        int batchSize = output.batchSize;

        LayerNormalizerForwardOutputGPU lnOutput = (LayerNormalizerForwardOutputGPU)output;
        LayerNormalizerBackpropagateOutputGPU backOutput = new LayerNormalizerBackpropagateOutputGPU(gradients, neurons, batchSize, stream);

        cublasHandle handle = CudaEngine.getCublasHandle(deviceId);

        synchronized(handle) {
            JCublas2.cublasSetStream(handle, new cudaStream_t(stream));
            
            // Beta gradients - sum upstream gradients across batch
            CudaFunctions.vector.reduceSum(backOutput.betaGradients, gradients, 
                                           neurons, batchSize, stream);
            
            // For gamma gradients, first multiply upstream with normalized values
            CUdeviceptr temp = CudaUtil.createFloatAsync(batchSize * neurons, stream);
            CudaFunctions.vector.multiply(gradients, lnOutput.normalized, temp, 
                                          batchSize * neurons, stream);

            // Then sum across batch
            CudaFunctions.vector.reduceSum(backOutput.gammaGradients, temp, 
                                           neurons, batchSize, stream);

            CudaUtil.freeAsync(temp, stream);
            
            // Step 4: If count > 1, average the gradients
            if (batchSize > 1) {
                Pointer alpha = Pointer.to(new float[]{1.0f / batchSize});
                JCublas2.cublasSscal(handle, neurons, alpha, backOutput.betaGradients, 1);
                JCublas2.cublasSscal(handle, neurons, alpha, backOutput.gammaGradients, 1);
            }        
        }
        
        // Call CUDA kernel for deltaLoss calculation
        CudaFunctions.normalization.LayerNormalizerBackpropagate(
            lnOutput.normalized,             // normalized values
            lnOutput.stds,                   // standard deviations
            gammaGPU,                        // gamma parameters
            gradients,                       // incoming gradients
            backOutput.deltaLoss,            // output: deltaloss   // maps to the same input gradients array
            neurons,
            output.batchSize,
            stream
        );

        return backOutput;
    }

    @Override
    public void applyGradients(BackpropagateOutput gradients, float learningRate, float weightDecay) {
        if(!(gradients instanceof LayerNormalizerBackpropagateOutput))
            throw new RuntimeException("LayerNormalizerBackpropagateOutput required for applying gradients");

        LayerNormalizerBackpropagateOutput lnGradients = (LayerNormalizerBackpropagateOutput)gradients;

        // Update gamma and beta for each neuron
        for(int i = 0; i < neurons; i++) {
            gamma[i] -= learningRate * lnGradients.gammaGradients[i];
            beta[i] -= learningRate * lnGradients.betaGradients[i];
        }
    }

    @Override
    public void applyGradientsGPU(BackpropagateOutputGPU gradients, float learningRate, float weightDecay, CUstream stream) {
        if(!(gradients instanceof LayerNormalizerBackpropagateOutputGPU))
            throw new RuntimeException("LayerNormalizerBackpropagateOutputGPU required for applying gradients");

        if(!gpuReady())
            throw new RuntimeException("Layer normalizer not prepared for GPU");

        cublasHandle handle = CudaEngine.getCublasHandle(deviceId);
        
        LayerNormalizerBackpropagateOutputGPU lnGradients = (LayerNormalizerBackpropagateOutputGPU)gradients;
        
        float alpha = -learningRate;

        synchronized(handle) {
            JCublas2.cublasSetStream(handle, new cudaStream_t(stream));

            // Update gamma using cuBLAS: gamma = gamma - learningRate * gammaGradients
            JCublas2.cublasSaxpy(handle, neurons, Pointer.to(new float[]{alpha}), lnGradients.gammaGradients, 1, gammaGPU, 1);

            // Update beta using cuBLAS: beta = beta - learningRate * betaGradients
            JCublas2.cublasSaxpy(handle, neurons, Pointer.to(new float[]{alpha}), lnGradients.betaGradients, 1, betaGPU, 1);
        }
    }
    
    @Override
    public boolean gpuReady() {
        return gammaGPU != null && betaGPU != null;
    }

    @Override
    protected void prepareGPU0(CUstream stream) {
        gammaGPU = CudaUtil.toGPUAsync(gamma, stream);
        betaGPU = CudaUtil.toGPUAsync(beta, stream);
    }

    @Override
    protected void freeGPU0(CUstream stream) {
        CudaUtil.freeAsync(gammaGPU, stream);
        CudaUtil.freeAsync(betaGPU, stream);

        gammaGPU = null;
        betaGPU = null;
    }

    @Override
    public int getInputSize() {
        return neurons;
    }

    @Override
    public int getOutputSize() {
        return neurons;
    }

    @Override
    public LayerNormalizer copy() {
        LayerNormalizer copy = new LayerNormalizer(neurons);
        System.arraycopy(gamma, 0, copy.gamma, 0, neurons);
        System.arraycopy(beta, 0, copy.beta, 0, neurons);
        return copy;
    }

    @Override
    public HashMap serialize() {
        HashMap obj = new HashMap();
        obj.put("type", "LayerNormalizer");
        obj.put("neurons", neurons);
        obj.put("gamma", util.base64encode(util.compress(util.toByteArray(gamma))));
        obj.put("beta", util.base64encode(util.compress(util.toByteArray(beta))));
        return obj;
    }

    public static LayerNormalizer deserialize(Map serialized) {
        int neurons = (Integer)serialized.get("neurons");

        LayerNormalizer normalizer = new LayerNormalizer(neurons);

        byte[] gammaBytes = util.decompress(util.base64decode((String)serialized.get("gamma")));
        byte[] betaBytes = util.decompress(util.base64decode((String)serialized.get("beta")));

        float[] gamma = util.toFloatArray(gammaBytes);
        float[] beta = util.toFloatArray(betaBytes);

        System.arraycopy(gamma, 0, normalizer.gamma, 0, neurons);
        System.arraycopy(beta, 0, normalizer.beta, 0, neurons);

        return normalizer;
    }

    @Override
    public void updateWeightsFromGPU() {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU");
        
        CUstream stream = CudaUtil.createStream();
        
        // Copy gamma and beta from GPU to CPU
        float[] gammaTemp = CudaUtil.fromGPUFloat(gammaGPU, gamma.length);
        float[] betaTemp = CudaUtil.fromGPUFloat(betaGPU, beta.length);
        
        System.arraycopy(gammaTemp, 0, gamma, 0, gamma.length);
        System.arraycopy(betaTemp, 0, beta, 0, beta.length);
        
        JCudaDriver.cuStreamDestroy(stream);
    }

    @Override
    public long getParametersCount() {
        return neurons * 2; // gamma and beta for each neuron
    }

    @Override
    public long getBackpropagateMemoryRequired(int batchSize) {
        return neurons * batchSize * Float.SIZE * 4; // prenorm, postnorm, means, variances
    }

    public float[] getGamma() {
        return gamma;
    }

    public float[] getBeta() {
        return beta;
    }
}
