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
package org.fjnn.adapter;

import java.util.HashMap;
import java.util.Map;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.adapter.output.AdapterBackpropagateOutput;
import org.fjnn.adapter.output.AdapterBackpropagateOutputGPU;
import org.fjnn.adapter.output.AdapterForwardOutput;
import org.fjnn.adapter.output.AdapterForwardOutputGPU;
import org.fjnn.base.output.FeedForwardOutput;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.base.ModelComponent;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.loss.Loss;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 */
public class PositionalEncoderAdapter extends ModelComponent {
    private final int unitDim;          // Number of features per unit (dimensionality)
    private final int unitCount;        // Number of units per input sequence
    private final int totalFeatures;    // Total number of features being processed
    private final int offset;
    
    private final float[][] positionalEncodings;    // Precomputed positional encodings
    private final float[] positionalEncodings1D;    // Flattened positional encodings (1D)
    
    private CUdeviceptr positionalEncodingsGPU;
    private boolean gpuReady;
    
    public PositionalEncoderAdapter(int unitDim, int unitCount) {
        this(unitDim, unitCount, 0);
    }
    
    public PositionalEncoderAdapter(int unitDim, int unitCount, int offset) {
        this.unitDim = unitDim;
        this.unitCount = unitCount;
        this.totalFeatures = unitDim * unitCount;
        this.offset = offset;
        this.positionalEncodings = generateSinusoidalEncodings(unitDim, unitCount, offset);
        this.positionalEncodings1D = flattenPositionalEncodings(positionalEncodings);
        this.gpuReady = false;
    }

    /*
        Positional Encoding Formula:
        For a given position `pos` and feature dimension `i`:
        PE(pos, 2i)   = sin(pos / 10000^(2i / featureSize))    // For even dimensions
        PE(pos, 2i+1) = cos(pos / 10000^(2i / featureSize))    // For odd dimensions

        - Alternates between sine (even indices) and cosine (odd indices).
        - `featureSize` is the total dimensionality of the encoding.
    */
    private float[][] generateSinusoidalEncodings(int unitDim, int unitCount, int offset) {
        float[][] encoding = new float[unitCount][unitDim];
        for (int pos = 0; pos < unitCount; pos++) {
            int adjusted = pos + offset;
            
            for (int i = 0; i < unitDim; i++) {
                if (i % 2 == 0) {
                    encoding[pos][i] = (float) Math.sin(adjusted / Math.pow(10000, (2.0 * i) / unitDim));
                } else {
                    encoding[pos][i] = (float) Math.cos(adjusted / Math.pow(10000, (2.0 * (i - 1)) / unitDim));
                }
            }
        }
        return encoding;
    }
    
    private float[] flattenPositionalEncodings(float[][] encodings2D) {
        int totalSize = unitCount * unitDim;
        float[] flattened = new float[totalSize];
        for (int i = 0; i < unitCount; i++) {
            System.arraycopy(encodings2D[i], 0, flattened, i * unitDim, unitDim);
        }
        return flattened;
    }

    @Override
    public AdapterForwardOutput feedForward(float[] input, int batchSize) {
        // Ensure input size matches the expected dimensions
        if (input.length != batchSize * totalFeatures) {
            throw new IllegalArgumentException(
                "Input size mismatch. Expected: " + (batchSize * unitCount * unitDim) + ", but got: " + input.length
            );
        }

        // Add positional encodings
        float[] encodedInput = new float[input.length];
        for(int i = 0; i < batchSize; i++) {
            for(int j = 0; j < unitCount; j++) {
                for(int k = 0; k < unitDim; k++) {
                    int idx = i * totalFeatures + j * unitDim + k;
                    encodedInput[idx] = input[idx] + positionalEncodings[j][k];
                }
            }
        }

        return new AdapterForwardOutput(totalFeatures, batchSize, encodedInput);
    }

    @Override
    public AdapterForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchSize, CUstream stream) {
        // Calculate the total number of elements
        long totalElements = batchSize * totalFeatures;

        // Allocate GPU memory for the output
        CUdeviceptr outputGPU = CudaUtil.createFloatAsync(totalElements, stream);

        // add encodings
        CudaFunctions.vector.addStride(input, positionalEncodingsGPU, outputGPU, totalFeatures, batchSize, stream);
        
        // Return the output
        return new AdapterForwardOutputGPU(totalFeatures, batchSize, outputGPU, false);
    }

    @Override
    public boolean gpuReady() {
        return gpuReady;
    }

    @Override
    protected void prepareGPU0(CUstream stream) {
        positionalEncodingsGPU = CudaUtil.toGPUAsync(positionalEncodings1D, stream);
        
        gpuReady = true;
    }

    @Override
    protected void freeGPU0(CUstream stream) {
        // Free positional encodings GPU memory
        CudaUtil.freeAsync(positionalEncodingsGPU, stream);
        
        gpuReady = false;
    }

    @Override
    public int getInputDim() {
        return totalFeatures;
    }

    @Override
    public int getOutputDim() {
        return totalFeatures;
    }

@Override
    public BackpropagateOutput backpropagate(FeedForwardOutput output, float[] deltaLoss) {
        // Since positional encodings are constant, simply pass back the deltaLoss unchanged
        return new AdapterBackpropagateOutput(totalFeatures, output.batchSize, deltaLoss);
    }

    @Override
    public BackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU output, CUdeviceptr deltaLoss, CUstream stream) {
        // Same logic for GPU: pass deltaLoss unchanged
        return new AdapterBackpropagateOutputGPU(totalFeatures, output.batchSize, deltaLoss, false);
    }

    @Override
    public ModelComponent copy() {
        return new PositionalEncoderAdapter(unitDim, unitCount);
    }
    
    @Override
    public HashMap serialize() {
       HashMap obj = new HashMap();

       // Store component type and main properties
       obj.put("type", "PositionalEncoderAdapter");
       obj.put("unitDim", unitDim);
       obj.put("unitCount", unitCount);
       obj.put("offset", offset);

       return obj;
    }

    public static PositionalEncoderAdapter deserialize(Map serialized) {
       int unitDim = (Integer)serialized.get("unitDim");
       int unitCount = (Integer)serialized.get("unitCount"); 
       int offset = (Integer)serialized.get("offset"); 
       
       return new PositionalEncoderAdapter(unitDim, unitCount, offset);
    }

    @Override
    public void updateWeightsFromGPU() {
        
    }
    
    @Override
    public long getParametersCount() {
        return totalFeatures;
    }
    
    @Override
    public long getBackpropagateMemoryRequired(int batchSize) {
        /* feedforward creates a new adjusted array */
        return totalFeatures * batchSize * Float.SIZE;
    }

    @Override
    public void applyGradients(BackpropagateOutput gradients, float learningRate, float weightDecay) {
        // no learnable parameters .. do nothing
    }

    @Override
    public void applyGradientsGPU(BackpropagateOutputGPU gradients, float learningRate, float weightDecay, CUstream stream) {
        // no learnable parameters .. do nothing
    }
}

