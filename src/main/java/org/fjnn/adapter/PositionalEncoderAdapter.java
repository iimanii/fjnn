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
    private final int featureSize;      // Number of features per unit (dimensionality)
    private final int featureCount;     // Number of units per input sequence
    private final int totalFeatures;    // Total number of features being processed
    
    private final float[][] positionalEncodings;    // Precomputed positional encodings
    private final float[] positionalEncodings1D;    // Flattened positional encodings (1D)
    
    private CUdeviceptr positionalEncodingsGPU;
    private boolean gpuReady;
    
    public PositionalEncoderAdapter(int featureSize, int featureCount) {
        this.featureSize = featureSize;
        this.featureCount = featureCount;
        this.totalFeatures = featureSize * featureCount;
        this.positionalEncodings = generateSinusoidalEncodings(featureSize, featureCount);
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
    private float[][] generateSinusoidalEncodings(int featureSize, int featureCount) {
        float[][] encoding = new float[featureCount][featureSize];
        for (int pos = 0; pos < featureCount; pos++) {
            for (int i = 0; i < featureSize; i++) {
                if (i % 2 == 0) {
                    encoding[pos][i] = (float) Math.sin(pos / Math.pow(10000, (2.0 * i) / featureSize));
                } else {
                    encoding[pos][i] = (float) Math.cos(pos / Math.pow(10000, (2.0 * (i - 1)) / featureSize));
                }
            }
        }
        return encoding;
    }
    
    private float[] flattenPositionalEncodings(float[][] encodings2D) {
        int totalSize = featureCount * featureSize;
        float[] flattened = new float[totalSize];
        for (int i = 0; i < featureCount; i++) {
            System.arraycopy(encodings2D[i], 0, flattened, i * featureSize, featureSize);
        }
        return flattened;
    }

    @Override
    public AdapterForwardOutput feedForward(float[] input, int batchCount) {
        // Ensure input size matches the expected dimensions
        if (input.length != batchCount * totalFeatures) {
            throw new IllegalArgumentException(
                "Input size mismatch. Expected: " + (batchCount * featureCount * featureSize) + ", but got: " + input.length
            );
        }

        // Add positional encodings
        float[] encodedInput = new float[input.length];
        for(int i = 0; i < batchCount; i++) {
            for(int j = 0; j < featureCount; j++) {
                for(int k = 0; k < featureSize; k++) {
                    int idx = i * totalFeatures + j * featureSize + k;
                    encodedInput[idx] = input[idx] + positionalEncodings[j][k];
                }
            }
        }

        return new AdapterForwardOutput(totalFeatures, batchCount, encodedInput);
    }

    @Override
    public AdapterForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchCount, CUstream stream) {
        // Calculate the total number of elements
        int totalElements = batchCount * totalFeatures;

        // Allocate GPU memory for the output
        CUdeviceptr outputGPU = CudaUtil.createFloatAsync(totalElements, stream);

        // add encodings
        CudaFunctions.vector.addStride(input, positionalEncodingsGPU, outputGPU, totalFeatures, batchCount, stream);
        
        // Return the output
        return new AdapterForwardOutputGPU(totalFeatures, batchCount, outputGPU, false);
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
    public int getInputSize() {
        return totalFeatures;
    }

    @Override
    public int getOutputSize() {
        return totalFeatures;
    }

@Override
    public BackpropagateOutput backpropagate(FeedForwardOutput output, float[] deltaLoss, float learningRate) {
        // Since positional encodings are constant, simply pass back the deltaLoss unchanged
        return new AdapterBackpropagateOutput(totalFeatures, output.batchCount, deltaLoss);
    }

    @Override
    public BackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU output, CUdeviceptr deltaLoss, float learningRate, CUstream stream) {
        // Same logic for GPU: pass deltaLoss unchanged
        return new AdapterBackpropagateOutputGPU(totalFeatures, output.batchCount, deltaLoss, false);
    }

    @Override
    public ModelComponent copy() {
        return new PositionalEncoderAdapter(featureSize, featureCount);
    }
}

