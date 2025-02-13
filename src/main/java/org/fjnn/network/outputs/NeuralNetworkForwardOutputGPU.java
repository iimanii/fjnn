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
package org.fjnn.network.outputs;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.activation.output.ActivationForwardOutput;
import org.fjnn.activation.output.ActivationForwardOutputGPU;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.normalizer.outputs.DropoutForwardOutputGPU;

/**
 *
 * @author ahmed
 */
public class NeuralNetworkForwardOutputGPU extends FeedForwardOutputGPU {
    public final int layers;
    
    public final CUdeviceptr[] layerInputs;                         // initial input to each layer
    public final FeedForwardOutputGPU[] normalizerOutputs;          // stores normalizer outputs for each layer
    public final ActivationForwardOutputGPU[] activationOutputs;    // activation results
    public final DropoutForwardOutputGPU[] dropoutOutputs;
    public final CUdeviceptr[] layerOutputs;                        // outputs from the layer activation
   
    public NeuralNetworkForwardOutputGPU(int outputDim, int batchSize, int layerCount) {
        super(outputDim, batchSize);
        
        this.layers = layerCount;
        this.layerInputs = new CUdeviceptr[layerCount];
        this.normalizerOutputs = new FeedForwardOutputGPU[layerCount];
        this.activationOutputs = new ActivationForwardOutputGPU[layerCount];
        this.dropoutOutputs = new DropoutForwardOutputGPU[layerCount];
        this.layerOutputs = new CUdeviceptr[layerCount];
    }

    @Override
    public CUdeviceptr output() {
        return layerOutputs[layerOutputs.length - 1];
    }

    @Override
    public void free() {
        /* first layer has initial input .. do not free */
        for (int i=1; i < layers; i++) {
            if(layerInputs[i] != null) {
                CudaUtil.free(layerInputs[i]);
                layerInputs[i] = null;
            }
            
            if(normalizerOutputs[i] != null) {
                normalizerOutputs[i].free();
                normalizerOutputs[i] = null;
            }
            
            if(activationOutputs[i] != null) {
                activationOutputs[i].free();
                activationOutputs[i] = null;
            }
            
           if(dropoutOutputs[i] != null) {
               dropoutOutputs[i].free();
               dropoutOutputs[i] = null;
           }
        }
    }

    @Override
    public void freeAsync(CUstream stream) {
        /* first layer has initial input .. do not free */
        for (int i=1; i < layers; i++) {
            if(layerInputs[i] != null) {
                CudaUtil.freeAsync(layerInputs[i], stream);
                layerInputs[i] = null;
            }
            
            if(normalizerOutputs[i] != null) {
                normalizerOutputs[i].freeAsync(stream);
                normalizerOutputs[i] = null;
            }
            
            if(activationOutputs[i] != null) {
                activationOutputs[i].freeAsync(stream);
                activationOutputs[i] = null;
            }
            
           if(dropoutOutputs[i] != null) {
               dropoutOutputs[i].freeAsync(stream);
               dropoutOutputs[i] = null;
           }
        }
    }
}
