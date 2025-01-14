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
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class NeuralNetworkForwardOutputGPU extends FeedForwardOutputGPU {
    
    public final int layers;
    public final CUdeviceptr[] preActivations; // Stores z_x (pre-activation) for each layer
    public final CUdeviceptr[] postActivations; // Stores a_x (post-activation) for each layer
    public final boolean[] zeroed; // Stores a_x (post-activation) for each layer

    public NeuralNetworkForwardOutputGPU(int batchSize, int batchCount, int layerCount) {
        super(batchSize, batchCount);
        
        this.layers = layerCount;
        this.preActivations = new CUdeviceptr[layerCount];
        this.postActivations = new CUdeviceptr[layerCount];
        this.zeroed = new boolean[layerCount];
    }

    @Override
    public CUdeviceptr output() {
        return postActivations[postActivations.length - 1];
    }

    @Override
    public void free() {
        /* do not free input (preActivations[0] / postActivation[0]) */
        for (int i = 1; i < preActivations.length; i++) {
            CudaUtil.free(preActivations[i]);
            if (preActivations[i] != postActivations[i]) {
                CudaUtil.free(postActivations[i]);
            }
            preActivations[i] = null;
            postActivations[i] = null;
        }
        preActivations[0] = null;
        postActivations[0] = null;
    }

    @Override
    public void freeAsync(CUstream stream) {
        /* do not free input (preActivations[0] / postActivation[0]) */
        for (int i = 1; i < preActivations.length; i++) {
            //                System.out.println(i + " " + preActivations[i] + " " + postActivations[i]);
            CudaUtil.freeAsync(preActivations[i], stream);
            if (preActivations[i] != postActivations[i]) {
                CudaUtil.freeAsync(postActivations[i], stream);
            }
            preActivations[i] = null;
            postActivations[i] = null;
        }
        preActivations[0] = null;
        postActivations[0] = null;
    }

    
}
