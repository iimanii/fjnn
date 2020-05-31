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

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import jcuda.driver.CUstream;
import org.fjnn.activation.Activation;
import org.fjnn.base.BaseLayer;
import org.fjnn.cuda.CudaEngine.CUdeviceptr2D;
import org.fjnn.serializer.LayerStub;

/**
 *
 * @author ahmed
 */
public abstract class ParallelLayer extends BaseLayer {
    /* number of similar layers */
    protected final int layersCount;
    
    protected final static ExecutorService THREAD_POOL = Executors.newCachedThreadPool();
    
    public ParallelLayer(int count, Activation activation, int neurons, int links, boolean hasBias, boolean[] condition) {
        super(activation, neurons, links, hasBias, condition);
        
        this.layersCount = count;
    }
    
    
    protected abstract float getWeight(int index, int from, int to);

    protected abstract void setWeight(int index, int from, int to, float value);

    protected abstract void setWeights(int index, float[][] values);

    protected abstract float[][] getWeights(int index);
    
    
    protected abstract float getBias(int index, int to);

    protected abstract void setBias(int index, int to, float value);
    
    protected abstract void setBiases(int index, float[] values);    
    
    protected abstract float[] getBiases(int index);

    
    protected abstract float[][] feedForward(float[][] input);
    
    protected abstract CUdeviceptr2D feedForwardGPU(CUdeviceptr2D ptr, CUstream stream);
    
        
    protected abstract LayerStub getStub(int index);
}
