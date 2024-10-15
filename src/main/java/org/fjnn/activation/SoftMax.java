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
package org.fjnn.activation;

import java.nio.FloatBuffer;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaModule;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.cuda.CUdeviceptr2D;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.util.intrinsic;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 */
public class SoftMax extends Activation {
    
    @Override
    public float compute(float input) {
        return 1.0f;
    }

    @Override
    public void compute(float[] input, int stride, int count) {
        for(int c=0; c < count; c++) {
            int from = c * stride;
            int to = from + stride;
            
            double sum = 0;
            float max = Float.NEGATIVE_INFINITY;

            for(int i=from; i < to; i++) {
                max = Math.max(input[i], max);
            }
//            
//            System.out.println("max: " + max);

            for(int i=from; i < to; i++) {
                input[i] = (float) Math.exp(input[i] - max);
                sum += input[i];
            }
            
//            System.out.println("sum: " + sum);
            if(sum == 0)
                for(int i=from; i < to; i++)
                    input[i] = 1.0f / input.length;
            else
                for(int i=from; i < to; i++)
                    input[i] /= sum;
        }
    }

    @Override
    public void computeGPU(CUdeviceptr ptr, long stride, long count, CUstream stream) {
        CudaFunctions.SoftMax(ptr, stride, count, stream);
    }

    /*
     * https://mmuratarat.github.io/2019-01-27/derivation-of-softmax-function
     * https://www.youtube.com/watch?v=09c7bkxpv9I
     */
    @Override
    public void derivative(float[] input, int from, int to) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void compute(FloatBuffer input, int stride, int count) {
        intrinsic.SoftMax(input, stride, count);
    }

    @Override
    public float derivative(float input) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void derivativeGPU(CUdeviceptr ptr, long stride, long size, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
