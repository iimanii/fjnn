/*
 * The MIT License
 *
 * Copyright 2024 ahmed.
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
package org.fjnn.loss;

import java.util.HashMap;
import java.util.Map;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;

/**
 *
 * @author ahmed
 */
public abstract class Loss {
    
    public static Loss MeanSquareError = new MeanSquareError();
    
    abstract public float compute(float[] output, float[] expected);
    
    abstract public void computeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream);
    
    abstract public float[] derivative(float[] output, float[] expected);
   
    abstract public void derivativeGPU(CUdeviceptr output, CUdeviceptr expected, CUdeviceptr result, long size, CUstream stream);
    
    public Map serialize() {
        HashMap result = new HashMap();
        result.put("type", getClass().getSimpleName());
        return result;
    }
    
    public static Loss deserialize(Map serialized) {
        String type = (String)serialized.get("type");

        switch(type) {
            case "MeanSquareError":
                return new MeanSquareError();
            case "BinaryCrossEntropy":
                return BinaryCrossEntropy.deserialize(serialized);
            case "WeightedMeanSquareError":
                throw new RuntimeException("WeightedMeanSquareError deserialization not implemented");
            case "FocalLoss":
                return FocalLoss.deserialize(serialized);
            default:
                throw new RuntimeException("Unknown loss type: " + type);
        }
    }
    
    abstract public String name();
}
