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

import java.io.Serializable;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.cuda.CudaEngine.CUdeviceptr2D;

/**
 *
 * @author ahmed
 */
public abstract class Activation implements Serializable {
    
    public abstract float compute(float input);
    
    
    public abstract void compute(float[] input);
    
    public abstract void computeGPU(CUdeviceptr ptr, int size, CUstream stream);
    
    public abstract void computeMultiGPU(CUdeviceptr2D ptr, int width, int height, CUstream stream);
    
    
    public abstract void computeConditional(float[] input, boolean[] compute);

    public abstract void computeGPUConditional(CUdeviceptr ptr, CUdeviceptr compute, int size, CUstream stream);
    
    public abstract void computeMultiGPUConditional(CUdeviceptr2D ptr, CUdeviceptr compute, int width, int height, CUstream stream);

    public static String toName(Activation activation) {
        if(activation == null)
            return null;
        
        return activation.getClass().getSimpleName();
    }
    
    public static Activation fromName(String name) {
        if(name == null)
            return null;
        
        switch(name.toLowerCase()) {
            case "relu":
                return new ReLU();
            case "sigmoid":
                return new Sigmoid();
            case "sin":
                return new Sin();
            case "softmax":
                return new SoftMax();
            case "step":
                return new Step();
            case "tanh":
                return new Tanh();
        }
        
        return null;
    }
    
// TODO
//    ActivationBiPolar,
//    ActivationBipolarSteepenedSigmoid,
//    ActivationClippedLinear,
//    ActivationCompetitive,
//    ActivationElliott,
//    ActivationElliottSymmetric,
//    ActivationGaussian,
//    ActivationLOG,
//    ActivationLinear,
//    ActivationRamp,
//    ActivationSteepenedSigmoid,
}
