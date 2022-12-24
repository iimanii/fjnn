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
import org.fjnn.cuda.CUdeviceptr2D;

/**
 *
 * @author ahmed
 */
public abstract class Activation implements Serializable {
    
    public abstract float compute(float input);
    
    public final void compute(float[] input) {
        compute(input, 0, input.length);
    }
    
    public abstract void compute(float[] input, int from, int to);
    
    public abstract void computeGPU(CUdeviceptr ptr, int size, CUstream stream);
    
    public abstract void computeConditional(float[] input, boolean[] compute);
    
    public final void derivative(float[] input) {
        derivative(input, 0, input.length);
    }
    
    public abstract void derivative(float[] input, int from, int to);
    
    public int minLayerSize() {
        return 0;
    }
    
    public String toName() {
        return this.getClass().getSimpleName();
    }
    
    public static String toName(Activation activation) {
        if(activation == null)
            return null;
        
        return activation.toName();
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
        
        if(name.toLowerCase().startsWith("complex"))
            return Complex.parse(name);
        
        return null;
    }
    
    static float FLOAT_MAX_VALUE = 1.0e20f;
    
    protected static float SafeInv(float a) {
        if(a == 0)
            return 0;
        
        return (float) Math.min(1.0f / a, FLOAT_MAX_VALUE);
    }
    
    protected static float SafePow2(float a) {
        return (float) Math.min(Math.pow(a, 2), FLOAT_MAX_VALUE);
    }
    
    protected static float SafePow3(float a) {
        if(a < 0)
            return (float) Math.max(Math.pow(a, 3), -FLOAT_MAX_VALUE);
        else
            return (float) Math.min(Math.pow(a, 3), FLOAT_MAX_VALUE);
    }
    
    protected static float SafeExp(float a) {
        return (float) Math.min(Math.exp(a), FLOAT_MAX_VALUE);
    }
    
    protected static float SafeLog(float a) {
        if(a <= 0)
            return 0;
        
        return (float) Math.max(Math.log(a), -FLOAT_MAX_VALUE);
    }

    protected static float MirrorLog(float a) {
        return (float) (Math.signum(a) * Math.max(Math.log(Math.abs(a)), -FLOAT_MAX_VALUE));
    }
    
    protected static float SafeSqrt(float a) {
        return (float) Math.sqrt(Math.max(0, a));
    }
    
    protected static float MirrorSqrt(float a) {
        int sign = a < 0 ? -1 : 1;
        return (float) (sign * Math.sqrt(Math.abs(a)));
    }
    
    
    
    public abstract void computeMultiGPU(CUdeviceptr2D ptr, int width, int height, CUstream stream);    

    public abstract void computeGPUConditional(CUdeviceptr ptr, CUdeviceptr compute, int size, CUstream stream, int count);
    
    public abstract void computeMultiGPUConditional(CUdeviceptr2D ptr, CUdeviceptr compute, int width, int height, CUstream stream);

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
