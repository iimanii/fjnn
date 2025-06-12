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
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.activation.output.ActivationForwardOutput;
import org.fjnn.activation.output.ActivationForwardOutputGPU;

/**
 *
 * @author ahmed
 */
public abstract class Activation implements Serializable {
    
    public abstract float compute(float input);
    
    public void compute(float[] input, int inputDim, int batchSize) {
        compute(input, input, inputDim, batchSize);
    }
    
    public abstract void compute(float[] input, float[] output, int inputDim, int batchSize);
    
    public ActivationForwardOutput feedForward(float[] input, int inputDim, int batchSize) {
        ActivationForwardOutput activationOutput = new ActivationForwardOutput(input, inputDim, batchSize);
        compute(activationOutput.preActivation, activationOutput.postActivation, inputDim, batchSize);
        
        return activationOutput;
    }
    
    public abstract float derivative(float preActivation, float postActivation);
    
    /* calculates derivative and place result in output */
    public abstract void derivative(float[] preActivation, float[] postActivation, float[] output, int inputDim, int batchSize);
        
    /* calculates derivative and multiplies it by gradient, puts result in gradient */
    public abstract void gradient(float[] preActivation, float[] postActivation, float[] gradient, int inputDim, int batchSize);
    
    
    public void computeGPU(CUdeviceptr input, int inputDim, int batchSize, CUstream stream) {
        computeGPU(input, input, inputDim, batchSize, stream);
    }
    
    public abstract void computeGPU(CUdeviceptr input, CUdeviceptr output, int inputDim, int batchSize, CUstream stream);
    
    public ActivationForwardOutputGPU feedForwardGPU(CUdeviceptr input, int inputDim, int batchSize, CUstream stream) {
        ActivationForwardOutputGPU activationOutput = new ActivationForwardOutputGPU(input, inputDim, batchSize, stream);
        computeGPU(activationOutput.preActivation, activationOutput.postActivation, inputDim, batchSize, stream);
        
        return activationOutput;
    }
        
    public abstract void derivativeGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr output, int inputDim, int batchSize, CUstream stream);
    
    public abstract void gradientGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, int inputDim, int batchSize, CUstream stream);

    
    public Map serialize() {
        HashMap obj = new HashMap();
        obj.put("type", getClass().getSimpleName());
        return obj;
    }
    
    public static Activation deserialize(Map serialized) {
        String type = (String)serialized.get("type");
        
        switch(type) {
            case "ReLU":
                return new ReLU();
            case "LeakyReLU":
                return LeakyReLU.deserialize(serialized);
            case "Sigmoid":
                return new Sigmoid();
            case "Sin":
                return new Sin();
            case "SoftMax":
                return new SoftMax();
            case "Step":
                return new Step();
            case "Tanh":
                return new Tanh();
            case "Linear":
                return new Linear();
            case "Swish":
                return new Swish();
            default:
                throw new RuntimeException("Unknown activation type: " + type);
        }
    }
    
    public String toName() {
        return this.getClass().getSimpleName();
    }
    
    public static String toName(Activation activation) {
        if(activation == null)
            return null;
        
        return activation.toName();
    }
    
    @Deprecated
    public static Activation fromName(String name) {
        if(name == null)
            return null;
        
        switch(name.toLowerCase()) {
            case "relu":
                return new ReLU();
            case "leakyrelu":
                return new LeakyReLU();
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
            case "linear":
                return new Linear();
            case "swish":
                return new Swish();
        }
        
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
    
    public void compute(FloatBuffer input, FloatBuffer output, int inputDim, int batchSize) {
        throw new UnsupportedOperationException("Not supported");
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
