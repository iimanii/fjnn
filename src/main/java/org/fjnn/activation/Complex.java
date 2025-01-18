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
import java.util.LinkedHashSet;
import java.util.SortedSet;
import java.util.TreeSet;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.cuda.CUdeviceptr2D;

/**
 * First x nodes represent mathematical functions 
 * @author ahmed
 */
public class Complex extends Activation {
    int count;
    Activation activation;
    function[] functions;

    @Override
    public void compute(FloatBuffer input, int stride, int count) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void computeGPU(CUdeviceptr ptr, long stride, long size, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public float derivative(float preactivation, float postactivation) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void derivative(float[] preactivation, float[] postactivation, float[] output, int stride, int count) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void derivativeGPU(CUdeviceptr preactivation, CUdeviceptr postactivation, CUdeviceptr output, long stride, long count, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void gradient(float[] preActivation, float[] postActivation, float[] gradient, int stride, int count) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void gradientGPU(CUdeviceptr preActivation, CUdeviceptr postActivation, CUdeviceptr gradient, long stride, long count, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    private static interface calc {
        float compute(float a);
    }
    
    public static enum function {
        linear((float a) -> {return a; }),
        inv((float a) -> { return SafeInv(a); }),
        pow2((float a) -> { return SafePow2(a); }),
        pow3((float a) -> { return SafePow3(a); }),
        log((float a) -> { return SafeLog(a); }),
        logmirror((float a) -> { return MirrorLog(a); }),
        sqrt((float a) -> { return SafeSqrt(a); }),
        sqrtmirror((float a) -> { return MirrorSqrt(a); });
        
        private final calc c;

        private function(calc c) {
            this.c = c;
        }
        
        float compute(float a) {
            return c.compute(a);
        }
    }

    public Complex(Activation activation, int count) {
        this.count = count;
        this.functions = function.values();
        this.activation = activation;
    }
    
    public Complex(Activation activation, int count, LinkedHashSet<function> functions) {
        this.count = count;
        this.functions = functions.toArray(new function[functions.size()]);
        this.activation = activation;
    }

    protected static Complex parse(String name) {
        String[] split = name.toLowerCase().split("-");
        
        if(split.length < 3)
            return null;
        
        if(!split[0].equals("complex"))
            return null;
        
        Activation activation = Activation.fromName(split[1]);
        int count = Integer.parseInt(split[2]);
        LinkedHashSet<function> s = new LinkedHashSet<>();
        
        for(int i=3; i < split.length; i++) {
            function f = function.valueOf(split[i]);
            if(f == null)
                throw new RuntimeException("Invalid Complex Function: " + split[i]);
            s.add(f);
        }
        
        return new Complex(activation, count, s);
    }
    
    @Override
    public String toName() {
        StringBuilder b = new StringBuilder();
        b.append(this.getClass().getSimpleName()).append("-")
                     .append(Activation.toName(activation)).append("-").append(count);
        
        for(function f : functions) {
            b.append("-").append(f.name());
        }
        
        return b.toString();
    }
    
    @Override
    public float compute(float input) {
        throw new RuntimeException("Function not supported");
    }
    
    @Override
    public void compute(float[] input, int from, int to) {
        int c = from;
        
        for(int i=0; i < count; i++) {
            for(function f : functions) {
                input[c] = f.compute(input[c]);
                c++;
            }
        }
        
        activation.compute(input, c, input.length);
    }
}
