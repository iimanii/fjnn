/*
 * The MIT License
 *
 * Copyright 2018 ahmed.
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
package org.fjnn.util;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/**
 *
 * @author ahmed
 */
public class intrinsic {
//    
//    static {
//        System.loadLibrary("intrinsic.dll");
//    }
    
    public static native ByteBuffer createAlignedDirectBuffer(int length, int alignment);
    public static native float dotProductByteBufferLoop(FloatBuffer a, FloatBuffer b, int length);
    public static native float dotProductByteBufferInline(FloatBuffer a, FloatBuffer b, int length);
    public static native float dotProductByteBufferAsm256(FloatBuffer a, FloatBuffer b, int length);
    public static native float dotProductByteBufferAsm256fma(FloatBuffer a, FloatBuffer b, int length);
    public static native float dotProductByteBufferMKL(FloatBuffer a, FloatBuffer b, int length);
    
    public static native void sgemv(FloatBuffer input, FloatBuffer result, FloatBuffer weights, FloatBuffer bias, int neurons, int links);
    public static native void sgemm(FloatBuffer inputs, int inputsCount, FloatBuffer result, FloatBuffer weights, FloatBuffer bias, int neurons, int links);

    public static native void ReLU(FloatBuffer input, int len);
    public static native void Sigmoid(FloatBuffer input, int len);
    public static native void Sin(FloatBuffer input, int len);
    public static native void SoftMax(FloatBuffer input, int len, int count);
    public static native void Step(FloatBuffer input, int len);
    public static native void Tanh(FloatBuffer input, int len);
    
    public static native void Add(FloatBuffer a, FloatBuffer b, int len);
}
