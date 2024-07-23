/*
 * The MIT License
 *
 * Copyright 2023 ahmed.
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
package org.fjnn.base;

import java.nio.FloatBuffer;
import jcuda.driver.CUdeviceptr;

/**
 *
 * @author ahmed
 */
public class NetworkInput {
    public static enum InputType {
        float_array,
        floatbuffer_array,
        deviceptr,
        deviceptr_list
    }
    
    float[] array;
    FloatBuffer array_buffer;
    CUdeviceptr deviceptr;
    CUdeviceptr[] deviceptr_list;
    
    public final int count;
    public final InputType type;

    public NetworkInput(float[] array, int count) {
        this.array = array;
        this.count = count;
        type = InputType.float_array;
    }
    
    public NetworkInput(FloatBuffer array_buffer, int count) {
        this.array_buffer = array_buffer;
        this.count = count;
        type = InputType.floatbuffer_array;
    }
    
    public NetworkInput(CUdeviceptr deviceptr, int count) {
        this.deviceptr = deviceptr;
        this.count = count;
        type = InputType.deviceptr;
    }
    
    public NetworkInput(CUdeviceptr[] deviceptr_list, int count) {
        this.deviceptr_list = deviceptr_list;
        this.count = count;
        type = InputType.deviceptr_list;
    }
    
    public float[] getInputArray() {
        return array;
    }
    
    public FloatBuffer getInputFloatBuffer() {
        return array_buffer;
    }
    
    public CUdeviceptr getInputDevicePtr() {
        return deviceptr;
    }
    
    public CUdeviceptr[] getInputDevicePtrList() {
        return deviceptr_list;
    }
}
