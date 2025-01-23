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
package org.fjnn.base.output;

/**
 *
 * @author ahmed
 */
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.cuda.CudaUtil;

public class BackpropagateOutputMapGPU {
    private final Map<Integer, BackpropagateOutputGPU> results;

    public BackpropagateOutputMapGPU() {
        this.results = new HashMap<>();
    }

    public void addOutput(int index, BackpropagateOutputGPU result) {
        if (results.containsKey(index)) {
            throw new IllegalArgumentException("Output at index " + index + " already exists.");
        }
        results.put(index, result);
    }

    public BackpropagateOutputGPU getOutput(int index) {
        return results.get(index);
    }

    public void free() {
        for (BackpropagateOutputGPU result : results.values()) {
            result.free();
        }
    }

    public void freeAsync(CUstream stream) {
        for (BackpropagateOutputGPU result : results.values()) {
            result.freeAsync(stream);
        }
    }
    
    public int size() {
        return results.size();
    }
}

