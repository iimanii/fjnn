/*
 * The MIT License
 *
 * Copyright 2025 ahmed.
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
package org.fjnn.trainer.backpropagate.outputs;

import jcuda.driver.CUstream;
import org.fjnn.network.outputs.NeuralNetworkBackpropagateOutputGPU;
import org.fjnn.network.outputs.NeuralNetworkForwardOutputGPU;

/**
 *
 * @author ahmed
 */
public class TrainingSessionOutputGPU {
    public final NeuralNetworkForwardOutputGPU forwardOutput;
    public final NeuralNetworkBackpropagateOutputGPU backwardOutput;
    public final float[] result;  // CPU copy for convenience
    public final double loss;
    
    public TrainingSessionOutputGPU(NeuralNetworkForwardOutputGPU forwardOutput, 
                                   NeuralNetworkBackpropagateOutputGPU backwardOutput,
                                   float[] result,
                                   double loss) {
        this.forwardOutput = forwardOutput;
        this.backwardOutput = backwardOutput;
        this.result = result;
        this.loss = loss;
    }
    
    /**
     * Free GPU memory for this session output
     */
    public void free() {
        if (forwardOutput != null) {
            forwardOutput.free();
        }
        if (backwardOutput != null) {
            backwardOutput.free();
        }
    }
    
    /**
     * Free GPU memory asynchronously
     */
    public void freeAsync(CUstream stream) {
        if (forwardOutput != null) {
            forwardOutput.freeAsync(stream);
        }
        if (backwardOutput != null) {
            backwardOutput.freeAsync(stream);
        }
    }
}