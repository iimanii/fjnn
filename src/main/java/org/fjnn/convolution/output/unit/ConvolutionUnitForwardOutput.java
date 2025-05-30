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
package org.fjnn.convolution.output.unit;

/**
 *
 * @author ahmed
 */
public class ConvolutionUnitForwardOutput {
    public final int unitCount;
    public final int outputOrigin;  /* the input position that maps to this unit's output[0] */
    
    public final float[] output;
    public final int outputSize;
    public final int batchSize;
    
    // For backpropagation
    public final float[] input;
    public final float[][] kernelOutputs;      // [kernel_index][raw_output]
    public final float[][] sigmoidOutputs;     // [kernel_index][sigmoid_output] or null for single kernel
    public final float[][] channelInputs;      // [kernel_index][channel_input] or null for single kernel
    
    // Single kernel constructor (for Kernel class)
    public ConvolutionUnitForwardOutput(float[] output, int outputSize, int batchSize, float[] forwardPassInputs, int unitCount) {
        this.unitCount = unitCount;
        this.outputOrigin = (unitCount - 1) / 2;
        
        this.output = output;
        this.outputSize = outputSize;
        this.batchSize = batchSize;
        
        this.input = forwardPassInputs;
        this.kernelOutputs = null;
        this.sigmoidOutputs = null;
        this.channelInputs = null;
    }
    
    // Multi-kernel constructor (for KernelGroup class)
    public ConvolutionUnitForwardOutput(float[] finalOutput, int outputSize, int batchSize,
                                        float[] forwardPassInputs, int unitCount,
                                        float[][] kernelOutputs, float[][] sigmoidOutputs,
                                        float[][] channelInputs) {
        this.unitCount = unitCount;
        this.outputOrigin = (unitCount - 1) / 2;
        
        this.output = finalOutput;
        this.outputSize = outputSize;
        this.batchSize = batchSize;
        
        this.input = forwardPassInputs;
        this.kernelOutputs = kernelOutputs;
        this.sigmoidOutputs = sigmoidOutputs;
        this.channelInputs = channelInputs;
    }
}
