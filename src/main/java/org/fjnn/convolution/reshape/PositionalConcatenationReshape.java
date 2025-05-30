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
package org.fjnn.convolution.reshape;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import org.fjnn.convolution.output.ConvolutionForwardOutput;
import org.fjnn.convolution.output.ConvolutionForwardOutputGPU;
import org.fjnn.convolution.output.layer.ConvolutionLayerForwardOutput;
import org.fjnn.convolution.output.layer.ConvolutionLayerForwardOutputGPU;
import org.fjnn.convolution.output.unit.ConvolutionUnitForwardOutput;
import org.fjnn.convolution.output.unit.ConvolutionUnitForwardOutputGPU;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */
public class PositionalConcatenationReshape implements OutputReshaper {
    
    @Override
    public ConvolutionForwardOutput reshape(ConvolutionLayerForwardOutput input) {
        int numKernels = input.unitOutputs.length;
        int batchSize = input.batchSize;
        
        // Find the unit with smallest output size (largest kernel)
        ConvolutionUnitForwardOutput minOutputKernel = input.unitOutputs[0];
        for (int i = 1; i < numKernels; i++) {
            if (input.unitOutputs[i].outputSize < minOutputKernel.outputSize)
                minOutputKernel = input.unitOutputs[i];
        }
        
        // Use this unit's range as reference
        int outputDim = minOutputKernel.outputSize * numKernels;
        float[] output = new float[outputDim * batchSize];
        
        for (int unitIdx = 0; unitIdx < numKernels; unitIdx++) {
            ConvolutionUnitForwardOutput unit = input.unitOutputs[unitIdx];
            int srcStart = minOutputKernel.outputOrigin - unit.outputOrigin;

            for (int batch = 0; batch < batchSize; batch++) {
                for (int pos = 0; pos < minOutputKernel.outputSize; pos++) {
                    int srcIdx = batch * unit.outputSize + srcStart + pos;
                    int dstIdx = batch * outputDim + unitIdx + pos * numKernels;
                    output[dstIdx] = unit.output[srcIdx];
                }
            }
        }

        return new ConvolutionForwardOutput(output, outputDim, batchSize, input, minOutputKernel.outputOrigin);
    }
    
    @Override
    public ConvolutionForwardOutputGPU reshapeGPU(ConvolutionLayerForwardOutputGPU input, CUstream stream) {
        int numKernels = input.unitOutputs.length;
        int batchSize = input.batchSize;

        // Find reference unit (smallest output)
        ConvolutionUnitForwardOutputGPU minOutputKernel = input.unitOutputs[0];
        for (int i = 1; i < numKernels; i++) {
            if (input.unitOutputs[i].outputSize < minOutputKernel.outputSize)
                minOutputKernel = input.unitOutputs[i];
        }

        int outputDim = minOutputKernel.outputSize * numKernels;
        CUdeviceptr result = CudaUtil.createFloatAsync(outputDim * batchSize, stream);

        // Copy each unit with strided pattern
        for (int unitIdx = 0; unitIdx < numKernels; unitIdx++) {
            ConvolutionUnitForwardOutputGPU unit = input.unitOutputs[unitIdx];
            int srcStart = minOutputKernel.outputOrigin - unit.outputOrigin;

            CudaFunctions.convolution.copyUnitStrided(
                unit.output, unit.outputSize, srcStart,
                result, outputDim, unitIdx, 
                numKernels, minOutputKernel.outputSize, batchSize, stream
            );
        }

        return new ConvolutionForwardOutputGPU(result, outputDim, batchSize, input, minOutputKernel.outputOrigin);
    }
    
    public float[][] backpropagate(ConvolutionForwardOutput reshapedOutput, float[] deltaLoss) {
        ConvolutionLayerForwardOutput originalInput = reshapedOutput.originalInput;
        int numUnits = originalInput.unitOutputs.length;
        int batchSize = reshapedOutput.batchSize;
        int outputDim = reshapedOutput.outputDim;
        int minOutputSize = outputDim / numUnits;
        int refFirstInput = reshapedOutput.outputOrigin;

        float[][] unitGradients = new float[numUnits][];

        // Distribute gradients to each unit
        for (int unitIdx = 0; unitIdx < numUnits; unitIdx++) {
            ConvolutionUnitForwardOutput unit = originalInput.unitOutputs[unitIdx];
            int srcStart = refFirstInput - unit.outputOrigin;

            unitGradients[unitIdx] = new float[unit.outputSize * batchSize];

            for (int batch = 0; batch < batchSize; batch++) {
                for (int pos = 0; pos < minOutputSize; pos++) {
                    int srcIdx = batch * outputDim + unitIdx + pos * numUnits;
                    int dstIdx = batch * unit.outputSize + srcStart + pos;
                    unitGradients[unitIdx][dstIdx] = deltaLoss[srcIdx];
                }
            }
        }

        return unitGradients;
    }
    
    public CUdeviceptr[] backpropagateGPU(ConvolutionForwardOutputGPU reshapedOutput, 
                                          CUdeviceptr deltaLoss, 
                                          CUstream stream) {
        ConvolutionLayerForwardOutputGPU originalInput = reshapedOutput.originalInput;
        int numUnits = originalInput.unitOutputs.length;
        int batchSize = reshapedOutput.batchSize;
        int outputDim = reshapedOutput.outputDim;
        int minOutputSize = outputDim / numUnits;
        int refFirstInput = reshapedOutput.outputOrigin;

        CUdeviceptr[] unitGradients = new CUdeviceptr[numUnits];

        // Distribute gradients to each unit using reverse strided copy
        for (int unitIdx = 0; unitIdx < numUnits; unitIdx++) {
            ConvolutionUnitForwardOutputGPU unit = originalInput.unitOutputs[unitIdx];
            int srcStart = refFirstInput - unit.outputOrigin;

            // Allocate gradient array for this unit
            unitGradients[unitIdx] = CudaUtil.createFloatAsync(unit.outputSize * batchSize, stream);

            // Clear the array first
            JCudaDriver.cuMemsetD32Async(unitGradients[unitIdx], 0, 
                                        unit.outputSize * batchSize, stream);

            // Copy gradients using reverse strided pattern
            CudaFunctions.convolution.copyUnitStridedReverse(
                deltaLoss,              // src: reshaped gradients
                unitGradients[unitIdx], // dst: unit gradients
                unitIdx,                // srcStart in reshaped (unit index)
                srcStart,               // dstStart in unit array
                outputDim,              // srcSize (reshaped dimension)
                numUnits,               // stride (spacing between elements)
                unit.outputSize,        // dstSize
                minOutputSize,          // count (elements to copy)
                batchSize,
                stream
            );
        }

        return unitGradients;
    }
}
