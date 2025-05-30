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
package org.fjnn.convolution;

import java.util.HashMap;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.jcublas.cublasHandle;
import org.fjnn.activation.Sigmoid;
import org.fjnn.convolution.output.unit.ConvolutionUnitBackpropagateOutput;
import org.fjnn.convolution.output.unit.ConvolutionUnitBackpropagateOutputGPU;
import org.fjnn.convolution.output.unit.ConvolutionUnitForwardOutput;
import org.fjnn.convolution.output.unit.ConvolutionUnitForwardOutputGPU;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;

/**
 *
 * @author ahmed
 */

/**
 * Multi-channel kernel group where each kernel processes its own input channel
 */
public class KernelGroup implements ConvolutionUnit {
    private final Kernel[] kernels;
    private final Sigmoid sigmoid;
    private boolean gpuReady;
    
    public final int inputStride;       /* each chunk of input must have this size */
    public final int unitCount;
    
    public KernelGroup(Kernel... kernels) {
        if (kernels.length < 2)
            throw new IllegalArgumentException("KernelGroup requires at least 2 kernels");
        
        this.kernels = kernels;
        this.sigmoid = new Sigmoid();
        this.gpuReady = false;
        
        // Validate all kernels have same unitCount
        unitCount = kernels[0].unitCount;
        for (int i = 1; i < kernels.length; i++) {
            if (kernels[i].unitCount != unitCount) {
                throw new IllegalArgumentException(
                    String.format("All kernels must have same unitCount. Expected %d but kernel %d has %d", unitCount, i, kernels[i].unitCount));
            }
        }
        
        // Calculate total input size
        int total = 0;
        for (Kernel kernel : kernels) {
            total += kernel.unitSize;
        }
        this.inputStride = total;
    }
    
    @Override
    public ConvolutionUnitForwardOutput feedForward(float[] input, int batchSize) {
        // Calculate sequence length and validate
        int batchLength = input.length / batchSize;
        if (batchLength % inputStride != 0) {
            throw new IllegalArgumentException("Sequence length must be multiple of total kernel width");
        }
        int numChunks = batchLength / inputStride;

        // Extract channels for each kernel
        float[][] channelInputs = new float[kernels.length][];
        for (int k = 0; k < kernels.length; k++) {
            channelInputs[k] = new float[kernels[k].unitSize * numChunks * batchSize];
        }

        // Extract interleaved data using loops
        for (int b = 0; b < batchSize; b++) {
            for (int chunk = 0; chunk < numChunks; chunk++) {
                int srcOffset = b * batchLength + chunk * inputStride;
                int channelOffset = 0;

                for (int k = 0; k < kernels.length; k++) {
                    int dstOffset = b * (kernels[k].unitSize * numChunks) + chunk * kernels[k].unitSize;

                    for (int i = 0; i < kernels[k].unitSize; i++) {
                        channelInputs[k][dstOffset + i] = input[srcOffset + channelOffset + i];
                    }
                    channelOffset += kernels[k].unitSize;
                }
            }
        }

        // Process all kernels and keep results
        ConvolutionUnitForwardOutput[] kernelResults = new ConvolutionUnitForwardOutput[kernels.length];
        for (int k = 0; k < kernels.length; k++) {
            kernelResults[k] = kernels[k].feedForward(channelInputs[k], batchSize);

            // Validate output size against first kernel
            if (k > 0 && kernelResults[k].outputSize != kernelResults[0].outputSize) {
                throw new RuntimeException("Kernel output sizes don't match: " + kernelResults[k].outputSize + " " + kernelResults[0].outputSize);
            }
        }

        int outputSize = kernelResults[0].outputSize;

        // Extract kernel outputs and process sigmoids
        float[][] kernelOutputs = new float[kernels.length][];
        float[][] sigmoidOutputs = new float[kernels.length][];
        for (int k = 0; k < kernels.length; k++) {
            kernelOutputs[k] = kernelResults[k].output;
            sigmoidOutputs[k] = new float[outputSize * batchSize];
            sigmoid.compute(kernelOutputs[k], sigmoidOutputs[k], outputSize, batchSize);
        }

        // AND operation (optimized)
        float[] finalOutput = new float[outputSize * batchSize];
        for (int i = 0; i < finalOutput.length; i++) {
            finalOutput[i] = sigmoidOutputs[0][i];
            for (int k = 1; k < kernels.length; k++) {
                finalOutput[i] *= sigmoidOutputs[k][i];
            }
        }

        return new ConvolutionUnitForwardOutput(finalOutput, outputSize, batchSize, input, unitCount, kernelOutputs, sigmoidOutputs, channelInputs);
    }
    
    @Override
    public ConvolutionUnitForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchSize, CUstream stream, cublasHandle handle) {
        // Calculate dimensions and validate
        long totalInputLength = CudaUtil.length(input) / CudaUtil.FLOAT_SIZE;
        int batchLength = (int)(totalInputLength / batchSize);

        if (batchLength % inputStride != 0)
            throw new IllegalArgumentException("Sequence length must be multiple of total kernel width");

        int totalChunks = batchLength / inputStride * batchSize;

        // Extract channels for each kernel
        int channelOffset = 0;
        CUdeviceptr[] channelInputs = new CUdeviceptr[kernels.length];
        for (int k = 0; k < kernels.length; k++) {
            int channelSize = kernels[k].unitSize * totalChunks;
            channelInputs[k] = CudaUtil.createFloatAsync(channelSize, stream);

            // Extract channel data using CUDA kernel
            CudaFunctions.convolution.extractChannel(input, channelInputs[k], 
                inputStride, kernels[k].unitSize, channelOffset, 
                totalChunks, stream
            );
            
            // Move to next channel offset
            channelOffset += kernels[k].unitSize;
        }

        // Process all kernels
        ConvolutionUnitForwardOutputGPU[] kernelResults = new ConvolutionUnitForwardOutputGPU[kernels.length];
        for (int k = 0; k < kernels.length; k++) {
            kernelResults[k] = kernels[k].feedForwardGPU(channelInputs[k], batchSize, stream, handle);

            // Validate output sizes match
            if (k > 0 && kernelResults[k].outputSize != kernelResults[0].outputSize) {
                throw new RuntimeException("Kernel output sizes don't match: " + 
                    kernelResults[k].outputSize + " " + kernelResults[0].outputSize);
            }
        }

        int outputSize = kernelResults[0].outputSize;
        int totalOutputSize = outputSize * batchSize;

        // Apply sigmoid to each kernel output
        CUdeviceptr[] kernelOutputs = new CUdeviceptr[kernels.length];
        CUdeviceptr[] sigmoidOutputs = new CUdeviceptr[kernels.length];

        for (int k = 0; k < kernels.length; k++) {
            kernelOutputs[k] = kernelResults[k].output;
            sigmoidOutputs[k] = CudaUtil.createFloatAsync(totalOutputSize, stream);
            sigmoid.computeGPU(kernelOutputs[k], sigmoidOutputs[k], totalOutputSize, 1, stream);
        }

        // AND operation - element-wise multiplication of all sigmoid outputs
        CUdeviceptr finalOutput = CudaUtil.copyFloatAsync(sigmoidOutputs[0], totalOutputSize, stream);
        for (int k = 1; k < kernels.length; k++) {
            CudaFunctions.vector.multiply(finalOutput, sigmoidOutputs[k], finalOutput, totalOutputSize, stream);
        }

        // Extract im2col matrices
        CUdeviceptr[] im2colMatrices = new CUdeviceptr[kernelResults.length];
        for (int i = 0; i < kernelResults.length; i++) {
            im2colMatrices[i] = kernelResults[i].im2colMatrix[0];
        }

        return new ConvolutionUnitForwardOutputGPU(
            finalOutput, outputSize, batchSize, input, unitCount,
            kernelOutputs, sigmoidOutputs, channelInputs, im2colMatrices
        );
    }

    @Override
    public ConvolutionUnitBackpropagateOutput backpropagate(ConvolutionUnitForwardOutput forwardOutput, float[] deltaLoss) {
        int outputSize = forwardOutput.outputSize;
        int batchSize = forwardOutput.batchSize;

        // Gradient for each sigmoid output (before AND operation)
        float[][] kernelGradients = new float[kernels.length][outputSize * batchSize];

        // Compute gradients for AND operation
        for (int i = 0; i < outputSize * batchSize; i++) {
            for (int k = 0; k < kernels.length; k++) {
                kernelGradients[k][i] = deltaLoss[i];
                // Multiply by all other sigmoid outputs
                for (int j = 0; j < kernels.length; j++) {
                    if (j != k)
                        kernelGradients[k][i] *= forwardOutput.sigmoidOutputs[j][i];
                }
            }
        }

        // Gradient for each kernel output (before sigmoid)
        for (int k = 0; k < kernels.length; k++) {
            sigmoid.gradient(forwardOutput.kernelOutputs[k], forwardOutput.sigmoidOutputs[k], 
                             kernelGradients[k], outputSize, batchSize);
        }
        
        // Backpropagate through each kernel
        ConvolutionUnitBackpropagateOutput[] kernelBackprops = new ConvolutionUnitBackpropagateOutput[kernels.length];
        for (int k = 0; k < kernels.length; k++) {
            /* reconstruct the kernelForwardOutput */
            ConvolutionUnitForwardOutput kernelForwardOutput = new ConvolutionUnitForwardOutput(forwardOutput.kernelOutputs[k], outputSize, batchSize, forwardOutput.channelInputs[k], unitCount);
            /* pass it to backpropagate */
            kernelBackprops[k] = kernels[k].backpropagate(kernelForwardOutput, kernelGradients[k]);
        }

        // Reconstruct input gradients (reverse of channel extraction)
        int channelInputLength = forwardOutput.channelInputs[0].length / batchSize;
        int numChunks = channelInputLength / kernels[0].unitSize;
        int sequenceLength = numChunks * inputStride;
        float[] inputGradients = new float[sequenceLength * batchSize];

        for (int b = 0; b < batchSize; b++) {
            for (int chunk = 0; chunk < numChunks; chunk++) {
                int dstOffset = b * sequenceLength + chunk * inputStride;
                int channelOffset = 0;

                for (int k = 0; k < kernels.length; k++) {
                    int srcOffset = b * (kernels[k].unitSize * numChunks) + chunk * kernels[k].unitSize;

                    for (int i = 0; i < kernels[k].unitSize; i++) {
                        inputGradients[dstOffset + channelOffset + i] = kernelBackprops[k].inputGradients[srcOffset + i];
                    }
                    channelOffset += kernels[k].unitSize;
                }
            }
        }

        // Collect weight and bias gradients
        float[][] weightGradients = new float[kernels.length][];
        float[] biasGradients = new float[kernels.length];
        for (int k = 0; k < kernels.length; k++) {
            weightGradients[k] = kernelBackprops[k].weightGradients[0];
            biasGradients[k] = kernelBackprops[k].biasGradients[0];
        }

        return new ConvolutionUnitBackpropagateOutput(inputGradients, sequenceLength, batchSize, weightGradients, biasGradients);
    }

    @Override
    public ConvolutionUnitBackpropagateOutputGPU backpropagateGPU(ConvolutionUnitForwardOutputGPU forwardOutput, 
                                                            CUdeviceptr deltaLoss,
                                                            CUstream stream,
                                                            cublasHandle handle) {
        int outputSize = forwardOutput.outputSize;
        int batchSize = forwardOutput.batchSize;
        int totalOutputSize = outputSize * batchSize;

        // Step 1: Compute gradients for AND operation
        // Each kernel gets deltaLoss multiplied by products of other sigmoid outputs
        CUdeviceptr[] kernelGradients = new CUdeviceptr[kernels.length];
        for (int k = 0; k < kernels.length; k++) {
            // Start with copy of deltaLoss
            kernelGradients[k] = CudaUtil.copyFloatAsync(deltaLoss, totalOutputSize, stream);

            // Multiply by all other sigmoid outputs (product rule for AND operation)
            for (int j = 0; j < kernels.length; j++) {
                if (j != k) {
                    CudaFunctions.vector.multiply(kernelGradients[k], 
                                                forwardOutput.sigmoidOutputs[j], 
                                                kernelGradients[k], 
                                                totalOutputSize, stream);
                }
            }
        }

        // Step 2: Apply sigmoid gradients to convert from sigmoid output gradients to raw kernel output gradients
        for (int k = 0; k < kernels.length; k++) {
            sigmoid.gradientGPU(forwardOutput.kernelOutputs[k], 
                               forwardOutput.sigmoidOutputs[k], 
                               kernelGradients[k], 
                               totalOutputSize, 1, stream);
        }

        // Step 3: Backpropagate through each individual kernel
        ConvolutionUnitBackpropagateOutputGPU[] kernelBackprops = new ConvolutionUnitBackpropagateOutputGPU[kernels.length];
        for (int k = 0; k < kernels.length; k++) {
            // Create individual kernel forward output from stored data
            ConvolutionUnitForwardOutputGPU kernelForwardOutput = new ConvolutionUnitForwardOutputGPU(
                forwardOutput.kernelOutputs[k], outputSize, batchSize, 
                forwardOutput.channelInputs[k], forwardOutput.im2colMatrix[k], unitCount
            );

            kernelBackprops[k] = kernels[k].backpropagateGPU(kernelForwardOutput, 
                                                            kernelGradients[k], 
                                                            stream, handle);
        }

        // Step 4: Reconstruct input gradients (reverse of channel extraction)
        // Calculate dimensions
        long totalInputLength = CudaUtil.length(forwardOutput.input) / CudaUtil.FLOAT_SIZE;
        int inputSize = (int)(totalInputLength / batchSize);

        // Create input gradients buffer
        CUdeviceptr inputGradients = CudaUtil.createFloatAsync((int)totalInputLength, stream);

        // Reconstruct input gradients by combining channel gradients
        // This reverses the channel extraction done in feedForwardGPU
        int numChunks = inputSize / inputStride;

        // For now, use a simple approach - we'll need a CUDA kernel for optimal performance
        for (int k = 0; k < kernels.length; k++) {
            int channelOffset = 0;
            for (int j = 0; j < k; j++) {
                channelOffset += kernels[j].unitSize;
            }

            // Distribute gradients from this kernel's channel back to input positions
            CudaFunctions.convolution.distributeChannelGradients(kernelBackprops[k].inputGradients, inputGradients,
                inputStride, kernels[k].unitSize, channelOffset,
                numChunks * batchSize, stream
            );
        }

        // Step 5: Collect weight and bias gradients from all kernels
        CUdeviceptr[] weightGradients = new CUdeviceptr[kernels.length];
        CUdeviceptr[] biasGradients = new CUdeviceptr[kernels.length];
        for (int k = 0; k < kernels.length; k++) {
            weightGradients[k] = kernelBackprops[k].weightGradients[0];
            biasGradients[k] = kernelBackprops[k].biasGradients[0];
        }

        // Clean up temporary gradient arrays
        for (int k = 0; k < kernels.length; k++) {
            CudaUtil.freeAsync(kernelGradients[k], stream);
        }

        // Return multi-kernel backprop output
        return new ConvolutionUnitBackpropagateOutputGPU(inputGradients, inputSize, batchSize, 
                                                        weightGradients, biasGradients);
    }
    /**
     * Update all kernels
     * @param backpropOutput
     * @param learningRate
    */
    public void updateWeights(ConvolutionUnitBackpropagateOutput backpropOutput, float learningRate) {
        for (int k = 0; k < kernels.length; k++) {
            kernels[k].updateWeights(backpropOutput.weightGradients[k], backpropOutput.biasGradients[k], learningRate);
        }
    }
    
    public void updateWeightsGPU(ConvolutionUnitBackpropagateOutputGPU backpropOutput, float learningRate, CUstream stream) {
        for (int k = 0; k < kernels.length; k++) {
            kernels[k].updateWeightsGPU(backpropOutput.weightGradients[k], backpropOutput.biasGradients[k], learningRate, stream);
        }
    }
    
    /**
    * Enable/disable Adam for all kernels
    */
    public void setUseAdam(boolean useAdam) {
        for (Kernel kernel : kernels) {
            kernel.setUseAdam(useAdam);
        }
    }
    
    @Override
    public int computeOutputSize(int inputSize) {
        // All kernels must have same unitCount, so output size is same as any kernel
        // Input is divided by inputStride (total channel size) to get number of chunks
        int numChunks = inputSize / inputStride;

        // Each kernel processes numChunks units and outputs (numChunks - unitCount + 1)
        // Since all kernels have same unitCount, use the first kernel's calculation
        return numChunks - kernels[0].unitCount + 1;
    }

    @Override
    public void prepareGPU(CUstream stream) {
        for (Kernel kernel : kernels) {
            kernel.prepareGPU(stream);
        }
        gpuReady = true;
    }
    
    @Override
    public void freeGPU(CUstream stream) {
        for (Kernel kernel : kernels) {
            kernel.freeGPU(stream);
        }
        gpuReady = false;
    }
    
    @Override
    public boolean gpuReady() {
        return gpuReady;
    }
    
    public Kernel getKernel(int index) { return kernels[index]; }

    @Override
    public int getStrideSize() {
        return inputStride;
    }
    
    @Override
    public void updateWeightsFromGPU() {
        for (Kernel kernel : kernels) {
            kernel.updateWeightsFromGPU();
        }
    }
}