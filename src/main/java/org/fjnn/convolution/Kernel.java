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

import java.util.Arrays;
import java.util.HashMap;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;
import org.fjnn.convolution.output.unit.ConvolutionUnitBackpropagateOutput;
import org.fjnn.convolution.output.unit.ConvolutionUnitBackpropagateOutputGPU;
import org.fjnn.convolution.output.unit.ConvolutionUnitForwardOutput;
import org.fjnn.convolution.output.unit.ConvolutionUnitForwardOutputGPU;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.util.Rng;

/**
 * Represents a single 1D convolutional kernel.
 *
 * @author ahmed
 */
public class Kernel implements ConvolutionUnit {
    // Kernel parameters
    public final int unitSize;   // Values per unit
    public final int unitCount;  // Units processed by kernel
    public final int width;      // Total size = unitSize * unitCount
    
    // Kernel weights and bias
    public float[] weights;
    public float bias;
    
    private boolean useAdam = false;       // Adam optimizer flag
    public float[] weightMomentum;         // m for weights
    public float[] weightVelocity;         // v for weights
    public float biasMomentum = 0.0f;      // m for bias
    public float biasVelocity = 0.0f;      // v for bias
    public int adamTimeStep = 0;           // t for bias correction

    // Adam hyperparameters
    private float beta1 = 0.9f;
    private float beta2 = 0.999f;
    private float adamEpsilon = 1e-8f;
    
    // GPU state
    public CUdeviceptr weightsGPU;
    public CUdeviceptr biasGPU;
    private boolean gpuReady;
    
    public CUdeviceptr weightMomentumGPU;
    public CUdeviceptr weightVelocityGPU;
    public CUdeviceptr biasMomentumGPU; 
    public CUdeviceptr biasVelocityGPU;

    /**
     * Creates a new 1D kernel with specified width.
     * @param width
     */
    public Kernel(int width) {
        this(1, width);
    }
    
    /**
    * Creates a 1D convolution kernel that processes units of data.
    * @param unitSize Number of values per unit
    * @param unitCount Number of units processed by this kernel
    */
    public Kernel(int unitSize, int unitCount) {
        this.unitSize = unitSize;
        this.unitCount = unitCount;
        this.width = unitSize * unitCount;
        this.weights = new float[width];
        this.bias = 0.0f;
        this.gpuReady = false;
    }
    
    /**
     * Computes output size for given input length.
     * @param inputSize
     * @return 
     */
    @Override
    public int computeOutputSize(int inputSize) {
        int numUnits = inputSize / unitSize;
        return numUnits - unitCount + 1;
    }
    
    /**
     * Forward pass on CPU for a single channel 1D input.
     * @param input
     * @param batchSize
     * @return KernelForwardOutput
     */
    @Override
    public ConvolutionUnitForwardOutput feedForward(float[] input, int batchSize) {
        if (input.length % batchSize != 0)
            throw new IllegalArgumentException("Input length must be divisible by batch size");
        
        if (input.length < width)
            throw new IllegalArgumentException("Input length must be atleast " + width);
    
        if (input.length % unitSize != 0)
            throw new IllegalArgumentException("Input length must be divisible by unit size");
        
        int inputDim = input.length / batchSize;
        int outputDim = computeOutputSize(inputDim);
        float[] output = new float[outputDim * batchSize];

        for (int batch = 0; batch < batchSize; batch++) {
            int inputOffset = batch * inputDim;
            int outputOffset = batch * outputDim;

            for (int i = 0; i < outputDim; i++) {
                float sum = bias;

                int startIdx = inputOffset + i * unitSize;
                for (int w = 0; w < width; w++) {
                    sum += input[startIdx + w] * weights[w];
                }

                output[outputOffset + i] = sum;
            }
        }
        
        return new ConvolutionUnitForwardOutput(output, outputDim, batchSize, input, unitCount);
    }
    
    @Override
    public ConvolutionUnitForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchSize, CUstream stream, cublasHandle handle) {
        long inputLength = CudaUtil.length(input) / CudaUtil.FLOAT_SIZE;
    
        if (inputLength % batchSize != 0)
            throw new IllegalArgumentException("Input length must be divisible by batch size");

        if (inputLength < width)
            throw new IllegalArgumentException("Input length must be at least " + width);

        if (inputLength % unitSize != 0)
            throw new IllegalArgumentException("Input length must be divisible by unit size");
        
        int inputDim = (int)(inputLength / batchSize);
        int outputDim = computeOutputSize(inputDim);
        // Total number of output elements across all batches
        int totalOutputSize = outputDim * batchSize;

        // Step 1: Transform input to im2col format
        CUdeviceptr im2colMatrix = im2colGPU(input, inputDim, batchSize, stream);

        // Step 2: Allocate output
        CUdeviceptr output = CudaUtil.createFloatAsync(totalOutputSize, stream);
        
        // Step 2: Perform matrix multiplication: im2col * weights = output
        // im2col: shape (batchSize*outputDim, width)
        // weights: shape (width, 1)
        // output: shape (batchSize*outputDim, 1)
        Pointer alpha = Pointer.to(new float[]{1.0f});
        Pointer beta = Pointer.to(new float[]{0.0f});
        
        synchronized(handle) {
            JCublas2.cublasSetStream(handle, new cudaStream_t(stream));

            // Perform: output = im2col * weights + bias
            int status = JCublas2.cublasSgemv(handle,
                cublasOperation.CUBLAS_OP_T,     // Transpose operation
                width, totalOutputSize,          // Matrix dimensions in column-major order (width, outputDim*batchSize)
                alpha,                           // Alpha scalar (1.0f)
                im2colMatrix, width,             // Matrix and leading dimension
                weightsGPU, 1,                   // Weights vector and increment
                beta,                            // Beta scalar (0.0f)
                output, 1);                      // Output vector and increment
//                System.err.println("cublasSgemm status: " + status + " " + JCublas2.cublasGetStatusName(status));
        }

        // Add bias
        CudaFunctions.vector.addStride(output, biasGPU, output, 1, totalOutputSize, stream);

        return new ConvolutionUnitForwardOutputGPU(output, outputDim, batchSize, input, im2colMatrix, unitCount);
    }
    
    /**
     * Computes gradients for weights, bias, and inputs via backpropagation.
     * @param forwardOutput output of forward pass
     * @param deltaLoss gradients w.r.t. layer outputs
     * @return weight and bias gradients
    */
    @Override
    public ConvolutionUnitBackpropagateOutput backpropagate(ConvolutionUnitForwardOutput forwardOutput, float[] deltaLoss) {
        float[] forwardPassInputs = forwardOutput.input;
        int inputSize = forwardPassInputs.length / forwardOutput.batchSize;
        int batchSize = forwardOutput.batchSize;
        
        float[] inputGradients = new float[forwardPassInputs.length];
        float[] weightGradients = new float[width];
        float biasGradient = 0.0f;

        int outputSize = computeOutputSize(inputSize);
            
        for (int c = 0; c < batchSize; c++) {
            int inputOffset = c * inputSize;
            int outputOffset = c * outputSize;

            // Compute gradients for this batch
            for (int i = 0; i < outputSize; i++) {
                // Bias gradient
                biasGradient += deltaLoss[outputOffset + i];

                // Weight gradients
                for (int k = 0; k < width; k++) {
                    weightGradients[k] += forwardPassInputs[inputOffset + i * unitSize + k] * deltaLoss[outputOffset + i];
                }

                // Input gradients
                for (int k = 0; k < width; k++) {
                    inputGradients[inputOffset + i * unitSize + k] += weights[k] * deltaLoss[outputOffset + i];
                }
            }
        }
                
        // Average gradients if multiple batches
        if (batchSize > 1) {
            for (int i = 0; i < weightGradients.length; i++)
                weightGradients[i] /= batchSize;
            biasGradient /= batchSize;
        }
    
        return new ConvolutionUnitBackpropagateOutput(inputGradients, inputSize, batchSize, weightGradients, biasGradient);
    }
    
    /**
    * GPU version of backpropagation
    * @param forwardOutput output from forward pass
    * @param deltaLoss gradients w.r.t. layer outputs on GPU
    * @param stream CUDA stream
    * @param handle cuBLAS handle
    * @return GPU gradients for weights, bias, and inputs
    */
    @Override
    public ConvolutionUnitBackpropagateOutputGPU backpropagateGPU(ConvolutionUnitForwardOutputGPU forwardOutput, 
                                 CUdeviceptr deltaLoss,
                                 CUstream stream,
                                 cublasHandle handle) {
        long inputLength = CudaUtil.length(forwardOutput.input) / CudaUtil.FLOAT_SIZE;
        int inputDim = (int)(inputLength / forwardOutput.batchSize);
        
        int outputDim = forwardOutput.outputSize;
        int batchSize = forwardOutput.batchSize;
        int totalOutputSize = outputDim * batchSize;
        int totalInputSize = inputDim * batchSize;
        
        // Get im2col matrix from forward pass
        CUdeviceptr im2colMatrix = forwardOutput.im2colMatrix[0];
        
        // Step 1: Compute weight gradients
        // Weight gradients = im2col^T * outputGradients (matrix-vector multiplication)
        CUdeviceptr weightGradients = CudaUtil.createFloatAsync(width, stream);

        synchronized (handle) {
            JCublas2.cublasSetStream(handle, new cudaStream_t(stream));

            // Perform: weightGradients = im2col * deltaLoss
            JCublas2.cublasSgemv(handle,
                               cublasOperation.CUBLAS_OP_N,     // No transpose on im2col
                               width, totalOutputSize,          // Matrix dimensions
                               Pointer.to(new float[]{1.0f}),   // Alpha
                               im2colMatrix, width,             // Input matrix
                               deltaLoss, 1,                    // Output gradients vector
                               Pointer.to(new float[]{0.0f}),   // Beta
                               weightGradients, 1);             // Weight gradients output
        }

        // Step 2: Compute bias gradient (sum of outputGradients)
        CUdeviceptr biasGradient = CudaUtil.createFloatAsync(1, stream);
        CudaFunctions.vector.reduceSum(biasGradient, deltaLoss, 1, totalOutputSize, stream);

        // Step 3: Compute input gradients
        CUdeviceptr inputGradients = CudaUtil.createFloatAsync(totalInputSize, stream);
        CudaFunctions.convolution.computeInputGradients(inputGradients, deltaLoss, weightsGPU, unitSize, inputDim, outputDim, width, batchSize, stream);

        // Step 4: Scale gradients by batch size if needed
        if (batchSize > 1) {
            float scaleFactor = 1.0f / batchSize;
            CudaFunctions.vector.scale(weightGradients, scaleFactor, width, stream);
            CudaFunctions.vector.scale(biasGradient, scaleFactor, 1, stream);
        }

        return new ConvolutionUnitBackpropagateOutputGPU(inputGradients, inputDim, batchSize, weightGradients, biasGradient);
    }
    
    /**
    * Enable/disable Adam optimizer
     * @param useAdam
    */
    public void setUseAdam(boolean useAdam) {
        this.useAdam = useAdam;
        if (useAdam && weightMomentum == null) {
            initAdamState();
        }
    }

    /**
     * Initialize Adam optimizer state
     */
    private void initAdamState() {
        weightMomentum = new float[width];
        weightVelocity = new float[width];
        Arrays.fill(weightMomentum, 0.0f);
        Arrays.fill(weightVelocity, 0.0f);
        biasMomentum = 0.0f;
        biasVelocity = 0.0f;
        adamTimeStep = 0;
    }
    
    public void updateWeights(float[] weightGradients, float biasGradient, float learningRate) {
        if (useAdam) {
            updateWeightsAdam(weightGradients, biasGradient, learningRate);
        } else {
             for (int i = 0; i < weights.length; i++) {
                 weights[i] -= learningRate * weightGradients[i];
             }
             bias -= learningRate * biasGradient;
        }
    }

   /**
    * Adam weight update
    */
    private void updateWeightsAdam(float[] weightGradients, float biasGradient, float learningRate) {
        adamTimeStep++;

        // Update weights
        for (int i = 0; i < weights.length; i++) {
            weightMomentum[i] = beta1 * weightMomentum[i] + (1 - beta1) * weightGradients[i];
            weightVelocity[i] = beta2 * weightVelocity[i] + (1 - beta2) * weightGradients[i] * weightGradients[i];

            float mHat = weightMomentum[i] / (1 - (float)Math.pow(beta1, adamTimeStep));
            float vHat = weightVelocity[i] / (1 - (float)Math.pow(beta2, adamTimeStep));

            weights[i] -= learningRate * mHat / ((float)Math.sqrt(vHat) + adamEpsilon);
        }

        // Update bias
        biasMomentum = beta1 * biasMomentum + (1 - beta1) * biasGradient;
        biasVelocity = beta2 * biasVelocity + (1 - beta2) * biasGradient * biasGradient;

        float biasMHat = biasMomentum / (1 - (float)Math.pow(beta1, adamTimeStep));
        float biasVHat = biasVelocity / (1 - (float)Math.pow(beta2, adamTimeStep));

        bias -= learningRate * biasMHat / ((float)Math.sqrt(biasVHat) + adamEpsilon);
    }
   
    public void updateWeightsGPU(CUdeviceptr weightGradients, CUdeviceptr biasGradient, float learningRate, CUstream stream) {
        if (useAdam) {
            updateWeightsAdamGPU(weightGradients, biasGradient, learningRate, stream);
        } else {
            // Update weights: w = w - lr * grad
            CudaFunctions.vector.add(weightsGPU, weightGradients, weightsGPU, -learningRate, width, stream);

            // Update bias: b = b - lr * grad
            CudaFunctions.vector.add(biasGPU, biasGradient, biasGPU, -learningRate, 1, stream);
        }
    }

    private void updateWeightsAdamGPU(CUdeviceptr weightGradients, CUdeviceptr biasGradient, float learningRate, CUstream stream) {
        adamTimeStep++;

        float beta1Power = (float)Math.pow(beta1, adamTimeStep);
        float beta2Power = (float)Math.pow(beta2, adamTimeStep);

        // Update weights on GPU
        CudaFunctions.convolution.adamUpdateWeights(weightsGPU, weightGradients, weightMomentumGPU, weightVelocityGPU,
                                       learningRate, beta1, beta2, adamEpsilon, beta1Power, beta2Power, width, stream);

        // Update bias on GPU
        CudaFunctions.convolution.adamUpdateBias(biasGPU, biasGradient, biasMomentumGPU, biasVelocityGPU,
                                    learningRate, beta1, beta2, adamEpsilon, beta1Power, beta2Power, stream);
    }
    
    public float[] getWeights() { return weights; }
    
    public void setWeights(float[] weights) {
        if (weights.length != this.width) {
            throw new IllegalArgumentException("Weight length must match kernel width");
        }
        this.weights = weights;
        this.gpuReady = false;
    }
    
    public float getBias() { return bias; }
    
    public void setBias(float bias) {
        this.bias = bias;
        this.gpuReady = false;
    }
    
    @Override
    public void initGaussian(float mean, float sd) {
        // Initialize weights with Gaussian distribution
        for (int i = 0; i < width; i++) {
            weights[i] = (float) Rng.gaussian(mean, sd);
        }

        // Initialize bias
        bias = (float) Rng.gaussian(mean, sd);

        gpuReady = false;
    }
    
    @Override
    public void initUniform(float min, float max) {
        // Initialize weights with uniform distribution
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float) (Rng.nextFloat()* (max - min) + min);
        }

        // Initialize bias to zero (common practice for uniform init)
        bias = 0.0f;

        // Mark GPU as not ready since weights changed
        gpuReady = false;
    }
    
    @Override
    public void xavier(float scalar) {
        int fanIn = width;
        int fanOut = 1;
        float std = scalar * (float) Math.sqrt(2.0 / (fanIn + fanOut));
        initGaussian(0.0f, std);
    }

    @Override
    public void kaiming(float scalar) {
        int fanIn = width;
        float std = scalar * (float) Math.sqrt(2.0 / fanIn);
        initGaussian(0.0f, std);
    }
    
    public HashMap<String, Object> serialize() {
        HashMap<String, Object> result = new HashMap<>();
        result.put("width", width);
        result.put("weights", weights);
        result.put("bias", bias);
        return result;
    }

    /**
     * Prepares the kernel for GPU computation with the specified CUDA stream.
     * 
     * @param stream The CUDA stream to use for GPU operations
     */
    public void prepareGPU(CUstream stream) {
        if (gpuReady)
            throw new RuntimeException("GPU resources already initialized");

        // Allocate and copy weights
        weightsGPU = CudaUtil.toGPUAsync(weights, stream);
        biasGPU = CudaUtil.toGPUAsync(new float[]{bias}, stream);
        
        // Initialize Adam state if needed
        if (useAdam) {
            // Allocate GPU Adam state
            weightMomentumGPU = CudaUtil.toGPUAsync(weightMomentum, stream);
            weightVelocityGPU = CudaUtil.toGPUAsync(weightVelocity, stream);

            biasMomentumGPU = CudaUtil.createFloatAsync(1, stream);
            biasVelocityGPU = CudaUtil.createFloatAsync(1, stream);
            biasMomentumGPU = CudaUtil.toGPUAsync(new float[]{biasMomentum}, stream);
            biasVelocityGPU = CudaUtil.toGPUAsync(new float[]{biasVelocity}, stream);
        }
        
        gpuReady = true;
    }
    
    /**
    * Releases GPU resources with the specified CUDA stream.
    * 
    * @param stream The CUDA stream to use for GPU operations
    */
    public void freeGPU(CUstream stream) {
        if (!gpuReady)
            return;

        // Free weights and bias memory
        CudaUtil.freeAsync(weightsGPU, stream);
        CudaUtil.freeAsync(biasGPU, stream);

        // Free Adam state if allocated
        if (useAdam) {
            CudaUtil.freeAsync(weightMomentumGPU, stream);
            CudaUtil.freeAsync(weightVelocityGPU, stream);
            CudaUtil.freeAsync(biasMomentumGPU, stream);
            CudaUtil.freeAsync(biasVelocityGPU, stream);
        }

        weightsGPU = null;
        biasGPU = null;
        weightMomentumGPU = null;
        weightVelocityGPU = null;
        biasMomentumGPU = null;
        biasVelocityGPU = null;
        gpuReady = false;
    }
    
    /**
     * Transforms input data for 1D convolution using im2col.
     * 
     * @param input The input data array
     * @param inputDim The dimension of each input (sequence length)
     * @param batchSize Number of batches in the input
     * @return The transformed im2col matrix
     */
    public float[] im2col(float[] input, int inputDim, int batchSize) {
        int outputDim = computeOutputSize(inputDim);
        float[] im2colOutput = new float[batchSize * outputDim * width];

        // For each batch
        for (int b = 0; b < batchSize; b++) {
            int batchInputOffset = b * inputDim;
            int batchOutputOffset = b * outputDim * width;

            // For each output position
            for (int i = 0; i < outputDim; i++) {
                // Extract window for current position
                for (int w = 0; w < width; w++) {
                    im2colOutput[batchOutputOffset + i * width + w] = input[batchInputOffset + i * unitSize + w];
                }
            }
        }

        return im2colOutput;
    }
    
    /**
     * GPU version of im2col transformation.
     * 
     * @param input The input data on GPU
     * @param inputDim The dimension of each input (sequence length)
     * @param batchSize Number of batches in the input
     * @param stream CUDA stream
     * @return Device pointer to the transformed im2col matrix
     */
    public CUdeviceptr im2colGPU(CUdeviceptr input, int inputDim, int batchSize, CUstream stream) {
        int outputDim = computeOutputSize(inputDim);

        // Call the CUDA implementation
        return CudaFunctions.convolution.im2col(
            input, inputDim, width, unitSize, outputDim, batchSize, stream);
    }

    @Override
    public boolean gpuReady() {
        return gpuReady;
    }

    @Override
    public int getStrideSize() {
        return unitSize;
    }
    
    @Override
    public void updateWeightsFromGPU() {
        if (!gpuReady())
            return;

        CUstream stream = CudaUtil.createStream();

        // Copy weights from GPU to CPU
        weights = CudaUtil.fromGPUFloatAsync(weightsGPU, width, stream);

        // Copy bias from GPU to CPU  
        float[] biasArray = CudaUtil.fromGPUFloatAsync(biasGPU, 1, stream);
        bias = biasArray[0];
        
        // If using Adam, also update momentum and velocity
        if (useAdam && weightMomentumGPU != null) {
            weightMomentum = CudaUtil.fromGPUFloatAsync(weightMomentumGPU, width, stream);
            weightVelocity = CudaUtil.fromGPUFloatAsync(weightVelocityGPU, width, stream);

            float[] biasMomArray = CudaUtil.fromGPUFloatAsync(biasMomentumGPU, 1, stream);
            float[] biasVelArray = CudaUtil.fromGPUFloatAsync(biasVelocityGPU, 1, stream);
            biasMomentum = biasMomArray[0];
            biasVelocity = biasVelArray[0];
        }
        
        JCudaDriver.cuStreamSynchronize(stream);
        CudaUtil.freeStream(stream);
    }
}
