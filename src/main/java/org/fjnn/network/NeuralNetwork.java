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
package org.fjnn.network;

import org.fjnn.network.gradient.ConnectionGradient;
import org.fjnn.network.gradient.ConnectionGradientGPU;
import org.fjnn.network.outputs.NeuralNetworkBackpropagateOutput;
import org.fjnn.network.outputs.NeuralNetworkForwardOutput;
import org.fjnn.network.outputs.NeuralNetworkForwardOutputGPU;
import org.fjnn.network.outputs.NeuralNetworkBackpropagateOutputGPU;
import org.fjnn.base.Network;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.cublasHandle;
import jcuda.jcurand.curandGenerator;
import org.fjnn.activation.Activation;
import org.fjnn.activation.Sigmoid;
import org.fjnn.base.output.FeedForwardOutput;
import org.fjnn.base.NetworkInput;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.loss.BinaryCrossEntropy;
import org.fjnn.loss.Loss;
import org.fjnn.network.Layer.crossOverMutateResult;
import org.fjnn.normalizer.Normalizer;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 */
public class NeuralNetwork extends Network<NeuralNetwork> {
    Layer[] layers;
    
    /* index of last layer */
    int last;
    
    /* true if CPU memory was emptied */
    boolean cpuFree;
    
    /* used for building the network */
    List<Layer> hidden;
    
    /* I:size->H:size[activation].....->O:size[activation] */
    String signature;
    
    /* note: not thread safe .. fix */
    boolean disableDropout;
    
    public NeuralNetwork(int input, int output, Activation outputActivation) {
        super(input, output, outputActivation, null);
        
        this.hidden = new ArrayList<>();
    }
    
    public NeuralNetwork(int input, int output, Activation outputActivation, Normalizer outputNormalizer) {
        super(input, output, outputActivation, outputNormalizer != null ? outputNormalizer.withNeurons(output) : null);
        
        this.hidden = new ArrayList<>();
    }
        
    NeuralNetwork(Layer[] layers) {
        super(layers[0].neurons, layers[layers.length-1].neurons, layers[layers.length-1].activation, layers[layers.length-1].normalizer);
        
        this.layers = layers;
        this.finalized = true;
        this.last = layers.length - 1;
    }
    
    @Override
    public NeuralNetwork copy() {
        return copy(true, true);
    }
    
    @Override
    public NeuralNetwork copy(boolean copyWeights, boolean createWeights) {
        Layer[] copied = new Layer[layers.length];
        
        for(int i=0; i < layers.length; i++) {
            copied[i] = layers[i].copy(copyWeights, createWeights);
        }
        
        return new NeuralNetwork(copied);
    }
    
    @Override
    public void copyWeights(NeuralNetwork n) {
        for(int i=0; i < layers.length; i++) {
            layers[i].copyWeights(n.layers[i]);
        }
    }
    
    /* for building the network */
    public NeuralNetwork addHiddenLayer(int neurons, Activation activation) {
        return addHiddenLayer(neurons, activation, null);
    }
    
    public NeuralNetwork addHiddenLayer(int neurons, Activation activation, Normalizer normalizer) {
        return addHiddenLayer(neurons, activation, normalizer, 0);
    }
    
    public NeuralNetwork addHiddenLayer(int neurons, Activation activation, Normalizer normalizer, float dropout) {
        if(finalized)
            throw new RuntimeException("network already finalized");
        
        normalizer = normalizer != null ? normalizer.withNeurons(neurons) : null;
        
        int layerIndex = hidden.size() + 1;   // +1 for input layer
        hidden.add(new Layer(layerIndex, neurons, activation, normalizer, dropout));
        return this;
    }
    
    public NeuralNetwork addConnection(int fromLayer, int toLayer) {
        if(!finalized)
            throw new RuntimeException("network layer structure is not finalized yet");
        
        Layer sourceLayer = layers[fromLayer];
        Layer targetLayer = layers[toLayer];
    
        int targetNeurons = targetLayer.neurons;
        
        // Add forward connection from source to target
        sourceLayer.addConnection(toLayer, targetNeurons);
        
        return this;
    }
    
    public final NeuralNetwork build() {
        if(finalized)
            throw new RuntimeException("network already finalized");

        /* input + output + hidden */
        layers = new Layer[hidden.size()+2];

        /* input layer */
        layers[0] = new Layer(0, inputSize, null, null, 0);

        Layer prev = layers[0];

        for(Layer layer : hidden) {
            // Add connection from previous layer to the current layer
            prev.addConnection(layer.index, layer.neurons);

            layers[layer.index] = layer;            
            prev = layer;
        }

        /* output layer */
        last = hidden.size() + 1;
        layers[last] = new Layer(last, outputSize, outputActivation, outputNormalizer, 0);

        // Add connection from the last hidden layer to the output layer
        prev.addConnection(last, outputSize);

        finalized = true;

        return this;
    }
    
    /* get functions */
    public Layer getInputLayer() {
        return layers[0];
    }
    
    public Layer getOutputLayer() {
        return layers[last];
    }
    
    public int getHiddenCount() {
        return layers.length - 2;
    }
    
    public Layer getLayer(int layer) {
        return layers[layer];
    }
        
    public int getLayerCount() {
        return layers.length;
    }
    
    /* total number of weights including bias */
    @Override
    public long getParametersCount() {
        long count = 0;
        
        for(Layer l : layers)
            count += l.getParameterCount();
        
        return count;
    }
    
    /* initialization functions */
    @Override
    public NeuralNetwork randomize(float min, float max) {
        for(Layer l : layers)
            l.initUniform(min, max);
        
        return this;
    }
    
    @Override
    public NeuralNetwork xavier() {
        return xavier(1);
    }
    public NeuralNetwork xavier(float scalar) {
        for(Layer l : layers)
            l.xavier(scalar);
        
        return this;
    }
    
    @Override
    public NeuralNetwork kaiming() {
        return kaiming(1);
    }
    public NeuralNetwork kaiming(int scalar) {
        for(Layer l : layers)
            l.kaiming(scalar);
        
        return this;
    }

    /*
     * Compute CPU
     */
    @Override
    public float[] compute(float[] input) {
        return compute(input, 1);
    }
    
    @Override
    public float[] compute(float[] input, int batchSize) {
        float[][] results = new float[getLayerCount()][];
        results[0] = util.copyArray(input);
        
        for(int i=0; i < layers.length; i++)
            layers[i].feedForward(results[i], batchSize, results);
        
        return results[last];
    }
       
    @Override
    public NeuralNetworkForwardOutput feedForward(float[] input, int batchSize) {
        NeuralNetworkForwardOutput activations = new NeuralNetworkForwardOutput(getOutputSize(), batchSize, getLayerCount());
        activations.layerInputs[0] = input;
        
        for (int i = 0; i < layers.length; i++)
            layers[i].feedForward(activations, disableDropout);

        return activations;
    }

    /*
     * https://www.mldawn.com/deriving-the-gradient-descent-rule-part-2/
     *
     *  1. Compute delta for the output layer using the derivative of the loss function with respect to the output.
     *  2. Calculate weight and bias gradients for the output layer using the delta and activations from the previous layer.
     *  3. Backpropagate the delta to the previous layer (hidden layers) using the weights of the current layer and the derivative of the activation function.
     *  4. Repeat steps 2-3 for each hidden layer, moving backward through the network.
     *  5. Accumulate gradients across all samples in the batch.
     *  6. After processing the entire batch, update weights and biases using the accumulated gradients and the learning rate.
     *  7. Clear the accumulated gradients after the update.
     */
    public NeuralNetworkBackpropagateOutput backpropagate(NeuralNetworkForwardOutput output, float[] truth, Loss lossFunction) {
        float[] outputPreActivationDelta = new float[outputSize * output.batchSize];
        
        /* handle special case */
        if(lossFunction instanceof BinaryCrossEntropy && layers[last].activation instanceof Sigmoid) {
            BinaryCrossEntropy bceLoss = (BinaryCrossEntropy)lossFunction;
            Sigmoid sigmoid = (Sigmoid) layers[last].activation;
            
            sigmoid.gradientCrossEntropy(output.activationOutputs[last].postActivation, truth, outputPreActivationDelta, bceLoss.alpha, bceLoss.beta, outputSize, output.batchSize);
            
            // If normalizer exists, apply its gradient
            BackpropagateOutput normOutput = null;

            if(outputNormalizer != null) {
                normOutput = outputNormalizer.backpropagate(output.normalizerOutputs[last], outputPreActivationDelta, outputSize, output.batchSize);
                outputPreActivationDelta = normOutput.deltaLoss();
            }

            NeuralNetworkBackpropagateOutput result = backpropagate0(output, outputPreActivationDelta);
            result.normalizerGradients[last] = normOutput;

            return result;
        } else {
            float[] result = output.output();
            float[] deltaLoss = lossFunction.derivative(result, truth);
            
            return backpropagate(output, deltaLoss);
        }
    }
    
    @Override
    public NeuralNetworkBackpropagateOutput backpropagate(FeedForwardOutput feedforward, float[] deltaLoss) {
        if(!(feedforward instanceof NeuralNetworkForwardOutput))
            throw new RuntimeException("Invalid feedforward output");
        
        NeuralNetworkForwardOutput output = (NeuralNetworkForwardOutput)feedforward;

        // Calculate pre-activation deltas for output layer 
        if(outputActivation != null)
            outputActivation.gradient(output.activationOutputs[last].preActivation, output.activationOutputs[last].postActivation, deltaLoss, outputSize, feedforward.batchSize);
        
        // If normalizer exists, apply its gradient
        BackpropagateOutput normOutput = null;
        
        if(outputNormalizer != null) {
            normOutput = outputNormalizer.backpropagate(output.normalizerOutputs[last], deltaLoss, outputSize, feedforward.batchSize);
            deltaLoss = normOutput.deltaLoss();
        }
        
        NeuralNetworkBackpropagateOutput result = backpropagate0(output, deltaLoss);
        result.normalizerGradients[last] = normOutput;
        
        return result;
    }
    
    public NeuralNetworkBackpropagateOutput backpropagate0(FeedForwardOutput feedforward, float[] outputPreActivationDelta) {
        if(!(feedforward instanceof NeuralNetworkForwardOutput))
            throw new RuntimeException("Invalid feedforward output");
        
        NeuralNetworkForwardOutput output = (NeuralNetworkForwardOutput)feedforward;
        
        NeuralNetworkBackpropagateOutput result = new NeuralNetworkBackpropagateOutput(getInputSize(), output.batchSize, getLayerCount());
        result.preActivationDeltas[last] = outputPreActivationDelta;

        // Step 4: Backpropagate through all hidden layers, except output layer
        for (int i = layers.length - 2; i >= 0; i--) {
            layers[i].backpropagate(output, result.preActivationDeltas, result.layerConnectionGradients.get(i), result.normalizerGradients);
        }
        
        return result;
    }
    
    @Override
    public void applyGradients(BackpropagateOutput gradients, float learningRate, float weightDecay) {
        if(!(gradients instanceof NeuralNetworkBackpropagateOutput))
            throw new RuntimeException("Invalid gradients object");
        
        NeuralNetworkBackpropagateOutput output = (NeuralNetworkBackpropagateOutput)gradients;
        
        List<Map<Integer, ConnectionGradient>> layerConnectionGradients = output.layerConnectionGradients;
        
        // Update weights and biases using the accumulated gradients
        for (int i = 0; i < layers.length; i++) {
            layers[i].updateWeights(layerConnectionGradients.get(i), output.normalizerGradients[i], learningRate, learningRate / 100.0f, weightDecay);
        }
    }
    /*
     * Compute GPU 
     */
    public float[] computeGPU(CUdeviceptr input) {
        return computeGPU(input, 1);
    }
    
    public float[] computeGPU(CUdeviceptr input, int batchSize) {
        return computeGPU(input, batchSize, true);
    }
    
    public float[] computeGPU(CUdeviceptr input, int batchSize, boolean memlock) {
        long memory = getGPUForwardMemoryRequired(batchSize);
        
        if(memlock)
            lockMemory(memory, deviceId);
        
        try {
            checkGPUContext();

            cublasHandle handle = CudaEngine.getCublasHandle(deviceId);

            CUdeviceptr[] layerResults = new CUdeviceptr[getLayerCount()];
            layerResults[0] = input;

            for(int i=0; i < layers.length; i++)
                layers[i].computeGPU(layerResults[i], batchSize, layerResults, null, handle);

            float[] result = CudaUtil.fromGPUFloat(layerResults[last], batchSize * outputSize);

            for(int i=1; i < layerResults.length; i++)
                CudaUtil.free(layerResults[i]);

            return result;
        } finally {
            if(memlock)
                releaseMemory(memory);
        }
    }
    
    public void computeGPU(CUdeviceptr input, CUdeviceptr output, int batchSize, CUstream stream) {
        checkGPUContext();
        
        cublasHandle handle = CudaEngine.getCublasHandle(deviceId);

        CUdeviceptr[] layerResults = new CUdeviceptr[getLayerCount()];
        layerResults[0] = input;
        layerResults[last] = output;
        
        JCudaDriver.cuMemsetD32Async(layerResults[last], 0, outputSize * batchSize, stream);
        
        for(int i=0; i < layers.length; i++)
            layers[i].computeGPU(layerResults[i], batchSize, layerResults, stream, handle);
        
        for(int i=1; i < layerResults.length-1; i++)
            CudaUtil.freeAsync(layerResults[i], stream);
    }
    
    @Override
    public NeuralNetworkForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchSize, CUstream stream) {
        cublasHandle handle = CudaEngine.getCublasHandle(deviceId);
        
        NeuralNetworkForwardOutputGPU activations = new NeuralNetworkForwardOutputGPU(getOutputSize(), batchSize, getLayerCount());
        activations.layerInputs[0] = input;

        for(int i=0; i < layers.length; i++) {
            layers[i].feedForwardGPU(activations, disableDropout, stream, handle);
        }
        
        return activations;
    }
    
    public NeuralNetworkBackpropagateOutputGPU backpropagateGPU(NeuralNetworkForwardOutputGPU output, CUdeviceptr truth, Loss lossFunction, CUstream stream) {
        checkGPUContext();
        
        // Step 1: Get result and calculate delta loss on GPU
        long totalOutputSize = outputSize * output.batchSize;
        CUdeviceptr outputPtr = output.output();
        CUdeviceptr deltaLoss = CudaUtil.createFloatAsync(totalOutputSize, stream);
        
        /* handle special case */
        if(lossFunction instanceof BinaryCrossEntropy && layers[last].activation instanceof Sigmoid) {
            BinaryCrossEntropy bceLoss = (BinaryCrossEntropy)lossFunction;
            Sigmoid sigmoid = (Sigmoid) layers[last].activation;
            
            sigmoid.gradientGPUCrossEntropy(output.activationOutputs[last].postActivation, truth, deltaLoss, bceLoss.alpha, bceLoss.beta, outputSize, output.batchSize, stream);
            
            // If normalizer exists, apply its gradient
            BackpropagateOutputGPU normOutput = null;

            if(outputNormalizer != null) {
                normOutput = outputNormalizer.backpropagateGPU(output.normalizerOutputs[last], deltaLoss, outputSize, output.batchSize, stream);
                deltaLoss = normOutput.deltaLoss();
            }

            NeuralNetworkBackpropagateOutputGPU result = backpropagateGPU0(output, deltaLoss, stream);
            result.normalizerGradients[last] = normOutput;

            return result;
        } else {
            lossFunction.derivativeGPU(outputPtr, truth, deltaLoss, totalOutputSize, stream);
        
            return backpropagateGPU(output, deltaLoss, stream);
        }
    }
    
    @Override
    public NeuralNetworkBackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU feedforward, CUdeviceptr deltaLoss, CUstream stream) {
        if(!(feedforward instanceof NeuralNetworkForwardOutputGPU))
            throw new RuntimeException("Invalid feedforward output");
        
        NeuralNetworkForwardOutputGPU output = (NeuralNetworkForwardOutputGPU) feedforward;
        
        // Calculate pre-activation delta for output layer 
        if(outputActivation != null)
           outputActivation.gradientGPU(output.activationOutputs[last].preActivation, 
                                        output.activationOutputs[last].postActivation, 
                                        deltaLoss, outputSize, feedforward.batchSize, stream);
        
        // If normalizer exists, apply its gradient
        BackpropagateOutputGPU normOutput = null;

        if(outputNormalizer != null) {
            normOutput = outputNormalizer.backpropagateGPU(output.normalizerOutputs[last], deltaLoss, outputSize, feedforward.batchSize, stream);
            deltaLoss = normOutput.deltaLoss();
        }

        NeuralNetworkBackpropagateOutputGPU result = backpropagateGPU0(output, deltaLoss, stream);
        result.normalizerGradients[last] = normOutput;

        return result;
    }
    
    private NeuralNetworkBackpropagateOutputGPU backpropagateGPU0(NeuralNetworkForwardOutputGPU feedforward, CUdeviceptr outputPreActivationDelta, CUstream stream) {        
        cublasHandle handle = CudaEngine.getCublasHandle(deviceId);
        
        // Create output structure with proper size
        NeuralNetworkBackpropagateOutputGPU result = new NeuralNetworkBackpropagateOutputGPU(getInputSize(), feedforward.batchSize, getLayerCount());
        result.preActivationDeltas[last] = outputPreActivationDelta;
        
        // Step 4: Backpropagate through layers, excluding output layer
        for (int i = layers.length - 2; i >= 0; i--) {
            layers[i].backpropagateGPU(feedforward, result.preActivationDeltas, result.layerConnectionGradients.get(i), result.normalizerGradients, stream, handle);
        }

        return result;
    }
    
    @Override
    public void applyGradientsGPU(BackpropagateOutputGPU gradients, float learningRate, float weightDecay, CUstream stream) {
        if(!(gradients instanceof NeuralNetworkBackpropagateOutputGPU))
            throw new RuntimeException("Invalid gradients object");
        
        NeuralNetworkBackpropagateOutputGPU output = (NeuralNetworkBackpropagateOutputGPU)gradients;
        
        cublasHandle handle = CudaEngine.getCublasHandle(deviceId);
        List<Map<Integer, ConnectionGradientGPU>> layerConnectionGradients = output.layerConnectionGradients;

        // Update weights and biases using the accumulated gradients
        for (int i = 0; i < layers.length; i++) {
            layers[i].updateWeightsGPU(layerConnectionGradients.get(i), output.normalizerGradients[i], learningRate, learningRate / 100.0f, weightDecay, stream, handle);
        }
    }
    
    /**
     * Memory required for network parameters (delegated to layers)
     * @return 
    */
    @Override
    public long getGPUPrepareMemoryRequired() {
        long memory = 0;
        
        for(Layer layer : layers) {
            memory += layer.getParameterCount();
        }
        
        return memory * CudaUtil.FLOAT_SIZE;
    }

    /**
     * Memory required during forward pass (delegated to layers)
     * @param batchSize
     * @return 
     */
    public long getGPUForwardMemoryRequired(int batchSize) {
        long memory = 0;
        for(Layer layer : layers) {
            memory += layer.getForwardMemoryRequired(batchSize);
        }
        return memory;
    }

    /**
     * Memory required during backward pass (initial deltaLoss + delegated to layers)
     * @param batchSize
     * @return 
     */
    public long getGPUBackwardMemoryRequired(int batchSize) {
        long memory = 0;

        // Initial deltaLoss allocation in backpropagateGPU()
        memory += outputSize * batchSize * CudaUtil.FLOAT_SIZE;

        // Delegate to layers
        for(Layer layer : layers) {
            memory += layer.getBackwardMemoryRequired(batchSize);
        }

        return memory;
    }
    
    /**
    * Total memory required during training (forward + backward)
     * @param batchSize
     * @return 
    */
    @Override
    public long getGPUTrainingMemoryRequired(int batchSize) {
       return getGPUForwardMemoryRequired(batchSize) + getGPUBackwardMemoryRequired(batchSize);
    }
    
//    
//    public double sumAbsWeightsGPU() {
//        if(!gpuReady())
//            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
//        
//        boolean prepareThread = CudaEngine.getThreadDeviceId() != deviceId;
//        
//        if(prepareThread)
//           CudaEngine.prepareThread(deviceId);
//        
//        CUstream stream = CudaEngine.aquireStream(deviceId);
//        float[][] results = new float[getLayerCount()][];
//        
//        for(int i=0; i < layers.length; i++) {
//            results[i] = layers[i].sumAbsWeightsGPU(stream);
//        }
//        
//        JCudaDriver.cuStreamSynchronize(stream);
//        
//        CudaEngine.releaseStream(deviceId, stream);
//
//        double result = 0;
//        for(float[] a : results)
//            for(float f : a)
//                result += f;
//        
//        for(int i=0; i < layers.length; i++)
//            CudaEngine.freeMempool(deviceId, layerResults[i]);
//        
//        if(prepareThread)
//           CudaEngine.finalizeThread();
//        
//        return result;
//    }
    
    /* add neurons to layer */
    public void expandLayer(int layer, int neurons) {
        expandLayer(layer, neurons, 0);
    }
    
    public void expandLayer(int layer, int neurons, float initialWeight) {
        if(deviceId != -1)
            freeGPU();
        
        layers[layer].addNeurons(neurons, initialWeight);
        
        if(layer != 0)
            layers[layer-1].addLinks(neurons, initialWeight);
        
        signature = null;
    }
    
    @Override
    public void crossOverMutate(NeuralNetwork a, NeuralNetwork b, float min, float max, double mutation) {
        for(int i=0; i < layers.length; i++)
            layers[i].crossOverMutate(a.getLayer(i), b.getLayer(i), min, max, mutation);
    }
    
    @Override
    public void crossOverMutateGPU(NeuralNetwork a, NeuralNetwork b, float min, float max, double mutation, boolean nocopy) {
        if(!a.gpuReady() || !b.gpuReady())
            throw new RuntimeException("Parent networks are not loaded to the GPU, please call prepareGPU first");
        
        if(a.deviceId != b.deviceId)
            throw new RuntimeException("Parent networks are on different GPUs");
        
        if(gpuReady())
            throw new RuntimeException("Network is gpu ready, please call freeGPU first");
        
        boolean invalidThreadId = CudaEngine.getThreadDeviceId() != deviceId;

        if(invalidThreadId)
            throw new RuntimeException("Invalid cuda context for device: " + deviceId + " must call CudaEngine.prepareThread(...)");
            
        this.deviceId = a.deviceId;
        
        long memory = getParametersCount() * 4 * CudaUtil.FLOAT_SIZE;
        
        /* 1x prepare gpu + 3x rng arrays */
        lockMemory(memory, deviceId);
        
        try {
            CUstream stream = CudaUtil.createStream();
            curandGenerator generator = CudaEngine.getCurandGenerator(deviceId);

            for(int i=0; i < layers.length; i++)
                layers[i].crossOverMutateGPU(a.getLayer(i), b.getLayer(i), min, max, mutation, nocopy, stream, generator);

            JCudaDriver.cuStreamSynchronize(stream);
            CudaUtil.freeStream(stream);

            for(int i=0; i < layers.length; i++)
                layers[i].freeGPURng();
        } finally {
            releaseMemory(memory * 3);
        }
    }
    
    @Override
    public void clipWeights(float min, float max) {
        for(int i=0; i < layers.length; i++)
            layers[i].clipWeights(min, max);
    }

    @Override
    public void clipWeightsGPU(float min, float max) {
        if(gpuReady())
            throw new RuntimeException("Network is gpu ready, please call freeGPU first");
        
        CudaEngine.prepareThread(deviceId);
        CUstream stream = CudaUtil.createStream();

        for(int i=0; i < layers.length; i++)
            layers[i].clipWeightsGPU(min, max, stream);

        JCudaDriver.cuStreamSynchronize(stream);
        CudaUtil.freeStream(stream);

        CudaEngine.finalizeThread();
    }    
    
    public void crossOverMutate(NeuralNetwork a, NeuralNetwork b, NeuralNetwork min, NeuralNetwork max, double mutation) {
        for(int i=0; i < layers.length; i++)
            layers[i].crossOverMutate(a.getLayer(i), b.getLayer(i), min.getLayer(i), max.getLayer(i), mutation);
    }
    
    public crossOverMutateResult crossOverMutate(NeuralNetwork a, NeuralNetwork b, 
                                NeuralNetwork minA, NeuralNetwork maxA, 
                                NeuralNetwork minB, NeuralNetwork maxB, 
                                double mutation) {
        
        crossOverMutateResult result = new crossOverMutateResult();
        
        for(int i=0; i < layers.length; i++)
            layers[i].crossOverMutate(a.getLayer(i), b.getLayer(i), 
                                      minA.getLayer(i), maxA.getLayer(i), 
                                      minB.getLayer(i), maxB.getLayer(i), 
                                      mutation, result);
        
        return result;
    }
    
    @Override
    public double compare(NeuralNetwork a) {
        double score = 0;
        
        for(int i=0; i < a.getLayerCount(); i++)
            score += layers[i].compare(a.getLayer(i));
        
        return score;
    }
    
//    public float compareGPU(NeuralNetwork a) {        
//        if(!gpuReady() || !a.gpuReady())
//            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
//        
//        CudaEngine.prepareThread(deviceId);
//        
//        float result = compareGPU(a, CudaEngine.getStream(deviceId));
//        
//        CudaEngine.finalizeThread();
//        
//        return result;
//    }
    
    public float compareGPU(NeuralNetwork a, CUstream stream) {
        if(!gpuReady() || !a.gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        float score = 0;
        
        for(int i=0; i < a.getLayerCount(); i++)
            score += layers[i].compareGPU(a.getLayer(i), stream);
        
        
        return score;
    }
    
    @Override
    protected void freeGPU0(CUstream stream) {
        for(Layer l : layers)
            l.freeGPU(stream);
        
        long memory = getParametersCount() * CudaUtil.FLOAT_SIZE;
        
        releaseMemory(memory);
    }
    
    /**
     * Average of all weights
     * @return 
     */
    public float mean() {
        float sum = 0;
        int count = 0;
        
        for (Layer l : layers) {
            for(Connection c : l.connections.values()) {
                float[] weights = c.weights;

                for(float w : weights)
                    sum += w;

                count += weights.length;

                float[] bias = c.biases;

                for(float b : bias)
                    sum += b;

                count += bias.length;
            }
        }
        
        return sum / count;
    }
    
    /**
     * Standard deviation of all weights
     * @param mean
     * @return 
     */
    public float sd(double mean) {
        float sd = 0;
        int count = 0;
        
        for (Layer l : layers) {
            for(Connection c : l.connections.values()) {
                float[] weights = c.weights;

                for(float w : weights)
                    sd += Math.pow(w - mean, 2);

                count += weights.length;

                float[] bias = c.biases;

                for(float b : bias)
                    sd += Math.pow(b - mean, 2);

                count += bias.length;
            }
        }
        
        return (float) Math.sqrt(sd / count);
    }
    
    public String getSignature() {
        if(signature != null)
            return signature;
        
        StringBuilder b = new StringBuilder();
        b.append("I:").append(layers[0].neurons).append(",");
        
        for(int i=1; i < layers.length-1; i++) {
            b.append("H:").append(layers[i].neurons);
            if(layers[i].activation != null)
                b.append("[").append(layers[i].activation.toName()).append("]");
            b.append(",");
        }
        
        b.append("O:").append(layers[last].neurons);
        if(layers[last].activation != null)
            b.append("[").append(layers[last].activation.toName()).append("]");
        
        signature = b.toString();
        
        return signature;
    }

    @Override
    public float[] compute(NetworkInput input) {
        switch(input.type) {
            case float_array:
                return compute(input.getInputArray(), input.count);
            case floatbuffer_array:
                return compute(input.getInputFloatBuffer(), input.count);
            case deviceptr:
                return computeGPU(input.getInputDevicePtr(), input.count);
            default:
                throw new RuntimeException("invalid input type for neuralnetwork: " + input.type);
        }
    }
    
    
    public boolean nativeReady() {
        boolean nativeReady = true;
        for(Layer l : layers)
            nativeReady &= l.nativeReady();
        
        return nativeReady;
    }
    
    public void prepareCPU() {
        for(Layer l : layers)
            l.prepareCPU();
    }
    
    public void freeCPU() {
        if(cpuFree)
            return;
        
        for(Layer l : layers)
            l.freeCPU();
        
        cpuFree = true;
    }
//    
//    @Override
//    public void updateWeightsFromGPU() {
//        checkGPUContext();
//        
//        CUstream stream = CudaUtil.createStream();
//        
//        for(Layer l : layers)
//            l.ensureCPU(stream);
//        
//        JCudaDriver.cuStreamSynchronize(stream);        
//        CudaUtil.freeStream(stream);
//    }
    
    @Override
    public void updateWeightsFromGPU() {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU");
        
        CudaEngine.prepareThread(deviceId);
        
        CUstream stream = CudaUtil.createStream();
        
        for(Layer l : layers)
            l.updateWeightsFromGPU(stream);
        
        JCudaDriver.cuStreamSynchronize(stream);
        
        CudaUtil.freeStream(stream);
        
        CudaEngine.finalizeThread();
    }

    @Override
    public HashMap serialize() {
        return serialize(new HashSet<>());
    }

    @Override
    public HashMap serialize(Set<String> ignoreProperties) {
        HashMap obj = super.serialize(ignoreProperties);
        
        List<HashMap> array = new ArrayList<>();
        
        obj.put("layers", array);
        
        for(Layer l: layers)
            array.add(l.serialize());
        
        obj.put("type", "NeuralNetwork");
        
        return obj;
    }
    
    public static NeuralNetwork deserialize(Map serialized) {
        List<HashMap> array = (List)serialized.get("layers");
        
        Layer[] layers = new Layer[array.size()];
        
        for(int i=0; i < array.size(); i++)
            layers[i] = Layer.deserialize(array.get(i));
        
        /* check reverse connections */
//        for(int i=0; i < layers.length; i++) {
//            Layer sourceLayer = layers[i];
//            for(int j : sourceLayer.connections.keySet()) {
//                Layer targetLayer = layers[j];
//                
//                if(!targetLayer.reverseConnections.contains(i)) {
//                    System.out.printf("adding reverse connection: %d <- %d\n", j, i);
//                    targetLayer.addReverseConnection(i);
//                }
//            }
//        }
        
        NeuralNetwork result = new NeuralNetwork(layers);
        
        result.properties.putAll((HashMap)serialized.get("properties"));
        return result;
    }
    
    @Override
    public boolean gpuReady() {
        boolean gpuReady = true;
        for(Layer l : layers)
            gpuReady &= l.gpuReady();
        
        return gpuReady;
    }

    @Override
    public void prepareGPU0(CUstream stream) {
        int device = CudaEngine.getThreadDeviceId();
        
        long memory = getParametersCount() * CudaUtil.FLOAT_SIZE;
        
        lockMemory(memory, device);

        for(Layer l : layers)
            l.prepareGPU(stream);
    }
    
    public void moveGPU(int newDevice) {
        if(!gpuReady())
            throw new RuntimeException("Network is not loaded to the GPU, please call prepareGPU first");
        
        if(deviceId == newDevice)
            throw new RuntimeException("Moving GPU to same device");
            
        freeGPU();
        
        CudaEngine.prepareThread(newDevice);
        
        prepareGPU();
        
        CudaEngine.finalizeThread();
    }    

    @Override
    public long getBackpropagateMemoryRequired(int batchSize) {
        throw new RuntimeException();
    }
    
    public void enableDropout() {
        disableDropout = false;
    }
    
    public void disableDropout() {
        disableDropout = true;
    }
    
    /**
     * Enable/disable Adam optimizer for all connections
     * @param useAdam
     * @return 
     */
    public NeuralNetwork setUseAdam(boolean useAdam) {
        for(Layer layer : layers) {
            for(Connection connection : layer.getConnections().values()) {
                connection.setUseAdam(useAdam);
            }
        }
        
        return this;
    }
    
    /**
     * Set Adam hyperparameters for all connections
     * @param beta1
     * @param beta2
     * @param epsilon
     */
    public void setAdamHyperparameters(float beta1, float beta2, float epsilon) {
        for(Layer layer : layers) {
            for(Connection connection : layer.getConnections().values()) {
                connection.beta1 = beta1;
                connection.beta2 = beta2;
                connection.adamEpsilon = epsilon;
            }
        }
    }

    @Override
    public float[] compute(FloatBuffer input) {
        return compute(input, 1);
    }
    
    @Override
    public float[] compute(FloatBuffer input, int batchSize) {
        if(!nativeReady())
            prepareCPU();
        
        FloatBuffer[] results = new FloatBuffer[getLayerCount()];
        results[0] = input;
        
        for(int i=0; i < layers.length; i++)
            layers[i].feedForward(results[i], batchSize, results);
        
        float[] result = new float[outputSize * batchSize];
        results[last].rewind();
        results[last].get(result);

        return result;
    }
    
}
