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
package org.fjnn.trainer.backpropagate;

import java.util.HashSet;
import org.fjnn.network.NeuralNetwork;
import org.fjnn.network.outputs.NeuralNetworkForwardOutput;
import org.fjnn.network.outputs.NeuralNetworkForwardOutputGPU;
import org.fjnn.network.outputs.NeuralNetworkBackpropagateOutput;
import org.fjnn.network.outputs.NeuralNetworkBackpropagateOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaUtil;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import java.util.List;
import jcuda.driver.CUevent;
import org.fjnn.trainer.backpropagate.outputs.TrainingSessionOutput;
import org.fjnn.trainer.backpropagate.outputs.TrainingSessionOutputGPU;

/**
 *
 * @author ahmed
 */
public class TrainingSession {
    private final NeuralNetwork network;
    private final Dataset dataset;
    private final TrainingConfig config;
    private final ProgressTracker progress;
    
    private int epoch = 0;
    
    public TrainingSession(NeuralNetwork network, Dataset dataset, TrainingConfig config) {
        validateConstructorParameters(network, dataset, config);
        
        this.network = network;
        this.dataset = dataset;
        this.config = config;
        this.progress = new ProgressTracker();
        
        // Auto-setup GPU if configured
        if (config.useGPU) {
            int device = config.devices[0]; // Use first device

            // Prepare network
            if (!network.gpuReady()) {
                CudaEngine.prepareThread(device);
                network.prepareGPU();
                CudaEngine.finalizeThread();
            }

            // Prepare dataset
            if (!dataset.gpuReady()) {
                dataset.prepareBatchesGPU(device);
            }
        }
    }
    
    /**
     * Run one training step through all batches
     * @return loss for this epoch
     */
    public double step() {
        return config.useGPU ? stepGPU() : stepCPU();
    }
    
    public double stepCPU() {
        if (!dataset.cpuReady())
            throw new IllegalStateException("Dataset not prepared - call dataset.prepareBatches() first");
        
        long forwardTime = 0;
        long backwardTime = 0;
        long updateTime = 0;
        long lossTime = 0;
        long stepTimeStart = System.nanoTime();
        
        double totalLoss = 0;
        
        for (int i = 0; i < dataset.batchCount; i++) {
            float[] input = dataset.getInput(i);
            float[] target = dataset.getTarget(i);
            
            // Forward pass
            long t1 = System.nanoTime();
            NeuralNetworkForwardOutput output = network.feedForward(input, dataset.batchSize);
            forwardTime += System.nanoTime() - t1;
            
            // Backward pass
            long t2 = System.nanoTime();
            NeuralNetworkBackpropagateOutput gradients = network.backpropagate(output, target, config.lossFunction);
            backwardTime += System.nanoTime() - t2;
            
            // Weight update
            long t3 = System.nanoTime();
            network.applyGradients(gradients, config.learningRate, config.weightDecay);
            updateTime += System.nanoTime() - t3;
            
            // Loss calculation
            long t4 = System.nanoTime();
            totalLoss += config.lossFunction.compute(output.output(), target);
            lossTime += System.nanoTime() - t4;
        }
        
        double avgLoss = totalLoss / dataset.batchCount;
        
        float stepTime = (System.nanoTime() - stepTimeStart);
        
        // Convert to milliseconds
        progress.recordMetrics(avgLoss, 
                            stepTime / 1e6f,
                         forwardTime / 1e6f, 
                        backwardTime / 1e6f, 
                          updateTime / 1e6f, 
                            lossTime / 1e6f);
        
        epoch++;
        return avgLoss;
    }
    
    /**
     * Manages CUDA events for timing GPU operations across multiple batches
     */
    private static class GPUTimingManager {
        private final CUevent[] forwardStartEvents;
        private final CUevent[] forwardEndEvents;
        private final CUevent[] backwardStartEvents;
        private final CUevent[] backwardEndEvents;
        private final CUevent[] updateStartEvents;
        private final CUevent[] updateEndEvents;
        private final CUevent[] lossStartEvents;
        private final CUevent[] lossEndEvents;
        
        private final int batchCount;
        
        public GPUTimingManager(int batchCount) {
            this.batchCount = batchCount;
            
            // Initialize event arrays
            forwardStartEvents = new CUevent[batchCount];
            forwardEndEvents = new CUevent[batchCount];
            backwardStartEvents = new CUevent[batchCount];
            backwardEndEvents = new CUevent[batchCount];
            updateStartEvents = new CUevent[batchCount];
            updateEndEvents = new CUevent[batchCount];
            lossStartEvents = new CUevent[batchCount];
            lossEndEvents = new CUevent[batchCount];
            
            // Create all events
            for (int i = 0; i < batchCount; i++) {
                forwardStartEvents[i] = new CUevent();
                forwardEndEvents[i] = new CUevent();
                backwardStartEvents[i] = new CUevent();
                backwardEndEvents[i] = new CUevent();
                updateStartEvents[i] = new CUevent();
                updateEndEvents[i] = new CUevent();
                lossStartEvents[i] = new CUevent();
                lossEndEvents[i] = new CUevent();
                
                JCudaDriver.cuEventCreate(forwardStartEvents[i], 0);
                JCudaDriver.cuEventCreate(forwardEndEvents[i], 0);
                JCudaDriver.cuEventCreate(backwardStartEvents[i], 0);
                JCudaDriver.cuEventCreate(backwardEndEvents[i], 0);
                JCudaDriver.cuEventCreate(updateStartEvents[i], 0);
                JCudaDriver.cuEventCreate(updateEndEvents[i], 0);
                JCudaDriver.cuEventCreate(lossStartEvents[i], 0);
                JCudaDriver.cuEventCreate(lossEndEvents[i], 0);
            }
        }
        
        public void recordForwardStart(int batchIndex, CUstream stream) {
            JCudaDriver.cuEventRecord(forwardStartEvents[batchIndex], stream);
        }
        
        public void recordForwardEnd(int batchIndex, CUstream stream) {
            JCudaDriver.cuEventRecord(forwardEndEvents[batchIndex], stream);
        }
        
        public void recordBackwardStart(int batchIndex, CUstream stream) {
            JCudaDriver.cuEventRecord(backwardStartEvents[batchIndex], stream);
        }
        
        public void recordBackwardEnd(int batchIndex, CUstream stream) {
            JCudaDriver.cuEventRecord(backwardEndEvents[batchIndex], stream);
        }
        
        public void recordUpdateStart(int batchIndex, CUstream stream) {
            JCudaDriver.cuEventRecord(updateStartEvents[batchIndex], stream);
        }
        
        public void recordUpdateEnd(int batchIndex, CUstream stream) {
            JCudaDriver.cuEventRecord(updateEndEvents[batchIndex], stream);
        }
        
        public void recordLossStart(int batchIndex, CUstream stream) {
            JCudaDriver.cuEventRecord(lossStartEvents[batchIndex], stream);
        }
        
        public void recordLossEnd(int batchIndex, CUstream stream) {
            JCudaDriver.cuEventRecord(lossEndEvents[batchIndex], stream);
        }
        
        /**
         * Calculate total elapsed times for all phases across all batches
         * @return array containing [forwardTime, backwardTime, updateTime, lossTime] in milliseconds
         */
        public float[] calculateElapsedTimes() {
            float forwardTime = 0;
            float backwardTime = 0;
            float updateTime = 0;
            float lossTime = 0;
            
            for (int i = 0; i < batchCount; i++) {
                float[] elapsedTime = new float[1];
                
                // Calculate forward time
                JCudaDriver.cuEventSynchronize(forwardEndEvents[i]);
                JCudaDriver.cuEventElapsedTime(elapsedTime, forwardStartEvents[i], forwardEndEvents[i]);
                forwardTime += elapsedTime[0];
                
                // Calculate backward time
                JCudaDriver.cuEventSynchronize(backwardEndEvents[i]);
                JCudaDriver.cuEventElapsedTime(elapsedTime, backwardStartEvents[i], backwardEndEvents[i]);
                backwardTime += elapsedTime[0];
                
                // Calculate update time
                JCudaDriver.cuEventSynchronize(updateEndEvents[i]);
                JCudaDriver.cuEventElapsedTime(elapsedTime, updateStartEvents[i], updateEndEvents[i]);
                updateTime += elapsedTime[0];
                
                // Calculate loss time
                JCudaDriver.cuEventSynchronize(lossEndEvents[i]);
                JCudaDriver.cuEventElapsedTime(elapsedTime, lossStartEvents[i], lossEndEvents[i]);
                lossTime += elapsedTime[0];
            }
            
            return new float[]{forwardTime, backwardTime, updateTime, lossTime};
        }
        
        /**
         * Destroy all CUDA events to free resources
         */
        public void cleanup() {
            for (int i = 0; i < batchCount; i++) {
                JCudaDriver.cuEventDestroy(forwardStartEvents[i]);
                JCudaDriver.cuEventDestroy(forwardEndEvents[i]);
                JCudaDriver.cuEventDestroy(backwardStartEvents[i]);
                JCudaDriver.cuEventDestroy(backwardEndEvents[i]);
                JCudaDriver.cuEventDestroy(updateStartEvents[i]);
                JCudaDriver.cuEventDestroy(updateEndEvents[i]);
                JCudaDriver.cuEventDestroy(lossStartEvents[i]);
                JCudaDriver.cuEventDestroy(lossEndEvents[i]);
            }
        }
    }
    
    public double stepGPU() {
        if (!dataset.gpuReady())
            throw new IllegalStateException("Dataset not prepared for GPU - call dataset.prepareBatchesGPU() first");

        if (!network.gpuReady())
            throw new IllegalStateException("Network not prepared for GPU - call network.prepareGPU() first");

        CUstream stream = CudaUtil.createStream();        
        CUdeviceptr[] batchLossGPU = new CUdeviceptr[dataset.batchCount];
        GPUTimingManager timingManager = new GPUTimingManager(dataset.batchCount);
        long stepTimeStart = System.nanoTime();
        
        for (int i = 0; i < dataset.batchCount; i++) {
            CUdeviceptr input = dataset.getInputGPU(i);
            CUdeviceptr target = dataset.getTargetGPU(i);

            // Forward pass - just record events, no sync
            timingManager.recordForwardStart(i, stream);
            NeuralNetworkForwardOutputGPU output = network.feedForwardGPU(input, dataset.batchSize, stream);
            timingManager.recordForwardEnd(i, stream);

            // Backward pass - just record events, no sync
            timingManager.recordBackwardStart(i, stream);
            NeuralNetworkBackpropagateOutputGPU gradients = network.backpropagateGPU(output, target, config.lossFunction, stream);
            timingManager.recordBackwardEnd(i, stream);
        
            // Weight update - just record events, no sync
            timingManager.recordUpdateStart(i, stream);
            network.applyGradientsGPU(gradients, config.learningRate, config.weightDecay, stream);
            timingManager.recordUpdateEnd(i, stream);
            
            // Loss computation on GPU
            timingManager.recordLossStart(i, stream);
            batchLossGPU[i] = CudaUtil.createFloatAsync(1, stream);
            long outputSize = network.getOutputDim() * dataset.batchSize;
            config.lossFunction.computeGPU(output.output(), target, batchLossGPU[i], outputSize, stream);
            timingManager.recordLossEnd(i, stream);

            // Free GPU memory for this batch
            output.freeAsync(stream);
            gradients.freeAsync(stream);
        }

        // Now synchronize and calculate all timing after the loop
        JCudaDriver.cuStreamSynchronize(stream);
        
        double totalLoss = 0;
        long lossStartTime = System.nanoTime();
        
        for (int i = 0; i < dataset.batchCount; i++) {
            float[] batchLoss = CudaUtil.fromGPUFloat(batchLossGPU[i], 1);
            double loss = batchLoss[0];

            // Check for NaN/Inf and throw exception
            if (Double.isNaN(loss) || Double.isInfinite(loss))
                throw new RuntimeException("NaN/Inf detected in loss at batch " + i + ", epoch " + epoch + ", loss: " + loss);

            totalLoss += loss;

            // Free loss memory
            CudaUtil.free(batchLossGPU[i]);
        }
            
        float[] timings = timingManager.calculateElapsedTimes();
    
        float forwardTime = timings[0];
        float backwardTime = timings[1];
        float updateTime = timings[2];
        float lossTime = (System.nanoTime() - lossStartTime) / 1e6f + timings[3];

        CudaUtil.freeStream(stream);
        timingManager.cleanup();

        double avgLoss = totalLoss / dataset.batchCount;
        
        float stepTime = (System.nanoTime() - stepTimeStart) / 1e6f;
        
        // Convert to milliseconds
        progress.recordMetrics(avgLoss, 
                               stepTime,
                               forwardTime,
                               backwardTime, 
                               updateTime, 
                               lossTime);

        epoch++;
        return avgLoss;
    }
    
    /**
     * Debug a single batch - returns detailed info
     * @param batchNumber which batch to debug
     * @return debug information for that batch
     */
    public TrainingSessionOutput debug(int batchNumber) {
        if (!dataset.cpuReady())
            throw new IllegalStateException("Dataset not prepared - call dataset.prepareBatches() first");
        
        if (batchNumber < 0 || batchNumber >= dataset.batchCount)
            throw new IllegalArgumentException("Batch number out of range: " + batchNumber);
        
        float[] input = dataset.getInput(batchNumber);
        float[] target = dataset.getTarget(batchNumber);
        
        // Forward pass
        NeuralNetworkForwardOutput forwardOutput = network.feedForward(input, dataset.batchSize);
        float[] result = forwardOutput.output();
        
        // Backward pass
        NeuralNetworkBackpropagateOutput backwardOutput = network.backpropagate(forwardOutput, target, config.lossFunction);
        
        // dont apply gradients
        // network.applyGradients(backwardOutput, config.learningRate, config.weightDecay);
        
        // Calculate loss for this batch
        double loss = config.lossFunction.compute(result, target);
        
        return new TrainingSessionOutput(forwardOutput, backwardOutput, result, loss);
    }
    
    /**
     * Debug a single batch on GPU - returns detailed info
     * @param batchNumber which batch to debug
     * @return debug information for that batch
     */
    public TrainingSessionOutputGPU debugGPU(int batchNumber) {
        if (!dataset.gpuReady())
            throw new IllegalStateException("Dataset not prepared for GPU - call dataset.prepareBatchesGPU() first");
        
        if (batchNumber < 0 || batchNumber >= dataset.batchCount)
            throw new IllegalArgumentException("Batch number out of range: " + batchNumber);
        
        CUdeviceptr input = dataset.getInputGPU(batchNumber);
        CUdeviceptr target = dataset.getTargetGPU(batchNumber);
        
        CUstream stream = CudaUtil.createStream();
        
        // Forward pass
        NeuralNetworkForwardOutputGPU forwardOutput = network.feedForwardGPU(input, dataset.batchSize, stream);
        
        // Backward pass
        NeuralNetworkBackpropagateOutputGPU backwardOutput = network.backpropagateGPU(forwardOutput, target, config.lossFunction, stream);
        
        // dont apply gradients
        // network.applyGradientsGPU(backwardOutput, config.learningRate, config.weightDecay, stream);
        
        // Compute loss on GPU
        CUdeviceptr lossGPU = CudaUtil.createFloatAsync(1, stream);
        long outputSize = network.getOutputDim() * dataset.batchSize;
        config.lossFunction.computeGPU(forwardOutput.output(), target, lossGPU, outputSize, stream);
        
        float[] result = CudaUtil.fromGPUFloatAsync(forwardOutput.output(), (int)outputSize, stream);
        float[] lossArray = CudaUtil.fromGPUFloatAsync(lossGPU, 1, stream);
        JCudaDriver.cuStreamSynchronize(stream);

        double loss = lossArray[0];
        CudaUtil.free(lossGPU);
        
        CudaUtil.freeStream(stream);
        
        return new TrainingSessionOutputGPU(forwardOutput, backwardOutput, result, loss);
    }

    /**
     * Evaluates network outcome on a dataset - returns average loss
     * @param dataset
     * @return 
     */
    public double evaluate(Dataset dataset) {
        return config.useGPU ? evaluateGPU(dataset) : evaluateCPU(dataset);
    }

    public double evaluateCPU(Dataset dataset) {
        if (!dataset.cpuReady())
            throw new IllegalStateException("Dataset not prepared - call dataset.prepareBatches() first");

        double totalLoss = 0;

        for (int i = 0; i < dataset.batchCount; i++) {
            float[] input = dataset.getInput(i);
            float[] target = dataset.getTarget(i);

            // Forward pass only - no backpropagation needed for evaluation
            NeuralNetworkForwardOutput output = network.feedForward(input, dataset.batchSize);

            // Calculate loss
            totalLoss += config.lossFunction.compute(output.output(), target);
        }

        return totalLoss / dataset.batchCount;
    }

    public double evaluateGPU(Dataset dataset) {
        if (!dataset.gpuReady())
            throw new IllegalStateException("Dataset not prepared for GPU - call dataset.prepareBatchesGPU() first");

        if (!network.gpuReady())
            throw new IllegalStateException("Network not prepared for GPU - call network.prepareGPU() first");

        CUstream stream = CudaUtil.createStream();

        double totalLoss = 0;

        for (int i = 0; i < dataset.batchCount; i++) {
            CUdeviceptr input = dataset.getInputGPU(i);
            CUdeviceptr target = dataset.getTargetGPU(i);

            // Forward pass only - no backpropagation needed for evaluation
            NeuralNetworkForwardOutputGPU output = network.feedForwardGPU(input, dataset.batchSize, stream);
            
            // Compute loss on GPU
            CUdeviceptr lossGPU = CudaUtil.createFloatAsync(1, stream);
            long outputSize = network.getOutputDim() * dataset.batchSize;
            config.lossFunction.computeGPU(output.output(), target, lossGPU, outputSize, stream);

            // Copy only the loss value
            float[] lossArray = CudaUtil.fromGPUFloatAsync(lossGPU, 1, stream);
            JCudaDriver.cuStreamSynchronize(stream);
            totalLoss += lossArray[0];

            CudaUtil.free(lossGPU);
            
            // Free GPU memory for this batch
            output.freeAsync(stream);
        }
        
        CudaUtil.freeStream(stream);

        return totalLoss / dataset.batchCount;
    }
    
    public enum TrainingStatus {
        CONTINUE("Training in progress"),
        STOP_MAX_TIME("Maximum training time reached"),
        STOP_MAX_EPOCHS("Maximum epochs reached"),
        STOP_LOSS_EXPLOSION("Loss explosion detected"),
        STOP_STAGNATION("No improvement for extended period"),
        STOP_CONVERGED("Loss has converged"),
        STOP_MINIMAL_IMPROVEMENT("Minimal improvement detected"),
        STOP_OSCILLATION("Unstable training - loss oscillating"),
        STOP_DEAD_NETWORK("Network not learning - weights not updating");

        public final String description;

        TrainingStatus(String description) {
            this.description = description;
        }
    }
    
    /**
     * Determines current training status based on multiple criteria
     * @return current training status
     */
    public TrainingStatus status() {
        List<Double> history = progress.getLossHistory();

        // 1. Max time limit reached
        if (progress.getElapsedTimeMs() > config.maxTimeMs) {
            return TrainingStatus.STOP_MAX_TIME;
        }

        // 2. Max epochs reached
        if (epoch >= config.maxEpochs) {
            return TrainingStatus.STOP_MAX_EPOCHS;
        }

        // 3. Must train minimum epochs before considering other stopping criteria
        if (epoch < config.minEpochs) {
            return TrainingStatus.CONTINUE;
        }
        
        if (history.size() >= 20) {
            double currentLoss = progress.getCurrentLoss();
            double loss20EpochsAgo = history.get(history.size() - 20);
            double initialLoss = history.get(0);

            double recentChange = Math.abs(currentLoss - loss20EpochsAgo);
            double recentRelativeChange = recentChange / Math.abs(loss20EpochsAgo);

            // No change whatsoever in 20 epochs = dead network
            if (recentRelativeChange < 1e-8) {
                double improvementFromInitial = (initialLoss - currentLoss) / Math.abs(initialLoss);
            
            // Dead network: Less than 5% improvement after 20+ epochs
            if (improvementFromInitial < 0.05)
                return TrainingStatus.STOP_DEAD_NETWORK;
            else
                return TrainingStatus.STOP_CONVERGED;
            }
        }     
//        // 5. Early stopping - no improvement for extended periods
//        if (progress.isStagnating()) {
//            return TrainingStatus.STOP_STAGNATION;
//        }

        // 6. Convergence detection - loss plateau over last 20 epochs
//        if (history.size() >= 20) {
//            double recentMean = 0;
//            int windowSize = 20;
//
//            for (int i = history.size() - windowSize; i < history.size(); i++) {
//                recentMean += history.get(i);
//            }
//            recentMean /= windowSize;
//
//            double recentVariance = 0;
//            for (int i = history.size() - windowSize; i < history.size(); i++) {
//                double diff = history.get(i) - recentMean;
//                recentVariance += diff * diff;
//            }
//            recentVariance /= windowSize;
//
//            double coefficientOfVariation = Math.sqrt(recentVariance) / Math.abs(recentMean);
//            if (coefficientOfVariation < 0.001) {
//                return TrainingStatus.STOP_CONVERGED;
//            }
//        }

//        // 7. Minimal improvement detection
//        if (history.size() >= 50) {
//            double recentSum = 0;
//            double olderSum = 0;
//            int windowSize = 10;
//
//            for (int i = history.size() - windowSize; i < history.size(); i++) {
//                recentSum += history.get(i);
//            }
//
//            for (int i = history.size() - 50; i < history.size() - 40; i++) {
//                olderSum += history.get(i);
//            }
//
//            double relativeImprovement = Math.abs(recentSum - olderSum) / Math.abs(olderSum);
//            if (relativeImprovement < 0.00001) {
//                return TrainingStatus.STOP_MINIMAL_IMPROVEMENT;
//            }
//        }

        // 8. Loss oscillation detection
        if (history.size() > 20) {
            int oscillationCount = 0;
            for (int i = history.size() - 9; i < history.size(); i++) {
                if ((history.get(i) > history.get(i-1)) != (history.get(i-1) > history.get(i-2))) {
                    oscillationCount++;
                }
            }
            if (oscillationCount >= 20) {
                return TrainingStatus.STOP_OSCILLATION;
            }
        }

        return TrainingStatus.CONTINUE;
    }
    
    /**
     * @return current epoch number
     */
    public int getEpoch() {
        return epoch;
    }
    
    /**
     * Returns progress tracker
     * @return 
     */
    public ProgressTracker getProgress() {
        return progress;
    }
    
    
/**
     * Validates constructor parameters
     */
    private static void validateConstructorParameters(NeuralNetwork network, Dataset dataset, TrainingConfig config) {
        if (network == null)
            throw new IllegalArgumentException("Neural network cannot be null");

        if (!network.isFinalized())
            throw new IllegalArgumentException("Network must be finalized (built) before training");
        
        if (dataset == null)
            throw new IllegalArgumentException("Dataset cannot be null");
        
        if (config == null)
            throw new IllegalArgumentException("Training configuration cannot be null");
        
        if (network.getInputDim() != dataset.inputDim)
            throw new IllegalArgumentException(String.format(
               "Network input size (%d) does not match dataset input dimension (%d)",
                network.getInputDim(), dataset.inputDim));
        
        if (network.getOutputDim() != dataset.targetDim)
            throw new IllegalArgumentException(String.format(
                "Network output size (%d) does not match dataset target dimension (%d)",
                network.getOutputDim(), dataset.targetDim));
    }
}
