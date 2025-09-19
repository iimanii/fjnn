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
package junit.training.loss;

import org.fjnn.network.NeuralNetwork;
import org.fjnn.activation.*;
import org.fjnn.loss.BirdLoss;
import org.fjnn.loss.MeanSquareError;
import org.fjnn.loss.Loss;
import org.fjnn.network.outputs.*;
import org.fjnn.normalizer.LayerNormalizer;
import org.fjnn.cuda.CudaEngine;
import org.junit.*;
import static org.junit.Assert.*;

/**
 * JUnit test demonstrating Bird Loss training scenarios and comparing
 * its behavior with other loss functions.
 * 
 * @author ahmed
 */
@Ignore
public class BirdLossTrainingTest {
    
    private static final float EPSILON = 1e-4f;
    
    @BeforeClass
    public static void setupCuda() {
        CudaEngine.init();
    }
    
    @Before
    public void prepareCudaThread() {
        CudaEngine.prepareThread(0);
    }
    
    @After
    public void finalizeCudaThread() {
        CudaEngine.finalizeThread();
    }
    
    @Test
    public void testBasicTraining() {
        System.out.println("=== Test Basic Bird Loss Training ===");
        
        // Create a simple network for regression
        NeuralNetwork network = new NeuralNetwork(10, 1, new Linear())
            .addHiddenLayer(20, new ReLU())
            .addHiddenLayer(10, new Tanh())
            .build()
            .xavier(1.0f);
        
        // Create bird loss with default parameters (alpha=0.25, beta=1000)
        BirdLoss loss = new BirdLoss(0.25f, 100f);
        
        // Generate synthetic data
        float[][] inputs = generateRandomData(50, 10);
        float[][] targets = generateRandomData(50, 1);
        
        // Track loss over epochs
        float initialLoss = 0.0f;
        float finalLoss = 0.0f;
        
        // Training loop
        float learningRate = 0.01f;
        for (int epoch = 0; epoch < 1000; epoch++) {
            float totalLoss = 0.0f;
            
            for (int i = 0; i < inputs.length; i++) {
                // Forward pass
                NeuralNetworkForwardOutput forward = network.feedForward(inputs[i], 1);
                
                // Calculate loss
                float lossValue = loss.compute(forward.output(), targets[i]);
                totalLoss += lossValue;
                
                // Backward pass
                NeuralNetworkBackpropagateOutput backward = network.backpropagate(forward, targets[i], loss);
                
                // Update weights
                network.applyGradients(backward, learningRate, 0.0001f);
            }
            
            float avgLoss = totalLoss / inputs.length;
            if (epoch == 0) initialLoss = avgLoss;
            if (epoch == 99) finalLoss = avgLoss;
        }
        
        // Assert that loss decreased
        assertTrue("Loss should decrease during training", finalLoss < initialLoss);
        assertTrue("Final loss should be significantly lower", finalLoss < initialLoss * 0.8f);
        
        System.out.printf("Initial loss: %.6f, Final loss: %.6f%n", initialLoss, finalLoss);
    }
    
    @Test
    public void testBirdLossVsMSE() {
        System.out.println("\n=== Test Bird Loss vs MSE Comparison ===");
        
        // Create loss functions
        BirdLoss birdLoss = new BirdLoss(0.1f, 500.0f);
        MeanSquareError mseLoss = new MeanSquareError();
        
        // Test different error magnitudes
        float[] errors = {0.0f, 0.01f, 0.05f, 0.1f, 0.5f, 1.0f};
        float[] output = new float[1];
        float[] expected = {0.0f};
        
        System.out.println("Error\tBird\tMSE\tRatio");
        for (float error : errors) {
            output[0] = error;
            float birdValue = birdLoss.compute(output, expected);
            float mseValue = mseLoss.compute(output, expected);
            float ratio = error == 0 ? 1.0f : birdValue / mseValue;
            
            System.out.printf("%.2f\t%.4f\t%.4f\t%.2f%n", error, birdValue, mseValue, ratio);
            
            // Bird loss should be relatively higher for small errors
            if (error > 0 && error < 0.1) {
                assertTrue("Bird loss should be relatively higher than MSE for small errors", ratio > 10.0f);
            }
        }
    }
    
    @Test
    public void testRobustRegression() {
        System.out.println("\n=== Test Robust Regression ===");
        
        // Network for robust regression
        NeuralNetwork network = new NeuralNetwork(5, 1, new Linear())
            .addHiddenLayer(32, new ReLU(), new LayerNormalizer(), 0.0f)
            .addHiddenLayer(16, new ReLU(), new LayerNormalizer(), 0.0f)
            .build()
            .kaiming(1);
        
        // Create a copy for MSE comparison
        NeuralNetwork networkMSE = network.copy(true, false);
        
        // Bird loss with moderate parameters for robustness
        BirdLoss birdLoss = new BirdLoss(0.3f, 150.0f);
        MeanSquareError mseLoss = new MeanSquareError();
        
        // Generate data with outliers
        int nSamples = 100;
        float[][] inputs = generateRandomData(nSamples, 5);
        float[][] targets = new float[nSamples][1];
        
        // Create linear relationship with noise and outliers
        for (int i = 0; i < nSamples; i++) {
            targets[i][0] = 0.5f * inputs[i][0] + 0.3f * inputs[i][1] - 0.2f * inputs[i][2];
            
            // Add noise
            targets[i][0] += (float)(Math.random() - 0.5) * 0.1f;
            
            // Add outliers (10% of data)
            if (i % 10 == 0) {
                targets[i][0] += (float)(Math.random() - 0.5) * 5.0f;
            }
        }
        
        // Train with bird loss
        float birdFinalLoss = trainAndEvaluate(network, inputs, targets, birdLoss, 50);
        
        // Train with MSE
        float mseFinalLoss = trainAndEvaluate(networkMSE, inputs, targets, mseLoss, 50);
        
        // Evaluate on clean test data (without outliers)
        float[][] testInputs = generateRandomData(20, 5);
        float[][] testTargets = new float[20][1];
        for (int i = 0; i < 20; i++) {
            testTargets[i][0] = 0.5f * testInputs[i][0] + 0.3f * testInputs[i][1] - 0.2f * testInputs[i][2];
        }
        
        float birdTestError = evaluateNetwork(network, testInputs, testTargets);
        float mseTestError = evaluateNetwork(networkMSE, testInputs, testTargets);
        
        System.out.printf("Training loss - Bird: %.4f, MSE: %.4f%n", birdFinalLoss, mseFinalLoss);
        System.out.printf("Test error - Bird Loss: %.4f, MSE: %.4f%n", birdTestError, mseTestError);
        
        // Bird loss should perform reasonably compared to MSE
        assertTrue("Bird loss should not be dramatically worse than MSE", 
                  birdTestError < mseTestError * 2.0f);
    }
    
    @Test
    public void testHighPrecisionTraining() {
        System.out.println("\n=== Test High Precision Training ===");
        
        // Network for high-precision classification
        NeuralNetwork network = new NeuralNetwork(20, 10, new Sigmoid())
            .addHiddenLayer(15, new ReLU())
            .build()
            .xavier(0.5f);
        
        // Bird loss with high α and β values to maintain strong learning pressure
        BirdLoss birdLoss = new BirdLoss(0.7f, 200.0f);
        
        // Generate sparse target data
        float[][] inputs = generateRandomData(50, 20);
        float[][] targets = new float[50][10];
        
        // Only a few outputs should be non-zero
        for (int i = 0; i < 50; i++) {
            targets[i][i % 10] = 1.0f;
        }
        
        // Train
        float learningRate = 0.005f;
        float initialAccuracy = 0.0f;
        float finalAccuracy = 0.0f;
        
        for (int epoch = 0; epoch < 100; epoch++) {
            float totalLoss = 0.0f;
            int correct = 0;
            
            for (int i = 0; i < inputs.length; i++) {
                NeuralNetworkForwardOutput forward = network.feedForward(inputs[i], 1);
                
                float lossValue = birdLoss.compute(forward.output(), targets[i]);
                totalLoss += lossValue;
                
                // Check accuracy
                int predicted = argmax(forward.output());
                int actual = argmax(targets[i]);
                if (predicted == actual) correct++;
                
                NeuralNetworkBackpropagateOutput backward = network.backpropagate(forward, targets[i], birdLoss);
                network.applyGradients(backward, learningRate, 0.001f);
            }
            
            float accuracy = (float)correct / inputs.length;
            if (epoch == 0) initialAccuracy = accuracy;
            if (epoch == 99) {
                finalAccuracy = accuracy;
                System.out.printf("Final loss: %.6f, Accuracy: %.2f%%%n", 
                    totalLoss / inputs.length, accuracy * 100);
            }
        }
        
        // Should achieve good accuracy
        assertTrue("Accuracy should improve", finalAccuracy > initialAccuracy);
        assertTrue("Should achieve reasonable accuracy", finalAccuracy > 0.5f);
    }
    
    @Test
    public void testParameterSensitivity() {
        System.out.println("\n=== Test Parameter Sensitivity ===");
        
        // Test how different alpha and beta values affect training
        float[][] alphaValues = {{0.1f, 100.0f}, {0.5f, 100.0f}, {1.0f, 100.0f}};
        float[][] betaValues = {{0.5f, 50.0f}, {0.5f, 200.0f}, {0.5f, 500.0f}};
        
        // Simple test data
        float[][] inputs = generateRandomData(20, 5);
        float[][] targets = generateRandomData(20, 1);
        
        System.out.println("Testing alpha sensitivity:");
        for (float[] params : alphaValues) {
            NeuralNetwork network = new NeuralNetwork(5, 1, new Linear())
                .addHiddenLayer(10, new ReLU())
                .build()
                .xavier(1.0f);
            
            BirdLoss loss = new BirdLoss(params[0], params[1]);
            float finalLoss = trainAndEvaluate(network, inputs, targets, loss, 30);
            System.out.printf("Alpha=%.1f, Beta=%.0f: Final loss=%.4f%n", 
                params[0], params[1], finalLoss);
        }
        
        System.out.println("\nTesting beta sensitivity:");
        for (float[] params : betaValues) {
            NeuralNetwork network = new NeuralNetwork(5, 1, new Linear())
                .addHiddenLayer(10, new ReLU())
                .build()
                .xavier(1.0f);
            
            BirdLoss loss = new BirdLoss(params[0], params[1]);
            float finalLoss = trainAndEvaluate(network, inputs, targets, loss, 30);
            System.out.printf("Alpha=%.1f, Beta=%.0f: Final loss=%.4f%n", 
                params[0], params[1], finalLoss);
        }
    }
    
    @Test
    public void testConvergenceBehavior() {
        System.out.println("\n=== Test Convergence Behavior ===");
        
        // Create networks for comparison
        NeuralNetwork networkBird = new NeuralNetwork(5, 1, new Linear())
            .addHiddenLayer(10, new ReLU())
            .build()
            .xavier(1.0f);
        
        NeuralNetwork networkMSE = networkBird.copy(true, false);
        
        BirdLoss birdLoss = new BirdLoss(0.5f, 200.0f);
        MeanSquareError mseLoss = new MeanSquareError();
        
        // Simple regression problem
        float[][] inputs = generateRandomData(30, 5);
        float[][] targets = new float[30][1];
        
        // Linear relationship
        for (int i = 0; i < 30; i++) {
            targets[i][0] = 0.3f * inputs[i][0] + 0.2f * inputs[i][1] - 0.1f * inputs[i][2];
        }
        
        float learningRate = 0.01f;
        
        // Track convergence
        float[] birdLosses = new float[50];
        float[] mseLosses = new float[50];
        
        for (int epoch = 0; epoch < 50; epoch++) {
            // Train Bird Loss network
            float birdTotalLoss = 0.0f;
            for (int i = 0; i < inputs.length; i++) {
                NeuralNetworkForwardOutput forward = networkBird.feedForward(inputs[i], 1);
                float lossValue = birdLoss.compute(forward.output(), targets[i]);
                birdTotalLoss += lossValue;
                
                NeuralNetworkBackpropagateOutput backward = networkBird.backpropagate(forward, targets[i], birdLoss);
                networkBird.applyGradients(backward, learningRate, 0.0001f);
            }
            birdLosses[epoch] = birdTotalLoss / inputs.length;
            
            // Train MSE network
            float mseTotalLoss = 0.0f;
            for (int i = 0; i < inputs.length; i++) {
                NeuralNetworkForwardOutput forward = networkMSE.feedForward(inputs[i], 1);
                float lossValue = mseLoss.compute(forward.output(), targets[i]);
                mseTotalLoss += lossValue;
                
                NeuralNetworkBackpropagateOutput backward = networkMSE.backpropagate(forward, targets[i], mseLoss);
                networkMSE.applyGradients(backward, learningRate, 0.0001f);
            }
            mseLosses[epoch] = mseTotalLoss / inputs.length;
        }
        
        // Analyze convergence
        float birdImprovement = birdLosses[0] - birdLosses[49];
        float mseImprovement = mseLosses[0] - mseLosses[49];
        
        System.out.printf("Bird Loss: Initial=%.4f, Final=%.4f, Improvement=%.4f%n", 
            birdLosses[0], birdLosses[49], birdImprovement);
        System.out.printf("MSE: Initial=%.4f, Final=%.4f, Improvement=%.4f%n", 
            mseLosses[0], mseLosses[49], mseImprovement);
        
        // Both should converge
        assertTrue("Bird loss should improve", birdImprovement > 0);
        assertTrue("MSE should improve", mseImprovement > 0);
        assertTrue("Final Bird loss should be reasonable", birdLosses[49] < birdLosses[0]);
    }
    
    /**
     * Helper method to train and evaluate a network
     */
    private float trainAndEvaluate(NeuralNetwork network, float[][] inputs, 
                                  float[][] targets, Loss loss, int epochs) {
        float learningRate = 0.01f;
        float finalLoss = 0.0f;
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalLoss = 0.0f;
            
            for (int i = 0; i < inputs.length; i++) {
                NeuralNetworkForwardOutput forward = network.feedForward(inputs[i], 1);
                float lossValue = loss.compute(forward.output(), targets[i]);
                totalLoss += lossValue;
                
                NeuralNetworkBackpropagateOutput backward = network.backpropagate(forward, targets[i], loss);
                network.applyGradients(backward, learningRate, 0.0001f);
            }
            
            if (epoch == epochs - 1) {
                finalLoss = totalLoss / inputs.length;
            }
        }
        
        return finalLoss;
    }
    
    /**
     * Evaluate network on test data using MSE
     */
    private float evaluateNetwork(NeuralNetwork network, float[][] inputs, float[][] targets) {
        float totalError = 0.0f;
        
        for (int i = 0; i < inputs.length; i++) {
            float[] output = network.compute(inputs[i]);
            float error = 0.0f;
            for (int j = 0; j < output.length; j++) {
                float diff = output[j] - targets[i][j];
                error += diff * diff;
            }
            totalError += error / output.length;
        }
        
        return totalError / inputs.length;
    }
    
    /**
     * Find index of maximum value
     */
    private int argmax(float[] array) {
        int maxIdx = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    /**
     * Generate random data for testing
     */
    private float[][] generateRandomData(int samples, int features) {
        float[][] data = new float[samples][features];
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < features; j++) {
                data[i][j] = (float)(Math.random() * 2 - 1); // Range [-1, 1]
            }
        }
        return data;
    }
}