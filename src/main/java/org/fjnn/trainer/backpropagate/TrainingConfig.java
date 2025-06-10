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

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.loss.Loss;

/**
 *
 * @author ahmed
 */
public class TrainingConfig {
    /**
     * Name of training session
     */
    public final String name;
    
    /**
     * Learning rate for gradient descent
     */
    public final float learningRate;
    
    /**
     * Weight decay (L2 regularization)
     */
    public final float weightDecay;
    
    /**
     * Loss function to use
     */
    public final Loss lossFunction;
    
    /**
     * Maximum training time in milliseconds
     */
    public final long maxTimeMs;
    
    /**
     * Minimum number of epochs before stopping
     */
    public final int minEpochs;
        
    /**
     * Max number of epochs
     */
    public final int maxEpochs;


    /**
     * GPU device ids to use
     */
    public final int[] devices;
    
    /**
     * Whether to use GPU training
     */
    public final boolean useGPU;
    
    public TrainingConfig(String name, float learningRate, float weightDecay, 
                         Loss lossFunction, long maxTimeMs, int minEpochs, int maxEpochs) {
        this(name, learningRate, weightDecay, lossFunction, maxTimeMs, minEpochs, maxEpochs, null);
    }
    
    public TrainingConfig(String name, float learningRate, float weightDecay, 
                          Loss lossFunction, long maxTimeMs, int minEpochs, int maxEpochs, boolean useGPU) {
        this(name, learningRate, weightDecay, lossFunction, maxTimeMs, minEpochs, maxEpochs, useGPU ? getAllAvailableDevices() : null);
    }
    
    public TrainingConfig(String name, float learningRate, float weightDecay,
                          Loss lossFunction, long maxTimeMs, int minEpochs, int maxEpochs, int[] devices) {
        validateConstructorParameters(name, learningRate, weightDecay, lossFunction, maxTimeMs, minEpochs, maxEpochs, devices);
        
        this.name = name;
        this.learningRate = learningRate;
        this.weightDecay = weightDecay;
        this.lossFunction = lossFunction;
        this.maxTimeMs = maxTimeMs;
        this.minEpochs = minEpochs;
        this.maxEpochs = maxEpochs;

        if (devices == null) {
            this.useGPU = false;
            this.devices = null;
        } else {
            this.useGPU = true;
            this.devices = devices.clone();
        }        
    }
    

    /**
     * Helper method to get all available GPU device IDs
     */
    private static int[] getAllAvailableDevices() {
        int deviceCount = CudaEngine.getDeviceCount();
        int[] allDevices = new int[deviceCount];
        
        for (int i = 0; i < deviceCount; i++) {
            allDevices[i] = i;
        }
        
        return allDevices;
    }
    
    private static void validateConstructorParameters(String name, float learningRate, float weightDecay, 
                                                      Loss lossFunction, long maxTimeMs, int minEpochs, int maxEpochs, int[] devices) {        
        if (name == null || name.trim().isEmpty())
            throw new IllegalArgumentException("Training session name cannot be null or empty");
        
        if (!Float.isFinite(learningRate) || learningRate <= 0)
            throw new IllegalArgumentException("Learning rate must be a positive finite number, got: " + learningRate);
        
        if (!Float.isFinite(weightDecay) || weightDecay < 0)
            throw new IllegalArgumentException("Weight decay must be a non-negative finite number, got: " + weightDecay);
        
        if (lossFunction == null)
            throw new IllegalArgumentException("Loss function cannot be null");
        
        if (maxTimeMs <= 0)
            throw new IllegalArgumentException("Maximum training time must be positive, got: " + maxTimeMs);
        
        if (minEpochs < 0)
            throw new IllegalArgumentException("Minimum epochs must be non-negative, got: " + minEpochs);
        
        if (maxEpochs < minEpochs)
            throw new IllegalArgumentException("Maximum epochs must not be less than minimum epochs, got: " + maxEpochs);

        /* Validate devices if specified */
        if (devices != null) {
            if (devices.length == 0)
                throw new IllegalArgumentException("Device array cannot be empty");
            
            int availableDevices = CudaEngine.getDeviceCount();
            Set<Integer> deviceSet = new HashSet<>();
            
            // Verify devices exist and check for duplicates in one pass
            for (int deviceId : devices) {
                if (deviceId < 0 || deviceId >= availableDevices)
                    throw new IllegalArgumentException("Device ID " + deviceId + " is invalid. Available devices: 0-" + (availableDevices - 1));
                
                if (!deviceSet.add(deviceId))
                    throw new IllegalArgumentException("Duplicate device ID found: " + deviceId);
            }            
        }
    }
    
    public HashMap serialize() {
        HashMap result = new HashMap();
        result.put("name", name);
        result.put("learningRate", learningRate);
        result.put("weightDecay", weightDecay);
        result.put("lossFunction", lossFunction.serialize());
        result.put("maxTimeMs", maxTimeMs);
        result.put("minEpochs", minEpochs);
        result.put("maxEpochs", maxEpochs);
        result.put("useGPU", useGPU);
        if (devices != null) {
            result.put("devices", devices);
        }
        return result;
    }    
    
    public static TrainingConfig deserialize(Map serialized) {
        String name = (String)serialized.get("name");
        float learningRate = (Float)serialized.get("learningRate");
        float weightDecay = (Float)serialized.get("weightDecay");
        Loss lossFunction = Loss.deserialize((HashMap)serialized.get("lossFunction"));
        long maxTimeMs = (Long)serialized.get("maxTimeMs");
        int minEpochs = (Integer)serialized.get("minEpochs");
        int maxEpochs = (Integer)serialized.get("maxEpochs");
        boolean useGPU = (Boolean)serialized.get("useGPU");

        int[] devices = null;
        if (serialized.containsKey("devices")) {
            devices = (int[])serialized.get("devices");
        }

        if (devices != null) {
            return new TrainingConfig(name, learningRate, weightDecay, lossFunction, maxTimeMs, minEpochs, maxEpochs, devices);
        } else if (useGPU) {
            return new TrainingConfig(name, learningRate, weightDecay, lossFunction, maxTimeMs, minEpochs, maxEpochs, true);
        } else {
            return new TrainingConfig(name, learningRate, weightDecay, lossFunction, maxTimeMs, minEpochs, maxEpochs);
        }
    }
}
