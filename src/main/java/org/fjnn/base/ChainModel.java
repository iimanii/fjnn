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
package org.fjnn.base;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author ahmed
 */
public class ChainModel {
    private final List<ModelComponent> components;

    // Constructor
    public ChainModel() {
        this.components = new ArrayList<>();
    }

    // Add a component to the chain
    public void addComponent(ModelComponent component) {
        components.add(component);
    }

    // Perform a forward pass through all components
    public FeedForwardContext forward(float[] inputs, int batchSize) {
        FeedForwardContext context = new FeedForwardContext();
        float[] currentOutput = inputs;

        for (ModelComponent component : components) {
            FeedForwardResult result = component.feedForward(currentOutput, batchSize);
            context.addResult(result);          // Store the result
            currentOutput = result.result();    // Update output for the next component
        }

        return context;
    }
//
//    // Perform a backward pass through all components
//    public float[] backward(float[] gradients, float learningRate) {
//        float[] currentGradients = gradients;
//        for (int i = components.size() - 1; i >= 0; i--) {
//            currentGradients = components.get(i).backward(currentGradients, learningRate);
//        }
//        return currentGradients;
//    }

    // Retrieve the list of components (optional utility)
    public List<ModelComponent> getComponents() {
        return components;
    }
}
