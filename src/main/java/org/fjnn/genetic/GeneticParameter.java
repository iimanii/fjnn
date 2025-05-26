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
package org.fjnn.genetic;

import java.util.List;
import org.fjnn.util.Rng;

/**
 *
 * @author ahmed
 */
public class GeneticParameter {
    public final String name;
    public final float min, max, initial;
    public final int index;
    
    // Adam-like statistics
    private float m = 0;
    private float v = 0;
    private float beta1 = 0.9f, beta2 = 0.999f;
    private int t = 0;
    private float adaptiveMutationStrength = 0.1f;
    private final float fixedMutationStrength;
    private final boolean useAdaptiveMutation;
    
    // Constructor for adaptive mutation
    public GeneticParameter(String name, float min, float max, float initial, int index) {
        this.name = name;
        this.min = min;
        this.max = max;
        this.initial = initial;
        this.index = index;
        this.useAdaptiveMutation = true;
        this.fixedMutationStrength = 0.1f; // Not used
    }
    
    // Constructor for fixed mutation
    public GeneticParameter(String name, float min, float max, float initial, int index, float mutationStrength) {
        this.name = name;
        this.min = min;
        this.max = max;
        this.initial = initial;
        this.index = index;
        this.useAdaptiveMutation = false;
        this.fixedMutationStrength = mutationStrength;
    }
    
    public void updateStatistics(List<GeneticIndividual> sortedPopulation) {
        if (!useAdaptiveMutation) 
            return; // Skip if using fixed mutation
        
        // Get top 20% performers (already sorted)
        int topCount = Math.max(1, sortedPopulation.size() / 5);
        
        // Calculate mean of top performers
        float mean = 0;
        for (int i = 0; i < topCount; i++) {
            mean += sortedPopulation.get(i).genes[index];
        }
        mean /= topCount;
        
        // Calculate variance of top performers
        float variance = 0;
        for (int i = 0; i < topCount; i++) {
            float diff = sortedPopulation.get(i).genes[index] - mean;
            variance += diff * diff;
        }
        variance /= topCount;
        
        // Update Adam moments
        t++;
        m = beta1 * m + (1 - beta1) * mean;
        v = beta2 * v + (1 - beta2) * variance;
        
        // Bias correction
        float m_hat = m / (float)(1 - Math.pow(beta1, t));
        float v_hat = v / (float)(1 - Math.pow(beta2, t));
        
        // Calculate adaptive mutation strength
        adaptiveMutationStrength = (float)(Math.abs(m_hat) / (Math.sqrt(v_hat) + 1e-8f));
    }
    
    public void mutate(GeneticIndividual individual, double mutationRate) {
        if (Rng.nextDouble() < mutationRate) {
            float current = individual.genes[index];
            float direction = Rng.nextBoolean() ? 1f : -1f;
            float strength = useAdaptiveMutation ? adaptiveMutationStrength : fixedMutationStrength;
            float newValue = current + direction * strength;
            individual.genes[index] = Math.max(min, Math.min(max, newValue));
        }
    }
    
    public float initialize() {
        return Rng.nextFloat(min, max);
    }
}