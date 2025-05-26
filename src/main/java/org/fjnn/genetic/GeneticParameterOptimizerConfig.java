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

/**
 *
 * @author ahmed
 */
public class GeneticParameterOptimizerConfig {
    public int populationSize = 100;
    public int tournamentSize = 3;
    public int eliteCount = 10;
    public double mutationRate = 0.1;
    public double mutationStrength = 0.1;
    public double crossoverRate = 0.8;
    public SelectionType selectionType = SelectionType.TOURNAMENT;
    public CrossoverType crossoverType = CrossoverType.UNIFORM;
    public MutationType mutationType = MutationType.GAUSSIAN;
    public boolean adaptiveMutation = true;
    public int stagnationThreshold = 10;
    public double mutationDecay = 0.95;
    public boolean useGPU = false;

    public enum SelectionType {
        TOURNAMENT,
        ROULETTE,
        RANK
    }

    public enum CrossoverType {
        UNIFORM,
        ARITHMETIC,
        SINGLE_POINT,
        BLEND_ALPHA
    }

    public enum MutationType {
        GAUSSIAN,
        UNIFORM,
        POLYNOMIAL
    }
}
