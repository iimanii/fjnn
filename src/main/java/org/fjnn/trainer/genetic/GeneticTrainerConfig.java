/*
 * The MIT License
 *
 * Copyright 2023 ahmed.
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
package org.fjnn.trainer.genetic;

/**
 *
 * @author ahmed
 */
public class GeneticTrainerConfig {
    
    public static enum SelectionCriteria {
        fitness,
        squareFitness,
        zscore,
        absoluteZscore
    }
    
    public static enum ComputeMode {
        plain,
        float_buffer,
        gpu,
        gpu_all
    }
    
    final ComputeMode computeMode;
    final int deviceId;
    
    final SelectionCriteria criteria;
    
    ComputeMode crossover;
    
    int tournametSelectionSize;
    boolean useTournamentSelection;
    
    boolean useStaticMutation;
    double staticMutationValue;    
    
    int startingID;
    int startingGeneration;
    int startingEpoch;

    public GeneticTrainerConfig(ComputeMode mode, SelectionCriteria criteria) {
        this(mode, criteria, -1);
    }
    
    public GeneticTrainerConfig(ComputeMode mode, SelectionCriteria criteria, int deviceId) {
        this.computeMode = mode;
        this.crossover = mode;
        this.criteria = criteria;
        this.deviceId = deviceId;
    }
    
    public GeneticTrainerConfig useTournamentSelection(int tournamentSize) {        
        this.useTournamentSelection = true;
        this.tournametSelectionSize = tournamentSize;
        
        return this;
    }
    
    public GeneticTrainerConfig useStaticMutation(double value) {
        this.useStaticMutation = true;
        this.staticMutationValue = value;
        
        return this;
    }    
    
    public GeneticTrainerConfig setCrossoverMode(ComputeMode mode) {
        this.crossover = mode;
        
        return this;
    }
    
    public void setStartingPoint(int startingID, int startingGeneration, int startingEpoch) {
        this.startingID = startingID;
        this.startingGeneration = startingGeneration;
        this.startingEpoch = startingEpoch;
    }
    
    void checkTournamentSize(int poolsize) {
        if(useTournamentSelection)
            if(tournametSelectionSize > poolsize || tournametSelectionSize < 2)
                throw new RuntimeException("Invalid tournament size: " + tournametSelectionSize);
    }
}
