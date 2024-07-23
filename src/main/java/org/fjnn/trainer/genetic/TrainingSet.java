/*
 * The MIT License
 *
 * Copyright 2022 ahmed.
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

import java.nio.FloatBuffer;
import jcuda.driver.CUdeviceptr;
import org.fjnn.base.NetworkInput;

/**
 *
 * @author ahmed
 */
public abstract class TrainingSet {

    public final static class TrainingResult {
        final double fitness; 
        final double[] detailed;

        public TrainingResult(double fitness, double[] detailed) {
            this.fitness = fitness;
            this.detailed = detailed;
        }
    }
    
    public abstract NetworkInput getInput(int batch);
    public abstract NetworkInput getAllInput();
    
//    public abstract float[] getInput(int batch);
//    public abstract float[] getAllInput();
//    
//    public abstract FloatBuffer getBufferInput(int batch);
//    public abstract FloatBuffer getAllBufferInput();
//    
    public abstract NetworkInput getGPUInput(int deviceId, int batch);
    public abstract NetworkInput getAllGPUInput(int deviceId);
    
    public abstract int getCount(int batch);
    public abstract int getAllCount();
    
    public abstract int getBatchCount();
    
    /* return fitness value for every training entry in the batch */
    public abstract TrainingResult calculateFitness(float[] result, int batch);
    
    /* return fitness value for every training entry */
    public abstract TrainingResult calculateAllFitness(float[] result);
}
