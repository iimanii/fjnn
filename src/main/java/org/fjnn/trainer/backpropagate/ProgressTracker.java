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

import java.util.ArrayList;
import java.util.List;
import org.fjnn.util.sma;

/**
 *
 * @author ahmed
 */
public class ProgressTracker {
    private final List<Double> lossHistory = new ArrayList<>();
    
    private final sma lossMA;
    private final sma epochTimeMA;
    private final sma forwardTimeMA;
    private final sma backwardTimeMA;
    private final sma updateTimeMA;
    private final sma lossTimeMA;
    
    private final long startTimeMs = System.currentTimeMillis();
    private double bestLoss = Double.POSITIVE_INFINITY;
    private int bestEpoch = -1;
    private int stagnationCount = 0;
    
    // Last recorded values
    private float lastEpochTime = 0;
    private float lastForwardTime = 0;
    private float lastBackwardTime = 0;
    private float lastUpdateTime = 0;
    private float lastLossTime = 0;
    
    public final double stagnationThreshold;
    
    public ProgressTracker() {
        this(1e-6);
    }
    
    public ProgressTracker(double stagnationThreshold) {
        this(stagnationThreshold, 10);
    }
    
    public ProgressTracker(double stagnationThreshold, int smoothingWindow) {
        this.stagnationThreshold = stagnationThreshold;
        this.lossMA = new sma(smoothingWindow);
        this.epochTimeMA = new sma(smoothingWindow);
        this.forwardTimeMA = new sma(smoothingWindow);
        this.backwardTimeMA = new sma(smoothingWindow);
        this.updateTimeMA = new sma(smoothingWindow);
        this.lossTimeMA = new sma(smoothingWindow);
    }
    
    public void recordMetrics(double loss, float epochTimeMs, float forwardTimeMs, float backwardTimeMs, float updateTimeMs, float lossTimeMs) {
        if (!Double.isFinite(loss))
            throw new IllegalArgumentException("Loss must be finite");
        
        lossHistory.add(loss);
        
        // Update SMAs
        lossMA.add(loss);
        epochTimeMA.add(epochTimeMs);
        forwardTimeMA.add(forwardTimeMs);
        backwardTimeMA.add(backwardTimeMs);
        updateTimeMA.add(updateTimeMs);
        lossTimeMA.add(lossTimeMs);
        
        // Update last values
        lastEpochTime = epochTimeMs;
        lastForwardTime = forwardTimeMs;
        lastBackwardTime = backwardTimeMs;
        lastUpdateTime = updateTimeMs;
        lastLossTime = lossTimeMs;
        
        if (loss < bestLoss - stagnationThreshold) {
            bestLoss = loss;
            bestEpoch = lossHistory.size() - 1;
            stagnationCount = 0;
        } else {
            stagnationCount++;
        }
    }
    
    public double getCurrentLoss() {
        return lossHistory.isEmpty() ? Double.POSITIVE_INFINITY : lossHistory.get(lossHistory.size() - 1);
    }
    
    public double getBestLoss() {
        return bestLoss;
    }
    
    public int getBestEpoch() {
        return bestEpoch;
    }
    
    public int getCurrentEpoch() {
        return lossHistory.size() - 1;
    }
    
    public long getElapsedTimeMs() {
        return System.currentTimeMillis() - startTimeMs;
    }
    
    public double getAverageLoss() {
        return lossMA.getNet();
    }
    
    public int getStagnationCount() {
        return stagnationCount;
    }
    
    public double getImprovementRate() {
        if (lossHistory.size() < 2) return 0.0;
        
        int epochs = Math.min(10, lossHistory.size());
        double recent = getCurrentLoss();
        double past = lossHistory.get(lossHistory.size() - epochs);
        
        return (recent - past) / epochs;
    }
    
    public float getLastForwardTime() {
        return lastForwardTime;
    }
    
    public float getLastBackwardTime() {
        return lastBackwardTime;
    }
    
    public float getLastUpdateTime() {
        return lastUpdateTime;
    }
    
    public float getLastLossTime() {
        return lastLossTime;
    }
    
    public float getLastEpochTime() {
        return lastEpochTime;
    }
    
    public double getAverageForwardTime() {
        return forwardTimeMA.getNet();
    }
    
    public double getAverageBackwardTime() {
        return backwardTimeMA.getNet();
    }
    
    public double getAverageUpdateTime() {
        return updateTimeMA.getNet();
    }
    
    public double getAverageLossTime() {
        return lossTimeMA.getNet();
    }
    
    public double getAverageEpochTime() {
        return epochTimeMA.getNet();
    }
    
    public double getEstimatedTimeRemaining(int totalEpochs) {
        if (lossHistory.isEmpty() || getCurrentEpoch() >= totalEpochs) return 0;
        
        int remaining = totalEpochs - getCurrentEpoch() - 1;
        double avgTimeMs = epochTimeMA.isValid() ? epochTimeMA.get() : getAverageEpochTime();
        return remaining * avgTimeMs / 1000.0; // seconds
    }
    
    public List<Double> getLossHistory() {
        return lossHistory;
    }
    
    public String summary() {
        StringBuilder sb = new StringBuilder();

        sb.append(String.format("Epoch: %d | ", getCurrentEpoch()));
        sb.append(String.format("Loss: %.6f (Best: %.6f @ %d) | ", 
            getCurrentLoss(), getBestLoss(), getBestEpoch()));

        // Add timing breakdown
        sb.append(String.format("F/B/U: %.0f/%.0f/%.0fms | ", 
            getLastForwardTime(), getLastBackwardTime(), getLastUpdateTime()));

        if (epochTimeMA.isValid())
            sb.append(String.format("Avg: %.0fms | ", getAverageEpochTime()));

        sb.append(String.format("Elapsed: %ds", getElapsedTimeMs() / 1000));

        sb.append(String.format(" | STAGNANTION %d", stagnationCount));

        return sb.toString();
    }
}