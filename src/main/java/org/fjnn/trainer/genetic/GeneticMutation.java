/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fjnn.trainer.genetic;

import java.util.ArrayList;
import java.util.List;
import org.fjnn.base.Network;
import org.fjnn.network.NeuralNetwork;
import org.fjnn.util.SMA;
import org.fjnn.util.statistics;

/**
 *
 * @author ahmed
 */
public class GeneticMutation {
    final SMA mutation;
    final double maxValue;
    final double minValue;
    final double initialValue;

    public GeneticMutation(int window, double initialValue, double minValue, double maxValue) {
        this.mutation = new SMA(window);
        this.maxValue = maxValue;
        this.minValue = minValue;
        this.initialValue = initialValue;
        
        this.mutation.set(initialValue);
    }
    
    public double get() {
        return this.mutation.get();
    }
    
    public void reset() {
        this.mutation.set(initialValue);
    }
    
    public void update(List<? extends Network> networks, String property) {
        ArrayList<Network> copy = new ArrayList<>(networks);
        
        GeneticUtil.sortDesc(copy, property);
        
        double sumWeights = 0;
        double sumMutations = 0;
        double countWeights = 0;
        
        double rate = networks.size() / 4;
        
        for(int i=0; i < networks.size(); i++) {
            Network n = networks.get(i);
            if(!n.hasProperty(GeneticTrainer.MUTATION_PROP))
                continue;
            
            double m = ((Number) n.getProperty(GeneticTrainer.MUTATION_PROP)).floatValue();
            double w = statistics.fastGaussian(i, 0, rate);
            
            sumMutations += m * w;
            sumWeights += w;
            countWeights++;
        }
        
        if(countWeights >= networks.size() / 2) {
            sumMutations /= sumWeights;
//            System.out.printf("mutation: %.6f -> %.6f\n", mutation.get() * 100, sumMutations * 100);
            mutation.add(sumMutations);
        }
        
        if(mutation.get() > maxValue)
           mutation.set(maxValue);
        else if(mutation.get() < minValue)
            mutation.set(minValue);
    }
    
    public void replenish(double rate) {
        double newMedian = mutation.get();
        newMedian *= rate;
        mutation.add(newMedian);

        if(mutation.get() > maxValue)
           mutation.set(maxValue);
    }
}
