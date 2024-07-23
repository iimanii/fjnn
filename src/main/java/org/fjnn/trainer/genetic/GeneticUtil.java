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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadLocalRandom;
import org.fjnn.base.Network;
import org.fjnn.network.NeuralNetwork;
import org.fjnn.util.Rng;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 */
public class GeneticUtil {
    
    public static void sortAsc(List<? extends Network> all, String property) {
        Collections.sort(all, (Network o1, Network o2) -> 
                Double.compare((double)o1.getProperty(property), (double)o2.getProperty(property)));
    }
    public static void sortDesc(List<? extends Network> all, String property) {
        Collections.sort(all, (Network o1, Network o2) -> 
                Double.compare((double)o2.getProperty(property), (double)o1.getProperty(property)));
    }

    public static <T extends Network> T pickRandom(List<T> pool, String property) {
        return pickRandom(pool, property, null);
    }
    public static <T extends Network> T pickRandom(List<T> pool, String property, Network avoid) {
        T n = null;
        
        double r = Rng.nextDouble();

        int index = 0;

        /* TODO: binary search */
        for(int i=0; i < pool.size(); i++) {
            if(r < (double) pool.get(i).getProperty(property)) {
                index = i;
                n = pool.get(i);
                break;
            }
        }
        
        if(n == avoid) {
            if(index > 0)
                n = pool.get(index - 1);
            else
                n = pool.get(index + 1);
        }
        
//        if(!n.hasProperty("pick"))
//            n.setProperty("pick", 0);
//        
//        n.setProperty("pick", (int)n.getProperty("pick") + 1);
        
//        System.out.println(String.format("%.4f", r) + " " + n.getProperty("id"));
        return n;
    }
    
    public static void incrementProperty(Network n, String property) {
        synchronized(n) {
            if(!n.hasProperty(property))
                n.setProperty(property, 0);
        
            n.setProperty(property, ((Number)n.getProperty(property)).intValue() + 1);
        }
    }

    public static <T extends Network> T tournamentPick(List<T> pool, int size, String property) {
        return tournamentPick(pool, size, property, null);
    }
    public static <T extends Network> T tournamentPick(List<T> pool, int size, String property, T avoid) {
        List<T> copy = new ArrayList<>(pool);
        
        if(avoid != null)
            copy.remove(avoid);

        Collections.shuffle(copy, ThreadLocalRandom.current());
        
        List<T> selected = new ArrayList<>(copy.subList(0, size));
        
        sortDesc(selected, property);
        
        return selected.get(0);
    }

    public static void commulativeDestribution(List<? extends Network> pool, String srcproperty, String resultproperty) {
        commulativeDestribution(pool, srcproperty, resultproperty, 1);
    }
    public static void commulativeDestribution(List<? extends Network> pool, String srcproperty, String resultproperty, double steepness) {
        sortAsc(pool, srcproperty);

        double min = (double) pool.get(0).getProperty(srcproperty);
        double sum = 0;
        
        for(Network n : pool) {
            double fitness = (double) n.getProperty(srcproperty) + 1 - min;
            n.setProperty(resultproperty, fitness);
            sum += fitness;
        }
        
        double commulative = 0;
        
        for(Network n : pool) {
            commulative += (double) n.getProperty(resultproperty);
            n.setProperty(resultproperty, Math.pow(commulative / sum, steepness));
        }        
    }
    
    public static void ArrayZScore(List<? extends Network> pool, String srcproperty, String resultproperty, boolean absolute) {
        int size = ((double[])pool.get(0).getProperty(srcproperty)).length;
        
        double[] mean = new double[size];
        double[] sd = new double[size];
        
        /* mean */
        for(Network n : pool) {
            double[] df = (double[])n.getProperty(srcproperty);
            
            for(int i=0; i < df.length; i++)
                mean[i] += df[i];
        }
        
        for(int i=0; i < mean.length; i++)
            mean[i] /= pool.size();

        /* standard deviation (using absolute values) */
        for(Network n : pool) {
            double[] df = (double[])n.getProperty(srcproperty);
            
            for(int i=0; i < df.length; i++) {
                double val = Math.abs(df[i] - mean[i]);
                if(!absolute)
                    val *= val;
                
                sd[i] += val;
            }
        }

        for(int i=0; i < mean.length; i++) {
            double val = sd[i] / pool.size();
            if(!absolute)
                Math.sqrt(val);
            
            sd[i] = val;
        }
        
        /* z-score */
        for(Network n : pool) {
            double[] df = (double[])n.getProperty(srcproperty);
            double[] z = new double[size];
            double sum = 0;
            
            for(int i=0; i < df.length; i++) {
                /* if all results are the same for a particular set .. sd will be equal to zero */
                if(sd[i] != 0)
                    z[i] = (df[i] - mean[i]) / sd[i];
                sum += z[i];
            }

            n.setProperty(resultproperty, sum);
        }
    }
    
    public static void squareProperty(List<? extends Network> pool, String srcproperty, String resultproperty)  {
        for(Network n : pool) {
            double property = (double) n.getProperty(srcproperty);
            n.setProperty(resultproperty, property * property);
        }
    }
}
