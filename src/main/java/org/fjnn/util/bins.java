/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fjnn.util;

import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

/**
 *
 * @author ahmed
 */
public class bins {
    final double binsize;
    final double halfbin;
    Map<Double, MutableDouble> map;
    
    public static class MutableDouble {
        double value;

        public MutableDouble(double value) {
            this.value = value;
        }
        
        void increment() {
            value += 1;
        }
        
        void add(double v) {
            this.value += v;
        }
        
        public double get() {
            return value;
        }
    }
    
    public bins(double ticksize) {
        this.binsize = ticksize;
        this.halfbin = ticksize / 2;
        this.map = new TreeMap<>();
    }
    
    public void add(double value) {
        add(value, 1);
    }
    
    public void add(double value, double weight) {
        double bin = Math.floor((value + halfbin) / binsize) * binsize;

        MutableDouble n = new MutableDouble(0);
        MutableDouble p = map.putIfAbsent(bin, n);
        n = p != null ? p : n;
        n.add(weight);
    }
    
    public Set<Map.Entry<Double, MutableDouble>> get() {
        return map.entrySet();
    }
    
    public void print(int maxWidth) {
        double max = 0;
        double total = 0;
        for(Map.Entry<Double, MutableDouble> e : map.entrySet()) {
            max = Math.max(max, Math.abs(e.getValue().get()));
            total += Math.abs(e.getValue().get());
        }
        
        for(Map.Entry<Double, MutableDouble> e : map.entrySet()) {
            double p = e.getValue().get() / total;
            System.out.printf("%+.3f\t%+.2f\t%+.2f\t%s\n",
                               e.getKey(),
                               e.getValue().get(),
                               p,
                               printSize(e.getValue().get() / max, maxWidth));
        }
    }
    
    public void printBins() {
        for(Map.Entry<Double, MutableDouble> e : map.entrySet()) {
            System.out.printf("%+.3f => %+.3f ", e.getKey(), e.getValue().get());
        }
        
        System.out.println();
    }

    private static String printSize(double d, int w) {
        int amount = (int) Math.round(Math.abs(d) * w);
        String c = d > 0 ? "#" : "*";
        
        StringBuilder b = new StringBuilder();
        for(int i=0; i < amount; i++)
            b.append(c);
        
        return b.toString();
    }
    
    public boolean isEmpty() {
        return map.isEmpty();
    }
    
    public int size() {
        return map.size();
    }
}
