/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fjnn.util;

import java.util.ArrayList;

/**
 *
 * @author ahmed
 */
public class sma {
    final ArrayList<Double> list;
    double sum;
    int window;

    public sma(int window) {
        this.window = window;
        this.list = new ArrayList<>();
    }

    public void set(double value) {
        for(int i=0; i < window; i++)
            add(value);
    }

    public void add(double x) {
        synchronized(list) {
            list.add(x);
            if(Double.isFinite(x))
                sum += x;

            if(list.size() > window) {
                double y = list.remove(0);

                if(Double.isFinite(y))
                    sum -= y;
            }
        }
    }
    
    public double peek(double x) {
        if(list.isEmpty())
            return -1;
        
        return (sum + x - list.get(0)) / list.size();
    }

    public double get() {
        if(list.size() < window)
            return -1;

        return sum / list.size();
    }

    public double getNet() {
        if(list.isEmpty())
            return 0;
        
        return sum / list.size();
    }

    public boolean isEmpty() {
        return list.isEmpty();
    }
    
    public void reset() {
        this.list.clear();
        this.sum = 0;
    }    

    public boolean isValid() {
        return list.size() >= window;
    }
}
