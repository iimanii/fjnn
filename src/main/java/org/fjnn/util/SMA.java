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
public class SMA {
    ArrayList<Double> list;
    double sum;
    int window;

    public SMA(int window) {
        this.window = window;
        this.list = new ArrayList<>();
    }

    public void set(double value) {
        for(int i=0; i < window; i++)
            add(value);
    }

    public void add(double x) {
        list.add(x);
        if(Double.isFinite(x))
            sum += x;

        if(list.size() > window) {
            double y = list.remove(0);
            
            if(Double.isFinite(y))
                sum -= y;
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
        return sum / list.size();
    }

    public void reset() {
        this.list = new ArrayList<>();
        this.sum = 0;
    }    
}
