/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fjnn.util;

/**
 *
 * @author ahmed
 */
public class statistics {
    
    static final double GAUSS_CONSTANT = Math.sqrt(2 * Math.PI);
    
    public static double fastGaussian(double v, double mean, double sd) {
        double e = -Math.pow((v - mean) / sd, 2.0) / 2.0;
        
        /* fast exp */
        final long tmp = (long) (1512775 * e + (1072693248 - 60801));
        double exp = Double.longBitsToDouble(tmp << 32);
        
        return exp / (sd * GAUSS_CONSTANT);
    }

    public static double mean(double[] values) {
        double mean = 0;
        
        for(double i : values)
            mean += i;
        
        mean /= values.length;
        
        return mean;
    }
    
    public static double sd(double[] values, double mean) {
        double sd = 0;
                
        for(double i : values) {
            sd += Math.abs(i - mean);
        }
        
        return sd / values.length;
    }
    
    public static bins gaussianBlur(double[] keys, double[] values, double step, int window, double sd) {
        bins result = new bins(step);
        int halfWindow = window / 2;
        
        for(int i=0; i < keys.length; i++) {
            double medianBin = Math.floor((keys[i] + step / 2) / step) * step;
            double mean = (keys[i] - medianBin) / step;
            double score = values[i];
            
            for(int j=-halfWindow; j <= halfWindow; j++) {
                double bin = medianBin + j * step;
                result.add(bin, score * statistics.fastGaussian(j, mean, sd));
            }
        }
        
        return result;
    }
}
