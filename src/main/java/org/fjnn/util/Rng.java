/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fjnn.util;

import java.util.SplittableRandom;
import java.util.concurrent.ThreadLocalRandom;

/**
 *
 * @author ahmed
 */
public class Rng {
    private static final SplittableRandom rng = new SplittableRandom();
    
    public static int nextInt(int max) {
        return (int)nextDouble(0, max);        
    }
    public static int nextInt(int min, int max) {
        return (int)nextDouble(min, max);
    }
    
    public static long nextLong(long max) {
        return (long)nextDouble(0, max);
    }
    public static long nextLong(long min, long max) {
        return (long)nextDouble(min, max);
    }
    
    public static float nextFloat() {
        return ThreadLocalRandom.current().nextFloat();
    }    
    public static float nextFloat(double min, double max) {
        double range = max - min;
        return (float)(nextFloat() * range + min);
    }

    public static double nextDouble() {
        return ThreadLocalRandom.current().nextDouble();
    }
    public static double nextDouble(double min, double max) {
        double range = max - min;
        return nextDouble() * range + min;
    }
    
    public static boolean nextBoolean() {
        return ThreadLocalRandom.current().nextBoolean();
    }

    public static double nextDoubleGaussian() {
        return nextDoubleGaussian(0, 0.2);
    }

    public static double nextDoubleGaussian(double min, double max) {
        double mean = (min + max) / 2;
        double stdDev = (max - min) / 10;
        
        return gaussian(mean, stdDev);
    }
    
    private static double gaussian(double mean, double stdDev) {
        double u, v, s;
        
        do {
            u = nextDouble(-1, 1);
            v = nextDouble(-1, 1);
            s = u * u + v * v;
        } while (s >= 1 || s == 0);
        
        double mul = Math.sqrt(-2.0 * Math.log(s) / s);
        return mean + stdDev * u * mul;
    }
}
