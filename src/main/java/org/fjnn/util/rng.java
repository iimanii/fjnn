/*
 * The MIT License
 *
 * Copyright 2018 Ahmed Tarek.
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
package org.fjnn.util;

import java.util.SplittableRandom;
import java.util.concurrent.ThreadLocalRandom;

/**
 *
 * @author ahmed
 */
public class rng {
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
        return nextDoubleGaussian(-1, 1);
    }

    public static double nextDoubleGaussian(double min, double max) {
        double mean = (min + max) / 2;
        double stdDev = (max - min) / 10;
        
        return gaussian(mean, stdDev);
    }
    
    public static double gaussian(double mean, double stdDev) {
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
