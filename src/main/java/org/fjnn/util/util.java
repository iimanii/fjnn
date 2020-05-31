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

import java.io.FileInputStream;
import java.io.IOException;

/**
 *
 * @author Ahmed Tarek
 */
public class util {    
        
    public static byte[] readFile(String filename) throws IOException {
        try (FileInputStream fs = new FileInputStream(filename)) {
            byte[] result = new byte[fs.available()];
            fs.read(result);
            fs.close();
            return result;
        }
    }
    
    /***
     * Read local resource from file
     * @param path
     * @return 
     * @throws java.io.IOException
     */
    public static byte[] readResource(String path) throws IOException {
        return readFile(util.class.getResource(path).getFile());
    }
    
    public static float[] copyArray(float[] array) {
        float[] copy = new float[array.length];

        System.arraycopy(array, 0, copy, 0, array.length);
        
        return copy;
    }
    
    public static float[][] copyArray(float[][] array) {
        float[][] copy = new float[array.length][];

        for(int i=0; i < array.length; i++) {
            copy[i] = new float[array[i].length];
            System.arraycopy(array[i], 0, copy[i], 0, array[i].length);
        }
        
        return copy;
    }
    
    public static float[] copyArray(float[] array, int length) {
        float[] copy = new float[length];

        System.arraycopy(array, 0, copy, 0, array.length);
        
        return copy;
    }
    
    public static void printArray(float[] array) {
        printArray(array, 0, array.length);
    }
    
    public static void printArray(float[] array, int start, int length) {
        for(int i=start; i < start+length; i++)
            System.out.print(array[i] + " ");
        System.out.println();
    }
}
