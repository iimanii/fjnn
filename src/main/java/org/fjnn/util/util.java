/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
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
