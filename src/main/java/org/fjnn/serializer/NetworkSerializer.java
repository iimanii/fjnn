/*
 * The MIT License
 *
 * Copyright 2018 ahmed.
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
package org.fjnn.serializer;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import org.fjnn.base.LayeredNetwork;
import org.fjnn.genetic.GeneticNetwork;
import org.fjnn.network.NeuralNetwork;

/**
 *
 * @author ahmed
 */
public class NetworkSerializer {
    
    public static void store(String filename, LayeredNetwork network) throws IOException {
        File file = new File(filename);
        
        FileOutputStream fos = new FileOutputStream(file);
        GZIPOutputStream zip = new GZIPOutputStream(fos);
        
        try(ObjectOutputStream oos = new ObjectOutputStream(zip)) {
            oos.writeObject(network.getStub());
        }
    }
    
    public static NeuralNetwork load(String filename) throws IOException, ClassNotFoundException {
        File file = new File(filename);

        FileInputStream fis = new FileInputStream(file);
        GZIPInputStream zip = new GZIPInputStream(fis);
        
        try(ObjectInputStream ois = new ObjectInputStream(zip)) {
            NetworkStub stub = (NetworkStub) ois.readObject();
            return new NeuralNetwork(stub);
        }
    }
    
    public static void storeGenetic(String filename, GeneticNetwork network) throws IOException {
        File file = new File(filename);
        
        FileOutputStream fos = new FileOutputStream(file);
        GZIPOutputStream zip = new GZIPOutputStream(fos);
        
        try(ObjectOutputStream oos = new ObjectOutputStream(zip)) {
            oos.writeObject(network.getStub());
        }
    }
    
    public static GeneticNetwork loadGenetic(String filename) throws IOException, ClassNotFoundException {
        File file = new File(filename);

        FileInputStream fis = new FileInputStream(file);
        GZIPInputStream zip = new GZIPInputStream(fis);
        
        try(ObjectInputStream ois = new ObjectInputStream(zip)) {
            GeneticStub stub = (GeneticStub) ois.readObject();
            return new GeneticNetwork(stub);
        }
    }
}
