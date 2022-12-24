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
package org.fjnn.network;

import java.util.Arrays;
import jcuda.driver.CUdeviceptr;
import org.fjnn.util.Rng;
import org.fjnn.util.util;
import org.json.JSONObject;

/**
 *
 * @author ahmed
 */
public class Connection {
    final public int neurons;
    final public int links;
    
    /**
     * Store everything in 1D array
     * [N(0-0) N(0-1) N(0-2) ... N(1-0) N(1-1) ....
     */
    float[] weights;

    /* bias in a separate array */
    float[] biases;
    
    /* weights stored on GPU */
    CUdeviceptr weightsGPU;

    /* bias stored on GPU */
    CUdeviceptr biasesGPU;
    
    boolean gpuReady;
    
    boolean disableBias;

    Connection(int neurons, int links) {
        this.neurons = neurons;
        this.links = links;
        this.weights = new float[neurons * links];
        this.biases = new float[links];
    }
    
    private Connection(int neurons, int links, float[] weights, float[] biases) {
        this.neurons = neurons;
        this.links = links;
        this.weights = weights;
        this.biases = biases;
        
        if(this.biases.length != links || this.weights.length != links * neurons)
            throw new RuntimeException("Inconsistent connection");
    }
    
    float[] feedForward(float[] input) {
        float[] output = new float[links];

        for(int i=0; i < neurons; i++) {
            int k = i * links;
            
            /* neurons */
            for(int j=0; j < links; j++)
                output[j] += input[i] * weights[k + j];
        }
        
        if(!disableBias)
            for(int i=0; i < links; i++)
                output[i] += biases[i];

//        for(int i=0; i < output.length; i++) {
//            System.out.print(String.format("%.3f ", output[i]));
//        }
//        
//        System.out.println();
        
        return output;        
    }
    
    public float getWeight(int from, int to) {
        if(from < 0 || from >= neurons)
            throw new ArrayIndexOutOfBoundsException();
        
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            
        
        return weights[from * links + to];
    }
    
    public float[] getWeights() {
        return weights;
    }
    
    public float[][] getWeights2D() {
        float[][] result = new float[neurons][links];
        
        for(int i=0; i < neurons; i++)
            for(int j=0; j < links; j++)
                result[i][j] = weights[i * neurons + j];
        
        return result;
    }

    public void setWeight(int from, int to, float value) {
        if(from < 0 || from >= neurons)
            throw new ArrayIndexOutOfBoundsException();
        
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();
        
        weights[from * links + to] = value;

        gpuReady = false;
    }
    
    public void setWeights(float[] values) {
        if(values.length != weights.length)
            throw new RuntimeException();
        
        weights = Arrays.copyOf(values, values.length);
        
        gpuReady = false;
    }
    
    public void setWeights2D(float[][] values) {
        for(int i=0; i < neurons; i++)
            for(int j=0; j < links; i++)
                weights[i * links + j] = values[i][j];
        
        gpuReady = false;
    }

    public float getBias(int to) {
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();  
        
        return biases[to];
    }
    
    public float[] getBias() {
        return biases;
    }

    public void setBias(int to, float value) {
        if(to < 0 || to >= links)
            throw new ArrayIndexOutOfBoundsException();            

        biases[to] = value;
        
        gpuReady = false;
    }
    
    public void setBias(float[] values) {
        if(values.length != links)
            throw new RuntimeException();

        System.arraycopy(values, 0, biases, 0, links);

        gpuReady = false;
    } 

    void randomize(float min, float max) {
        int len = neurons * links;
        
        for(int i=0; i < len; i++)
            weights[i] = Rng.nextFloat(min, max);
        
        for(int i=0; i < links; i++)
            biases[i] = Rng.nextFloat(min, max);
    }

    void free() {
        weights = null;
        biases = null;
    }

    void crossOverMutate(Connection a, Connection b, double min, double max, double mutation) {
        float[] wa = a.weights;
        float[] wb = b.weights;

        for(int j=0; j < wa.length; j++) {
            float w = Rng.nextBoolean() ? wa[j] : wb[j];

            if(Rng.nextDouble() < mutation)
                w = w + (float) Rng.nextDouble(min, max);

            weights[j] = w;
        }
        
        float[] ba = a.biases;
        float[] bb = b.biases;

        for(int j=0; j < ba.length; j++) {
            float w = Rng.nextBoolean() ? ba[j] : bb[j];

            if(Rng.nextDouble() < mutation)
                w = w + (float) Rng.nextDouble(min, max);

            biases[j] = w;
        }
    }

    JSONObject serialize() {
        JSONObject result = new JSONObject();

        result.put("neurons", neurons);
        result.put("links", links);
        
        result.put("weights", util.base64encode(util.compress(util.toByteArray(weights))));
        result.put("biases", util.base64encode(util.compress(util.toByteArray(biases))));
        
        return result;
    }
    
    static Connection deserialize(JSONObject obj) {
        int neurons = obj.getInt("neurons");
        int links = obj.getInt("links");
        
        float[] weights = util.toFloatArray(util.decompress(util.base64decode(obj.getString("weights"))));
        float[] biases = util.toFloatArray(util.decompress(util.base64decode(obj.getString("biases"))));
        
        return new Connection(neurons, links, weights, biases);
    }

    Connection copy(boolean withoutWeights) {
        if(withoutWeights)
            return new Connection(neurons, links);
        
        float[] wc = Arrays.copyOf(weights, weights.length);
        float[] bc = Arrays.copyOf(biases, biases.length);
        
        return new Connection(neurons, links, wc, bc);
    }
    
    float compare(Connection c) {
        float score = 0;
        
        float[] wa = c.getWeights();

        for(int j=0; j < wa.length; j++)            
            score += Math.abs(wa[j] - weights[j]);

        float[] ba = c.getBias();

        for(int j=0; j < ba.length; j++)
            score += Math.abs(ba[j] - biases[j]);            
        
        return score;
    }

    public boolean hasBias() {
        return !disableBias;
    }
}
