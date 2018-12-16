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
package org.fjnn.parallel;

import java.util.ArrayList;
import java.util.List;
import org.fjnn.activation.Activation;

/**
 *
 * @author ahmed
 */
public class MultiNetworkBuilder {
    List<ParallelLayerBuilder> layerBuilders;
    boolean threadSafe;
    int size;
    
    public MultiNetworkBuilder(int size, boolean threadSafe) {
        this.layerBuilders = new ArrayList<>();
        this.threadSafe = threadSafe;
        this.size = size;
    }

    public ParallelLayer[] buildLayers() {
        ParallelLayer[] layers = new ParallelLayer[layerBuilders.size()];
        
        int i_output = layers.length - 1;
        
        for(int i=0; i < i_output; i++) {
            ParallelLayerBuilder l = layerBuilders.get(i);
            ParallelLayerBuilder n = layerBuilders.get(i+1);
            layers[i] = l.build(size, n.size(), threadSafe);
        }
        
        layers[i_output] = layerBuilders.get(i_output).buildOutput(size, threadSafe);
        
        return layers;
    }

    public void addLayer(int neurons, Activation activation, boolean hasBias, boolean[] condition) {
        layerBuilders.add(new ParallelLayerBuilder(neurons, activation, hasBias, condition));
    }
    
//    public void addLayer(LayerStub stub) {
//        layerBuilders.add(new ParallelLayerBuilder(stub));
//    }
}
