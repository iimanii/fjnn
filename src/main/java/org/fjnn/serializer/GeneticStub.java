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
package org.fjnn.serializer;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import org.fjnn.activation.Activation;
import org.fjnn.genetic.Innovation;

/**
 *
 * @author ahmed
 */
public class GeneticStub implements Serializable {
    public final int inputSize;
    public final int outputSize;
    public final List<Innovation> innovations;
    public final List<GeneticConnectionStub> connections;
    public final Map<String, Object> properties;
    public final String outputActivation;
    public final String hiddenActivation;

    public GeneticStub(int inputSize, int outputSize, 
                       List<Innovation> innovations, 
                       List<GeneticConnectionStub> connections, 
                       Map<String, Object> properties, 
                       String outputActivation, String hiddenActivation) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.innovations = innovations;
        this.connections = connections;
        this.properties = properties;
        this.outputActivation = outputActivation;
        this.hiddenActivation = hiddenActivation;
    }
}
