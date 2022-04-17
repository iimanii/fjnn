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
package org.fjnn.genetic;

import java.io.Serializable;
import org.fjnn.genetic.GeneticConfig.mutation;

/**
 *
 * @author ahmed
 */
public class Innovation implements Serializable {    
    private static final long serialVersionUID = 7659747973936368991l;
    
    public final mutation m;
    public final String from;
    public final String to;

    public Innovation(mutation m, String from, String to) {
        this.m = m;
        this.from = from;
        this.to = to;
    }
    
    public String id() {
        return getId(m, from, to);
    }
    
    public static String getId(mutation m, GeneticNode from, GeneticNode to) {
        return from.id + ":" + to.id + ":" + m.toString();
    }
    
    public static String getId(mutation m, String from, String to) {
        return from + ":" + to + ":" + m.toString();
    }
}
