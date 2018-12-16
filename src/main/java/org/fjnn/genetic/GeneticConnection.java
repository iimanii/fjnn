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
package org.fjnn.genetic;

import java.io.Serializable;

/**
 *
 * @author ahmed
 */
public class GeneticConnection implements Serializable {
    private static final long serialVersionUID = -7782604400128420487l;
    
    float weight;
    boolean disabled;

    final String id;
    final GeneticNode from;
    final GeneticNode to;

    public GeneticConnection(GeneticNode from, GeneticNode to, float weight) {
        this.id = from.name + "-" + to.name;
        this.weight = weight;
        this.disabled = false;
        this.from = from;
        this.to = to;
    }
    
    public void enable() {
        from.enable(to);
    }
    
    public void disable() {
        from.disable(to);
    }
    
    public String getId() {
        return id;
    }
    
    public float getWeight() {
        return weight;
    }
    
    public boolean isDisabled() {
        return disabled;
    }
}
