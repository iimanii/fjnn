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
import java.util.Objects;

/**
 *
 * @author ahmed
 */
public class GeneticConnection implements Serializable {
    public float weight;
    public boolean disabled;

    public final String id;
    public final GeneticNode from;
    public final GeneticNode to;
    
    private final int hashcode;

    public GeneticConnection(GeneticNode from, GeneticNode to, float weight) {
        this.id = createId(from, to);
        this.weight = weight;
        this.disabled = false;
        this.from = from;
        this.to = to;
        this.hashcode = id.hashCode();
    }
    
    public String from() {
        return from.id;
    }
    
    public String to() {
        return to.id;
    }
    
    static final String createId(String from, String to) {
        return from + "-" + to;
    }
    
    static final String createId(GeneticNode from, GeneticNode to) {
        return createId(from.id, to.id);
    }
    
    @Override
    public int hashCode() {
        return hashcode;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;

        if (obj == null)
            return false;

        if (getClass() != obj.getClass())
            return false;

        final GeneticConnection other = (GeneticConnection) obj;
        
        return Objects.equals(this.id, other.id);
    }
}
