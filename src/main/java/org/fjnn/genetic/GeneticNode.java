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
public class GeneticNode implements Serializable {
    private static final long serialVersionUID = 441567370194619765l;

    public int index;
    public final String id;
    public final NodeType type;
    
    private final int hashcode;

    public enum NodeType {
        INPUT("I"),
        OUTPUT("O"),
        HIDDEN("H");
        
        String prefix;

        private NodeType(String prefix) {
            this.prefix = prefix;
        }
    }

    GeneticNode(String id, NodeType t) {
        this.id = id;
        this.type = t;
        this.hashcode = id.hashCode();
    }
    
    static String createId(int i, NodeType t) {
        return t.prefix + i;
    }
    
    @Override
    public String toString() {
        return id;
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

        final GeneticNode other = (GeneticNode) obj;
        
        return Objects.equals(this.id, other.id);
    }
}
