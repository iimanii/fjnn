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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.fjnn.util.Rng;

/**
 *
 * @author ahmed
 */
class GeneticNode implements Serializable {
    private static final long serialVersionUID = 441567370194619765l;
    
    final int id;
    final String name;
    final NodeType type;
    
    List<GeneticConnection> in;
    List<GeneticConnection> out;
    GeneticConnection[] _in;
    
    /* list of all connection out nodes */
    List<GeneticNode> to;

    /* list of non disabled links */
    private List<GeneticNode> enabled;
    
    private Map<Integer, GeneticConnection> fastAccessIn;
    private Map<Integer, GeneticConnection> fastAccessOut;
    
    enum NodeType {
        INPUT,
        OUTPUT,
        HIDDEN;
    }
    
//    GeneticNode(String id) {
//        this(id, false);
//    }

    GeneticNode(int id, String name, NodeType t) {
        this.id = id;
        this.type = t;
        this.name = name;

        in = new ArrayList<>();
        out = new ArrayList<>();
        _in = new GeneticConnection[0];
        
        to = new ArrayList<>();
        enabled = new ArrayList<>();
        
        fastAccessIn = new HashMap<>();
        fastAccessOut = new HashMap<>();
    }
        
    GeneticConnection connect(GeneticNode node) {
        return connect(node, 1.0f);
    }
    
    GeneticConnection connect(GeneticNode node, float w) {
        if(to(node) != null)
           throw new RuntimeException("Already Connected: " + id + " " + node.id);
        
        GeneticConnection c = new GeneticConnection(this, node, w);
        out.add(c);
        fastAccessOut.put(node.id, c);
        to.add(node);
        enabled.add(node);

        node.in.add(c);
        node._in = new GeneticConnection[node.in.size()];
        node.in.toArray(node._in);
        node.fastAccessIn.put(id, c);
        
        return c;
    }
    
    void to(GeneticNode node, float w) {
        fastAccessOut.get(node.id).weight = w;
    }
    
    void enable(GeneticNode node) {
        GeneticConnection c = fastAccessOut.get(node.id);

        if(!c.disabled)
            return;
        
        c.disabled = false;
        enabled.add(node);
    }
    
    void disable(GeneticNode node) {
        GeneticConnection c = fastAccessOut.get(node.id);
        
        if(c.disabled)
           return;

        c.disabled = true;
        enabled.remove(node);
    }
    
    GeneticConnection to(GeneticNode node) {
        return fastAccessOut.get(node.id);
    }
    
    GeneticConnection from(GeneticNode node) {
        return fastAccessIn.get(node.id);        
    }
    
    GeneticConnection getRandom() {
        if(enabled.isEmpty())
            return null;
        
        int n = enabled.get(Rng.nextInt(enabled.size())).id;
        return fastAccessOut.get(n);
    }
}
