/*
 * The MIT License
 *
 * Copyright 2020 ahmed.
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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * @author ahmed
 */
public class GeneticConnectionMap {
    
    private static final ArrayList EMPTY = new ArrayList<>();
    private static final GeneticConnection[] EMPTY_ARRAY = {};
    
    private final List<GeneticConnection> connections;
    private final List<GeneticConnection> enabled;
    private final List<GeneticConnection> disabled;
    private final Map<String, GeneticConnection> fastAccessConnections;
    
    /* node index is unique .. Use node index for access */
    private List<GeneticConnection>[] from;
    private List<GeneticConnection>[] to;
    /* this is needed for faster computation */
    private GeneticConnection[][] to0;
    
    private List<GeneticNode>[] next;
    private List<GeneticNode>[] prev;

    public GeneticConnectionMap(int initialCapacity) {
        connections = new ArrayList<>();
        enabled = new ArrayList<>();
        disabled = new ArrayList<>();
        fastAccessConnections = new HashMap<>();
        
        from = new ArrayList[initialCapacity];
        to = new ArrayList[initialCapacity];
        to0 = new GeneticConnection[initialCapacity][];
        
        next = new ArrayList[initialCapacity];
        prev = new ArrayList[initialCapacity];
    }
    
    boolean add(GeneticConnection c) {
        if(fastAccessConnections.containsKey(c.id))
            return false;
        
        connections.add(c);
        fastAccessConnections.put(c.id, c);
        
        if(c.disabled)
            disabled.add(c);
        else
            enabled.add(c);
        
        addFrom(c);
        addTo(c);
        addNext(c);
        addPrev(c);
        
        return true;
    }
    
    boolean update(GeneticConnection c) {
        if(!fastAccessConnections.containsKey(c.id))
            return false;
        
        if(c.disabled) {
            if(disabled.indexOf(c) > -1)
                throw new RuntimeException();
            enabled.remove(c);
            disabled.add(c);
        } else {
            if(enabled.indexOf(c) > -1)
                throw new RuntimeException();
            enabled.add(c);
            disabled.remove(c);            
        }
        
        if(enabled.size() + disabled.size() > connections.size())
            throw new RuntimeException();
        
        return true;
    }
    
    public List<GeneticNode> next(GeneticNode n) {
        if(n.index < next.length)
            return avoidNull(next[n.index]);
        
        return EMPTY;
    }
    
    public List<GeneticNode> prev(GeneticNode n) {
        if(n.index < prev.length)
            return avoidNull(prev[n.index]);
        
        return EMPTY;
    }    
    
    public List<GeneticConnection> from(GeneticNode n) {
        if(n.index < from.length)
            return avoidNull(from[n.index]);
        
        return EMPTY;
    }    
    
    public List<GeneticConnection> to(GeneticNode n) {
        if(n.index < to.length)
            return avoidNull(to[n.index]);
        
        return EMPTY;
    }
    
    public GeneticConnection[] to0(GeneticNode n) {
        if(n.index < to0.length) {
            GeneticConnection[] r = to0[n.index];
            return r == null ? EMPTY_ARRAY : r;
        }
        
        return EMPTY_ARRAY;
    }
    
    public List<GeneticConnection> enabled() {
        return enabled;
    }
    
    public List<GeneticConnection> disabled() {
        return disabled;
    }
    
    public List<GeneticConnection> all() {
        return connections;
    }
    
    public GeneticConnection get(String id) {
        return fastAccessConnections.get(id);
    }
    
    public GeneticConnection get(GeneticNode from, GeneticNode to) {
        return get(GeneticConnection.createId(from, to));
    }
    
    public GeneticConnection get(String from, String to) {
        return get(GeneticConnection.createId(from, to));
    }
    
    public int count() {
        return connections.size();
    }
    
    public List avoidNull(List l) {
        return l == null ? EMPTY : l;
    }
    
    private void addFrom(GeneticConnection c) {
        int index = c.from.index;
        
        if(from.length <= index)
            from = Arrays.copyOf(from, (int) Math.ceil(index * 1.5));
        
        List<GeneticConnection> _from = from[index];
        if(_from == null) {
            _from = new ArrayList<>();
            from[index] = _from;
        }
        _from.add(c);
    }

    private void addTo(GeneticConnection c) {
        int index = c.to.index;
        
        if(to.length <= index)
            to = Arrays.copyOf(to, (int) Math.ceil(index * 1.5));

        List<GeneticConnection> _to = to[index];
        if(_to == null) {
            _to = new ArrayList<>();
            to[index] = _to;
        }
        _to.add(c);
        
        if(to0.length <= index)
            to0 = Arrays.copyOf(to0, (int) Math.ceil(index * 1.5));
        
        to0[index] = _to.toArray(new GeneticConnection[_to.size()]);
    }

    private void addNext(GeneticConnection c) {
        int index = c.from.index;
                
        if(next.length <= index)
            next = Arrays.copyOf(next, (int) Math.ceil(index * 1.5));
        
        List<GeneticNode> _next = next[index];
        if(_next == null) {
            _next = new ArrayList<>();
            next[index] = _next;
        }
        _next.add(c.to);
    }

    private void addPrev(GeneticConnection c) {
        int index = c.to.index;

        if(prev.length <= index)
            prev = Arrays.copyOf(prev, (int) Math.ceil(index * 1.5));
        
        List<GeneticNode> _prev = prev[index];
        if(_prev == null) {
            _prev = new ArrayList<>();
            prev[index] = _prev;
        }
        _prev.add(c.from);
    }
}
