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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.atomic.AtomicInteger;
import org.fjnn.activation.Activation;
import org.fjnn.base.Network;
import org.fjnn.genetic.GeneticConfig.mutation;
import org.fjnn.genetic.GeneticNode.NodeType;
import org.fjnn.network.NeuralNetwork;
import org.fjnn.serializer.GeneticStub;
import org.fjnn.util.Rng;
import org.fjnn.util.intrinsic;

/**
 *
 * @author ahmed
 */
public class GeneticNetwork extends Network {
    public List<GeneticNode> inputs;
    public List<GeneticNode> hidden;
    public List<GeneticNode> outputs;
    public List<GeneticNode> all;

    public List<GeneticConnection> connections;
    public List<GeneticConnection> enabled;
    public List<GeneticConnection> disabled;
    
    protected List<Innovation> innovations;
    
    private Set<String> paths;
    
    private Map<Integer, GeneticNode> fastAccessNodes;
    private Map<String, GeneticNode> fastAccessIndex;
    private Map<String, Innovation> fastAccessInnovation;
    private Map<String, GeneticConnection> fastAccessConnections;
    
    private GeneticNode[] topSort;
    /* nodes and connections are never removed .. 
       we can use their count as test if topSort field is valid */
    private int lastTopSortNodeCount;
    private int lastTopSortConnectionCount;
    
    private static final String BIAS_NAME = "Bias";
    
    private final int BIAS_ID;
    
    private Activation outputActivation;
    private Activation hiddenActivation;
    
    private NeuralNetwork network;
    private Map<String, mapping> networkMapping;
    
    private AtomicInteger nodeCounter;
    
    class mapping {
        final int layer;
        final int from;
        final int to;

        public mapping(int layer, int from, int to) {
            this.layer = layer;
            this.from = from;
            this.to = to;
        }
    }
    
    /**
     * Constructor
     * @param input
     * @param output 
     */
    public GeneticNetwork(int input, int output) {
        this(input, output, null, null);
    }
    
    /**
     * Constructor
     * @param input
     * @param output
     * @param outputActivation
     * @param hiddenActivation
     */
    public GeneticNetwork(int input, int output, Activation outputActivation, Activation hiddenActivation) {
        inputs = new ArrayList<>();
        hidden = new ArrayList<>();
        outputs = new ArrayList<>();
        all = new ArrayList<>();
        
        connections = new ArrayList<>();
        enabled = new ArrayList<>();
        disabled = new ArrayList<>();
        
        innovations = new ArrayList<>();
        
        paths = new HashSet<>();
        networkMapping = new HashMap<>();
        
        fastAccessNodes = new HashMap<>();
        fastAccessIndex = new HashMap<>();
        fastAccessInnovation = new HashMap<>();
        fastAccessConnections = new HashMap<>();
        
        this.outputActivation = outputActivation;
        this.hiddenActivation = hiddenActivation;
        
        nodeCounter = new AtomicInteger();
        
        for(int i=0; i < output; i++) {
            GeneticNode o = new GeneticNode(nodeCounter.incrementAndGet(), "O"+i, NodeType.OUTPUT);
            addNode(o);
        }
        
        for(int i=0; i < input; i++) {
            GeneticNode n = new GeneticNode(nodeCounter.incrementAndGet(), "I"+i, NodeType.INPUT);
            addNode(n);
        }
        
        /* Bias Node */
        BIAS_ID = nodeCounter.incrementAndGet();
        GeneticNode bias = new GeneticNode(BIAS_ID, BIAS_NAME, NodeType.INPUT);
        addNode(bias);
    }

    public GeneticNetwork(GeneticStub stub) {
        this(stub.inputSize, stub.outputSize, stub.outputActivation, stub.hiddenActivation);
        
        for(Innovation i : stub.innovations)
            applyInnovation(i);
        
        for(GeneticConnection c : stub.connections)
            setWeight(c.from.name, c.to.name, c.weight);
        
        outputActivation = stub.outputActivation;
        hiddenActivation = stub.hiddenActivation;
        
        properties.putAll(stub.properties);
    }
    
    /**
     * Copy constructor
     * @return 
     */
    public GeneticNetwork copy() {
        GeneticNetwork result = new GeneticNetwork(inputs.size()-1, outputs.size());
        
        for(Innovation i : innovations)
            result.applyInnovation(i);
        
        for(GeneticConnection c : connections)
            result.setWeight(c.from.name, c.to.name, c.weight);
        
        result.outputActivation = outputActivation;
        result.hiddenActivation = hiddenActivation;
        
        result.properties.putAll(properties);

        return result;
    }
    
    /**
     * Connect all unconnected inputs to outputs
     */
    public void connectInputs(float value) {
        for(GeneticNode n : inputs) {
            for(GeneticNode o : outputs) {
                if(n.to(o) == null) {
                    connect(n, o, value);
                    updateInnovationList(mutation.ADD_CONNECTION, n.id, o.id);
                }
            }
        }
    }
    
    /**
     * Randomize all weights between [-1 1]
     */
    public void randomize() {
        for(GeneticConnection c : connections)
            c.weight = Rng.nextFloat(-1, 1);
    }
    
    /**
     * adds an input
     * @param count
     * @return 
     */
    public String[] addInput(int count) {
        String[] ids = new String[count];
        
        for(int i=0; i < count; i++) {
            ids[i] = getInputNodeID(getInputSize());
            GeneticNode n = new GeneticNode(nodeCounter.incrementAndGet(), ids[i], NodeType.INPUT);
            addNode(n);
        }
        
        return ids;
    }
    
    /**
     * Randomly selects a mutation and apply it
     * @return 
     */
    public mutation mutate() {
        return mutate(false);
    }
    
    
    public mutation mutate(boolean smart) {
        mutation m = GeneticConfig.getRandomMutation();
        
        switch(m) {
            case ADD_NODE:
                return addNode() ? m : null;
            case ADD_CONNECTION:
                return (smart ? addConnectionSmart() : addConnection()) ? m : null;
            case ENABLE_CONNECTION:
                return enableConnection() ? m : null;
            case DISABLE_CONNECTION:
                return (smart ? disableConnectionSmart() : disableConnection()) ? m : null;
        }
        
        return null;
    }
    
    /**
     * Randomly selects a connection and adds a node in between
     * @return 
     */
    public boolean addNode() {
        GeneticNode from = selectRandom(outputs);        
        GeneticConnection c = from.getRandom();
        
        if(c == null) {
            System.out.println("Failed to add node: " + from.id);
            return false;
        }
        
        GeneticNode to = c.to;
        
        from.disable(to);
        disabled.add(c);
        enabled.remove(c);
        
        GeneticNode n = new GeneticNode(nodeCounter.incrementAndGet(), "H"+hidden.size(), NodeType.HIDDEN);
        addNode(n);
        
        connect(from, n, c.weight);
        connect(n, to);
        
        updateInnovationList(mutation.ADD_NODE, from.id, to.id);
        
        return true;
    }
    
    /**
     * Randomly adds a connection between unconnected nodes
     * @return 
     */
    public boolean addConnection() {
        GeneticNode from = selectRandom(outputs);
        GeneticNode to = selectRandom(inputs, from.to);
        
        if(to == null || from == to)
            return false;
        
        if(!pathExists(to, from)) {
            connect(from, to, 1.0f);
            updateInnovationList(mutation.ADD_CONNECTION, from.id, to.id);
            return true;
        }
        
        return false;
    }

    /**
     * Selects least connected node and adds a connection
     * @return 
     */    
    public boolean addConnectionSmart() {
        List<GeneticNode> l = new ArrayList<>();
        int c = Integer.MAX_VALUE;
        
        for(GeneticNode n : fastAccessNodes.values()) {
            if(n.type == NodeType.OUTPUT)
                continue;
            
            if(n.out.size() < c) {
                l.clear();
                l.add(n);
                c = n.out.size();
            } else if(n.out.size() == c) {
                l.add(n);
            }
        }
        
        GeneticNode from = l.get(Rng.nextInt(l.size()));
        GeneticNode to = selectRandom(inputs, from.to);
        
        if(to == null || from == to)
            return false;
        
        if(!pathExists(to, from)) {
            connect(from, to, 1.0f);
            updateInnovationList(mutation.ADD_CONNECTION, from.id, to.id);
            return true;
        }
        
        return false;        
    }

    /**
     * Makes a connection enabled
     * @return 
     */
    public synchronized boolean enableConnection() {
        if(disabled.isEmpty())
            return false;

        int rand = Rng.nextInt(disabled.size());
        
        GeneticConnection c = disabled.get(rand);
        c.enable();
        
        enabled.add(disabled.remove(rand));
        
        /* make sure innovations dont repeat */
        updateInnovationList(mutation.ENABLE_CONNECTION, c.from.id, c.to.id);
                    
        return true;
    }
    
    /**
     * Disables a connection randomly
     * @return 
     */
    public synchronized boolean disableConnection() {
        if(enabled.isEmpty())
            return false;

        int rand = Rng.nextInt(enabled.size());
        
        GeneticConnection c = enabled.get(rand);
        c.disable();
        
        disabled.add(enabled.remove(rand));

        updateInnovationList(mutation.DISABLE_CONNECTION, c.from.id, c.to.id);

        return true;
    }
    
    /**
     * Disables least valuable connection (closest to zero)
     * @return 
     */
    public synchronized boolean disableConnectionSmart() {
        if(enabled.isEmpty())
            return false;
        
        double min = Integer.MAX_VALUE;
        int selection = -1;
        
        for(int i=0; i < enabled.size(); i++) {
            float weight = Math.abs(enabled.get(i).weight);
            if(weight < min) {
                min = Math.abs(weight);
                selection = i;
            }
        }
        
        GeneticConnection c = enabled.get(selection);
        c.disable();
        
        disabled.add(enabled.remove(selection));

        updateInnovationList(mutation.DISABLE_CONNECTION, c.from.id, c.to.id);

        return true;
    }
    
    
    /**
     * Compute result for a certain input
     * @param input
     * @return 
     */
    @Deprecated
    public float[] _compute(float[] input) {
        float[] values = new float[nodeCounter.get()+1];
        boolean[] computed = new boolean[nodeCounter.get()+1];
        
        for(int i=0; i < input.length; i++) {
            int id = inputs.get(i).id;
            values[id] = input[i];
            computed[id] = true;
        }
        
        values[BIAS_ID] = 1.0f;
        computed[BIAS_ID] = true;
        
        float[] result = new float[outputs.size()];
        
        for(int i=0; i < outputs.size(); i++)
            result[i] = compute(outputs.get(i), values, computed);
        
        if(outputActivation != null)
            outputActivation.compute(result);
        
        return result;
    }

    public float[] compute(float[] input) {
        float[] values = new float[nodeCounter.get()+1];
        
        for(int i=0; i < input.length; i++) {
            int id = inputs.get(i).id;
            values[id] = input[i];
        }
        
        values[BIAS_ID] = 1.0f;
        
        for(GeneticNode n : topSort())
            values[n.id] = compute0(n, values);
        
        float[] result = new float[outputs.size()];
        
        for(int i=0; i < outputs.size(); i++)
            result[i] = values[outputs.get(i).id];
        
        if(outputActivation != null)
            outputActivation.compute(result);
        
        return result;
    }
    
    public native float[] compute1(float[] input);
    
    /**
     * Converts the genetic network to a feed forward network
     * @param threadSafe
     * @return 
     */
    public NeuralNetwork getNetwork(boolean threadSafe) {
        if(network != null && network.isThreadSafe() == threadSafe)
            return network;
        
        networkMapping.clear();
        
        List<List<Integer>> layers = new ArrayList<>();
        List<Set<GeneticConnection>> layerLinks = new ArrayList<>();
        
        Set<Integer> from = new HashSet<>();
        Set<Integer> to = new HashSet<>();
        Set<Integer> processed = new HashSet<>();
        Set<Integer> unconnected = new HashSet<>();
        
        for(GeneticNode n : inputs) {
            from.add(n.id);
            processed.add(n.id);

            for(GeneticConnection e : n.out)
                if(!e.disabled)
                    to.add(e.to.id);
        }
        
        for(GeneticNode n : hidden) {
            boolean connected = false;
            
            for(GeneticConnection c : n.in) {
                if(!c.disabled) {
                    connected = true;
                    break;
                }
            }
            
            if(!connected)
                unconnected.add(n.id);
        }
        
        do {
            Set<Integer> temp = new HashSet<>();

            List<Integer> layer = new ArrayList<>();
            layers.add(layer);
            
            Set<GeneticConnection> links = new HashSet<>();
            layerLinks.add(links);

            /* Process "to" nodes and add to temp */
            for(Iterator<Integer> it = to.iterator(); it.hasNext();) {
                int s = it.next();
                
                GeneticNode n = fastAccessNodes.get(s);
                
                boolean p = true;
                
                for(GeneticConnection e : n.in) {
                    if(e.disabled)
                        continue;
                    
                    int f = e.from.id;
                    if(!unconnected.contains(f))
                        p &= from.contains(f);
                }
                
                if(p) {
                    it.remove();
                    processed.add(s);
                    temp.add(s);
                    
                    for(GeneticConnection e : n.in) {
                        if(!e.disabled)
                            links.add(e);
                    }
                }
            }

            /* remove nodes that are no longer needed as input */
            for(Iterator<Integer> it = from.iterator(); it.hasNext();) {
                int s = it.next();
                layer.add(s);
                
                GeneticNode n = fastAccessNodes.get(s);
                
                if(n.type != NodeType.OUTPUT) {
                    boolean d = true;

                    for(GeneticConnection e : n.out) {
                        if(!e.disabled)
                            d &= processed.contains(e.to.id);
                    }

                    if(d)
                        it.remove();
                }
            }
            
            /* update from and to with temp */
            from.addAll(temp);
            
            for(int s : temp) {
                GeneticNode n = fastAccessNodes.get(s);

                for(GeneticConnection t : n.out) {
                    if(!t.disabled)
//                        System.out.println("Processed: " + t.to.id + " " + to.contains(t.to.id));
//                    else
                        to.add(t.to.id);
                }
            }
        } while(!to.isEmpty());
        
        /* output layer */
        List<Integer> output = new ArrayList<>();
        layers.add(output);
        
        for(GeneticNode o : outputs)
            output.add(o.id);
        
        /* compute activation conditions */
        List<boolean[]> conditions = new ArrayList<>();
        
        for(int i=0; i < layers.size() - 2; i++) {
            List<Integer> current = layers.get(i);
            List<Integer> next = layers.get(i+1);
            Set<GeneticConnection> currentLinks = layerLinks.get(i);
            
            boolean[] b = new boolean[next.size()];

            for(int j=0; j < next.size(); j++) {
                GeneticNode nodeNext = fastAccessNodes.get(next.get(j));
                b[j] = false;

                for(int k=0; k < current.size(); k++) {
                    GeneticNode nodeCurrent = fastAccessNodes.get(current.get(k));
                    
                    GeneticConnection c = nodeCurrent.to(nodeNext);

                    if(currentLinks.contains(c) && nodeCurrent != nodeNext && nodeNext.type != NodeType.OUTPUT) {
                        b[j] = true;
                        break;
                    }                    
                }
            }

            conditions.add(b);
        }        
        
        
        /* build the actual network */
        NeuralNetwork n = new NeuralNetwork(threadSafe);
        
        int layerCount = layers.size();
        
        /* input */
        n.addLayer(layers.get(0).size() - 1, null, true);
        
        for(int i=1; i < layerCount-1; i++)
            n.addLayer(layers.get(i).size(), hiddenActivation, false, conditions.get(i-1));

        /* output */        
        n.addLayer(layers.get(layerCount-1).size(), outputActivation, false);
        
        n.build();
        
        /* build input layer in order I0 - Bias */
        List<Integer> next = layers.get(1);
        Set<GeneticConnection> currentLinks = layerLinks.get(0);
        
        for(int j=0; j < inputs.size()-1; j++) {
            GeneticNode nodeCurrent = inputs.get(j);

            for(int k=0; k < next.size(); k++) {
                GeneticNode nodeNext = fastAccessNodes.get(next.get(k));

                GeneticConnection c = nodeCurrent.to(nodeNext);

                if(currentLinks.contains(c)) {
                   n.setWeight(0, j, k, c.weight);
                   networkMapping.put(c.id, new mapping(0, j, k));
                }

                if(nodeCurrent == nodeNext)
                    n.setWeight(0, j, k, 1);
            }
        }
             
        GeneticNode biasNode = fastAccessNodes.get(BIAS_ID);

        for (int k = 0; k < next.size(); k++) {
            GeneticNode nodeNext = fastAccessNodes.get(next.get(k));

            GeneticConnection c = biasNode.to(nodeNext);

            if (c != null && !c.disabled) {
                n.setBias(0, k, c.weight);
                networkMapping.put(c.id, new mapping(0, -1, k));                
            }

            if (biasNode == nodeNext) {
                n.setBias(0, k, 1);
            }
        }
        
        for(int i=1; i < layers.size() - 1; i++) {
            List<Integer> current = layers.get(i);
            next = layers.get(i+1);
            currentLinks = layerLinks.get(i);
                    
            for(int j=0; j < current.size(); j++) {
                GeneticNode nodeCurrent = fastAccessNodes.get(current.get(j));

                for(int k=0; k < next.size(); k++) {
                    GeneticNode nodeNext = fastAccessNodes.get(next.get(k));

                    GeneticConnection c = nodeCurrent.to(nodeNext);

                    if(currentLinks.contains(c)) {
                       n.setWeight(i, j, k, c.weight);
                       networkMapping.put(c.id, new mapping(i, j, k));                       
                    }

                    if(nodeCurrent == nodeNext)
                        n.setWeight(i, j, k, 1);
                }
            }
        }
        
        network = n;
        
        return n;
    }
    
        
    /**
     * 
     * @param fromName
     * @param toName
     * @param weight 
     */
    public void setWeight(String fromName, String toName, float weight) {
        GeneticNode from = fastAccessIndex.get(fromName);
        GeneticNode to = fastAccessIndex.get(toName);
        
        GeneticConnection c = from.to(to);
        
        c.weight = weight;

        if(network != null && !c.disabled) {
            mapping m = networkMapping.get(c.id);
            
            if(m.from == -1)
                network.setBias(m.layer, m.to, weight);
            else
                network.setWeight(m.layer, m.from, m.to, weight);
        }
    }
    
    /**
     * 
     * @param fromID
     * @param toID
     * @param weight 
     */
    public void setWeight(int fromID, int toID, float weight) {
        GeneticNode from = fastAccessNodes.get(fromID);
        GeneticNode to = fastAccessNodes.get(toID);
        
        GeneticConnection c = from.to(to);

        c.weight = weight;

        if(network != null && !c.disabled) {
            mapping m = networkMapping.get(c.id);
            
            if(m.from == -1)
                network.setBias(m.layer, m.to, weight);
            else
                network.setWeight(m.layer, m.from, m.to, weight);
        }
    }
    
    /**
     * 
     * @param connectionID
     * @param weight 
     */
    public void setWeight(String connectionID, float weight) {
        GeneticConnection c = fastAccessConnections.get(connectionID);
        c.weight = weight;
        
        if(network != null && !c.disabled) {
            mapping m = networkMapping.get(connectionID);
            
            if(m.from == -1)
                network.setBias(m.layer, m.to, weight);
            else
                network.setWeight(m.layer, m.from, m.to, weight);
        }
    }
    
    /**
     * 
     * @param connectionID
     * @return 
     */
    public float getWeight(String connectionID) {
        return fastAccessConnections.get(connectionID).weight;
    }

    /**
     * Private functions
     */
    
    private float compute(GeneticNode n, float[] values, boolean[] computed) {
        float sum = 0;
        
        for(GeneticConnection c : n.in) {
            GeneticNode from = c.from;
            
            if(c.disabled)
                continue;
            
            if(!computed[from.id]) {
                values[from.id] = compute(from, values, computed);
                computed[from.id] = true;
            }
            
            sum += values[from.id] * c.weight;
        }
        
        if(hiddenActivation != null && n.type == NodeType.HIDDEN)
            sum = hiddenActivation.compute(sum);
        
        return sum;
    }
        
    private float compute0(GeneticNode n, float[] values) {
//        int len = n.in.size();
//        
//        float[] weights = new float[n._in.length];
//        float[] inputs  = new float[n._in.length];
        float sum = 0;
        
        for(GeneticConnection c : n._in) {//int i=0; i < len; i++) {
//            GeneticConnection c = n.in.get(i);
            GeneticNode from = c.from;
            
            if(c.disabled)
                continue;
//            
//            if(!computed[from.id]) {
//                for(GeneticNode q : topSort())
//                    System.out.println(q.name);
//                
//                throw new RuntimeException("Should not happen " + from.name + " " + n.name);
//            }
            
            sum += c.weight * values[from.id];
//            weights[i] = c.weight;
//            inputs[i]  = values[from.id];
        }
                
//        float sum = len == 0 ? 0 : intrinsic.dotProduct(weights, inputs);
        
        if(hiddenActivation != null && n.type == NodeType.HIDDEN)
            sum = hiddenActivation.compute(sum);
        
        return sum;
    }
        
    private GeneticNode selectRandom(Collection<GeneticNode>... exclude) {
        Set<Integer> banned = new HashSet<>();
        
        for(Collection<GeneticNode> nodes : exclude)
            for(GeneticNode n : nodes)
                banned.add(n.id);
        
        List<Integer> pool = new ArrayList<>();
        
        for(int s : fastAccessNodes.keySet())
            if(!banned.contains(s))
                pool.add(s);
        
        if(pool.isEmpty())
            return null;
        
        int selection = pool.get(Rng.nextInt(pool.size()));
        
        return fastAccessNodes.get(selection);
    }
    
    private boolean pathExists(GeneticNode n1, GeneticNode n2) {
        Queue<GeneticNode> queue = new LinkedList<>();
        Set<GeneticNode> visited = new HashSet<>();
        
        queue.add(n1);
        
        while(!queue.isEmpty()) {
            GeneticNode n = queue.poll();
            
            if(visited.contains(n))
                continue;

            String c = n.id + "-" + n2.id;

            if(paths.contains(c))
                return true;

            if(n.to(n2) != null) {
                paths.add(c);
                return true;
            }

            visited.add(n);

            for(GeneticNode next : n.to) {
                queue.add(next);
            }
        }
        
        return false;
    }
    
    GeneticNode[] topSort() {
        if(lastTopSortConnectionCount == connections.size() && lastTopSortNodeCount == nodeCounter.get())
            return topSort;
        
        ArrayList<GeneticNode> result = new ArrayList<>();
        Stack<GeneticNode> nodes = new Stack<>();
        Set<GeneticNode> visited = new HashSet<>();
        
        for(GeneticNode n : all) {
            if(visited.contains(n))
                continue;
            
            nodes.push(n);
            
            while(!nodes.isEmpty()) {
                GeneticNode current = nodes.peek();

                boolean done = true;
                
                for(GeneticNode next : current.to) {
                    if(!visited.contains(next)) {
                        nodes.push(next);
                        done = false;
                        break;
                    }
                }
                
                if(done) {
                    nodes.pop();
                    visited.add(current);
                    
                    if(current.type != NodeType.INPUT)
                        result.add(0, current);
                }
            }
        }
        
        topSort = new GeneticNode[result.size()];
        result.toArray(topSort);
        lastTopSortConnectionCount = connections.size();
        lastTopSortNodeCount = nodeCounter.get();
        
        return topSort;
    }
    
    private boolean applyInnovation(Innovation i) {
        switch(i.m) {
            case ADD_NODE:
                return addNode(i.fromName, i.toName);
            case ADD_CONNECTION:
                return addConnection(i.fromName, i.toName);
            case ENABLE_CONNECTION:
                return enableConnection(fastAccessIndex.get(i.fromName).id, fastAccessIndex.get(i.toName).id);
            case DISABLE_CONNECTION:
                return disableConnection(fastAccessIndex.get(i.fromName).id, fastAccessIndex.get(i.toName).id);
        }
        
        throw new RuntimeException();
    }
        
    public String getInputNodeID(int index) {
        return "I" + index;
    }
    
    public String getOutputNodeID(int index) {
        return "O" + index;
    }
    
    public String getHiddenNodeID(int index) {
        return "H" + index;
    }
    
    public boolean addNode(String fromName, String toName) {
        return addNode(fastAccessIndex.get(fromName).id, fastAccessIndex.get(toName).id);
    }
    
    boolean addNode(int fromID, int toID) {
        GeneticNode from = fastAccessNodes.get(fromID);
        GeneticNode to = fastAccessNodes.get(toID);

        GeneticConnection c = from.to(to);

        from.disable(to);
        
        GeneticNode n = new GeneticNode(nodeCounter.incrementAndGet(), "H"+hidden.size(), NodeType.HIDDEN);
        addNode(n);

        connect(from, n, c.weight);
        connect(n, to);
    
        updateInnovationList(mutation.ADD_NODE, fromID, toID);
    
        return true;
    }
    
    public boolean addConnection(String fromName, String toName) {
        return addConnection(fastAccessIndex.get(fromName).id, fastAccessIndex.get(toName).id);
    }
    
    boolean addConnection(int fromID, int toID) {
        GeneticNode n1 = fastAccessNodes.get(fromID);
        GeneticNode n2 = fastAccessNodes.get(toID);
        
        if(n1 == n2)
            return false;
        
        if(!pathExists(n2, n1)) {
            connect(n1, n2);
            updateInnovationList(mutation.ADD_CONNECTION, fromID, toID);
            return true;
        }
        
        return false;        
    }

    synchronized boolean enableConnection(int fromID, int toID) {
        GeneticNode from = fastAccessNodes.get(fromID);
        GeneticNode to = fastAccessNodes.get(toID);
        
        GeneticConnection c = from.to(to);
        c.enable();
        
        int i = disabled.indexOf(c);
        
        if(i > -1)
           disabled.remove(i);
        
        i = enabled.indexOf(c);
        
        if(i == -1)
            enabled.add(c);

        updateInnovationList(mutation.ENABLE_CONNECTION, fromID, toID);

        return true;
    }
    
    synchronized boolean disableConnection(int fromID, int toID) {
        GeneticNode from = fastAccessNodes.get(fromID);
        GeneticNode to = fastAccessNodes.get(toID);
        
        GeneticConnection c = from.to(to);
        c.disable();
        
        int i = enabled.indexOf(c);
        
        if(i > -1)
           enabled.remove(i);
        
        i = disabled.indexOf(c);
        
        if(i == -1)
            disabled.add(c);

        updateInnovationList(mutation.DISABLE_CONNECTION, fromID, toID);

        return true;
    }
    
    private void connect(GeneticNode a, GeneticNode b) {
        connect(a, b, 1.0f);
    }
    
    private void connect(GeneticNode a, GeneticNode b, float weight) {
        GeneticConnection c = a.connect(b, weight);
        connections.add(c);
        fastAccessConnections.put(c.id, c);
        enabled.add(c);
    }

    @Override
    public int getInputSize() {
        return inputs.size() - 1;
    }

    @Override
    public int getOutputSize() {
        return outputs.size();
    }
    
    public GeneticStub getStub() {
        return new GeneticStub(getInputSize(), getOutputSize(), innovations, connections, properties, outputActivation, hiddenActivation);
    }
    
    public int getInnovationCount() {
        return innovations.size();
    }
    
    private void updateInnovationList(mutation m, int fromID, int toID) {
        String hash = fromID + ":" + toID + ":" + m.toString();

        if(m == mutation.ENABLE_CONNECTION || m == mutation.DISABLE_CONNECTION) {
            mutation c = m == mutation.ENABLE_CONNECTION ? mutation.DISABLE_CONNECTION : mutation.ENABLE_CONNECTION;

            String cHash = fromID + ":" + toID + ":" + c.toString();

            Innovation previous = fastAccessInnovation.get(cHash);
            
            if(previous != null) {
                System.out.println("Found enable disable: " + hash);
                innovations.remove(previous);
                fastAccessInnovation.remove(cHash);
                return;
            }            
        }

        Innovation i = new Innovation(m, fastAccessNodes.get(fromID).name, fastAccessNodes.get(toID).name);
        innovations.add(i);
        fastAccessInnovation.put(hash, i);
        
        if(network != null) {
            network.freeGPU();
            network = null;
            networkMapping.clear();
        }
    }
    
    private void addNode(GeneticNode n) {
        fastAccessNodes.put(n.id, n);
        fastAccessIndex.put(n.name, n);
        
        switch(n.type) {
            case INPUT:
                inputs.add(n);
                break;
            case HIDDEN:
                hidden.add(n);
                break;
            case OUTPUT:
                outputs.add(n);
                break;
        }
        
        all.add(n);
    }
    
    public List<Innovation> getInnovations() {
        return innovations;
    }
    
    public int getConnectionsCount() {
        return fastAccessConnections.size();
    }
    
    public int getNodeCount() {
        return fastAccessNodes.size();
    }
    
    public int getHiddenCount() {
        return hidden.size();
    }
}
