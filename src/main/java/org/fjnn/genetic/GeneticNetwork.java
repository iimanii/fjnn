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

import java.nio.FloatBuffer;
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
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import org.fjnn.activation.Activation;
import org.fjnn.base.Network;
import org.fjnn.base.NetworkInput;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.base.output.FeedForwardOutput;
import org.fjnn.base.output.FeedForwardOutputGPU;
import org.fjnn.genetic.GeneticConfig.mutation;
import org.fjnn.genetic.GeneticNode.NodeType;
import org.fjnn.loss.Loss;
import org.fjnn.network.Connection;
import org.fjnn.network.NeuralNetwork;
import org.fjnn.util.Rng;

/**
 *
 * @author ahmed
 */
public class GeneticNetwork extends Network<GeneticNetwork> {
    public List<GeneticNode> inputs;
    public List<GeneticNode> hidden;
    public List<GeneticNode> outputs;
    public List<GeneticNode> all;

    public GeneticConnectionMap connectionMap;
    
    protected List<Innovation> innovations;
    
    private AtomicInteger inputCounter;
    private AtomicInteger outputCounter;
    private AtomicInteger hiddenCounter;
    private AtomicInteger nodeCounter;
    
    private Map<String, GeneticNode> fastAccessNodes;
    
    /**
     * Innovations can duplicate on the same ID in some circumstances
     * 1- Adding a node on the same connection multiple times (multiple ADD_NODE)
     * 2- Enabling a connection after a node is added (multiple ENABLE_CONNECTION)
     * this list contains the last innovation per ID
     */
    Map<String, Innovation> fastAccessInnovation;
    Map<String, AtomicInteger> fastAccessInnovationCount;
    
    private GeneticNode[] topSort;
    
    /* nodes and connections are never removed .. 
       we can use their count as test if topSort field is valid */
    private int lastTopSortNodeCount;
    private int lastTopSortConnectionCount;
    
    private static final String BIAS_NAME = "Bias";
    private final int BIAS_ID;
    
    private Activation hiddenActivation;
    
    private NeuralNetwork network;
    private Map<String, mapping> networkMapping;

    @Override
    public float[] compute(NetworkInput input) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public FeedForwardOutput feedForward(float[] input, int batchCount) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public FeedForwardOutputGPU feedForwardGPU(CUdeviceptr input, int batchCount, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public BackpropagateOutput backpropagate(FeedForwardOutput output, float[] deltaLoss) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public BackpropagateOutputGPU backpropagateGPU(FeedForwardOutputGPU output, CUdeviceptr deltaLoss, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public HashMap serialize() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void updateWeightsFromGPU() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public long getParametersCount() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public long getBackpropagateMemoryRequired(int batchCount) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void applyGradients(BackpropagateOutput gradients, float learningRate) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void applyGradientsGPU(BackpropagateOutputGPU gradients, float learningRate, CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    static class mapping {
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
        super(input, output, outputActivation, null);
        
        inputs = new ArrayList<>();
        hidden = new ArrayList<>();
        outputs = new ArrayList<>();
        all = new ArrayList<>();
        
        innovations = new ArrayList<>();
        
        networkMapping = new HashMap<>();
        
        fastAccessNodes = new HashMap<>();
        fastAccessInnovation = new HashMap<>();
        fastAccessInnovationCount = new HashMap<>();
        
        this.hiddenActivation = hiddenActivation;
        
        inputCounter  = new AtomicInteger();
        outputCounter = new AtomicInteger();
        hiddenCounter = new AtomicInteger();
        nodeCounter   = new AtomicInteger();
        
        for(int i=0; i < input; i++)
            createNode(inputCounter.getAndIncrement(), NodeType.INPUT);
        
        for(int i=0; i < output; i++)
            createNode(outputCounter.getAndIncrement(), NodeType.OUTPUT);
        
        /* Bias Node */
        BIAS_ID = createNode(BIAS_NAME, NodeType.INPUT).index;
        
        connectionMap = new GeneticConnectionMap(nodeCounter.get());
    }

    /**
     * 
     * @param stub 
     */
    public GeneticNetwork(GeneticStub stub) {
        this(stub.inputSize, stub.outputSize, Activation.fromName(stub.outputActivation), Activation.fromName(stub.hiddenActivation));
        
        for(Innovation i : stub.innovations)
            applyInnovation(i);
        
        for(GeneticConnectionStub c : stub.connections) {
            setWeight(c.from, c.to, c.weight);
            
            /* TODO: remove this after testing */
            GeneticNode from = fastAccessNodes.get(c.from);
            GeneticNode to = fastAccessNodes.get(c.to);
            if(c.disabled != connectionMap.get(from, to).disabled)
                throw new RuntimeException();
        }
        
        properties.putAll(stub.properties);
    }
    
    /**
     * Returns network stub
     * @return 
     */
    public GeneticStub getStub() {
        List<GeneticConnectionStub> connectionStubs = new ArrayList<>();

        for(GeneticConnection c : connectionMap.all()) {
            connectionStubs.add(new GeneticConnectionStub(c.from.id, c.to.id, c.weight, c.disabled));
        }
        return new GeneticStub(getInputSize(), getOutputSize(), 
                               innovations, connectionStubs, properties, 
                               Activation.toName(outputActivation), Activation.toName(hiddenActivation));
    }
    
    /**
     * Copy constructor
     * @return 
     */
    public GeneticNetwork copy() {
        GeneticNetwork result = new GeneticNetwork(inputCounter.get(), outputCounter.get());
        
        for(Innovation i : innovations)
            result.applyInnovation(i);
        
        for(GeneticConnection c : connectionMap.all())
            result.setWeight(c.from.id, c.to.id, c.weight);
        
        result.hiddenActivation = hiddenActivation;
        
        result.properties.putAll(properties);

        return result;
    }
    
    /**
     * Randomize all weights between [-1 1]
     * @param min
     * @param max
     */
    @Override
    public void randomize(float min, float max) {
        for(GeneticConnection c : connectionMap.all())
            c.weight = Rng.nextFloat(min, max);
    }
    
    /**
     * adds more inputs
     * @param count
     */
    public void addInputs(int count) {
        for(int i=0; i < count; i++)
            createNode(inputCounter.getAndIncrement(), NodeType.INPUT);
    }
    
    /**
     * Connect all unconnected inputs to outputs
     * @param weight
     */
    public void connectAllInputs(float weight) {
        for(GeneticNode n : inputs) {
            for(GeneticNode o : outputs) {
                if(connectionMap.get(n, o) == null) {
                    createConnection(n, o, weight);
                    createInnovation(mutation.ADD_CONNECTION, n, o);
                }
            }
        }
    }
    
    /**
     * Randomly selects a mutation and apply it
     * @return 
     */
    public Innovation mutate() {
        return mutate(false);
    }
    public Innovation mutate(boolean smart) {
        mutation m = GeneticConfig.getRandomMutation();

        switch(m) {
            case ADD_NODE:
                return addHiddenNode();
            case ADD_CONNECTION:
                return smart ? addConnectionSmart() : addConnection();
            case ENABLE_CONNECTION:
                return enableConnection();
            case DISABLE_CONNECTION:
                return smart ? disableConnectionSmart() : disableConnection();
        }

        return null;
    }
    
    /**
     * Randomly adds a hidden node between a connection
     * return true on success
     * @return 
     */
    public Innovation addHiddenNode() {
        List<GeneticConnection> enabled = connectionMap.enabled();
        
        if(enabled.isEmpty())
            return null;
        
        GeneticConnection c = enabled.get(Rng.nextInt(enabled.size()));
        return addHiddenNode(c);
    }
    public Innovation addHiddenNode(String from, String to) {
        GeneticConnection c = connectionMap.get(from, to);
        
        if(c == null || c.disabled)
            return null;
        
        return addHiddenNode(c);
    }
    
    /**
     * Randomly adds a connection between unconnected nodes
     * @return 
     */
    public Innovation addConnection() {
        GeneticNode from = selectRandomExcept(outputs);
        GeneticNode to = selectRandomExcept(inputs, connectionMap.next(from));
        
        GeneticConnection c = createConnection(from, to, 1.0f);
        
        if(c == null)
            return null;
        
        return createInnovation(mutation.ADD_CONNECTION, from, to);
    }
    
    /**
     * Selects least connected node and adds a connection
     * @return 
     */
    public Innovation addConnectionSmart() {
        List<GeneticNode> l = new ArrayList<>();
        int s = Integer.MAX_VALUE;
        
        for(GeneticNode n : fastAccessNodes.values()) {
            if(n.type == NodeType.OUTPUT)
                continue;
            
            int size = connectionMap.from(n).size();
            if(size < s) {
                l.clear();
                l.add(n);
                s = size;
            } else if(size == s) {
                l.add(n);
            }
        }
        
        GeneticNode from = l.get(Rng.nextInt(l.size()));
        GeneticNode to = selectRandomExcept(inputs, connectionMap.next(from));

        GeneticConnection c = createConnection(from, to, 1.0f);
        
        if(c == null)
            return null;

        return createInnovation(mutation.ADD_CONNECTION, from, to);
    }
    public Innovation addConnection(String from, String to, float weight) {
        GeneticNode fromNode = fastAccessNodes.get(from);
        GeneticNode toNode = fastAccessNodes.get(to);
                
        GeneticConnection c = createConnection(fromNode, toNode, weight);
        
        if(c == null)
            return null;

        return createInnovation(mutation.ADD_CONNECTION, fromNode, toNode);
    }
    
    /**
     * Makes a random connection enabled
     * @return 
     */
    public Innovation enableConnection() {
        List<GeneticConnection> disabled = connectionMap.disabled();

        if(disabled.isEmpty())
            return null;

        int rand = Rng.nextInt(disabled.size());
        
        GeneticConnection c = disabled.get(rand);
        c.disabled = false;
        connectionMap.update(c);
        
        return createInnovation(mutation.ENABLE_CONNECTION, c.from, c.to);
    }
    public Innovation enableConnection(String from, String to) {
        GeneticConnection c = connectionMap.get(from, to);
        
        if(!c.disabled)
            return null;
        
        c.disabled = false;
        connectionMap.update(c);
        
        return createInnovation(mutation.ENABLE_CONNECTION, c.from, c.to);
    }
    
    /**
     * Disables a connection randomly
     * @return 
     */
    public Innovation disableConnection() {
        List<GeneticConnection> enabled = connectionMap.enabled();

        if(enabled.isEmpty())
            return null;

        int rand = Rng.nextInt(enabled.size());
        
        GeneticConnection c = enabled.get(rand);
        c.disabled = true;
        connectionMap.update(c);
        
        return createInnovation(mutation.DISABLE_CONNECTION, c.from, c.to);
    }
    /**
     * Disables least valuable connection (closest to zero)
     * @return 
     */
    public Innovation disableConnectionSmart() {
        List<GeneticConnection> enabled = connectionMap.enabled();
        
        if(enabled.isEmpty())
            return null;
        
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
        c.disabled = true;
        connectionMap.update(c);
        
        return createInnovation(mutation.DISABLE_CONNECTION, c.from, c.to);
    }
    public Innovation disableConnection(String from, String to) {
        GeneticConnection c = connectionMap.get(from, to);
        
        if(c.disabled)
            return null;
        
        c.disabled = true;
        connectionMap.update(c);
        
        return createInnovation(mutation.DISABLE_CONNECTION, c.from, c.to);
    }
    
    /**
     *  Uses topsort instead of recursion which is alot faster
     * @param input
     * @return 
    */
    public float[] compute(float[] input) {
        float[] values = new float[nodeCounter.get()+1];
        
        for(int i=0; i < input.length; i++) {
            int index = inputs.get(i).index;
            values[index] = input[i];
        }
        
        values[BIAS_ID] = 1.0f;
        
        for(GeneticNode n : topSort())
            values[n.index] = compute(n, values);
        
        float[] result = new float[outputs.size()];
        
        for(int i=0; i < outputs.size(); i++)
            result[i] = values[outputs.get(i).index];
        
        if(outputActivation != null)
            outputActivation.compute(result, result, result.length, 1);
        
        return result;
    }
    
    /**
     * Converts the genetic network to a feed forward network
     * @param threadSafe
     * @return 
     */
    public NeuralNetwork getNetwork() {
        if(network != null)
            return network;
        
        networkMapping.clear();
        
        List<List<GeneticNode>> layers = new ArrayList<>();
        List<Set<GeneticConnection>> layerLinks = new ArrayList<>();
        
        Set<GeneticNode> from = new HashSet<>();
        Set<GeneticNode> to = new HashSet<>();
        Set<GeneticNode> processed = new HashSet<>();
        Set<GeneticNode> unconnected = new HashSet<>();
        
        for(GeneticNode n : inputs) {
            from.add(n);
            processed.add(n);

            for(GeneticConnection e : connectionMap.from(n))
                if(!e.disabled)
                    to.add(e.to);
        }
        
        for(GeneticNode n : hidden) {
            boolean connected = false;
            
            for(GeneticConnection c : connectionMap.to(n)) {
                if(!c.disabled) {
                    connected = true;
                    break;
                }
            }
            
            if(!connected)
                unconnected.add(n);
        }
        
        do {
            Set<GeneticNode> temp = new HashSet<>();

            List<GeneticNode> layer = new ArrayList<>();
            layers.add(layer);
            
            Set<GeneticConnection> links = new HashSet<>();
            layerLinks.add(links);

            /* Process "to" nodes and add to temp */
            for(Iterator<GeneticNode> it = to.iterator(); it.hasNext();) {
                GeneticNode n = it.next();
                
                boolean p = true;
                
                for(GeneticConnection e : connectionMap.to(n)) {
                    if(e.disabled)
                        continue;
                    
                    if(!unconnected.contains(e.from))
                        p &= from.contains(e.from);
                }
                
                if(p) {
                    it.remove();
                    processed.add(n);
                    temp.add(n);
                    
                    for(GeneticConnection e : connectionMap.to(n)) {
                        if(!e.disabled)
                            links.add(e);
                    }
                }
            }

            /* remove nodes that are no longer needed as input */
            for(Iterator<GeneticNode> it = from.iterator(); it.hasNext();) {
                GeneticNode n = it.next();
                layer.add(n);
                
                if(n.type != NodeType.OUTPUT) {
                    boolean d = true;

                    for(GeneticConnection e : connectionMap.from(n)) {
                        if(!e.disabled)
                            d &= processed.contains(e.to);
                    }

                    if(d)
                        it.remove();
                }
            }
            
            /* update from and to with temp */
            from.addAll(temp);
            
            for(GeneticNode n : temp) {
                for(GeneticConnection t : connectionMap.from(n)) {
                    if(!t.disabled)
//                        System.out.println("Processed: " + t.to.id + " " + to.contains(t.to.id));
//                    else
                        to.add(t.to);
                }
            }
        } while(!to.isEmpty());
        
        /* output layer */
        List<GeneticNode> output = new ArrayList<>();
        layers.add(output);
        
        for(GeneticNode o : outputs)
            output.add(o);
        
        /* compute activation conditions */
        List<boolean[]> conditions = new ArrayList<>();
        
        for(int i=0; i < layers.size() - 2; i++) {
            List<GeneticNode> current = layers.get(i);
            List<GeneticNode> next = layers.get(i+1);
            Set<GeneticConnection> currentLinks = layerLinks.get(i);
            
            boolean[] b = new boolean[next.size()];

            for(int j=0; j < next.size(); j++) {
                GeneticNode nodeNext = next.get(j);
                b[j] = false;

                for(int k=0; k < current.size(); k++) {
                    GeneticNode nodeCurrent = current.get(k);
                    
                    GeneticConnection c = connectionMap.get(nodeCurrent, nodeNext);

                    if(currentLinks.contains(c) && nodeCurrent != nodeNext && nodeNext.type != NodeType.OUTPUT) {
                        b[j] = true;
                        break;
                    }                    
                }
            }

            conditions.add(b);
        }        
        
        
        /* build the actual network */
        int layerCount = layers.size();
        int iSize = layers.get(0).size() - 1;
        int oSize = layers.get(layerCount-1).size();
        
        NeuralNetwork n = new NeuralNetwork(iSize, oSize, outputActivation);
        
        for(int i=1; i < layerCount-1; i++)
            n.addHiddenLayer(layers.get(i).size(), hiddenActivation);
        
        n.build();
        
        /* build input layer in order I0 - Bias */
        List<GeneticNode> next = layers.get(1);
        Set<GeneticConnection> currentLinks = layerLinks.get(0);
        
        Connection layerConnection = n.getLayer(0).getConnection();
        
        for(int j=0; j < inputs.size()-1; j++) {
            GeneticNode nodeCurrent = inputs.get(j);

            for(int k=0; k < next.size(); k++) {
                GeneticNode nodeNext = next.get(k);

                GeneticConnection c = connectionMap.get(nodeCurrent, nodeNext);

                if(currentLinks.contains(c)) {
                   layerConnection.setWeight(j, k, c.weight);
                   networkMapping.put(c.id, new mapping(0, j, k));
                }

                if(nodeCurrent == nodeNext)
                    layerConnection.setWeight(j, k, 1);
            }
        }
             
        GeneticNode biasNode = fastAccessNodes.get(BIAS_NAME);

        for (int k = 0; k < next.size(); k++) {
            GeneticNode nodeNext = next.get(k);

            GeneticConnection c = connectionMap.get(biasNode, nodeNext);

            if (c != null && !c.disabled) {
                layerConnection.setBias(k, c.weight);
                networkMapping.put(c.id, new mapping(0, -1, k));                
            }

            if (biasNode == nodeNext) {
                layerConnection.setBias(k, 1);
            }
        }
        
        for(int i=1; i < layers.size() - 1; i++) {
            List<GeneticNode> current = layers.get(i);
            next = layers.get(i+1);
            currentLinks = layerLinks.get(i);
            layerConnection = n.getLayer(i).getConnection();
            
            for(int j=0; j < current.size(); j++) {
                GeneticNode nodeCurrent = current.get(j);

                for(int k=0; k < next.size(); k++) {
                    GeneticNode nodeNext = next.get(k);

                    GeneticConnection c = connectionMap.get(nodeCurrent, nodeNext);

                    if(currentLinks.contains(c)) {
                       layerConnection.setWeight(j, k, c.weight);
                       networkMapping.put(c.id, new mapping(i, j, k));                       
                    }

                    if(nodeCurrent == nodeNext)
                        layerConnection.setWeight(j, k, 1);
                }
            }
        }
        
        network = n;
        
        return n;
    }
        
    /**
     *
     * @param from
     * @param to
     * @param weight 
     * @return  
     */
    public boolean setWeight(String from, String to, float weight) {
        GeneticConnection c = connectionMap.get(from, to);
        
        if(c == null)
            return false;
        
        c.weight = weight;

        if(network != null && !c.disabled) {
            mapping m = networkMapping.get(c.id);
            
            if(m.from == -1)
                network.getLayer(m.layer).getConnection().setBias(m.to, weight);
            else
                network.getLayer(m.layer).getConnection().setWeight(m.from, m.to, weight);
        }
        
        return true;
    }
   
    public float getWeight(String from, String to) {
        return connectionMap.get(from, to).weight;
    }
    
    
    private float compute(GeneticNode n, float[] values) {
        float sum = 0;
        
        for(GeneticConnection c : connectionMap.to0(n)) {
            GeneticNode from = c.from;
            
            if(c.disabled)
                continue;
            
            sum += c.weight * values[from.index];
        }
        
        if(hiddenActivation != null && n.type == NodeType.HIDDEN)
            sum = hiddenActivation.compute(sum);
        
        return sum;
    }
    
    public GeneticNode[] topSort() {
        if(lastTopSortConnectionCount == connectionMap.count() && lastTopSortNodeCount == nodeCounter.get())
            return topSort;
        
        ArrayList<GeneticNode> result = new ArrayList<>();
        Stack<GeneticNode> nodes = new Stack<>();
        Set<GeneticNode> visited = new HashSet<>();
        
        for(GeneticNode n : all) {
            if(n.type == NodeType.INPUT)
                continue;
            
            if(visited.contains(n))
                continue;
            
            nodes.push(n);
            
            while(!nodes.isEmpty()) {
                GeneticNode current = nodes.peek();
                
                for(GeneticNode prev : connectionMap.prev(current)) {
                    if(prev.type == NodeType.INPUT)
                        continue;
                    
                    if(!visited.contains(prev))
                        nodes.push(prev);
                }
                
                if(current == nodes.peek()) {
                    visited.add(nodes.pop());
                    result.add(current);
                }
            }
        }
        
        topSort = new GeneticNode[result.size()];
        result.toArray(topSort);
        lastTopSortConnectionCount = connectionMap.count();
        lastTopSortNodeCount = nodeCounter.get();
        
        return topSort;
    }
    
    private Innovation addHiddenNode(GeneticConnection c) {
        if(c.disabled)
            throw new RuntimeException();
        
        GeneticNode n = createNode(hiddenCounter.getAndIncrement(), NodeType.HIDDEN);
        
        createConnection(c.from, n, c.weight);
        createConnection(n, c.to, 1.0f);
        
        c.disabled = true;
        connectionMap.update(c);
        
        return createInnovation(mutation.ADD_NODE, c.from, c.to);
    }
    
    /* Innovation Functions */
    private Innovation createInnovation(mutation m, GeneticNode from, GeneticNode to) {
        Innovation i = new Innovation(m, from.id, to.id);
        
        /* make sure innovations dont repeat
        * Remove this --> ENABLE_CONNECTION ... (new) DISABLE_CONNECTION
        * Remove this --> DISABLE_CONNECTION ... (new) ENABLE_CONNECTION
        * This will never happen -->  ENABLE_CONNECTION ... ADD_NODE ... (new) DISABLE_CONNECTION
        * because ADD_NODE disables the connection  
        */
        if(m == mutation.ENABLE_CONNECTION || m == mutation.DISABLE_CONNECTION) {
            mutation mp = m == mutation.ENABLE_CONNECTION ? mutation.DISABLE_CONNECTION : mutation.ENABLE_CONNECTION;

            String pid = Innovation.getId(mp, from, to);
            Innovation p = fastAccessInnovation.get(pid);

            if(p != null) {
//                System.out.println("Removing: " + p.id());
                innovations.remove(p);
                fastAccessInnovation.remove(pid);
                AtomicInteger count = fastAccessInnovationCount.get(pid);
                count.decrementAndGet();
                
                if(count.get() == 0)
                    fastAccessInnovationCount.remove(pid);
    
                return i;
            }
        }

        String id = Innovation.getId(m, from, to);
        
        innovations.add(i);
        fastAccessInnovation.put(id, i);
        fastAccessInnovationCount.putIfAbsent(id, new AtomicInteger());
        fastAccessInnovationCount.get(id).incrementAndGet();
                
        if(network != null) {
            network.freeGPU();
            network = null;
            networkMapping.clear();
        }
        
        return i;
    }

    private boolean applyInnovation(Innovation i) {
        switch(i.m) {
            case ADD_NODE:
                return addHiddenNode(i.from, i.to) != null;
            case ADD_CONNECTION:
                return addConnection(i.from, i.to, 1.0f) != null;
            case ENABLE_CONNECTION:
                return enableConnection(i.from, i.to) != null;
            case DISABLE_CONNECTION:
                return disableConnection(i.from, i.to) != null;
        }
        
        return false;
    }

    /* Node functions */
    private GeneticNode createNode(int index, NodeType type) {
        return createNode(GeneticNode.createId(index, type), type);
    }
    private GeneticNode createNode(String id, NodeType type) {
        GeneticNode n = new GeneticNode(id, type);
        
        switch(type) {
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
        fastAccessNodes.put(id, n);
        
        n.index = nodeCounter.getAndIncrement();
        
        return n;
    }
    
    /* Connection functions */
    private GeneticConnection createConnection(GeneticNode from, GeneticNode to, float weight) {
        if(from == null || to == null || from == to)
            return null;

        if(connectionMap.get(from, to) != null)
            return null;
        
        /* dont allow cycles */
        if(pathExists(to, from))
            return null;
        
        GeneticConnection c = new GeneticConnection(from, to, weight);
        connectionMap.add(c);
        
        return c;
    }
    private boolean pathExists(GeneticNode from, GeneticNode to) {
        Queue<GeneticNode> queue = new LinkedList<>();
        Set<GeneticNode> visited = new HashSet<>();
        
        queue.add(from);
        
        while(!queue.isEmpty()) {
            GeneticNode n = queue.poll();
            
            if(visited.contains(n))
                continue;
            
            if(connectionMap.get(n, to) != null)
                return true;

            visited.add(n);

            for(GeneticNode next : connectionMap.next(n))
                queue.add(next);
        }
        
        return false;
    }

    public String getInputNode(int index) {
        return GeneticNode.createId(index, NodeType.INPUT);
    }
    
    public String getOutputNode(int index) {
        return GeneticNode.createId(index, NodeType.OUTPUT);
    }
    
    public String getHiddenNode(int index) {
        return GeneticNode.createId(index, NodeType.HIDDEN);
    }
    
    public Activation getActivationHidden() {
        return hiddenActivation;
    }
    
    public Activation getActivationOutput() {
        return outputActivation;
    }
    
    public List<Innovation> getInnovations() {
        return innovations;
    }
    
    public int getConnectionsCount() {
        return connectionMap.count();
    }
    
    public int getNodeCount() {
        return fastAccessNodes.size();
    }
    
    public int getHiddenCount() {
        return hiddenCounter.get();
    }
    
    public int getInnovationCount() {
        return innovations.size();
    }
    
    private GeneticNode selectRandomExcept(Collection<GeneticNode>... exclude) {
        Set<GeneticNode> banned = new HashSet<>();
        
        for(Collection<GeneticNode> nodes : exclude)
            for(GeneticNode n : nodes)
                banned.add(n);
        
        List<GeneticNode> pool = new ArrayList<>();
        
        for(GeneticNode n : all)
            if(!banned.contains(n))
                pool.add(n);
        
        if(pool.isEmpty())
            return null;
        
        return pool.get(Rng.nextInt(pool.size()));
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
            int id = inputs.get(i).index;
            values[id] = input[i];
            computed[id] = true;
        }
        
        values[BIAS_ID] = 1.0f;
        computed[BIAS_ID] = true;
        
        float[] result = new float[outputs.size()];
        
        for(int i=0; i < outputs.size(); i++)
            result[i] = _compute(outputs.get(i), values, computed);
        
        if(outputActivation != null)
            outputActivation.compute(result, result, result.length, 1);
        
        return result;
    }

    /**
     * Private functions
     */
    @Deprecated
    private float _compute(GeneticNode n, float[] values, boolean[] computed) {
        float sum = 0;
        
        for(GeneticConnection c : connectionMap.to(n)) {
            GeneticNode from = c.from;
            
            if(c.disabled)
                continue;
            
            if(!computed[from.index]) {
                values[from.index] = _compute(from, values, computed);
                computed[from.index] = true;
            }
            
            sum += values[from.index] * c.weight;
        }
        
        if(hiddenActivation != null && n.type == NodeType.HIDDEN)
            sum = hiddenActivation.compute(sum);
        
        return sum;
    }
    

    @Override
    public float[] compute(float[] input, int count) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public float[] compute(FloatBuffer input) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public float[] compute(FloatBuffer input, int count) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void clipWeights(float clipMin, float clipMax) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void ensureCPU() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean gpuReady() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    protected void prepareGPU0(CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void clipWeightsGPU(float clipMin, float clipMax) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    protected void freeGPU0(CUstream stream) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public void copyWeights(GeneticNetwork n) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public GeneticNetwork copy(boolean copyWeights, boolean createWeights) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double compare(GeneticNetwork n0) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void crossOverMutate(GeneticNetwork n0, GeneticNetwork n1, float f, float mutationAmount, double m) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void crossOverMutateGPU(GeneticNetwork n0, GeneticNetwork n1, float min, float max, double mutation, boolean nocopy) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    @Override
    public long getWeightsCount() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public long getGPUComputeMemoryRequired(int count) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public long getGPUPrepareMemoryRequired() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void kaiming() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public HashMap serialize(Set<String> ignoreProperties) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
