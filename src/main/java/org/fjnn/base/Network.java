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
package org.fjnn.base;


import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author ahmed
 */

public abstract class Network {
    protected Map<String, Object> properties;
    
    public Network() {
        this.properties = new HashMap<>();
    }
    
    /**
     * 
     * @param name
     * @return 
     */
    public Object getProperty(String name) {
        return properties.get(name);
    }
    
    /**
     * 
     * @param name
     * @param object 
     */
    public void setProperty(String name, Object object) {
        properties.put(name, object);
    }
    
    /**
     * 
     * @param name
     * @return 
     */
    public boolean hasProperty(String name) {
        return properties.containsKey(name);
    }
    
    /**
     * 
     * @return Number of inputs
     */
    public abstract int getInputSize();
    
    /**
     * 
     * @return Number of outputs
     */
    public abstract int getOutputSize();    
}
