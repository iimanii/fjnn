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

import org.fjnn.util.Rng;

/**
 *
 * @author ahmed
 */
public class GeneticNetworkConfig {
    
    public enum mutation {
        ADD_NODE (0.1f),
        ADD_CONNECTION (0.1f),
        ENABLE_CONNECTION (0.5f),
        DISABLE_CONNECTION (0.5f);
        
        double value;
        double prob;
        double comm;
        
        mutation(float value) {
            this.value = value;
        }
        
        static void update() {
            double sum = 0;
            
            mutation[] list = mutation.values();
            
            for(mutation m : list)
                sum += m.value;
            
            float commulative = 0;
            
            for(mutation m : list) {
                m.prob = m.value / sum;
                commulative += m.prob;                
                m.comm = commulative;
            }
        }
        
        static {
            update();
        }
        
        public double getProb() {
            return prob;
        }
    };
    
    public static mutation getRandomMutation() {
        float r = Rng.nextFloat();
        
        for(mutation m : mutation.values())
            if(m.comm > r)
                return m;
        
        return null;
    }
    
    public static void setMutationProbability(double ADD_NODE, double ADD_CONNECTION, double ENABLE_CONNECTION, double DISABLE_CONNECTION) {
        mutation.ADD_NODE.value = ADD_NODE;
        mutation.ADD_CONNECTION.value = ADD_CONNECTION;
        mutation.ENABLE_CONNECTION.value = ENABLE_CONNECTION;
        mutation.DISABLE_CONNECTION.value = DISABLE_CONNECTION;
        
        mutation.update();
    }
}
