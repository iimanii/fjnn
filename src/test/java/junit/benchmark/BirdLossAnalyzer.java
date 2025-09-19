/*
 * The MIT License
 *
 * Copyright 2024 ahmed.
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
package junit.benchmark;

import org.fjnn.loss.BirdLoss;
import org.fjnn.loss.MeanSquareError;

/**
 * Analyzer for BirdLoss function behavior across different parameter combinations.
 * Helps understand how alpha and beta parameters affect the loss function shape.
 * 
 * @author ahmed
 */
public class BirdLossAnalyzer {
    
    /**
     * Java 8 compatible string repeat function
     */
    private static String repeatString(String str, int count) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < count; i++) {
            sb.append(str);
        }
        return sb.toString();
    }
    
    /**
     * Generate and display BirdLoss values across different parameter combinations
     */
    public static void analyzeBirdLoss() {
        System.out.println("=== BirdLoss Parameter Analysis ===");
        System.out.println("Function: L(x) = alpha * ln(beta*x^2 + 1)");
        System.out.println("Range: x in [-2, 2]\n");
        
        // Parameter combinations to test
        float[] alphaValues = {0.1f, 0.15f, 0.25f, 0.5f, 0.75f};
        float[] betaValues = {10.0f, 50.0f, 100.0f, 500.0f, 1000.0f};
        
        // Test points across the range [-2, 2]
        float[] testPoints = {-2.0f, -1.5f, -1.0f, -0.5f, -0.1f, -0.01f, 0.0f, 
                             0.01f, 0.1f, 0.5f, 1.0f, 1.5f, 2.0f};
        
        // Compare with MSE for reference
        MeanSquareError mse = new MeanSquareError();
        
        // Test each alpha value with default beta
        System.out.println("ALPHA SENSITIVITY (beta = 100):");
        System.out.printf("%-6s", "x");
        System.out.printf("%10s", "a=0.1");
        System.out.printf("%10s", "a=0.15");
        System.out.printf("%10s", "a=0.25");
        System.out.printf("%10s", "a=0.5");
        System.out.printf("%10s", "a=0.75");
        System.out.printf("%10s", "MSE");
        System.out.println();
        System.out.println(repeatString("-", 66));
        
        for (float x : testPoints) {
            System.out.printf("%-6.2f", x);
            for (float alpha : alphaValues) {
                BirdLoss bird = new BirdLoss(alpha, 100.0f);
                float[] pred = {x};
                float[] target = {0.0f};
                float loss = bird.compute(pred, target);
                System.out.printf("%10.4f", loss);
            }
            // Add MSE for comparison
            float[] pred = {x};
            float[] target = {0.0f};
            float mseLoss = mse.compute(pred, target);
            System.out.printf("%10.4f", mseLoss);
            System.out.println();
        }
        
        System.out.println("\n" + repeatString("=", 70) + "\n");
        
        // Test each beta value with default alpha
        System.out.println("BETA SENSITIVITY (alpha = 0.5):");
        System.out.printf("%-6s", "x");
        System.out.printf("%10s", "b=10");
        System.out.printf("%10s", "b=50");
        System.out.printf("%10s", "b=100");
        System.out.printf("%10s", "b=500");
        System.out.printf("%10s", "b=1000");
        System.out.printf("%10s", "MSE");
        System.out.println();
        System.out.println(repeatString("-", 66));
        
        for (float x : testPoints) {
            System.out.printf("%-6.2f", x);
            for (float beta : betaValues) {
                BirdLoss bird = new BirdLoss(0.5f, beta);
                float[] pred = {x};
                float[] target = {0.0f};
                float loss = bird.compute(pred, target);
                System.out.printf("%10.4f", loss);
            }
            // Add MSE for comparison
            float[] pred = {x};
            float[] target = {0.0f};
            float mseLoss = mse.compute(pred, target);
            System.out.printf("%10.4f", mseLoss);
            System.out.println();
        }
    }
    
    /**
     * Analyze derivative behavior (gradient) across parameters
     */
    public static void analyzeGradients() {
        System.out.println("\n" + repeatString("=", 70));
        System.out.println("=== BirdLoss Gradient Analysis ===");
        System.out.println("Derivative: dL/dx = 2*alpha*beta*x / (beta*x^2 + 1)");
        System.out.println("Range: x in [-2, 2]\n");
        
        float[] testPoints = {-2.0f, -1.0f, -0.5f, -0.1f, 0.0f, 0.1f, 0.5f, 1.0f, 2.0f};
        float[] alphaValues = {0.1f, 0.25f, 0.75f};
        float[] betaValues = {50.0f, 100.0f, 500.0f};
        
        MeanSquareError mse = new MeanSquareError();
        
        // Gradient comparison with different alphas
        System.out.println("GRADIENT COMPARISON - ALPHA VARIATION (beta = 100):");
        System.out.printf("%-6s", "x");
        System.out.printf("%12s", "a=0.1");
        System.out.printf("%12s", "a=0.25");
        System.out.printf("%12s", "a=0.75");
        System.out.printf("%12s", "MSE_grad");
        System.out.println();
        System.out.println(repeatString("-", 54));
        
        for (float x : testPoints) {
            System.out.printf("%-6.1f", x);
            for (float alpha : alphaValues) {
                BirdLoss bird = new BirdLoss(alpha, 100.0f);
                float[] pred = {x};
                float[] target = {0.0f};
                float[] grad = bird.derivative(pred, target);
                System.out.printf("%12.4f", grad[0]);
            }
            // MSE gradient for comparison
            float[] pred = {x};
            float[] target = {0.0f};
            float[] mseGrad = mse.derivative(pred, target);
            System.out.printf("%12.4f", mseGrad[0]);
            System.out.println();
        }
        
        System.out.println("\nGRADIENT COMPARISON - BETA VARIATION (alpha = 0.5):");
        System.out.printf("%-6s", "x");
        System.out.printf("%12s", "b=50");
        System.out.printf("%12s", "b=100");
        System.out.printf("%12s", "b=500");
        System.out.printf("%12s", "MSE_grad");
        System.out.println();
        System.out.println(repeatString("-", 54));
        
        for (float x : testPoints) {
            System.out.printf("%-6.1f", x);
            for (float beta : betaValues) {
                BirdLoss bird = new BirdLoss(0.5f, beta);
                float[] pred = {x};
                float[] target = {0.0f};
                float[] grad = bird.derivative(pred, target);
                System.out.printf("%12.4f", grad[0]);
            }
            // MSE gradient for comparison
            float[] pred = {x};
            float[] target = {0.0f};
            float[] mseGrad = mse.derivative(pred, target);
            System.out.printf("%12.4f", mseGrad[0]);
            System.out.println();
        }
    }
    
    /**
     * Analyze parameter effects on small vs large errors
     */
    public static void analyzeErrorRegimes() {
        System.out.println("\n" + repeatString("=", 70));
        System.out.println("=== Error Regime Analysis ===");
        System.out.println("Comparing BirdLoss vs MSE for different error magnitudes\n");
        
        float[] errors = {0.001f, 0.01f, 0.05f, 0.1f, 0.2f, 0.5f, 1.0f, 2.0f};
        float[][] paramCombos = {
            {0.1f, 100.0f},   // Conservative
            {0.15f, 500.0f},  // Gentle
            {0.25f, 1000.0f}, // Default
            {0.75f, 200.0f}   // Aggressive
        };
        
        MeanSquareError mse = new MeanSquareError();
        
        System.out.printf("%-8s", "Error");
        System.out.printf("%10s", "MSE");
        System.out.printf("%12s", "Bird(0.1,100)");
        System.out.printf("%13s", "Bird(0.15,500)");
        System.out.printf("%15s", "Bird(0.25,1000)");
        System.out.printf("%13s", "Bird(0.75,200)");
        System.out.println();
        System.out.println(repeatString("-", 83));
        
        for (float error : errors) {
            System.out.printf("%-8.3f", error);
            
            // MSE
            float[] pred = {error};
            float[] target = {0.0f};
            float mseLoss = mse.compute(pred, target);
            System.out.printf("%10.4f", mseLoss);
            
            // Different BirdLoss configurations
            for (int i = 0; i < paramCombos.length; i++) {
                float[] params = paramCombos[i];
                BirdLoss bird = new BirdLoss(params[0], params[1]);
                float birdLoss = bird.compute(pred, target);
                if (i == 0) {
                    System.out.printf("%12.4f", birdLoss);  // Bird(0.1,100)
                } else if (i == 1) {
                    System.out.printf("%13.4f", birdLoss);  // Bird(0.15,500)
                } else if (i == 2) {
                    System.out.printf("%15.4f", birdLoss);  // Bird(0.25,1000)
                } else {
                    System.out.printf("%13.4f", birdLoss);  // Bird(0.75,200)
                }
            }
            System.out.println();
        }
        
        // Show ratios compared to MSE
        System.out.println("\nRATIO TO MSE (BirdLoss/MSE):");
        System.out.printf("%-8s", "Error");
        System.out.printf("%12s", "Bird(0.1,100)");
        System.out.printf("%13s", "Bird(0.15,500)");
        System.out.printf("%15s", "Bird(0.25,1000)");
        System.out.printf("%13s", "Bird(0.75,200)");
        System.out.println();
        System.out.println(repeatString("-", 73));
        
        for (float error : errors) {
            System.out.printf("%-8.3f", error);
            
            float[] pred = {error};
            float[] target = {0.0f};
            float mseLoss = mse.compute(pred, target);
            
            for (int i = 0; i < paramCombos.length; i++) {
                float[] params = paramCombos[i];
                BirdLoss bird = new BirdLoss(params[0], params[1]);
                float birdLoss = bird.compute(pred, target);
                float ratio = mseLoss > 0 ? birdLoss / mseLoss : Float.POSITIVE_INFINITY;
                if (i == 0) {
                    System.out.printf("%12.2f", ratio);  // Bird(0.1,100)
                } else if (i == 1) {
                    System.out.printf("%13.2f", ratio);  // Bird(0.15,500)
                } else if (i == 2) {
                    System.out.printf("%15.2f", ratio);  // Bird(0.25,1000)
                } else {
                    System.out.printf("%13.2f", ratio);  // Bird(0.75,200)
                }
            }
            System.out.println();
        }
    }
    
    /**
     * Show mathematical properties and behavior summary
     */
    public static void showMathematicalProperties() {
        System.out.println("\n" + repeatString("=", 70));
        System.out.println("=== Mathematical Properties ===");
        System.out.println();
        
        System.out.println("BirdLoss Function: L(x) = alpha * ln(beta*x^2 + 1)");
        System.out.println("Derivative: dL/dx = 2*alpha*beta*x / (beta*x^2 + 1)");
        System.out.println();
        
        System.out.println("Parameter Effects:");
        System.out.println("* alpha: Scales the overall loss magnitude");
        System.out.println("  - Higher alpha -> Higher loss values");
        System.out.println("  - Typical range: 0.1 - 0.75 (keeping alpha < 1.0)");
        System.out.println();
        
        System.out.println("* beta: Controls sensitivity to small errors");
        System.out.println("  - Higher beta -> More sensitive to small errors");
        System.out.println("  - Higher beta -> Less sensitive to large errors (log saturation)");
        System.out.println("  - Typical range: 50 - 1000");
        System.out.println();
        
        System.out.println("Key Properties:");
        System.out.println("* Symmetric: L(-x) = L(x)");
        System.out.println("* Always positive: L(x) >= 0");
        System.out.println("* Minimum at x = 0: L(0) = alpha * ln(1) = 0");
        System.out.println("* Logarithmic growth prevents explosion for large errors");
        System.out.println("* Maintains significant loss for small errors (vs MSE's quadratic decay)");
        System.out.println();
        
        System.out.println("Comparison to MSE:");
        System.out.println("* Small errors: BirdLoss >> MSE (maintains learning pressure)");
        System.out.println("* Large errors: BirdLoss < MSE (robust to outliers)");
        System.out.println("* Crossover point depends on alpha and beta parameters");
    }
    
    /**
     * Main analysis method - runs all analyses
     */
    public static void runCompleteAnalysis() {
        analyzeBirdLoss();
        analyzeGradients();
        analyzeErrorRegimes();
        showMathematicalProperties();
        
        System.out.println("\n" + repeatString("=", 70));
        System.out.println("Analysis complete!");
        System.out.println("Recommended starting parameters (alpha < 1.0):");
        System.out.println("* Conservative: alpha=0.1, beta=100 (gentle learning pressure)");
        System.out.println("* Gentle: alpha=0.15, beta=500 (mild learning pressure)");
        System.out.println("* Balanced: alpha=0.25, beta=1000 (default, good general purpose)");
        System.out.println("* Aggressive: alpha=0.75, beta=200 (strong learning pressure)");
    }
    
    /**
     * Example usage and testing
     */
    public static void main(String[] args) {
        runCompleteAnalysis();
    }
}