/*
 * The MIT License
 *
 * Copyright 2025 ahmed.
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
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUstream;
import jcuda.jcurand.curandGenerator;
import org.fjnn.base.output.BackpropagateOutput;
import org.fjnn.base.output.BackpropagateOutputGPU;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.cuda.CudaFunctions;
import org.fjnn.cuda.CudaUtil;
import org.fjnn.util.rng;

/**
 * Genetic Parameter Optimizer
 * Evolves continuous parameters using genetic algorithms
 * 
 * @author ahmed
 */
public class GeneticParameterOptimizer {
//    
//    public static class Parameter {
//        public final String name;
//        public final float min;
//        public final float max;
//        public final float initialValue;
//        public final boolean isLog;  // Use log scale for parameter space
//        
//        public Parameter(String name, float min, float max, float initialValue) {
//            this(name, min, max, initialValue, false);
//        }
//        
//        public Parameter(String name, float min, float max, float initialValue, boolean isLog) {
//            this.name = name;
//            this.min = min;
//            this.max = max;
//            this.initialValue = initialValue;
//            this.isLog = isLog;
//        }
//    }
//    
//    public static class Individual {
//        public final float[] genes;
//        public double fitness;
//        public final Map<String, Object> metadata;
//        
//        public Individual(int paramCount) {
//            this.genes = new float[paramCount];
//            this.fitness = Double.NEGATIVE_INFINITY;
//            this.metadata = new HashMap<>();
//        }
//        
//        public Individual(float[] genes) {
//            this.genes = Arrays.copyOf(genes, genes.length);
//            this.fitness = Double.NEGATIVE_INFINITY;
//            this.metadata = new HashMap<>();
//        }
//        
//        public Individual copy() {
//            Individual copy = new Individual(this.genes);
//            copy.fitness = this.fitness;
//            copy.metadata.putAll(this.metadata);
//            return copy;
//        }
//    }
//    
//    public static class GeneticConfig {
//        
//    }
//    
//    private final Parameter[] parameters;
//    private final GeneticConfig config;
//    private final List<Individual> population;
//    private final Function<float[], Double> fitnessFunction;
//    
//    private int generation;
//    private double bestFitness;
//    private Individual bestIndividual;
//    private int stagnationCount;
//    private double currentMutationRate;
//    
//    // GPU resources
//    private CUdeviceptr populationGPU;
//    private CUdeviceptr fitnessGPU;
//    private CUdeviceptr tempGPU;
//    private curandGenerator randomGenerator;
//    private boolean gpuReady;
//    
//    public GeneticParameterOptimizer(Parameter[] parameters, 
//                                   Function<float[], Double> fitnessFunction,
//                                   GeneticConfig config) {
//        this.parameters = Arrays.copyOf(parameters, parameters.length);
//        this.fitnessFunction = fitnessFunction;
//        this.config = config;
//        this.population = new ArrayList<>(config.populationSize);
//        this.generation = 0;
//        this.bestFitness = Double.NEGATIVE_INFINITY;
//        this.currentMutationRate = config.mutationRate;
//        
//        initializePopulation();
//    }
//    
//    public GeneticParameterOptimizer(Parameter[] parameters, 
//                                   Function<float[], Double> fitnessFunction) {
//        this(parameters, fitnessFunction, new GeneticConfig());
//    }
//    
//    private void initializePopulation() {
//        for (int i = 0; i < config.populationSize; i++) {
//            Individual individual = new Individual(parameters.length);
//            
//            for (int j = 0; j < parameters.length; j++) {
//                Parameter param = parameters[j];
//                
//                if (i == 0) {
//                    // First individual uses initial values
//                    individual.genes[j] = param.initialValue;
//                } else {
//                    // Random initialization
//                    if (param.isLog) {
//                        float logMin = (float) Math.log(param.min);
//                        float logMax = (float) Math.log(param.max);
//                        float logValue = Rng.nextFloat(logMin, logMax);
//                        individual.genes[j] = (float) Math.exp(logValue);
//                    } else {
//                        individual.genes[j] = Rng.nextFloat(param.min, param.max);
//                    }
//                }
//            }
//            
//            population.add(individual);
//        }
//    }
//    
//    public void evolve(int generations) {
//        for (int i = 0; i < generations; i++) {
//            evolveStep();
//        }
//    }
//    
//    public void evolveStep() {
//        // Evaluate fitness
//        evaluatePopulation();
//        
//        // Update best individual
//        updateBest();
//        
//        // Check for stagnation
//        checkStagnation();
//        
//        // Create next generation
//        List<Individual> nextGeneration = createNextGeneration();
//        
//        // Replace population
//        population.clear();
//        population.addAll(nextGeneration);
//        
//        generation++;
//        
//        // Log progress
//        if (generation % 10 == 0) {
//            System.out.printf("Generation %d: Best fitness = %.6f, Mutation rate = %.4f%n", 
//                            generation, bestFitness, currentMutationRate);
//        }
//    }
//    
//    private void evaluatePopulation() {
//        if (config.useGPU && gpuReady) {
//            evaluatePopulationGPU();
//        } else {
//            evaluatePopulationCPU();
//        }
//    }
//    
//    private void evaluatePopulationCPU() {
//        for (Individual individual : population) {
//            if (individual.fitness == Double.NEGATIVE_INFINITY) {
//                individual.fitness = fitnessFunction.apply(individual.genes);
//            }
//        }
//    }
//    
//    private void evaluatePopulationGPU() {
//        // TODO: Implement batch GPU evaluation if fitness function supports it
//        evaluatePopulationCPU();
//    }
//    
//    private void updateBest() {
//        for (Individual individual : population) {
//            if (individual.fitness > bestFitness) {
//                bestFitness = individual.fitness;
//                bestIndividual = individual.copy();
//                stagnationCount = 0;
//            }
//        }
//    }
//    
//    private void checkStagnation() {
//        stagnationCount++;
//        
//        if (config.adaptiveMutation && stagnationCount >= config.stagnationThreshold) {
//            currentMutationRate *= config.mutationDecay;
//            currentMutationRate = Math.max(currentMutationRate, 0.001);
//            stagnationCount = 0;
//        }
//    }
//    
//    private List<Individual> createNextGeneration() {
//        List<Individual> nextGeneration = new ArrayList<>(config.populationSize);
//        
//        // Elitism - keep best individuals
//        List<Individual> elite = selectElite();
//        nextGeneration.addAll(elite);
//        
//        // Generate offspring
//        while (nextGeneration.size() < config.populationSize) {
//            Individual parent1 = selectParent();
//            Individual parent2 = selectParent();
//            
//            Individual[] offspring = crossover(parent1, parent2);
//            
//            for (Individual child : offspring) {
//                mutate(child);
//                nextGeneration.add(child);
//                
//                if (nextGeneration.size() >= config.populationSize) {
//                    break;
//                }
//            }
//        }
//        
//        return nextGeneration;
//    }
//    
//    private List<Individual> selectElite() {
//        population.sort((a, b) -> Double.compare(b.fitness, a.fitness));
//        
//        List<Individual> elite = new ArrayList<>();
//        for (int i = 0; i < Math.min(config.eliteCount, population.size()); i++) {
//            elite.add(population.get(i).copy());
//        }
//        
//        return elite;
//    }
//    
//    private Individual selectParent() {
//        switch (config.selectionType) {
//            case TOURNAMENT:
//                return tournamentSelection();
//            case ROULETTE:
//                return rouletteSelection();
//            case RANK:
//                return rankSelection();
//            default:
//                return tournamentSelection();
//        }
//    }
//    
//    private Individual tournamentSelection() {
//        Individual best = null;
//        double bestFitness = Double.NEGATIVE_INFINITY;
//        
//        for (int i = 0; i < config.tournamentSize; i++) {
//            Individual candidate = population.get(Rng.nextInt(population.size()));
//            if (candidate.fitness > bestFitness) {
//                bestFitness = candidate.fitness;
//                best = candidate;
//            }
//        }
//        
//        return best;
//    }
//    
//    private Individual rouletteSelection() {
//        // Find min fitness for offset
//        double minFitness = population.stream()
//                .mapToDouble(ind -> ind.fitness)
//                .min()
//                .orElse(0.0);
//        
//        // Calculate total fitness
//        double totalFitness = population.stream()
//                .mapToDouble(ind -> ind.fitness - minFitness + 1e-10)
//                .sum();
//        
//        double rand = Rng.nextDouble() * totalFitness;
//        double sum = 0;
//        
//        for (Individual individual : population) {
//            sum += individual.fitness - minFitness + 1e-10;
//            if (sum >= rand) {
//                return individual;
//            }
//        }
//        
//        return population.get(population.size() - 1);
//    }
//    
//    private Individual rankSelection() {
//        population.sort((a, b) -> Double.compare(a.fitness, b.fitness));
//        
//        int totalRank = population.size() * (population.size() + 1) / 2;
//        int rand = Rng.nextInt(totalRank);
//        
//        int sum = 0;
//        for (int i = 0; i < population.size(); i++) {
//            sum += i + 1;
//            if (sum >= rand) {
//                return population.get(i);
//            }
//        }
//        
//        return population.get(population.size() - 1);
//    }
//    
//    private Individual[] crossover(Individual parent1, Individual parent2) {
//        if (Rng.nextDouble() > config.crossoverRate) {
//            return new Individual[]{parent1.copy(), parent2.copy()};
//        }
//        
//        switch (config.crossoverType) {
//            case UNIFORM:
//                return uniformCrossover(parent1, parent2);
//            case ARITHMETIC:
//                return arithmeticCrossover(parent1, parent2);
//            case SINGLE_POINT:
//                return singlePointCrossover(parent1, parent2);
//            case BLEND_ALPHA:
//                return blendAlphaCrossover(parent1, parent2);
//            default:
//                return uniformCrossover(parent1, parent2);
//        }
//    }
//    
//    private Individual[] uniformCrossover(Individual parent1, Individual parent2) {
//        Individual child1 = new Individual(parameters.length);
//        Individual child2 = new Individual(parameters.length);
//        
//        for (int i = 0; i < parameters.length; i++) {
//            if (Rng.nextBoolean()) {
//                child1.genes[i] = parent1.genes[i];
//                child2.genes[i] = parent2.genes[i];
//            } else {
//                child1.genes[i] = parent2.genes[i];
//                child2.genes[i] = parent1.genes[i];
//            }
//        }
//        
//        return new Individual[]{child1, child2};
//    }
//    
//    private Individual[] arithmeticCrossover(Individual parent1, Individual parent2) {
//        Individual child1 = new Individual(parameters.length);
//        Individual child2 = new Individual(parameters.length);
//        
//        float alpha = Rng.nextFloat(0, 1);
//        
//        for (int i = 0; i < parameters.length; i++) {
//            child1.genes[i] = alpha * parent1.genes[i] + (1 - alpha) * parent2.genes[i];
//            child2.genes[i] = (1 - alpha) * parent1.genes[i] + alpha * parent2.genes[i];
//            
//            // Clamp to bounds
//            Parameter param = parameters[i];
//            child1.genes[i] = Math.max(param.min, Math.min(param.max, child1.genes[i]));
//            child2.genes[i] = Math.max(param.min, Math.min(param.max, child2.genes[i]));
//        }
//        
//        return new Individual[]{child1, child2};
//    }
//    
//    private Individual[] singlePointCrossover(Individual parent1, Individual parent2) {
//        Individual child1 = new Individual(parameters.length);
//        Individual child2 = new Individual(parameters.length);
//        
//        int crossoverPoint = Rng.nextInt(parameters.length);
//        
//        for (int i = 0; i < parameters.length; i++) {
//            if (i < crossoverPoint) {
//                child1.genes[i] = parent1.genes[i];
//                child2.genes[i] = parent2.genes[i];
//            } else {
//                child1.genes[i] = parent2.genes[i];
//                child2.genes[i] = parent1.genes[i];
//            }
//        }
//        
//        return new Individual[]{child1, child2};
//    }
//    
//    private Individual[] blendAlphaCrossover(Individual parent1, Individual parent2) {
//        Individual child1 = new Individual(parameters.length);
//        Individual child2 = new Individual(parameters.length);
//        
//        float alpha = 0.5f;
//        
//        for (int i = 0; i < parameters.length; i++) {
//            float gene1 = parent1.genes[i];
//            float gene2 = parent2.genes[i];
//            
//            float min = Math.min(gene1, gene2);
//            float max = Math.max(gene1, gene2);
//            float range = max - min;
//            
//            float low = min - alpha * range;
//            float high = max + alpha * range;
//            
//            // Clamp to parameter bounds
//            Parameter param = parameters[i];
//            low = Math.max(param.min, low);
//            high = Math.min(param.max, high);
//            
//            child1.genes[i] = Rng.nextFloat(low, high);
//            child2.genes[i] = Rng.nextFloat(low, high);
//        }
//        
//        return new Individual[]{child1, child2};
//    }
//    
//    private void mutate(Individual individual) {
//        for (int i = 0; i < parameters.length; i++) {
//            if (Rng.nextDouble() < currentMutationRate) {
//                mutateGene(individual, i);
//            }
//        }
//    }
//    
//    private void mutateGene(Individual individual, int geneIndex) {
//        Parameter param = parameters[geneIndex];
//        float currentValue = individual.genes[geneIndex];
//        
//        switch (config.mutationType) {
//            case GAUSSIAN:
//                gaussianMutation(individual, geneIndex, param, currentValue);
//                break;
//            case UNIFORM:
//                uniformMutation(individual, geneIndex, param);
//                break;
//            case POLYNOMIAL:
//                polynomialMutation(individual, geneIndex, param, currentValue);
//                break;
//        }
//    }
//    
//    private void gaussianMutation(Individual individual, int geneIndex, Parameter param, float currentValue) {
//        float range = param.max - param.min;
//        float sigma = (float) (config.mutationStrength * range);
//        
//        if (param.isLog) {
//            // Log-space mutation
//            float logValue = (float) Math.log(currentValue);
//            float logMin = (float) Math.log(param.min);
//            float logMax = (float) Math.log(param.max);
//            float logRange = logMax - logMin;
//            float logSigma = (float) (config.mutationStrength * logRange);
//            
//            float newLogValue = logValue + Rng.nextGaussian() * logSigma;
//            newLogValue = Math.max(logMin, Math.min(logMax, newLogValue));
//            individual.genes[geneIndex] = (float) Math.exp(newLogValue);
//        } else {
//            float newValue = currentValue + Rng.nextGaussian() * sigma;
//            individual.genes[geneIndex] = Math.max(param.min, Math.min(param.max, newValue));
//        }
//    }
//    
//    private void uniformMutation(Individual individual, int geneIndex, Parameter param) {
//        if (param.isLog) {
//            float logMin = (float) Math.log(param.min);
//            float logMax = (float) Math.log(param.max);
//            float logValue = Rng.nextFloat(logMin, logMax);
//            individual.genes[geneIndex] = (float) Math.exp(logValue);
//        } else {
//            individual.genes[geneIndex] = Rng.nextFloat(param.min, param.max);
//        }
//    }
//    
//    private void polynomialMutation(Individual individual, int geneIndex, Parameter param, float currentValue) {
//        float eta = 20.0f; // Distribution index
//        float range = param.max - param.min;
//        
//        float delta1 = (currentValue - param.min) / range;
//        float delta2 = (param.max - currentValue) / range;
//        
//        float rnd = Rng.nextFloat();
//        float mutPow = 1.0f / (eta + 1.0f);
//        
//        float deltaq;
//        if (rnd <= 0.5) {
//            float val = 2.0f * rnd + (1.0f - 2.0f * rnd) * Math.pow(1.0f - delta1, eta + 1);
//            deltaq = (float) Math.pow(val, mutPow) - 1.0f;
//        } else {
//            float val = 2.0f * (1.0f - rnd) + 2.0f * (rnd - 0.5f) * Math.pow(1.0f - delta2, eta + 1);
//            deltaq = 1.0f - (float) Math.pow(val, mutPow);
//        }
//        
//        float newValue = currentValue + deltaq * range;
//        individual.genes[geneIndex] = Math.max(param.min, Math.min(param.max, newValue));
//    }
//    
//    public void prepareGPU() {
//        if (gpuReady) return;
//        
//        int deviceId = CudaEngine.getThreadDeviceId();
//        if (deviceId == -1) {
//            throw new RuntimeException("No GPU context available");
//        }
//        
//        // Allocate GPU memory
//        long populationSize = config.populationSize * parameters.length;
//        populationGPU = CudaUtil.createFloat(populationSize);
//        fitnessGPU = CudaUtil.createFloat(config.populationSize);
//        tempGPU = CudaUtil.createFloat(populationSize * 2);
//        
//        // Create random generator
//        randomGenerator = CudaEngine.getCurandGenerator(deviceId);
//        
//        gpuReady = true;
//    }
//    
//    public void freeGPU() {
//        if (!gpuReady) return;
//        
//        CudaUtil.free(populationGPU);
//        CudaUtil.free(fitnessGPU);
//        CudaUtil.free(tempGPU);
//        
//        populationGPU = null;
//        fitnessGPU = null;
//        tempGPU = null;
//        randomGenerator = null;
//        
//        gpuReady = false;
//    }
//    
//    // Getter methods
//    public Individual getBestIndividual() {
//        return bestIndividual != null ? bestIndividual.copy() : null;
//    }
//    
//    public double getBestFitness() {
//        return bestFitness;
//    }
//    
//    public int getGeneration() {
//        return generation;
//    }
//    
//    public List<Individual> getPopulation() {
//        List<Individual> copy = new ArrayList<>();
//        for (Individual ind : population) {
//            copy.add(ind.copy());
//        }
//        return copy;
//    }
//    
//    public Parameter[] getParameters() {
//        return Arrays.copyOf(parameters, parameters.length);
//    }
//    
//    public Map<String, Float> getBestParameters() {
//        if (bestIndividual == null) return null;
//        
//        Map<String, Float> params = new HashMap<>();
//        for (int i = 0; i < parameters.length; i++) {
//            params.put(parameters[i].name, bestIndividual.genes[i]);
//        }
//        return params;
//    }
//    
//    public void setFitness(Individual individual, double fitness) {
//        individual.fitness = fitness;
//    }
//    
//    public void reset() {
//        population.clear();
//        generation = 0;
//        bestFitness = Double.NEGATIVE_INFINITY;
//        bestIndividual = null;
//        stagnationCount = 0;
//        currentMutationRate = config.mutationRate;
//        initializePopulation();
//    }
}