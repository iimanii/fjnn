/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fjnn.trainer.genetic;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicLong;
import jcuda.driver.CUdeviceptr;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.network.NeuralNetwork;
import org.fjnn.trainer.genetic.GeneticTrainerConfig.ComputeMode;
import org.fjnn.trainer.genetic.TrainingSet.TrainingResult;
import org.fjnn.util.Rng;

/**
 *
 * @author ahmed
 */
public class GeneticTrainer {

    public static class trainPerformance {
        public float computeTime;
        public float mutationTime;
        public float finalizationTime;
        public float totalTime;

        public trainPerformance(float computeTime, float mutationTime, float finalizationTime) {
            this.computeTime = computeTime;
            this.mutationTime = mutationTime;
            this.finalizationTime = finalizationTime;
            this.totalTime = computeTime + mutationTime;
        }
    }
    
    /* maximum networks in pool */
    public final int poolsize;
    
    /* first network node to be submited */
    public final NeuralNetwork initial;
    
    final TrainingSet trainingSet;
    
    /* best performing network */
    NeuralNetwork best;
            
    /* best fitness based on last epoch */
    private double bestFitness;
    
    /* calculated average fitness based on last epoch */
    private double averageFitness;

    /* mutated each generation */
    List<NeuralNetwork> pool;

    /* how many times did we mutate / crossover */
    int generation;
    
    /* how many times did we run over the whole training set */
    int epoch;
    
    /* last id in the pool */
    AtomicLong lastID;
    
    /* mutation */
    GeneticMutation mutation;
    
    final GeneticTrainerConfig config;
    
    trainPerformance lastPerformance;
    
    ExecutorService threadPool;
    
    public static final String ID_PROP = "id";
    public static final String PARENT_PROP = "parent";
    public static final String GENERATION_PROP = "generation";
    public static final String FITNESS_PROP = "fitness";
    public static final String FITNESS_SQUARED_PROP = "fitness-squared";
    public static final String DETAILED_FITNESS_PROP = "detailed-fitness";
    public static final String COMMULATIVE = "commulative:%s";
    public static final String MUTATION_PROP = "mutation";
    public static final String CHILD_COUNTER = "child-counter";
    public static final String ABSOLUTE_Z_SCORE = "absolute-z-score";
    public static final String SQUARE_Z_SCORE = "square-z-score";
    public static final String OVERALL_FITNESS_PROP = "overall-fitness";
    public static final String OVERALL_DETAILED_FITNESS_PROP = "overall-detailed-fitness";
    
    public static final float MUTATION_CONSTANT = 2;
    
    private boolean ownExecutor;
    
    public GeneticTrainer(NeuralNetwork base, int poolsize, TrainingSet trainingSet, GeneticTrainerConfig config) {
        this(Arrays.asList(base), poolsize, trainingSet, config);
    }
    
    public GeneticTrainer(List<NeuralNetwork> networks, int poolsize, TrainingSet trainingSet, GeneticTrainerConfig config) {
        config.checkTournamentSize(poolsize);
        
        this.poolsize = poolsize;
        this.pool = new ArrayList<>(networks);
        this.lastID = new AtomicLong(config.startingID);
        this.threadPool = Executors.newCachedThreadPool();
        this.ownExecutor = true;
        this.trainingSet = trainingSet;
        this.initial = networks.get(0);
        this.mutation = new GeneticMutation(5, 0.5, 0.00001, 1);
        this.config = config;
        this.generation = config.startingGeneration;
        this.epoch = config.startingEpoch;
        
        /* first calculate all fitness */
        ArrayList<Future<?>> futures = new ArrayList<>();
        
        for(NeuralNetwork n : networks) {
            if(!n.hasProperty(ID_PROP))
                n.setProperty(ID_PROP, lastID.incrementAndGet());
        }
        
        int missing = poolsize - networks.size();
        for(int i=0; i < missing; i++) {
            NeuralNetwork n = this.initial.copy(false);
            n.setProperty(ID_PROP, lastID.incrementAndGet());
            n.randomize(-1, 1);
            pool.add(n);
        }
        
        for(NeuralNetwork n : pool) {
            if(!n.hasProperty(FITNESS_PROP)) {
                futures.add(threadPool.submit(() -> {
                    TrainingResult output = compute(n, 0);

                    n.setProperty(FITNESS_PROP, output.fitness);
                    n.setProperty(DETAILED_FITNESS_PROP, output.detailed);
                }));
            }
        }
        
        waitForAll(futures);        
        
        ArrayList<NeuralNetwork> sort = new ArrayList<>(pool);    
        if(trainingSet.getBatchCount() > 1) {
            /* calculate fitness for the pool over the whole dataset */
            calculateOverallFitness(sort);
    
            GeneticUtil.sortDesc(sort, OVERALL_FITNESS_PROP);
            this.best = sort.get(0);
            this.bestFitness = (double) best.getProperty(OVERALL_FITNESS_PROP);
            this.averageFitness = calculateAverageFitness(sort, OVERALL_FITNESS_PROP);
        } else {
            GeneticUtil.sortDesc(sort, FITNESS_PROP);
            this.best = sort.get(0);
            this.bestFitness = (double) best.getProperty(FITNESS_PROP);
            this.averageFitness = calculateAverageFitness(sort, FITNESS_PROP);
        }
    }
    
    private TrainingResult compute(NeuralNetwork n, int batch) {
        float[] result;
        
        int batchsize = batch < 0 ? trainingSet.getAllCount() : trainingSet.getCount(batch);
        
        switch(config.computeMode) {
            case float_buffer:
                FloatBuffer bufferInput = batch < 0 ? trainingSet.getAllBufferInput() : trainingSet.getBufferInput(batch);
                result = n.compute(bufferInput, batchsize);
                break;
            case gpu:
                if(!n.gpuReady())
                    n.prepareGPU(config.deviceId);
                
                CUdeviceptr gpuInput = batch == -1 ? trainingSet.getAllGPUInput(config.deviceId) : trainingSet.getGPUInput(config.deviceId, batch);
                result = n.computeGPU(gpuInput, batchsize);
                break;                
            case gpu_all:
                if(!n.gpuReady())
                    n.prepareGPU();
                
                gpuInput = batch < 0 ? trainingSet.getAllGPUInput(n.getGPUDeviceId()) : trainingSet.getGPUInput(n.getGPUDeviceId(), batch);
                result = n.computeGPU(gpuInput, batchsize);
                break;                
            default:
                float[] input = trainingSet.getInput(batch);
                result = n.compute(input);
        }
        
        return batch < 0 ? trainingSet.calculateAllFitness(result) : trainingSet.calculateFitness(result, batch);
    }
    
    public void train() {
        trainPerformance currentTrainPerformance = new trainPerformance(0, 0, 0);
        
        for(int i=0; i < trainingSet.getBatchCount(); i++) {
            long time_1 = System.nanoTime();

            generation++;

            String property = FITNESS_PROP;

            switch(config.criteria) {
                case absoluteZscore:
                    GeneticUtil.ArrayZScore(pool, DETAILED_FITNESS_PROP, ABSOLUTE_Z_SCORE, true);
                    property = ABSOLUTE_Z_SCORE;
                    break;
                case squareFitness:
                    GeneticUtil.squareProperty(pool, FITNESS_PROP, FITNESS_SQUARED_PROP);
                    property = FITNESS_SQUARED_PROP;
                    break;
                case zscore:
                    GeneticUtil.ArrayZScore(pool, DETAILED_FITNESS_PROP, SQUARE_Z_SCORE, false);
                    property = SQUARE_Z_SCORE;
                    break;
            }

            if(!config.useStaticMutation)
                mutation.update(pool, property);

            double maxMutation = config.useStaticMutation ? config.staticMutationValue : Math.min(mutation.get() * 2, 1);
            
            /* crossover mutate */
            boolean crossOverGPU = config.crossover == ComputeMode.gpu && config.computeMode != ComputeMode.gpu_all;

            List<NeuralNetwork> newGeneration = config.useTournamentSelection ? 
                    createNewGenerationTS(maxMutation, config.tournametSelectionSize, property, crossOverGPU) : 
                    createNewGenerationRW(maxMutation, property, crossOverGPU);

            long time_2 = System.nanoTime();

            /* first calculate all fitness */
            ArrayList<Future<?>> futures = new ArrayList<>();
            newGeneration.addAll(pool);
            
            int batchId = i;
            
            for(NeuralNetwork n : newGeneration) {
                if(trainingSet.getBatchCount() > 1 || !n.hasProperty(FITNESS_PROP)) {
                    futures.add(threadPool.submit(() -> {                    
                        TrainingResult output = compute(n, batchId);

                        n.setProperty(FITNESS_PROP, output.fitness);
                        n.setProperty(DETAILED_FITNESS_PROP, output.detailed);
                    }));
                }
            }

            waitForAll(futures);

            /* pick top performing networks as next generation */
            GeneticUtil.sortDesc(newGeneration, FITNESS_PROP);
            pool = new ArrayList<>(newGeneration.subList(0, poolsize));

            futures = new ArrayList<>();

            if(crossOverGPU) {
                for(int j=0; j < poolsize; j++) {
                    NeuralNetwork n = newGeneration.get(j);
                    futures.add(threadPool.submit(() -> {
                        n.ensureCPU();
                    }));
                }
            }

            if(config.computeMode == ComputeMode.gpu || config.computeMode == ComputeMode.gpu_all) {
                for(int j=poolsize; j < newGeneration.size(); j++) {
                    NeuralNetwork n = newGeneration.get(j);
                    futures.add(threadPool.submit(() -> {
                        n.freeGPU();
                    }));
                }
            }

            waitForAll(futures);

            long time_3 = System.nanoTime();
            currentTrainPerformance.computeTime += (time_3 - time_2) / 1e6f;
            currentTrainPerformance.mutationTime += (time_2 - time_1) / 1e6f;
        }
        
        long time_4 = System.nanoTime();
        
        ArrayList<NeuralNetwork> sort = new ArrayList<>(pool);  
        if(trainingSet.getBatchCount() > 1) {
            /* calculate fitness for the pool over the whole dataset */
            calculateOverallFitness(sort);
      
            GeneticUtil.sortDesc(sort, OVERALL_FITNESS_PROP);
            this.best = sort.get(0);
            this.bestFitness = (double) best.getProperty(OVERALL_FITNESS_PROP);
            this.averageFitness = calculateAverageFitness(sort, OVERALL_FITNESS_PROP);
        } else {
            GeneticUtil.sortDesc(sort, FITNESS_PROP);
            this.best = sort.get(0);
            this.bestFitness = (double) best.getProperty(FITNESS_PROP);
            this.averageFitness = calculateAverageFitness(sort, FITNESS_PROP);
        }
        
        long time_5 = System.nanoTime();
        
        currentTrainPerformance.finalizationTime += (time_5 - time_4) / 1e6f;
        lastPerformance = currentTrainPerformance;
        epoch++;
    }
    
    public trainPerformance getLastPerformanceCounters() {
        return lastPerformance;
    }
    
    /* using tournament selection */
    private List<NeuralNetwork> createNewGenerationTS(double maxMutation, int tournamentSize, String property, boolean gpu) {
        List<NeuralNetwork> result = new ArrayList<>();

        ArrayList<Future<?>> futures = new ArrayList<>();
        
        /* crossover mutate */
        for(int i=0; i < poolsize; i++) {
            NeuralNetwork n = best.copy(false, !gpu);
            n.setProperty(ID_PROP, lastID.incrementAndGet());
            result.add(n);
            double m = Rng.nextDouble(0, maxMutation);
            
            NeuralNetwork n0 = GeneticUtil.tournamentPick(pool, tournamentSize, property);
            NeuralNetwork n1 = GeneticUtil.tournamentPick(pool, tournamentSize, property, n0);
            
            /* select parents */
            futures.add(threadPool.submit(() -> {    
                if(gpu)
                    n.crossOverMutateGPU(n0, n1, -MUTATION_CONSTANT, MUTATION_CONSTANT, m, true);
                else
                    n.crossOverMutate(n0, n1, -MUTATION_CONSTANT, MUTATION_CONSTANT, m);
                
                GeneticUtil.incrementProperty(n0, CHILD_COUNTER);
                GeneticUtil.incrementProperty(n1, CHILD_COUNTER);
                
                String parent = String.format("[%s]%.3f x [%s]%.3f [%.3f]",
                                    n0.getProperty(ID_PROP),
                                    n0.getProperty(FITNESS_PROP),
                                    n1.getProperty(ID_PROP),
                                    n1.getProperty(FITNESS_PROP),
                                    m*100);
                
                n.setProperty(GENERATION_PROP, generation);
                n.setProperty(MUTATION_PROP, m);
                n.setProperty(PARENT_PROP, parent);
            }));
        }

        waitForAll(futures);
        
        return result;
    }
    
    /* using roulette wheel */
    private List<NeuralNetwork> createNewGenerationRW(double maxMutation, String property, boolean gpu) {
        List<NeuralNetwork> result = new ArrayList<>();

        ArrayList<Future<?>> futures = new ArrayList<>();
        
        String commulativeProprety = String.format(COMMULATIVE, property);
        GeneticUtil.commulativeDestribution(pool, property, commulativeProprety);
        GeneticUtil.sortAsc(pool, commulativeProprety);
        
        /* crossover mutate */
        for(int i=0; i < poolsize; i++) {
            NeuralNetwork n = best.copy(false, !gpu);
            n.setProperty(ID_PROP, lastID.incrementAndGet());
            double m = Rng.nextDouble(0, maxMutation);
            result.add(n);
            
            NeuralNetwork n0 = GeneticUtil.pickRandom(pool, commulativeProprety);
            NeuralNetwork n1 = GeneticUtil.pickRandom(pool, commulativeProprety, n0);
            
            /* select parents */
            futures.add(threadPool.submit(() -> {
                if(gpu)
                    n.crossOverMutateGPU(n0, n1, -MUTATION_CONSTANT, MUTATION_CONSTANT, m, true);
                else
                    n.crossOverMutate(n0, n1, -MUTATION_CONSTANT, MUTATION_CONSTANT, m);

                GeneticUtil.incrementProperty(n0, CHILD_COUNTER);
                GeneticUtil.incrementProperty(n1, CHILD_COUNTER);
                
                String parent = String.format("[%s]%.3f x [%s]%.3f [%.3f]",
                                    n0.getProperty(ID_PROP),
                                    n0.getProperty(FITNESS_PROP),
                                    n1.getProperty(ID_PROP),
                                    n1.getProperty(FITNESS_PROP),
                                    m*100);
                
                n.setProperty(GENERATION_PROP, generation);
                n.setProperty(MUTATION_PROP, m);
                n.setProperty(PARENT_PROP, parent);
            }));
        }

        waitForAll(futures);
        
        return result;        
    }
    
    private void calculateOverallFitness(List<NeuralNetwork> networks) {
        ArrayList<Future<?>> futures = new ArrayList<>();
        
        for(NeuralNetwork n : networks) {
            futures.add(threadPool.submit(() -> {
                TrainingResult output = compute(n, -1);

                n.setProperty(OVERALL_FITNESS_PROP, output.fitness);
                n.setProperty(OVERALL_DETAILED_FITNESS_PROP, output.detailed);
            }));
        }
        
        waitForAll(futures);
    }

    private double calculateAverageFitness(List<NeuralNetwork> networks, String property) {
        double total = 0;
        double count = 0;
        
        for(NeuralNetwork n : networks) {
            total += (double) n.getProperty(property);
            count++;
        }
        
        return total / count;
    }

    public void setExecutor(ExecutorService executor) {
        this.threadPool = executor;
        this.ownExecutor = false;
    }
    
    public double getDiversity() {
        double score = 0;
        ArrayList<Future<Float>> futures = new ArrayList<>();
        
        int size = pool.size();
        for(int i=0; i < size; i++) {
            for(int j=i+1; j < size; j++) {
                
                final int x = i;
                final int y = j;
                futures.add(threadPool.submit(() -> {
                    return pool.get(x).compare(pool.get(y));
                }));
            }
        }
        
        for(Future<Float> f : futures)
            try {
                score += f.get();
            } catch (ExecutionException | InterruptedException ex) {
                throw new RuntimeException(ex);
            }
        
        score /= (size * (size-1)) / 2;
        return score;
    }
    
    public double getInitialFitness() {
        return (double) initial.getProperty(FITNESS_PROP);
    }
    
    public double getBestFitness() {
        return bestFitness;
    }
    
    /* returns average fitness of generation size */
    public double getAverageFitness() {
        return averageFitness;
    }
    
    public NeuralNetwork getBest() {
        return best;
    }
    
    public List<NeuralNetwork> getPool() {
        return new ArrayList<>(pool);
    }
    
    public double getMutationRate() {
        return mutation.get();
    }
    
    private void randomize(NeuralNetwork n0, NeuralNetwork n1, double mutation) {
        n0.crossOverMutate(n0, n1, -1, 1, mutation);
    }

    public int getGeneration() {
        return generation;
    }
    
    public int getEpoch() {
        return epoch;
    }
    
    static void waitForAll(ArrayList<Future<?>> futures) {
        for(Future f : futures) {
            try {
                f.get();
            } catch (InterruptedException | ExecutionException ex) {
                ex.getCause().printStackTrace(System.out);
                throw new RuntimeException("Shit happened: " + ex.toString());
            }
        }
    }
    
    static <T> List<T> getAll(List<Future<T>> futures) {
        List<T> result = new ArrayList<>();

        for(Future<T> f : futures) {
            try {
                result.add(f.get());
            } catch (InterruptedException | ExecutionException ex) {
                ex.getCause().printStackTrace(System.out);
                throw new RuntimeException("Shit happened: " + ex.toString());
            }
        }
        
        return result;
    }
    
    public void finish() {
        if(ownExecutor)
            threadPool.shutdown();
        
        for(NeuralNetwork n : pool)
            n.freeGPU();
    }

    public long getLastID() {
        return lastID.get();
    }

    public void replenishMutationRate(int i) {
        this.mutation.replenish(i);
    }
}
