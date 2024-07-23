/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.fjnn.trainer.genetic;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicLong;
import jcuda.driver.CUdeviceptr;
import org.fjnn.base.Network;
import org.fjnn.base.NetworkInput;
import org.fjnn.cuda.CudaEngine;
import org.fjnn.trainer.genetic.GeneticTrainerConfig.ComputeMode;
import org.fjnn.trainer.genetic.TrainingSet.TrainingResult;
import org.fjnn.util.Rng;
import org.fjnn.util.util;

/**
 *
 * @author ahmed
 * @param <T>
 */
public class GeneticTrainer <T extends Network> {

    public static class trainPerformance {
        public float computeTime;
        public float mutationTime;
        public float finalizationTime;
        public float totalTime;

        public trainPerformance(float computeTime, float mutationTime, float finalizationTime) {
            this.computeTime = computeTime;
            this.mutationTime = mutationTime;
            this.finalizationTime = finalizationTime;
            this.totalTime = computeTime + mutationTime + finalizationTime;
        }

        private void updateTotalTime() {
            this.totalTime = computeTime + mutationTime + finalizationTime;
        }
    }
    
    /* maximum networks in pool */
    public final int poolsize;
    
    final TrainingSet trainingSet;
    
    /* best performing network */
    T best;
            
    /* best fitness based on last epoch */
    private double bestFitness;
    
    /* calculated average fitness based on last epoch */
    private double averageFitness;

    /* mutated each generation */
    List<T> pool;

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
    public static final String PARENT_MUTATION_A = "parent-mutation-a";
    public static final String PARENT_MUTATION_B = "parent-mutation-b";
    public static final String PARENT_GENERATION_A = "parent-generation-a";
    public static final String PARENT_GENERATION_B = "parent-generation-b";
    public static final String GENERATION_PROP = "generation";
    public static final String FITNESS_PROP = "fitness";
    public static final String FITNESS_SQUARED_PROP = "fitness-squared";
    public static final String DETAILED_FITNESS_PROP = "detailed-fitness";
    public static final String DETAILED_FITNESS_HASH_PROP = "detailed-fitness-hash";
    public static final String COMMULATIVE = "commulative:%s";
    public static final String MUTATION_PROP = "mutation";
    public static final String CHILD_COUNTER = "child-counter";
    public static final String ABSOLUTE_Z_SCORE = "absolute-z-score";
    public static final String SQUARE_Z_SCORE = "square-z-score";
    public static final String OVERALL_FITNESS_PROP = "overall-fitness";
    public static final String OVERALL_DETAILED_FITNESS_PROP = "overall-detailed-fitness";
    
    private boolean ownExecutor;
    
    public GeneticTrainer(T base, int poolsize, TrainingSet trainingSet, GeneticTrainerConfig config) {
        this(Arrays.asList(base), poolsize, trainingSet, config);
    }
    
    public GeneticTrainer(List<T> networks, int poolsize, TrainingSet trainingSet, GeneticTrainerConfig config) {
        config.checkTournamentSize(poolsize);
        
        this.poolsize = poolsize;
        this.pool = new ArrayList<>(networks);
        this.lastID = new AtomicLong(config.startingID);
        this.threadPool = Executors.newCachedThreadPool();
        this.ownExecutor = true;
        this.trainingSet = trainingSet;
        this.mutation = new GeneticMutation(5, 0.5, 0.00001, 1);
        this.config = config;
        this.generation = config.startingGeneration;
        this.epoch = config.startingEpoch;
        
        /* first calculate all fitness */
        ArrayList<Future<?>> futures = new ArrayList<>();
        
        for(T n : networks) {
            if(!n.hasProperty(ID_PROP))
                n.setProperty(ID_PROP, lastID.incrementAndGet());
            
            if(!n.hasProperty(GENERATION_PROP))
                n.setProperty(GENERATION_PROP, generation);
            
            if(!n.hasProperty(MUTATION_PROP))
                n.setProperty(MUTATION_PROP, 1.0);
        }
        
        int missing = poolsize - networks.size();
        for(int i=0; i < missing; i++) {
            T n = (T)networks.get(0).copy(false);
            n.setProperty(ID_PROP, lastID.incrementAndGet());
            n.setProperty(GENERATION_PROP, generation);
            n.setProperty(MUTATION_PROP, 1.0);
            n.randomize(-1, 1);
            pool.add(n);
        }
        
        System.out.println("preprocessing... ");
        long time = System.nanoTime();
        
        for(T n : pool) {
            if(!n.hasProperty(FITNESS_PROP)) {
                futures.add(threadPool.submit(() -> {
                    TrainingResult output = compute(n, -1);

                    n.setProperty(FITNESS_PROP, output.fitness);
                    n.setProperty(DETAILED_FITNESS_PROP, output.detailed);
                }));
            }
        }
        
        util.waitForAll(futures);       
        
        long time_2 = System.nanoTime();
        System.out.println("preprocessing done: " + (time_2 - time) / 1e6f); 
        
        copyOverallFitness(pool);
        
        GeneticUtil.sortDesc(pool, FITNESS_PROP);
        this.best = pool.get(0);
        this.bestFitness = (double) best.getProperty(FITNESS_PROP);
        this.averageFitness = calculateAverageFitness(pool, FITNESS_PROP);
    }
    
    private TrainingResult compute(T n, int batch) {
        float[] result;
        
        switch(config.computeMode) {
            case gpu:
                if(!n.gpuReady())
                    n.prepareGPU(config.deviceId);
                
                NetworkInput gpuInput = batch == -1 ? trainingSet.getAllGPUInput(config.deviceId) : trainingSet.getGPUInput(config.deviceId, batch);
                result = n.compute(gpuInput);
                break;
            case gpu_all:
                if(!n.gpuReady())
                    n.prepareGPU();
                
                gpuInput = batch < 0 ? trainingSet.getAllGPUInput(n.getGPUDeviceId()) : trainingSet.getGPUInput(n.getGPUDeviceId(), batch);
                result = n.compute(gpuInput);
                break;
            default:
                NetworkInput input = batch == -1 ? trainingSet.getAllInput() : trainingSet.getInput(batch);
                result = n.compute(input);
                break;
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

            double maxMutation = config.useStaticMutation ? config.mutationChance : Math.min(mutation.get() * 2, 1);
            
            /* crossover mutate */
            boolean crossOverGPU = config.crossover == ComputeMode.gpu && config.computeMode != ComputeMode.gpu_all;
            
            List<T> newGeneration = config.useTournamentSelection ? 
                    createNewGenerationTS(maxMutation, config.tournametSelectionSize, property, crossOverGPU) : 
                    createNewGenerationRW(maxMutation, property, crossOverGPU);

            long time_2 = System.nanoTime();
            
            boolean cleanGPU = config.computeMode == ComputeMode.gpu || config.computeMode == ComputeMode.gpu_all;

            if(config.eliteLimit >= 0) {
                ArrayList<T> sort = new ArrayList<>(pool);
                GeneticUtil.sortDesc(sort, FITNESS_PROP);
                
                for(int j=0; j < config.eliteLimit; j++)
                    newGeneration.add(sort.get(j));
                
                if(cleanGPU) {
                    ArrayList<Future<?>> futures = new ArrayList<>();
                    
                    for(int j=config.eliteLimit; j < pool.size(); j++) {
                        T n = pool.get(j);
                        futures.add(threadPool.submit(() -> {
                            n.freeGPU();
                        }));
                    }
                    
                    util.waitForAll(futures);                    
                }
            } else
                newGeneration.addAll(pool);
            
            /* first calculate all fitness */
            ArrayList<Future<?>> futures = new ArrayList<>();
            
            int batchId = i;
            
            for(T n : newGeneration) {
                if(trainingSet.getBatchCount() > 1 || !n.hasProperty(FITNESS_PROP)) {
                    futures.add(threadPool.submit(() -> {                    
                        TrainingResult output = compute(n, batchId);

                        n.setProperty(FITNESS_PROP, output.fitness);
                        n.setProperty(DETAILED_FITNESS_PROP, output.detailed);
                    }));
                }
            }

            util.waitForAll(futures);

            /* pick top performing networks as next generation */
            if(!config.allowEqualSolutions) {
                newGeneration = filterEqualSolutions(newGeneration, cleanGPU, poolsize);
            }
            
            GeneticUtil.sortDesc(newGeneration, FITNESS_PROP);
            pool = new ArrayList<>(newGeneration.subList(0, poolsize));

            futures = new ArrayList<>();

            if(crossOverGPU) {
                for(int j=0; j < poolsize; j++) {
                    T n = newGeneration.get(j);
                    futures.add(threadPool.submit(() -> {
                        n.ensureCPU();
                    }));
                }
            }

            if(cleanGPU) {
                for(int j=poolsize; j < newGeneration.size(); j++) {
                    T n = newGeneration.get(j);
                    futures.add(threadPool.submit(() -> {
                        n.freeGPU();
                    }));
                }
            }

            util.waitForAll(futures);

            long time_3 = System.nanoTime();
            currentTrainPerformance.computeTime += (time_3 - time_2) / 1e6f;
            currentTrainPerformance.mutationTime += (time_2 - time_1) / 1e6f;
        }

        long time_4 = System.nanoTime();
        
        ArrayList<T> sort = new ArrayList<>(pool);  
        if(trainingSet.getBatchCount() > 1) {
            /* calculate fitness for the pool over the whole dataset */
            calculateOverallFitness(sort);      
        } else {
            copyOverallFitness(sort);
        }
        
        GeneticUtil.sortDesc(sort, OVERALL_FITNESS_PROP);
        this.best = sort.get(0);
        this.bestFitness = (double) best.getProperty(OVERALL_FITNESS_PROP);
        this.averageFitness = calculateAverageFitness(sort, OVERALL_FITNESS_PROP);
        
        long time_5 = System.nanoTime();
        
        currentTrainPerformance.finalizationTime += (time_5 - time_4) / 1e6f;
        currentTrainPerformance.updateTotalTime();
        lastPerformance = currentTrainPerformance;
        epoch++;
    }
    
    public trainPerformance getLastPerformanceCounters() {
        return lastPerformance;
    }
    
    /* using tournament selection */
    private List<T> createNewGenerationTS(double maxMutation, int tournamentSize, String property, boolean gpu) {
        List<T> result = new ArrayList<>();

        ArrayList<Future<?>> futures = new ArrayList<>();
        
        /* crossover mutate */
        for(int i=0; i < poolsize; i++) {
            T n = (T)best.copy(false, !gpu);
            n.setProperty(ID_PROP, lastID.incrementAndGet());
            result.add(n);
            double m = maxMutation;//Rng.nextDouble(0, maxMutation);
            
            futures.add(threadPool.submit(() -> {
                /* select parents */
                T n0 = GeneticUtil.tournamentPick(pool, tournamentSize, property);
                T n1 = GeneticUtil.tournamentPick(pool, tournamentSize, property, n0);
            
                if(gpu) {
                    n.crossOverMutateGPU(n0, n1, -config.mutationAmount, config.mutationAmount, m, true);
                    
                    if(config.clipWeights)
                        n.clipWeightsGPU(config.clipMin, config.clipMax);
                } else {
                    n.crossOverMutate(n0, n1, -config.mutationAmount, config.mutationAmount, m);
                    
                    if(config.clipWeights)
                        n.clipWeights(config.clipMin, config.clipMax);
                }
                
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
                n.setProperty(PARENT_MUTATION_A, n0.getProperty(MUTATION_PROP));
                n.setProperty(PARENT_MUTATION_B, n1.getProperty(MUTATION_PROP));
                n.setProperty(PARENT_GENERATION_A, n0.getProperty(GENERATION_PROP));
                n.setProperty(PARENT_GENERATION_B, n1.getProperty(GENERATION_PROP));
            }));
        }

        util.waitForAll(futures);
        
        return result;
    }
    
    /* using roulette wheel */
    private List<T> createNewGenerationRW(double maxMutation, String property, boolean gpu) {
        List<T> result = new ArrayList<>();

        ArrayList<Future<?>> futures = new ArrayList<>();
        
        String commulativeProprety = String.format(COMMULATIVE, property);
        GeneticUtil.commulativeDestribution(pool, property, commulativeProprety);
        GeneticUtil.sortAsc(pool, commulativeProprety);
        
        /* crossover mutate */
        for(int i=0; i < poolsize; i++) {
            T n = (T)best.copy(false, !gpu);
            n.setProperty(ID_PROP, lastID.incrementAndGet());
            double m = maxMutation;//Rng.nextDouble(0, maxMutation);
            result.add(n);
            
            T n0 = GeneticUtil.pickRandom(pool, commulativeProprety);
            T n1 = GeneticUtil.pickRandom(pool, commulativeProprety, n0);
            
            /* select parents */
            futures.add(threadPool.submit(() -> {
                if(gpu)
                    n.crossOverMutateGPU(n0, n1, -config.mutationAmount, config.mutationAmount, m, true);
                else
                    n.crossOverMutate(n0, n1, -config.mutationAmount, config.mutationAmount, m);

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
                n.setProperty(PARENT_MUTATION_A, n0.getProperty(MUTATION_PROP));
                n.setProperty(PARENT_MUTATION_B, n1.getProperty(MUTATION_PROP));
                n.setProperty(PARENT_GENERATION_A, n0.getProperty(GENERATION_PROP));
                n.setProperty(PARENT_GENERATION_B, n1.getProperty(GENERATION_PROP));
            }));
        }

        util.waitForAll(futures);
        
        return result;        
    }
    
    private void copyOverallFitness(List<T >networks) {
        ArrayList<Future<?>> futures = new ArrayList<>();
        
        for(T n : networks) {
            futures.add(threadPool.submit(() -> {
                n.setProperty(OVERALL_FITNESS_PROP, n.getProperty(FITNESS_PROP));
                n.setProperty(OVERALL_DETAILED_FITNESS_PROP, n.getProperty(DETAILED_FITNESS_PROP));
            }));
        }
        
        util.waitForAll(futures);
    }
    
    private void calculateOverallFitness(List<T> networks) {
        ArrayList<Future<?>> futures = new ArrayList<>();
        
        for(T n : networks) {
            futures.add(threadPool.submit(() -> {
                TrainingResult output = compute(n, -1);

                n.setProperty(OVERALL_FITNESS_PROP, output.fitness);
                n.setProperty(OVERALL_DETAILED_FITNESS_PROP, output.detailed);
            }));
        }
        
        util.waitForAll(futures);
    }

    private double calculateAverageFitness(List<T> networks, String property) {
        double total = 0;
        double count = 0;
        
        for(T n : networks) {
            total += (double) n.getProperty(property);
            count++;
        }
        
        return total / count;
    }

    private List<T> filterEqualSolutions(List<T> array, boolean cleanGPU, int minSize) {
        HashMap<Double, List> map = new HashMap<>();
        
        for(T n : array) {
            double fitness = (double) n.getProperty(FITNESS_PROP);
            if(!map.containsKey(fitness))
                map.put(fitness, new ArrayList<>());
            
            map.get(fitness).add(n);
        }
        
        List<Future> futures = new ArrayList<>();
        
        for(Map.Entry<Double, List> e : map.entrySet()) {
            futures.add(threadPool.submit(() -> {
                List<T> list = e.getValue();
                
                if(list.size() == 1)
                    return list;
                
                List<T> result = new ArrayList<>();
                
                List<Future<?>> f = new ArrayList<>();

                for(T n : list) {
                    f.add(threadPool.submit(() -> {
                        double[] detailed = (double[])n.getProperty(DETAILED_FITNESS_PROP);
                        n.setProperty(DETAILED_FITNESS_HASH_PROP, util.doubleArrayHash(detailed));
                    }));
                }
                
                util.waitForAll(f);
                
                Set<Integer> hashes = new HashSet<>();
                
                int filtered = 0;
                
                for(T n : list) {
                    int hash = (int) n.getProperty(DETAILED_FITNESS_HASH_PROP);
                    
                    if(!hashes.contains(hash)) {
                        result.add(n);
                        hashes.add(hash);
                    } else
                        filtered++;
                }
                
//                if(filtered > 0)
//                    System.out.printf("Filtered %d / %d %.3f\n", filtered, list.size(), result.get(0).getProperty(FITNESS_PROP));
                
                return result;
            }));
        }
        
        List<T> result = new ArrayList<>();
        
        for(Future<List<T>> f : futures)
            try {
                result.addAll(f.get());
            } catch (InterruptedException | ExecutionException ex) {
                throw new RuntimeException(ex);
            }
        
        int filtered = array.size() - result.size();
        
        if(filtered > 0)
            System.out.println("Filtered same solution: " + filtered);
        
        int minRemain = minSize - result.size();
        
        if(minRemain > 0) {
            System.out.println("Adding min size: " + (minSize - result.size()));
            
            ArrayList<T> rest = new ArrayList<>(array);
            for(T n : result)
                rest.remove(n);
            
            Collections.shuffle(rest, ThreadLocalRandom.current());
            
            for(int i=0; i < minRemain; i++)
                result.add(rest.get(i));
        }
        
        List<Future<?>> freeFutures = new ArrayList<>();
        
        /* free gpu */
        if(cleanGPU)
            for(T n : array)
                if(!result.contains(n))
                    freeFutures.add(threadPool.submit(() -> {
                        n.freeGPU();
                    }));
        
        util.waitForAll(freeFutures);
        
        return result;
    }
    
    public void setExecutor(ExecutorService executor) {
        this.threadPool = executor;
        this.ownExecutor = false;
    }
    
    public double getDiversity() {
        double score = 0;
        ArrayList<Future<Double>> futures = new ArrayList<>();
        
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
        
        for(Future<Double> f : futures)
            try {
                score += f.get();
            } catch (ExecutionException | InterruptedException ex) {
                throw new RuntimeException(ex);
            }
        
        score /= (size * (size-1)) / 2;
        return score;
    }
    
    public double getBestFitness() {
        return bestFitness;
    }
    
    /* returns average fitness of generation size */
    public double getAverageFitness() {
        return averageFitness;
    }
    
    public T getBest() {
        return best;
    }
    
    public List<T> getPool() {
        return new ArrayList<>(pool);
    }
    
    public double getMutationRate() {
        return mutation.get();
    }
    
    private void randomize(T n0, T n1, double mutation) {
        n0.crossOverMutate(n0, n1, -1, 1, mutation);
    }

    public int getGeneration() {
        return generation;
    }
    
    public int getEpoch() {
        return epoch;
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
        
        for(T n : pool)
            n.freeGPU();
    }

    public long getLastID() {
        return lastID.get();
    }

    public void replenishMutationRate(int i) {
        this.mutation.replenish(i);
    }
}
