# Fast Java Neural Network library

Lightweight neural networks implementation

Supports execution on multiple GPUs

Includes an implementation for NEAT

http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

# How to use

Include fjnn jar found in target (or compile your own version)
Must include jcuda for GPU usage

Initialize
```
    int hiddencount = 5;
    int inputsize  = 10;
    int hiddensize = 10;
    int outputsize = 2;

    NeuralNetwork network = new NeuralNetwork(false);

    /* input */
    network.addLayer(inputsize, null, true);

    /* hidden layers */
    for(int i=0; i < 3; i++)
        network.addLayer(hiddensize, new Sigmoid(), true);

    network.addLayer(outputsize, new SoftMax(), true);
    
    /* must call build*/
    network.build();

    // network.setWeight(...)
    // network.setBias(...)
    // network.compute(inputData)

```

# Using GPU (must include Cuda libraries)
```
    /* must be called before anything */
    CudaEngine.init();
    
    // Initialization code here
    NeuralNetwork ...
    
    // Must call prepare GPU after chaning any of the weights / biases
    // you can optionally select a device for multiple GPU systems
    network.prepareGPU(2);

    float[] result = network.computeGPU(inputData);

    // call when you are done with the network to free GPU resources
    network.freeGPU();
```

# Running multiple neural networks in parallel (all must have the same structure)
```
    MultiNetwork multi = new MultiNetwork(10, false);

    /* input */
    network.addLayer(inputsize, null, true);

    /* hidden layers */
    for(int i=0; i < 3; i++)
        network.addLayer(hiddensize, new Sigmoid(), true);

    network.addLayer(outputsize, new SoftMax(), true);

    // one can also create multinetwork based on a neuralnetwork
    // MultiNetwork multi = new MultiNetwork(10, false, baseNetwork);
    
    multi.build();

    multi.setWeights(0, weights0);
    multi.setBiases(0, biases0);

    multi.setWeights(1, weights1);
    multi.setBiases(1, biases1);

    float[][] input2D = ...

    float[][] resultCPU = network.compute(input2D);

    multi.prepareGPU();
    float[][] resultGPU = network.computeGPU(input2D);
```

# Using multiple GPUs
By default, neural networks / multi networks will run on 1 GPU,
In order to use multiple GPUs you must use a NetworkPool
NetworkPool will evenly distributed all networks across the GPUs

```
    NetworkPool pool = new NetworkPool(size, false);
        
    /* input */
    pool.addLayer(inputsize, null, true);
    
    /* hidden layers */
    for(int i=0; i < 3; i++)
        pool.addLayer(hiddensize, new Sigmoid(), true);

    pool.addLayer(outputsize, new SoftMax(), true);

    pool.build();

    /* set weights for all your networks */
    pool.getNetwork(0).setWeights(...);
    pool.getNetwork(1).setWeights(...);
    ...

    /* compute all GPU */
    pool.prepareGPU();
    float[][] resultGPU = pool.computeGPU(input2D);
```

# NEAT
```
```
    
# Converting NEAT to a layered neural network
```
```

# Visualization

You can use a library like graphstream to visualize the network
An example of this can be found in test/examples

# Contribution

-   Creating Unit tests