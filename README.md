# Fast Java Neural Network library

A lightweight implementation of neural network to run on GPU

Includes an implementation for NEAT

http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

# How to use

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
    // network.compute()

```

# Cuda
```
    /* must be called before anything */
    CudaEngine.init();
    
    // Initialization code here

    float[] result = network.computeGPU(inputData);
```
# NEAT
