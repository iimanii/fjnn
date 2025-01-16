# Fast Java Neural Network library

Lightweight neural networks implementation supporting:
- CPU and GPU execution with multi-GPU support
- Chain model architecture for complex networks
- Genetic Algorithm / NEAT implementation

http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

# Installation

Include fjnn jar found in target (or compile your own version)
Must include jcuda for GPU usage

Initialize
```
    int hiddencount = 5;
    int inputsize  = 10;
    int hiddensize = 10;
    int outputsize = 2;

    NeuralNetwork network = new NeuralNetwork(inputSize, outputSize, new Sigmoid());
    network.addHiddenLayer(hiddenSize, new ReLU());

    /* hidden layers */
    for(int i=0; i < 3; i++)
        network.addLayer(hiddensize, new Sigmoid(), true);

    /* must call build*/
    network.build();
    
    /* Initialize weights */
    network.xavier();

    // network.setWeight(...)
    // network.setBias(...)

    // Feed forward
    float[] result = network.compute(inputData);

    // Backpropagation 
    float[] truth = new float[outputSize];
    float learningRate = 0.1f;
    BackpropagateOutput backpropResult = network.backpropagate(feedforwardResult, truth, Loss.MeanSquareError, learningRate);

    // Access results
    float[] output = feedforwardResult.output();
    float[] deltaLoss = backpropResult.deltaLoss();
```

# Using GPU (must include Cuda libraries)
```
    /* initialize the cuda engine */
    CudaEngine.init();

    // utility functions to get info about the system
    int deviceCount = CudaEngine.getDeviceCount();
    System.out.println("GPU " + 0 + ":");
    System.out.println("Multiprocessors: " + CudaEngine.getMultiProcessorCount(0));
    System.out.println("Clock Rate: " + CudaEngine.getDeviceProperties(0).clockRate / 1e3f + " MHz");
    System.out.println("Free Memory: " + CudaEngine.getFreeMemory(0) / 1e9f + " GB");
    
    // Initialization code here
    NeuralNetwork ...
    
    // Select GPU device for current thread .. must be called before any GPU operations on the current thread
    CudaEngine.prepareThread(0);
    

    // Transfer weights to GPU
    network.prepareGPU();

    // Perform GPU forward pass and backpropagation using streams
    CUstream stream = CudaUtil.createStream();

    CUdeviceptr inputGPU = CudaUtil.toGPUAsync(input, stream);

    NeuralNetworkForwardOutputGPU feedforwardResultGPU = network.feedForwardGPU(inputGPU, 1, stream);

    // GPU Backpropagation
    float[] truth = new float[outputSize];
    CUdeviceptr targetGPU = CudaUtil.toGPUAsync(target, stream);
    float learningRate = 0.1f;
    
    // This call will automatically update the weights with gradients
    BackpropagateOutputGPU backpropResultGPU = network.backpropagateGPU(feedforwardResultGPU, 
        truth, Loss.MeanSquareError, learningRate);

    feedforwardResultGPU.freeAsync(stream);
    backpropResultGPU.freeAsync(stream);

    JCudaDriver.cuStreamSynchronize(stream);

    // Free GPU resources when done
    network.freeGPU();
```

# Chain Model Architecture
Chain models allow connecting multiple components (networks, adapters) in sequence
```
    // Create chain model
    ChainModel chain = new ChainModel(inputSize);

    // Add neural network component
    NeuralNetwork nn = new NeuralNetwork(....);
    ...

    chain.addComponent(nn);
    
    // Add positional encoder
    PositionalEncoderAdapter encoder = new PositionalEncoderAdapter(featureSize, featureCount);
    chain.addComponent(encoder);

    // modify input shape if necessary to match the next component
    chain.addBatchSizeAdapter(newBatchSize);

    // Add another neural network component
    NeuralNetwork nn2 = new NeuralNetwork(....);
    ...
    chain.addComponent(nn);
    
    // get back to the correct batchSize
    chain.restoreBatchCount();
    
    // Feed forward
    FeedForwardOutputMap result = chain.feedForward(input, batchSize, batchCount);

    // Backpropagation
    BackpropagateOutputMap backpropResult = chain.backpropagate(result, truth, 
        batchSize, batchCount, Loss.MeanSquareError, learningRate);
```

GPU Support
```
    // Prepare model for GPU
    chain.prepareGPU(stream);

    // GPU Operations
    FeedForwardOutputMapGPU resultGPU = chain.feedForwardGPU(inputGPU, batchSize, batchCount, stream);
    BackpropagateOutputMapGPU backpropResultGPU = chain.backpropagateGPU(resultGPU, truthGPU, 
        batchSize, batchCount, Loss.MeanSquareError, learningRate, stream);

    // Free resources
    chain.freeGPU();
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

- Creating Unit tests
- Better documentation