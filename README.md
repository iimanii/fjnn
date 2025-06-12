# FJNN - Fast Java Neural Network

A Java neural network library with CPU and GPU acceleration support for building traditional feedforward networks, 1D convolutional networks, and evolutionary neural networks.

## Requirements

- Java 8+
- GPU support requires CUDA 11.2+

## Installation

**Build from Source**

1. Build with Maven:
```bash
mvn clean install
```
This will compile the library, run tests, and install it to your local Maven repository.

2. Add to your Maven project:
```xml
<dependency>
    <groupId>org.fjnn</groupId>
    <artifactId>fjnn</artifactId>
    <version>1.0.0</version>
</dependency>
```

3. For Gradle projects, add the built JAR as a local dependency:
```gradle
dependencies {
    implementation files('/path/to/fjnn/target/fjnn-1.0.0.jar')
}
```

## Quick Start

**Basic Neural Network**
```java
import org.fjnn.network.NeuralNetwork;
import org.fjnn.activation.*;

// Create network: 10 inputs -> 50 hidden -> 20 hidden -> 3 outputs
NeuralNetwork network = new NeuralNetwork(10, 3, new Sigmoid())
    .addHiddenLayer(50, new ReLU())
    .addHiddenLayer(20, new Tanh())
    .build()
    .kaiming(1.0f); // Kaiming initialization

// Compute output
float[] output = network.compute(new float[]{1.0f, 0.5f, -0.2f, /*...*/});
```

## Training

**Basic Training**
```java
import org.fjnn.loss.Loss;
import org.fjnn.network.outputs.*;

for (int epoch = 0; epoch < 1000; epoch++) {
    // Forward pass
    NeuralNetworkForwardOutput forwardOutput = network.feedForward(inputData, batchSize);
    
    // Backward pass with loss calculation
    NeuralNetworkBackpropagateOutput backpropOutput = network.backpropagate(
        forwardOutput, targetData, Loss.MeanSquareError);
    
    // Apply gradients
    float learningRate = 0.01f;
    float weightDecay = 0.0001f;
    network.applyGradients(backpropOutput, learningRate, weightDecay);
    
    if (epoch % 100 == 0) {
        System.out.println("Epoch " + epoch);
    }
}
```

**Networks with Normalization and Dropout**
```java
import org.fjnn.normalizer.LayerNormalizer;

// Add layer normalization and dropout
NeuralNetwork network = new NeuralNetwork(100, 10, new ReLU())
    .addHiddenLayer(50, new ReLU(), new LayerNormalizer(), 0.3f) // 30% dropout
    .addHiddenLayer(25, new Tanh(), new LayerNormalizer(), 0.2f) // 20% dropout
    .build()
    .xavier(1.0f);
```

## GPU Operations

```java
import org.fjnn.cuda.*;
import jcuda.driver.*;

// Initialize CUDA engine (once per application)
CudaEngine.init();

// Get GPU system information
int deviceCount = CudaEngine.getDeviceCount();
System.out.println("Available GPUs: " + deviceCount);

for (int i = 0; i < deviceCount; i++) {
    System.out.println("GPU " + i + ":");
    System.out.println("  Multiprocessors: " + CudaEngine.getMultiProcessorCount(i));
    System.out.println("  Clock Rate: " + CudaEngine.getDeviceProperties(i).clockRate / 1e3f + " MHz");
    System.out.println("  Free Memory: " + CudaEngine.getFreeMemory(i) / 1e9f + " GB");
}

// Prepare thread for GPU operations (must be called before any GPU operations)
CudaEngine.prepareThread(0); // GPU device 0

// Create CUDA stream for async operations
CUstream stream = CudaUtil.createStream();

// Prepare network for GPU (copies weights from CPU to GPU)
network.prepareGPU(stream);

// Create GPU memory and copy data asynchronously
CUdeviceptr gpuInput = CudaUtil.toGPUAsync(inputData, stream);

// Compute on GPU
float[] results = network.computeGPU(gpuInput, batchSize);

// For training operations with streams
NeuralNetworkForwardOutputGPU forwardOutput = network.feedForwardGPU(gpuInput, batchSize, stream);

// GPU backpropagation
CUdeviceptr targetGPU = CudaUtil.toGPUAsync(targetData, stream);
NeuralNetworkBackpropagateOutputGPU backpropOutput = network.backpropagateGPU(
    forwardOutput, targetGPU, Loss.MeanSquareError, stream);

// Apply gradients
network.applyGradientsGPU(backpropOutput, learningRate, weightDecay, stream);

// Synchronize stream to ensure completion
JCudaDriver.cuStreamSynchronize(stream);

// Sync weights back to CPU if needed
network.updateWeightsFromGPU();

// Clean up GPU memory
CudaUtil.freeAsync(gpuInput, stream);
forwardOutput.freeAsync(stream);
backpropOutput.freeAsync(stream);

// Free resources
CudaUtil.freeStream(stream);
network.freeGPU();

// Essential cleanup for thread
CudaEngine.finalizeThread();
```

## Trainer

**Dataset Setup**
```java
import org.fjnn.trainer.backpropagate.Dataset;

// Create dataset from your data
float[] inputs = {...};    // Your input data
float[] targets = {...};   // Your target data
int inputDim = 100;        // Features per sample
int outputDim = 10;        // Outputs per sample  
int batchSize = 32;

Dataset dataset = new Dataset(inputs, targets, inputDim, outputDim, batchSize);

// Prepare for CPU training
dataset.prepareBatches();

// Or prepare for GPU training
dataset.prepareBatchesGPU(0); // GPU device 0
```

**CPU Training with TrainingSession**
```java
import org.fjnn.trainer.backpropagate.*;
import org.fjnn.loss.*;

TrainingConfig config = new TrainingConfig(
    0.001f,                     // learning rate
    0.0001f,                    // weight decay  
    new BinaryCrossEntropy(),   // loss function
    300000,                     // max time (5 min)
    50,                         // min epochs
    1000                        // max epochs
);

TrainingSession session = new TrainingSession(network, dataset, config);

while (session.status() == TrainingSession.TrainingStatus.CONTINUE) {
    double loss = session.step();
    
    // Check progress and statistics
    ProgressTracker progress = session.getProgress();
    int epoch = progress.getCurrentEpoch();
    
    System.out.println("Epoch " + epoch + ", Loss: " + loss);
    System.out.println("Best Loss: " + progress.getBestLoss());
    System.out.println("Forward: " + progress.getLastForwardTime() + "ms");
}
```

**GPU Training**
```java
import jcuda.driver.*;

// Initialize CUDA (once per application)
CudaEngine.init();

// GPU training configuration
TrainingConfig gpuConfig = new TrainingConfig(
    0.01f, 0.0001f, new MeanSquareError(), 
    60000, 10, 100,
    new int[]{0}        // Use GPU device 0
);

// Prepare thread for GPU operations
CudaEngine.prepareThread(0);

// Create CUDA stream
CUstream stream = CudaUtil.createStream();

// Prepare dataset for GPU (copies batches to GPU memory)
dataset.prepareBatchesGPU(0);

TrainingSession session = new TrainingSession(network, dataset, gpuConfig);

while (session.status() == TrainingSession.TrainingStatus.CONTINUE) {
    double loss = session.step();
    
    ProgressTracker progress = session.getProgress();
    System.out.println("Epoch " + progress.getCurrentEpoch() + ", Loss: " + loss);
    System.out.println("GPU Forward: " + progress.getLastForwardTime() + "ms");
    System.out.println("GPU Backward: " + progress.getLastBackwardTime() + "ms");
}

CudaUtil.destroyStream(stream);
CudaEngine.finalizeThread();
```

## Advanced Features

**1D Convolutional Networks**
```java
import org.fjnn.convolution.*;
import org.fjnn.convolution.reshape.PositionalConcatenationReshape;

// Create kernels for different pattern sizes
Kernel smallKernel = new Kernel(unitSize, 3);  // 3-width patterns
Kernel largeKernel = new Kernel(unitSize, 7);  // 7-width patterns

// Create kernel group with AND logic
KernelGroup group = new KernelGroup(unitSize, smallKernel, largeKernel);

// Build convolution layer
ConvolutionLayer convLayer = new ConvolutionLayer(group);

// Add reshaper for output formatting
PositionalConcatenationReshape reshaper = new PositionalConcatenationReshape();
```

**Chain Models with Positional Encoding**
```java
import org.fjnn.base.ChainModel;
import org.fjnn.adapter.PositionalEncoderAdapter;
import org.fjnn.adapter.InputDimAdapter;

// Create sequence processing pipeline
ChainModel sequenceModel = new ChainModel(inputSize);

// Add transformer-style positional encoding
sequenceModel.addComponent(new PositionalEncoderAdapter(128, maxSeqLength));

// Add input dimension reshaping if needed
sequenceModel.addComponent(new InputDimAdapter(inputDim, targetDim));

// Add convolutional feature extraction
sequenceModel.addComponent(convLayer);

// Add final classification network
sequenceModel.addComponent(network);

// Process sequence data
float[] result = sequenceModel.feedForward(sequenceData, inputSize, batchSize)
                              .getLastOutput();
```

## Evolutionary Neural Networks

**⚠️ Currently under maintenance** - The evolutionary network implementation may not work correctly or contain bugs due to ongoing library updates.

**Parameter Evolution**
```java
import org.fjnn.trainer.genetic.*;

// Evolve weights of existing network architectures
NeuralNetwork baseNetwork = new NeuralNetwork(inputSize, outputSize, new Sigmoid())
    .addHiddenLayer(50, new ReLU())
    .build()
    .xavier(1.0f);

// Custom training set for fitness evaluation
class MyTrainingSet extends TrainingSet {
    public TrainingResult calculateFitness(float[] result, int batch) {
        double fitness = evaluateFitness(result);
        return new TrainingResult(fitness, new double[]{fitness});
    }
    // ... other required methods
}

// Configure parameter evolution
GeneticTrainerConfig config = new GeneticTrainerConfig(
    GeneticTrainerConfig.ComputeMode.plain, 
    GeneticTrainerConfig.SelectionCriteria.fitness)
    .setMutationAmount(0.1f)
    .setEliteLimit(5);

GeneticTrainer<NeuralNetwork> trainer = new GeneticTrainer<>(baseNetwork, 50, trainingSet, config);
trainer.train();
```

**NEAT-style Topology Evolution**

Based on [Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)

```java
import org.fjnn.genetic.*;

// Create NEAT-style genetic network for topology evolution
GeneticNetwork genetic = new GeneticNetwork(10, 5, new Tanh(), new ReLU());

// Configure mutation probabilities for topology changes
GeneticNetworkConfig.setMutationProbability(
    0.1,  // ADD_NODE - create new neurons
    0.15, // ADD_CONNECTION - create new links  
    0.3,  // ENABLE_CONNECTION - reactivate links
    0.2   // DISABLE_CONNECTION - deactivate links
);

// Evolve the network topology
Innovation mutation = genetic.mutate();

// Use genetic network directly for computation
float[] output = genetic.compute(input);

// Convert evolved topology to standard NeuralNetwork for training/inference
NeuralNetwork evolved = genetic.getNetwork();
```

## Platform Support

- **Operating Systems**: Windows, Linux, macOS
- **Hardware**: CPU (x86_64, ARM), GPU (CUDA 11.2+)
- **Build Tools**: Maven, Gradle

## Examples

See the [examples repository](link-to-examples) for complete working examples and tutorials.