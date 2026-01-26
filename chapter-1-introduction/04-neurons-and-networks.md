# Neurons and Networks

## The Building Block: Artificial Neuron

An **artificial neuron** mimics biological neurons by taking inputs, weighing their importance, and producing an output.

```
BIOLOGICAL NEURON:              ARTIFICIAL NEURON:
                                
  Dendrites (inputs)              Inputs (x₁, x₂, x₃)
       ↓                               ↓
   Cell Body                      Weighted Sum
  (processing)                    + Bias
       ↓                               ↓
   Axon (output)                  Activation Function
                                        ↓
                                    Output
```

## Mathematical Model of a Neuron

```
        x₁ ──w₁──┐
                  │
        x₂ ──w₂──┤
                  ├──→ Σ ──→ σ(z) ──→ output
        x₃ ──w₃──┤
                  │
        bias ─────┘

z = (w₁×x₁ + w₂×x₂ + w₃×x₃) + bias
output = σ(z)
```

**Formula:**
```
z = Σᵢ(wᵢ × xᵢ) + b
output = activation(z)
```

Where:
- `xᵢ` = inputs
- `wᵢ` = weights (importance of each input)
- `b` = bias (threshold adjustment)
- `σ` = activation function

## Simple Neuron Implementation

```python
import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        # Random initialization
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
    
    def forward(self, inputs):
        # Weighted sum
        z = np.dot(self.weights, inputs) + self.bias
        # Activation (ReLU)
        output = max(0, z)
        return output

# Example usage
neuron = Neuron(num_inputs=3)
inputs = np.array([1.0, 2.0, 3.0])
output = neuron.forward(inputs)

print(f"Weights: {neuron.weights}")
print(f"Bias: {neuron.bias}")
print(f"Output: {output}")

# Example output:
# Weights: [0.5, -0.3, 0.8]
# Bias: 0.1
# z = 0.5×1 + (-0.3)×2 + 0.8×3 + 0.1 = 2.6
# output = max(0, 2.6) = 2.6
```

## Weights: Importance Values

**Weights** determine how important each input is.

```
High positive weight (+2.5):  Input is very important (amplified)
Small positive weight (+0.1):  Input matters a little
Zero weight (0):               Input is ignored
Negative weight (-1.5):        Input has inverse effect
```

**Example: Spam Detection Neuron**
```python
# Inputs: [contains_"free", contains_"urgent", from_known_sender]
weights = [0.9, 0.7, -1.2]  # What each feature means
bias = -0.5

email = [1, 1, 0]  # Has "free", has "urgent", unknown sender
z = 0.9×1 + 0.7×1 + (-1.2)×0 + (-0.5) = 1.1
output = max(0, 1.1) = 1.1  # Likely spam!

email2 = [1, 1, 1]  # Same words, but from known sender
z = 0.9×1 + 0.7×1 + (-1.2)×1 + (-0.5) = -0.1
output = max(0, -0.1) = 0  # Probably not spam
```

## Bias: Threshold Adjustment

**Bias** shifts the activation threshold.

```
Without bias (b=0):     With positive bias (b=2):
Neuron fires when       Neuron fires MORE easily
input sum > 0          (already +2 head start)

With negative bias (b=-2):
Neuron fires LESS easily
(needs to overcome -2)
```

**Example:**
```python
# Same inputs, different biases
inputs = [0.5, 0.5]
weights = [1.0, 1.0]

# High bias: Neuron fires easily
bias = 2.0
z = 1.0×0.5 + 1.0×0.5 + 2.0 = 3.0  # Fires!

# Low bias: Neuron harder to activate
bias = -2.0
z = 1.0×0.5 + 1.0×0.5 + (-2.0) = -1.0  # Doesn't fire
```

## Activation Functions

**Activation functions** introduce non-linearity, enabling networks to learn complex patterns.

### ReLU (Rectified Linear Unit)

```
Most popular activation function.

ReLU(z) = max(0, z)

Graph:     output
           ↑
         3 │     ╱
         2 │   ╱
         1 │ ╱
         0 │╱________
          -2 -1 0 1 2  z
```

```python
def relu(z):
    return max(0, z)  # or np.maximum(0, z) for arrays

# Examples:
relu(3.0) = 3.0   # Positive passes through
relu(-2.0) = 0    # Negative becomes 0
relu(0) = 0
```

**Why ReLU?**
- Simple and fast
- Avoids vanishing gradient problem
- Works well in practice

### Sigmoid

```
Squashes values to (0, 1) range.

σ(z) = 1 / (1 + e⁻ᶻ)

Graph:     output
           ↑
         1 │    ─────
       0.5 │  ╱
         0 │──────
          -4 -2 0 2 4  z
```

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Examples:
sigmoid(0) = 0.5     # Middle point
sigmoid(5) = 0.993   # Large positive → near 1
sigmoid(-5) = 0.007  # Large negative → near 0
```

**Use case:** Output layer for binary classification (probabilities).

### Tanh (Hyperbolic Tangent)

```
Squashes to (-1, 1) range.

tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)

Graph:     output
           ↑
         1 │    ─────
         0 │  ╱
        -1 │──────
          -4 -2 0 2 4  z
```

```python
def tanh(z):
    return np.tanh(z)

# Examples:
tanh(0) = 0          # Zero-centered
tanh(3) = 0.995
tanh(-3) = -0.995
```

**Use case:** Hidden layers when you want zero-centered outputs.

### Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 100)

relu_output = np.maximum(0, z)
sigmoid_output = 1 / (1 + np.exp(-z))
tanh_output = np.tanh(z)

# All on same input range, different outputs!
```

## Neural Networks: Layers of Neurons

A **neural network** connects multiple neurons in layers.

```
INPUT LAYER     HIDDEN LAYER      OUTPUT LAYER
    x₁             h₁                 y₁
     ●─────────────●
     │╲           ╱│╲                ╱
     │ ╲         ╱ │ ╲              ╱
     │  ╲       ╱  │  ╲            ╱
    x₂   ╲     ╱   h₂──────────────●──→ output
     ●────╲───╱────●            ╱
     │     ╲ ╱    ╱│          ╱
     │      ╳    ╱ │        ╱
     │     ╱ ╲  ╱  │      ╱
    x₃───╱───╲╱───h₃────╱
     ●───────────●

  3 inputs    3 neurons     1 output
              (hidden)
```

### Forward Pass

Data flows from input → hidden → output.

```python
class NeuralNetwork:
    def __init__(self):
        # Layer 1: 3 inputs → 4 hidden neurons
        self.W1 = np.random.randn(4, 3)  # 4x3 weight matrix
        self.b1 = np.random.randn(4)     # 4 biases
        
        # Layer 2: 4 hidden → 2 outputs
        self.W2 = np.random.randn(2, 4)  # 2x4 weight matrix
        self.b2 = np.random.randn(2)     # 2 biases
    
    def forward(self, x):
        # Hidden layer
        z1 = self.W1 @ x + self.b1      # @ is matrix multiplication
        h = np.maximum(0, z1)            # ReLU activation
        
        # Output layer
        z2 = self.W2 @ h + self.b2
        output = z2                      # No activation (regression)
        
        return output

# Usage
net = NeuralNetwork()
input_data = np.array([1.0, 2.0, 3.0])
prediction = net.forward(input_data)
print(f"Prediction: {prediction}")
```

## Weight Matrices: Efficient Computation

Instead of individual neurons, we use **matrices** for efficiency.

```
Single neuron (slow):
for each neuron:
    output = Σ(weight × input) + bias

Matrix form (fast, parallel):
outputs = W @ inputs + biases
```

**Example:**
```python
# 3 inputs → 4 neurons

# Individual way (slow):
h1 = w1_1*x1 + w1_2*x2 + w1_3*x3 + b1
h2 = w2_1*x1 + w2_2*x2 + w2_3*x3 + b2
h3 = w3_1*x1 + w3_2*x2 + w3_3*x3 + b3
h4 = w4_1*x1 + w4_2*x2 + w4_3*x3 + b4

# Matrix way (fast):
W = [[w1_1, w1_2, w1_3],
     [w2_1, w2_2, w2_3],
     [w3_1, w3_2, w3_3],
     [w4_1, w4_2, w4_3]]

h = W @ x + b  # Single line!

# GPU can do this in parallel → 1000x faster
```

## Deep Networks: Multiple Layers

**Deep learning** = Many layers stacked together.

```
┌───────────────────────────────────────┐
│          DEEP NEURAL NETWORK          │
├───────────────────────────────────────┤
│                                       │
│  Input (raw pixels)                   │
│    ↓                                  │
│  Layer 1 (edges, corners)             │
│    ↓                                  │
│  Layer 2 (shapes, textures)           │
│    ↓                                  │
│  Layer 3 (parts: eyes, ears)          │
│    ↓                                  │
│  Layer 4 (objects: faces, cars)       │
│    ↓                                  │
│  Output (classification)              │
│                                       │
└───────────────────────────────────────┘

Each layer learns increasingly abstract features!
```

## Real-World Example: MNIST Classifier

```python
import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 784 input pixels → 128 hidden → 64 hidden → 10 outputs
        self.layer1 = nn.Linear(784, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Flatten image
        x = x.view(-1, 784)
        
        # Layer 1
        x = self.layer1(x)
        x = self.relu(x)
        
        # Layer 2
        x = self.layer2(x)
        x = self.relu(x)
        
        # Output layer
        x = self.layer3(x)
        return x

# Count parameters
model = MNISTNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Output: 109,386 parameters

# Breakdown:
# Layer1: 784×128 + 128 = 100,480
# Layer2: 128×64 + 64 = 8,256
# Layer3: 64×10 + 10 = 650
# Total: 109,386
```

## Analogy: Factory Assembly Line

```
Neural Network = Assembly Line

Input Layer:        Raw materials arrive
Hidden Layer 1:     Basic processing
Hidden Layer 2:     Assembly
Hidden Layer 3:     Quality control
Output Layer:       Final product

Each "worker" (neuron):
- Has specific job (weights define what to look for)
- Passes result to next station (activation)
- Entire line works together (forward pass)
```

## Key Concepts

### 1. Layer Depth
More layers = Can learn more complex patterns

### 2. Layer Width
More neurons per layer = More representational capacity

### 3. Architecture
How layers are connected and organized

### 4. Parameters
All weights and biases in the network

## Parameter Count Formula

For a fully connected layer:
```
Parameters = (inputs × outputs) + outputs
                 ↑                    ↑
              weights              biases
```

**Example:**
```
Layer: 100 inputs → 50 outputs
Weights: 100 × 50 = 5,000
Biases: 50
Total: 5,050 parameters
```

## Key Takeaway

```
Neuron = Basic computational unit
  - Takes weighted sum of inputs
  - Adds bias
  - Applies activation function
  
Network = Collection of neurons in layers
  - Each layer transforms data
  - Deep networks learn hierarchical features
  - Millions/billions of parameters work together
  
Magic happens through:
  ✓ Right architecture (how neurons connect)
  ✓ Right parameters (learned values)
  ✓ Right activation functions (non-linearity)
  
Result: Can approximate incredibly complex functions!
```
