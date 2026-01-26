# Parameters Explained

## What Are Parameters?

**Parameters** are the learnable numbers in a neural network that encode intelligence.

```
Parameters = Weights + Biases

These numbers:
✓ Are learned from data (not hand-coded)
✓ Encode patterns and knowledge
✓ ARE the model's intelligence
✓ Are what makes a trained model valuable
```

## Parameter Composition

```
┌────────────────────────────────────┐
│      NEURAL NETWORK PARAMETERS     │
├────────────────────────────────────┤
│                                    │
│  WEIGHTS (W)                       │
│  - Connection strengths            │
│  - Feature importance              │
│  - Pattern detectors               │
│  - 95%+ of parameters              │
│                                    │
│  BIASES (b)                        │
│  - Threshold adjustments           │
│  - Activation offsets              │
│  - ~5% of parameters               │
│                                    │
└────────────────────────────────────┘
```

## Weight Matrices: The Core

**Weight matrix** connects one layer to the next.

```
Layer with 3 neurons → Layer with 4 neurons

        n₁ ──w₁₁──→ m₁
        │  ╲╱       ╱│
        │  ╱╲      ╱ │
        │ ╱  ╲    ╱  │
        n₂────w₂₃──→ m₂
        │ ╲  ╱    ╱  │
        │  ╲╱    ╱   │
        │  ╱╲   ╱    │
        n₃────────→  m₃
                ╱    │
               ╱     │
              ╱      m₄

Weight matrix W (4×3):
     n₁   n₂   n₃
m₁ [w₁₁  w₁₂  w₁₃]
m₂ [w₂₁  w₂₂  w₂₃]
m₃ [w₃₁  w₃₂  w₃₃]
m₄ [w₄₁  w₄₂  w₄₃]

Total: 12 weights
```

## Counting Parameters

### Formula

For a fully connected layer:
```
Parameters = (input_size × output_size) + output_size
                    ↑                          ↑
                 Weights                    Biases
```

### Examples

```python
import torch.nn as nn

# Example 1: Simple layer
layer = nn.Linear(100, 50)
params = 100 * 50 + 50 = 5,050 parameters

# Example 2: Deeper network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)   # 784×256 + 256 = 200,960
        self.fc2 = nn.Linear(256, 128)   # 256×128 + 128 = 32,896
        self.fc3 = nn.Linear(128, 10)    # 128×10 + 10 = 1,290
        # Total: 235,146 parameters

# Count parameters
model = Net()
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")
# Output: 235,146
```

## What Do Parameters Represent?

### Weights: Feature Detectors

Each weight learns to detect specific patterns.

```python
# Example: Simple digit recognition
# First layer weights learn to detect edges

W1[0] might detect vertical edges:
[[ 1,  0, -1],
 [ 1,  0, -1],
 [ 1,  0, -1]]

W1[1] might detect horizontal edges:
[[ 1,  1,  1],
 [ 0,  0,  0],
 [-1, -1, -1]]

W1[2] might detect diagonal edges:
[[ 1,  0,  0],
 [ 0,  1,  0],
 [ 0,  0,  1]]
```

### Before Training vs After Training

```python
import numpy as np

# Before training (random)
W_before = np.random.randn(10, 784)
print("Before training:")
print(W_before[0, :10])  # First 10 weights of neuron 0
# Output: [ 0.0234, -0.1834,  0.9237, -0.0123, ...]
# Meaningless random numbers!

# After training (learned)
# (simplified representation)
W_after = trained_model.fc1.weight.data.numpy()
print("\nAfter training:")
print(W_after[0, :10])
# Output: [ 0.7492, -0.0923,  1.3374,  0.0012, ...]
# Carefully tuned to detect specific features!
```

## Real-World Model Sizes

```
┌─────────────────────────────────────────────┐
│          MODEL PARAMETER COUNTS             │
├─────────────────────────────────────────────┤
│                                             │
│ Small MNIST classifier:     ~100K params    │
│ ResNet-18 (image):          ~11M params     │
│ BERT-base (text):           ~110M params    │
│ GPT-2 (text):               ~1.5B params    │
│ GPT-3 (text):               ~175B params    │
│ GPT-4 (estimated):          ~1.76T params   │
│                                             │
│ 1B params ≈ 4GB of memory (float32)         │
│                                             │
└─────────────────────────────────────────────┘
```

## Parameter Storage

```python
# How parameters are stored in memory

# Single precision (float32): 4 bytes per parameter
model_size_bytes = num_parameters * 4

# Example: GPT-3 (175B parameters)
size_gb = 175_000_000_000 * 4 / (1024**3)
print(f"GPT-3 size: {size_gb:.0f} GB")
# Output: ~700 GB (in float32)

# Half precision (float16): 2 bytes per parameter
size_gb_half = 175_000_000_000 * 2 / (1024**3)
print(f"GPT-3 size (float16): {size_gb_half:.0f} GB")
# Output: ~350 GB
```

## Parameter Initialization

Why and how we initialize parameters.

```python
# Bad: All zeros
W = np.zeros((128, 784))  # All neurons learn same thing!

# Bad: All same value
W = np.ones((128, 784)) * 0.5  # Still symmetry problem

# Good: Random small values
W = np.random.randn(128, 784) * 0.01

# Better: Xavier/Glorot initialization
# (accounts for layer sizes)
W = np.random.randn(128, 784) / np.sqrt(784)

# Best for ReLU: He initialization
W = np.random.randn(128, 784) * np.sqrt(2.0 / 784)
```

**Why random initialization matters:**
```
All zeros → All neurons compute same thing → No learning
Random → Each neuron learns different features → Rich representation
```

## Parameter Updates During Training

```python
# Training updates parameters iteratively

# Initial (random)
W = np.random.randn(100, 784)
print(f"Initial W[0,0]: {W[0,0]:.4f}")
# Output: 0.0234

# After 1 update
gradient = compute_gradient()
W = W - learning_rate * gradient
print(f"After update 1: {W[0,0]:.4f}")
# Output: 0.0198 (changed slightly)

# After 1000 updates
for i in range(1000):
    gradient = compute_gradient()
    W = W - learning_rate * gradient

print(f"After 1000 updates: {W[0,0]:.4f}")
# Output: 0.7492 (significantly different, optimized!)
```

## Visualizing Parameter Updates

```
Training Progress (one weight):

Epoch 0:    W = 0.023 ●────────────────────
                                         target

Epoch 10:   W = 0.145 ────●──────────────
                                         target

Epoch 50:   W = 0.623 ──────────●────────
                                         target

Epoch 100:  W = 0.742 ─────────────●─────
                                         target

Epoch 500:  W = 0.750 ──────────────●────
                                         target

Converges to optimal value!
```

## Where Intelligence Lives

```
┌──────────────────────────────────────────┐
│     WHERE IS THE INTELLIGENCE?           │
├──────────────────────────────────────────┤
│                                          │
│  NOT in the code:                        │
│  ✗ Architecture is just structure        │
│  ✗ Forward pass is just matrix math      │
│  ✗ Anyone can write this code            │
│                                          │
│  YES in the parameters:                  │
│  ✓ 175B carefully tuned numbers          │
│  ✓ Trained on trillions of tokens        │
│  ✓ Millions of dollars in compute        │
│  ✓ Encodes language understanding        │
│                                          │
│  Code: Open source, free                 │
│  Parameters: Proprietary, valuable       │
│                                          │
└──────────────────────────────────────────┘
```

## Practical Example: Understanding a Trained Weight

```python
# Load trained model
import torch
model = torch.load('trained_model.pth')

# Examine first layer weights
W1 = model.fc1.weight.data  # Shape: (128, 784)

# What does neuron 5 respond to?
neuron_5_weights = W1[5]  # 784 weights

# Reshape to image (28×28)
weight_image = neuron_5_weights.reshape(28, 28)

# Visualize (high weights = strong response)
import matplotlib.pyplot as plt
plt.imshow(weight_image, cmap='RdBu')
plt.title('Neuron 5 detects this pattern')
plt.show()

# Might show: Horizontal line detector
#   [blue  blue  blue  ...]  ← negative weights
#   [red   red   red   ...]  ← positive weights  
#   [blue  blue  blue  ...]  ← negative weights
# This neuron fires when it sees horizontal edges!
```

## Matrix Multiplication: Efficient Computation

```python
# Why weight matrices are efficient

# Slow way: Loop through neurons
output = []
for neuron_idx in range(128):
    neuron_output = 0
    for input_idx in range(784):
        neuron_output += W[neuron_idx, input_idx] * input[input_idx]
    neuron_output += bias[neuron_idx]
    output.append(neuron_output)

# Fast way: Matrix multiplication
output = W @ input + bias  # Single operation!

# GPU can parallelize this: 100-1000× faster
```

## Parameter Sharing

Some architectures share parameters for efficiency.

```python
# Convolutional layer: Shares weights across image
# Instead of 1M unique weights, uses same 9 weights everywhere

conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
# Only 3×3×32 = 288 weights (+ 32 biases)
# Applied across entire image → Spatial invariance

# vs Fully connected:
fc = nn.Linear(784, 256)
# 784×256 = 200,704 weights!

# Sharing reduces parameters and improves generalization
```

## Embeddings: Special Parameters

```python
# Word embeddings: Parameters that represent words

vocab_size = 50000
embedding_dim = 300

embedding_layer = nn.Embedding(vocab_size, embedding_dim)
# Parameters: 50000 × 300 = 15,000,000

# Each word gets a learnable vector
word_vector = embedding_layer(word_id)
# Shape: (300,)

# Example (learned):
# "king" → [0.2, 0.7, -0.3, 0.1, ...]
# "queen" → [0.3, 0.6, -0.2, 0.15, ...]
# Similar words have similar vectors!
```

## Analogy: Recipe vs Ingredients

```
NEURAL NETWORK = COOKING

Architecture (code):     Recipe instructions
  - "Mix ingredients"    Fixed, doesn't change
  - "Bake at 350°F"      Anyone can follow
  - "Cool for 10 min"    

Parameters (numbers):    Ingredient proportions
  - 2.5 cups flour       Precisely measured
  - 1.75 tsp salt        Critical values
  - 3.2 oz butter        Learned through practice

Training:                Recipe development
  - Try different amounts
  - Taste and adjust
  - Optimize proportions

The intelligence is in knowing EXACTLY how much of each ingredient!
```

## Key Formulas

**Parameter count (fully connected):**
```
P = (n_in × n_out) + n_out
```

**Memory requirement:**
```
Memory (bytes) = num_params × bytes_per_param
```

**Parameter update:**
```
θ_new = θ_old - learning_rate × gradient
```

**Gradient computation:**
```
gradient = ∂Loss/∂θ
```

## Key Takeaway

```
Parameters = The Intelligence

🔢 Numbers, not code
  - Weights: 95%+ of parameters
  - Biases: ~5% of parameters

💾 What gets saved
  - Model file = Parameter values
  - Architecture = Separate code

🎓 What training learns
  - Optimize billions of numbers
  - Find patterns in data
  - Encode knowledge

💰 What's valuable
  - Code: Free/open source
  - Trained parameters: Worth millions

🧠 Where intelligence lives
  - NOT in the architecture
  - YES in the parameter values
  - 175B perfectly tuned numbers

Training = Finding the right numbers
Model = The found numbers + How to use them
```
