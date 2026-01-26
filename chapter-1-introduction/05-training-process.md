# Training Process

## What is Training?

**Training** is the process of adjusting a model's parameters to minimize prediction errors on training data.

```
Before Training:         After Training:
Parameters = Random  →   Parameters = Learned
Model = Dumb            Model = Smart
```

## The Training Loop

```
┌─────────────────────────────────────────┐
│         TRAINING CYCLE                  │
├─────────────────────────────────────────┤
│                                         │
│  1. Initialize: Random parameters       │
│         ↓                               │
│  2. Forward Pass: Make predictions      │
│         ↓                               │
│  3. Calculate Loss: How wrong?          │
│         ↓                               │
│  4. Backward Pass: Compute gradients    │
│         ↓                               │
│  5. Update: Adjust parameters           │
│         ↓                               │
│  6. Repeat: Until loss is low           │
│                                         │
└─────────────────────────────────────────┘
```

## Step-by-Step Process

### Step 1: Random Initialization

Start with random parameter values.

```python
import numpy as np

# Random weights and biases
W1 = np.random.randn(128, 784) * 0.01  # Small random values
b1 = np.zeros(128)
W2 = np.random.randn(10, 128) * 0.01
b2 = np.zeros(10)

print("Initial W1[0,0]:", W1[0,0])  # e.g., 0.0043 (random)
```

**Why random?**
- Breaking symmetry (all neurons should learn different features)
- Starting point for optimization

### Step 2: Forward Pass

Compute predictions by passing data through the network.

```python
def forward_pass(x, W1, b1, W2, b2):
    # Layer 1
    z1 = W1 @ x + b1
    h1 = np.maximum(0, z1)  # ReLU
    
    # Layer 2
    z2 = W2 @ h1 + b2
    output = z2
    
    return output, (z1, h1, z2)  # Save for backward pass

# Example
x = train_data[0]  # First training example
y_true = train_labels[0]  # True label

y_pred, cache = forward_pass(x, W1, b1, W2, b2)
print(f"Prediction: {y_pred}")  # Random initially
print(f"True label: {y_true}")
```

### Step 3: Calculate Loss

Measure how wrong the prediction is.

```
Loss = Distance between prediction and truth
```

**Mean Squared Error (Regression):**
```python
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

loss = mse_loss(y_pred, y_true)
print(f"Loss: {loss}")  # High initially (model is random)
```

**Cross-Entropy Loss (Classification):**
```python
def cross_entropy_loss(logits, y_true):
    # Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # Negative log likelihood
    loss = -np.log(probs[y_true])
    return loss

loss = cross_entropy_loss(y_pred, y_true)
```

### Step 4: Backpropagation

Compute gradients: how each parameter affects the loss.

```
Gradient = ∂Loss/∂parameter

Tells us:
- Which direction to adjust parameter
- How much to adjust it
```

**Chain Rule:**
```
∂Loss/∂W1 = ∂Loss/∂output × ∂output/∂h1 × ∂h1/∂W1
```

```python
def backward_pass(x, y_true, y_pred, cache, W2):
    z1, h1, z2 = cache
    m = x.shape[0]  # Batch size
    
    # Output layer gradient
    dz2 = y_pred - y_true  # Derivative of loss
    dW2 = dz2 @ h1.T / m
    db2 = np.mean(dz2, axis=1)
    
    # Hidden layer gradient
    dh1 = W2.T @ dz2
    dz1 = dh1 * (z1 > 0)  # ReLU derivative
    dW1 = dz1 @ x.T / m
    db1 = np.mean(dz1, axis=1)
    
    return dW1, db1, dW2, db2

# Compute gradients
grads = backward_pass(x, y_true, y_pred, cache, W2)
dW1, db1, dW2, db2 = grads

print(f"Gradient dW1[0,0]: {dW1[0,0]}")  # e.g., -0.0023
```

### Step 5: Gradient Descent

Update parameters in the direction that reduces loss.

```
Formula:
θ_new = θ_old - learning_rate × gradient
```

```python
# Hyperparameter
learning_rate = 0.01

# Update weights and biases
W1 = W1 - learning_rate * dW1
b1 = b1 - learning_rate * db1
W2 = W2 - learning_rate * dW2
b2 = b2 - learning_rate * db2

print(f"Updated W1[0,0]: {W1[0,0]}")  # Slightly different now
```

**Visualization:**
```
Loss surface:
       Loss
         ↑
      ╱│╲│╱╲
     ╱ │ │ │ ╲
    ╱  │ ●─→ ╲     ● = current position
   ╱   │  ↓   ╲    → = gradient direction
  ╱────┼──●────╲   ↓ = move toward minimum
       parameter

Gradient points to steepest descent
```

### Step 6: Iterate

Repeat for many epochs (passes through entire dataset).

```python
def train(X_train, y_train, epochs=100, learning_rate=0.01):
    # Initialize parameters
    W1 = np.random.randn(128, 784) * 0.01
    b1 = np.zeros(128)
    W2 = np.random.randn(10, 128) * 0.01
    b2 = np.zeros(10)
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Loop through all training examples
        for i in range(len(X_train)):
            x = X_train[i]
            y_true = y_train[i]
            
            # Forward pass
            y_pred, cache = forward_pass(x, W1, b1, W2, b2)
            
            # Calculate loss
            loss = cross_entropy_loss(y_pred, y_true)
            total_loss += loss
            
            # Backward pass
            dW1, db1, dW2, db2 = backward_pass(x, y_true, y_pred, cache, W2)
            
            # Update parameters
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
        
        # Print progress
        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return W1, b1, W2, b2

# Train the model
W1, b1, W2, b2 = train(X_train, y_train, epochs=10)
```

## Full Example with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    for batch_x, batch_y in train_loader:
        # 1. Forward pass
        outputs = model(batch_x)
        
        # 2. Calculate loss
        loss = criterion(outputs, batch_y)
        
        # 3. Zero gradients (important!)
        optimizer.zero_grad()
        
        # 4. Backward pass (compute gradients)
        loss.backward()
        
        # 5. Update parameters
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# After training, parameters are optimized!
```

## Batch Training

Process multiple examples together for efficiency.

```
Single example (slow):
for each example:
    forward → backward → update

Batch (fast):
for each batch of 32 examples:
    forward all 32 → average gradients → update
```

```python
# Mini-batch gradient descent
batch_size = 32

for epoch in range(epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    
    for i in range(0, len(X_train), batch_size):
        # Get batch
        batch_idx = indices[i:i+batch_size]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]
        
        # Forward pass (vectorized for batch)
        y_pred, cache = forward_pass(X_batch, W1, b1, W2, b2)
        
        # Backward pass (average over batch)
        grads = backward_pass(X_batch, y_batch, y_pred, cache, W2)
        
        # Update
        W1 -= learning_rate * grads[0]
        # ... update other parameters
```

## Optimization Algorithms

### Stochastic Gradient Descent (SGD)

```python
# Basic SGD
θ = θ - η∇L
```

### SGD with Momentum

```python
# Accumulates velocity
v = β×v + ∇L
θ = θ - η×v
```

### Adam (Adaptive Moment Estimation)

```python
# Most popular optimizer
# Adapts learning rate per parameter
m = β₁×m + (1-β₁)×∇L          # First moment
v = β₂×v + (1-β₂)×(∇L)²       # Second moment
θ = θ - η×m/√(v + ε)
```

```python
# PyTorch usage
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Learning Rate

**Critical hyperparameter**: controls update step size.

```
Too small (η=0.0001):      Too large (η=1.0):
  Loss                       Loss
   ↑                          ↑
   │╲                         │  ╱╲
   │ ╲                        │ ╱  ╲╱╲
   │  ╲                       │╱      ╲
   │   ╲___                   │        ╱╲
   └────────→                 └───────────→
   Slow convergence          Diverges (unstable)

Just right (η=0.01):
  Loss
   ↑
   │╲
   │ ╲___
   │     ╲___
   └────────→
   Fast, stable convergence
```

## Training Visualization

```
EPOCH 1:  Loss = 2.305 (random predictions)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EPOCH 2:  Loss = 1.823 (learning...)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EPOCH 5:  Loss = 0.892 (getting better)
━━━━━━━━━━━━━━━━

EPOCH 10: Loss = 0.312 (pretty good)
━━━━━

EPOCH 50: Loss = 0.098 (excellent!)
━━
```

## Why Only Parameters Change

```
ARCHITECTURE (stays fixed):
┌─────────────────────────┐
│ Layer 1: 784→128 (ReLU) │  ← Structure defined by programmer
│ Layer 2: 128→10         │  ← Never changes
└─────────────────────────┘

PARAMETERS (change during training):
┌─────────────────────────┐
│ W1: [0.02, -0.13, ...]  │  ← Random initially
│ b1: [0, 0, ...]         │
│ W2: [0.05, 0.21, ...]   │  ← Optimized during training
│ b2: [0, 0, ...]         │
└─────────────────────────┘

Training adjusts 109,386 numbers (parameters)
Architecture (2 layers, ReLU) never changes!
```

## Analogy: Tuning a Radio

```
Training a neural network = Tuning a radio

Architecture:    Radio hardware (fixed)
Parameters:      Tuning knobs (adjustable)
Loss:            Static/noise level
Gradient:        Which way to turn knobs
Learning rate:   How much to turn each knob
Training:        Turning knobs to reduce static

After training: All knobs perfectly tuned → Clear signal!
```

## Common Challenges

### Overfitting
Model memorizes training data, fails on new data.

**Solution:** Regularization, dropout, more data

### Underfitting  
Model too simple, can't learn patterns.

**Solution:** Bigger model, more layers, train longer

### Vanishing Gradients
Gradients become too small in deep networks.

**Solution:** ReLU, batch normalization, residual connections

### Exploding Gradients
Gradients become too large.

**Solution:** Gradient clipping, careful initialization

## Key Formulas

**Forward Pass:**
```
z = Wx + b
a = σ(z)
```

**Loss (MSE):**
```
L = (1/n)Σ(ŷᵢ - yᵢ)²
```

**Gradient Descent:**
```
θ ← θ - η∇L(θ)
```

**Backpropagation (chain rule):**
```
∂L/∂W = ∂L/∂output × ∂output/∂W
```

## Key Takeaway

```
Training Process:

1. Start: Random parameters (dumb model)
2. Loop:
   - Forward: Make predictions
   - Loss: Measure errors
   - Backward: Compute gradients
   - Update: Adjust parameters
3. Result: Optimized parameters (smart model)

The architecture never changes!
Only the parameter values change!

After millions of updates:
- Parameters encode learned knowledge
- Model makes accurate predictions
- Intelligence emerges from optimized numbers
```
