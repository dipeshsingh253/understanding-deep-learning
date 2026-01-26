# Key Concepts

## Inference

**Inference**: Using a trained model to make predictions on new data (without updating parameters).

```
TRAINING:                    INFERENCE:
Input + Label               Input only
    ↓                          ↓
Update parameters          Fixed parameters
    ↓                          ↓
Learning                   Prediction

Slow, requires GPU         Fast, can use CPU
Changes model              Model unchanged
Needs labeled data         Works on unlabeled
```

```python
# Training
for epoch in range(100):
    for x, y in training_data:
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()  # Update parameters
        optimizer.step()

# Inference (after training)
model.eval()  # Set to evaluation mode
with torch.no_grad():  # Don't compute gradients
    prediction = model(new_input)
    # Model parameters don't change!
```

## Tensors

**Tensor**: Multi-dimensional array of numbers (generalization of matrices).

```
DIMENSIONS:

Scalar (0D):    5

Vector (1D):    [1, 2, 3, 4]

Matrix (2D):    [[1, 2, 3],
                 [4, 5, 6]]

Tensor (3D):    [[[1, 2],
                  [3, 4]],
                 [[5, 6],
                  [7, 8]]]

Tensor (4D+):   ... and beyond!
```

### Common Tensor Shapes

```python
import torch

# Image: [batch, channels, height, width]
image_batch = torch.randn(32, 3, 224, 224)
# 32 images, 3 color channels (RGB), 224×224 pixels

# Text: [batch, sequence_length]
text_batch = torch.randint(0, 10000, (32, 128))
# 32 sentences, each with 128 tokens

# Video: [batch, frames, channels, height, width]
video_batch = torch.randn(8, 16, 3, 128, 128)
# 8 videos, 16 frames each, 3 channels, 128×128 pixels
```

### Tensor Operations

```python
import numpy as np

# Creation
a = torch.tensor([1, 2, 3])
b = torch.zeros(3, 4)
c = torch.randn(2, 3, 4)  # Random normal

# Shape
print(a.shape)  # torch.Size([3])
print(c.shape)  # torch.Size([2, 3, 4])

# Reshaping
x = torch.randn(12)
y = x.view(3, 4)      # Reshape to 3×4
z = x.view(-1, 2)     # Auto-calculate first dim: 6×2

# Slicing
tensor = torch.randn(4, 5, 6)
print(tensor[0])       # First element: shape (5, 6)
print(tensor[:, 0])    # First column: shape (4, 6)
print(tensor[..., 0])  # Last dim: shape (4, 5)

# Math operations
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

c = a + b           # Element-wise: [5, 7, 9]
d = a * b           # Element-wise: [4, 10, 18]
e = a @ b           # Dot product: 32 (1×4 + 2×5 + 3×6)
```

### Why Tensors?

```
1. GPU Acceleration
   Tensors enable parallel computation
   1000× faster than loops

2. Automatic Differentiation
   Frameworks track operations
   Automatically compute gradients

3. Batch Processing
   Process multiple examples simultaneously
   Efficient use of hardware
```

## Noise

**Noise**: Random variation added to data or models.

### Types of Noise

#### 1. Data Noise
Natural variation in input data.

```python
# Clean data
clean_image = load_image('photo.jpg')

# Add Gaussian noise
noise = torch.randn_like(clean_image) * 0.1
noisy_image = clean_image + noise
```

#### 2. Training Noise (Regularization)
Intentionally added during training to improve generalization.

```python
# Dropout: Randomly zero out activations
dropout = nn.Dropout(p=0.5)  # Drop 50% of neurons
x = dropout(x)  # Random masking

# Data augmentation: Random transformations
def augment(image):
    if random() < 0.5:
        image = flip_horizontal(image)
    image = rotate(image, angle=random(-15, 15))
    image = add_noise(image, std=0.1)
    return image
```

#### 3. Sampling Noise (Generative Models)
Random input for generation.

```python
# VAE: Sample from latent distribution
mu, logvar = encoder(image)
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)  # Random noise
z = mu + eps * std  # Stochastic latent code

# Generate new image
generated = decoder(z)  # Different each time!
```

### Benefits of Noise

```
✓ Prevents overfitting
✓ Improves generalization
✓ Enables stochasticity in generation
✓ Makes models more robust
```

## Epochs

**Epoch**: One complete pass through the entire training dataset.

```
Training Dataset: 1000 examples
Batch Size: 100

1 Epoch = 10 batches (1000 / 100)

Training for 50 epochs:
- Each example seen 50 times
- Model has 50 opportunities to learn
```

```python
# Training loop with epochs

num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Average Loss: {avg_loss:.4f}")
```

### Epoch vs Iteration

```
Iteration (step): One batch processed
Epoch: All batches processed once

Example:
1000 examples, batch size 100

Iterations per epoch: 10
After 5 epochs: 50 iterations total

Epoch 1: iterations 1-10
Epoch 2: iterations 11-20
Epoch 3: iterations 21-30
Epoch 4: iterations 31-40
Epoch 5: iterations 41-50
```

## Batching

**Batch**: Group of training examples processed together.

```
┌──────────────────────────────────────┐
│         BATCHING STRATEGIES          │
├──────────────────────────────────────┤
│                                      │
│  Full Batch (batch = all data):     │
│  - Slow                              │
│  - Smooth convergence                │
│  - Memory intensive                  │
│                                      │
│  Stochastic (batch = 1):             │
│  - Fast updates                      │
│  - Noisy convergence                 │
│  - Poor GPU utilization              │
│                                      │
│  Mini-Batch (batch = 32-256):        │
│  - Good balance ✓                    │
│  - Efficient GPU use                 │
│  - Stable convergence                │
│                                      │
└──────────────────────────────────────┘
```

### Example

```python
# Dataset
X_train = torch.randn(1000, 784)  # 1000 examples
y_train = torch.randint(0, 10, (1000,))

# Create DataLoader for batching
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(X_train, y_train)
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training with batches
for epoch in range(10):
    for batch_x, batch_y in train_loader:
        print(f"Batch shape: {batch_x.shape}")  # torch.Size([32, 784])
        
        # Process entire batch at once (parallel)
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        
        # One update per batch
        loss.backward()
        optimizer.step()
```

### Batch Size Effects

```
Small batch (8):
+ Fast iterations
+ More updates per epoch
- Noisy gradients
- Underfits hardware

Large batch (512):
+ Stable gradients
+ Better hardware utilization
- Slower iterations
- May converge to sharp minima
- Higher memory requirements

Optimal: 32-256 for most tasks
```

## Training vs Inference Differences

```
┌────────────────────────────────────────────┐
│  TRAINING          vs       INFERENCE      │
├────────────────────────────────────────────┤
│                                            │
│  Needs labels               No labels      │
│  Updates params             Frozen params  │
│  Slow (hours/days)          Fast (ms)      │
│  Requires GPU               Can use CPU    │
│  Uses dropout               No dropout     │
│  Batch norm (train)         Batch norm     │
│                             (eval)         │
│  Stochastic                 Deterministic  │
│  Large batches              Single/small   │
│  Backpropagation            Forward only   │
│                                            │
└────────────────────────────────────────────┘
```

```python
# Training mode
model.train()  # Enable dropout, batch norm training
for x, y in train_data:
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()  # Compute gradients
    optimizer.step()  # Update parameters

# Inference mode
model.eval()  # Disable dropout, use batch norm stats
with torch.no_grad():  # Don't track gradients
    for x in test_data:
        pred = model(x)  # Just predict
        # No loss, no gradients, no updates
```

## GPU Acceleration

**GPU (Graphics Processing Unit)**: Specialized processor for parallel computation.

```
CPU vs GPU:

CPU:                          GPU:
┌──┐                         ┌──┬──┬──┬──┬──┬──┬──┬──┐
│  │ Few powerful cores      │  │  │  │  │  │  │  │  │
└──┘                         ├──┼──┼──┼──┼──┼──┼──┼──┤
                             │  │  │  │  │  │  │  │  │
Sequential tasks             ├──┼──┼──┼──┼──┼──┼──┼──┤
Fast per operation           │  │  │  │  │  │  │  │  │
                             └──┴──┴──┴──┴──┴──┴──┴──┘
                             Thousands of simple cores
                             Parallel tasks
                             Slower per core,
                             much faster overall
```

```python
# Move model and data to GPU

# Check if GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to GPU
model = model.to(device)

# Move data to GPU
data = data.to(device)
labels = labels.to(device)

# Everything happens on GPU now
output = model(data)  # GPU computation (fast!)

# Speedup example:
# CPU: 10 hours
# GPU: 30 minutes (20× faster)
```

## Practical Code Example

Putting it all together:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Create tensors
X_train = torch.randn(10000, 784)
y_train = torch.randint(0, 10, (10000,))

# 2. Setup batching
dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 3. Define model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),  # Noise for regularization
    nn.Linear(256, 10)
)

# 4. Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 5. Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training loop (epochs)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # Training mode
    total_loss = 0
    
    for batch_x, batch_y in train_loader:
        # Move batch to GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 7. Inference
model.eval()  # Evaluation mode
with torch.no_grad():
    test_input = torch.randn(1, 784).to(device)
    prediction = model(test_input)
    predicted_class = torch.argmax(prediction)
    print(f"Predicted class: {predicted_class}")
```

## Key Takeaway

```
Essential Concepts:

🔮 Inference
  - Use trained model
  - No parameter updates
  - Fast prediction

🔢 Tensors
  - Multi-dimensional arrays
  - Enable GPU acceleration
  - Foundation of deep learning

🎲 Noise
  - Random variation
  - Improves generalization
  - Enables generation

📅 Epochs
  - Complete data passes
  - More epochs = more learning
  - Risk: overfitting

📦 Batches
  - Process examples in groups
  - Balance speed and stability
  - Typical: 32-256 examples

⚡ GPU
  - Parallel processing
  - 10-100× faster than CPU
  - Essential for training

Master these → Understand deep learning workflow!
```
