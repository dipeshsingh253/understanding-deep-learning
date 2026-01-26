# Latent Variables

## What Are Latent Variables?

**Latent variables** are hidden representations learned by a model that capture essential features of data in compressed form.

```
Latent = "Hidden" or "Not directly observed"

Input → Encoder → Latent Variables → Decoder → Output
                      ↑
                Hidden representation
               (compressed, meaningful)
```

## Core Concept

```
┌─────────────────────────────────────────┐
│         LATENT VARIABLES                │
├─────────────────────────────────────────┤
│                                         │
│  Observed Data (high-dimensional):      │
│  - Images: 1000s of pixels              │
│  - Text: 1000s of words                 │
│  - Audio: Millions of samples           │
│                                         │
│           ↓ Compression ↓               │
│                                         │
│  Latent Space (low-dimensional):        │
│  - 10-1000 meaningful numbers           │
│  - Captures essence of data             │
│  - Removes redundancy                   │
│                                         │
│           ↓ Reconstruction ↓            │
│                                         │
│  Reconstructed Data:                    │
│  - Close to original                    │
│  - Generated from latent code           │
│                                         │
└─────────────────────────────────────────┘
```

## Visualization

```
HIGH-DIMENSIONAL DATA          LATENT SPACE
(e.g., 784 pixels)            (e.g., 2 dimensions)

●  ●  ●  ●  ●                      ●
●  ●  ●  ●  ●                    ● ● ●
●  ●  ●  ●  ●    Encode         ●  ●  ●
●  ●  ●  ●  ●    ──────→          ● ●
28×28 = 784                        2D

Much easier to:
- Visualize
- Interpret
- Manipulate
- Generate new samples
```

## Autoencoders: Learning Latent Representations

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        
        # Encoder: 784 → 32 (compression)
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),  # Bottleneck
        )
        
        # Decoder: 32 → 784 (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Compress to latent space
        latent = self.encoder(x)
        
        # Reconstruct from latent space
        reconstructed = self.decoder(latent)
        
        return reconstructed, latent

# Usage
model = Autoencoder(latent_dim=32)
image = torch.randn(784)  # Input image

reconstructed, latent_code = model(image)

print(f"Original: {image.shape}")        # torch.Size([784])
print(f"Latent: {latent_code.shape}")    # torch.Size([32])
print(f"Reconstructed: {reconstructed.shape}")  # torch.Size([784])

# 784 → 32 → 784
# Compressed 24× while preserving key information!
```

## What Makes Latent Variables Special?

### 1. Dimensionality Reduction

```
Original: 1000 dimensions (redundant, noisy)
Latent: 10 dimensions (essential, meaningful)
```

### 2. Disentanglement

Each latent dimension captures a distinct feature.

```python
# Example: Face generation
# latent[0] → Age
# latent[1] → Gender
# latent[2] → Smile
# latent[3] → Glasses
# latent[4] → Hair color
# ...

# Change one dimension = Change one feature
latent_code[2] += 1.0  # Add more smile
latent_code[4] -= 0.5  # Darken hair

new_face = decoder(latent_code)  # Modified face!
```

### 3. Smooth Interpolation

```python
# Interpolate between two images

latent_A = encoder(image_A)  # Person A
latent_B = encoder(image_B)  # Person B

# Linear interpolation
for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
    latent_interpolated = (1-alpha)*latent_A + alpha*latent_B
    interpolated_image = decoder(latent_interpolated)
    # Smooth transition from A to B!
```

## Real-World Example: VAE (Variational Autoencoder)

```python
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        
        # Encoder outputs mean and variance
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)      # Mean
        self.fc_logvar = nn.Linear(400, latent_dim)  # Log variance
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # Sample from distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # z ~ N(mu, std²)
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        # Encode to distribution
        mu, logvar = self.encode(x)
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decode(z)
        
        return reconstructed, mu, logvar

# VAE can GENERATE new samples!
vae = VAE(latent_dim=20)

# Sample random latent code
z = torch.randn(20)  # Random point in latent space
generated_image = vae.decode(z)  # New image!
```

## Latent Space Visualization

```
2D Latent Space for MNIST:

        Digit clusters:
        
    7  7    1  1        4  4
   7    7  1  1  1    4    4
    7  7    1  1        4  4
    
  2  2                    9  9
 2    2                  9    9
  2  2                    9  9
  
    3  3    0  0        5  5
   3    3  0    0      5    5
    3  3    0  0        5  5

Similar digits cluster together!
Smooth transitions between clusters!
```

## Generative Models: Text-to-Image

```python
# Simplified text-to-image model architecture

class TextToImage:
    def __init__(self):
        self.text_encoder = TextEncoder()    # Text → latent
        self.image_decoder = ImageDecoder()  # Latent → image
    
    def generate(self, text_prompt):
        # Encode text to latent space
        latent_code = self.text_encoder(text_prompt)
        # latent_code: 512-dimensional vector
        
        # Decode latent to image
        image = self.image_decoder(latent_code)
        
        return image

# Usage
model = TextToImage()
image = model.generate("A cat wearing sunglasses")

# Process:
# "A cat wearing sunglasses" → [0.2, -0.5, 0.8, ...] → 🖼️
#                                     ↑
#                            Latent representation
#                    (captures: cat, sunglasses, style)
```

## Latent Arithmetic

Manipulating latent codes enables semantic operations.

```python
# Famous example: Word2Vec
# king - man + woman = queen

# Visual example: Face latent space
latent_smiling_man = encoder(smiling_man_image)
latent_neutral_man = encoder(neutral_man_image)
latent_neutral_woman = encoder(neutral_woman_image)

# Compute "smile vector"
smile_vector = latent_smiling_man - latent_neutral_man

# Add smile to woman
latent_smiling_woman = latent_neutral_woman + smile_vector
smiling_woman_image = decoder(latent_smiling_woman)
# Result: Woman smiling!
```

## Dimensionality Comparison

```
┌───────────────────────────────────────────┐
│      DIMENSIONALITY EXAMPLES              │
├───────────────────────────────────────────┤
│                                           │
│  Input → Latent → Compression Ratio       │
│                                           │
│  MNIST images:                            │
│  784 → 32 → 24.5×                         │
│                                           │
│  Large images:                            │
│  1024×1024×3 → 512 → 6144×                │
│                                           │
│  Text (BERT):                             │
│  30000 vocab → 768 → 39×                  │
│                                           │
│  Audio:                                   │
│  16000 samples/sec → 128 → 125×           │
│                                           │
└───────────────────────────────────────────┘
```

## Training Autoencoders

```python
# Training to learn latent representations

model = Autoencoder(latent_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Reconstruction loss

for epoch in range(100):
    for batch in train_loader:
        # Forward pass
        reconstructed, latent = model(batch)
        
        # Loss: How different is reconstruction?
        loss = criterion(reconstructed, batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# After training:
# - Encoder learned to compress meaningfully
# - Decoder learned to reconstruct from compressed form
# - Latent space is structured and meaningful
```

## Use Cases

### 1. Dimensionality Reduction
```python
# Visualize high-dimensional data
latent_codes = encoder(high_dim_data)
plt.scatter(latent_codes[:, 0], latent_codes[:, 1])
plt.title("Data in 2D latent space")
```

### 2. Anomaly Detection
```python
# Unusual data has high reconstruction error
reconstructed, latent = model(test_data)
error = torch.mean((reconstructed - test_data)**2)
if error > threshold:
    print("Anomaly detected!")
```

### 3. Data Generation
```python
# Sample random latent codes → Generate new data
random_latent = torch.randn(10, latent_dim)
generated_samples = decoder(random_latent)
```

### 4. Transfer Learning
```python
# Use learned encoder for another task
pretrained_encoder = autoencoder.encoder
features = pretrained_encoder(images)
classifier = train_classifier(features, labels)
```

## Latent Space Structure

```
UNSTRUCTURED (before training):
Random noise, no meaning

  ●   ●     ●
    ●   ●       ●
  ●       ●   ●
      ●     ●

STRUCTURED (after training):
Organized, meaningful regions

  [Dogs]      [Cats]
    ●●●         ◆◆◆
    ●●●         ◆◆◆
    
  [Birds]     [Fish]
    ▲▲▲         ○○○
    ▲▲▲         ○○○
    
Smooth transitions between regions!
```

## Analogy: Compression Algorithm

```
Latent Variables = Smart Compression

ZIP file compression:
  - Removes redundancy
  - Keeps essential info
  - Can reconstruct original

Latent representations:
  - Removes pixel redundancy
  - Keeps semantic meaning
  - Can reconstruct (approximately)
  - BONUS: Can generate new samples!

ZIP: "dog.jpg" → [compressed bits] → "dog.jpg"
VAE: Dog image → [32 numbers] → Dog image
     Also: [random 32 numbers] → New dog image!
```

## Key Formulas

**Autoencoder:**
```
z = Encoder(x)
x̂ = Decoder(z)
Loss = ||x - x̂||²
```

**VAE:**
```
μ, σ² = Encoder(x)
z ~ N(μ, σ²)
x̂ = Decoder(z)
Loss = ||x - x̂||² + KL(N(μ,σ²) || N(0,1))
```

**Latent interpolation:**
```
z_interpolated = (1-α)z₁ + αz₂, α ∈ [0,1]
```

## Key Takeaway

```
Latent Variables = Compressed Meaningful Representations

🗜️ Compression
  - High-dimensional → Low-dimensional
  - Remove redundancy
  - Keep essential features

🎯 Structure
  - Similar data → Similar latent codes
  - Smooth interpolation
  - Disentangled features

🎨 Generation
  - Sample latent space → New data
  - Control features independently
  - Enables creativity

🔧 Applications
  - Dimensionality reduction
  - Anomaly detection
  - Data generation
  - Feature learning
  - Transfer learning

Magic: Transform complex data into simple,
       meaningful numbers that capture essence!
```
