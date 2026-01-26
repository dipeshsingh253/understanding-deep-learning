# Unsupervised Learning

## Definition

**Unsupervised Learning**: Training a model on unlabeled data to discover hidden patterns, structures, or relationships.

```
Input (X) → Model finds patterns → Structure/Groups/Representations
```

## Core Difference from Supervised Learning

```
SUPERVISED:
Input + Label → Learn mapping
"Cat image" + "cat" → Predict labels

UNSUPERVISED:
Input only → Find patterns
"Cat image" → Discover that some images are similar
```

## Main Types

### 1. Clustering

**Group similar data points together.**

```
┌─────────────────────────────────────────┐
│           CLUSTERING                    │
├─────────────────────────────────────────┤
│                                         │
│  Unlabeled Data:                        │
│      •  •    ◆  ◆                      │
│    •  •  •   ◆  ◆  ◆                   │
│      •  •       ◆  ◆                   │
│                                         │
│  Algorithm discovers groups:            │
│   ┌─────────┐  ┌─────────┐            │
│   │ •  •    │  │ ◆  ◆    │            │
│   │•  •  •  │  │◆  ◆  ◆  │            │
│   │ •  •    │  │  ◆  ◆   │            │
│   └─────────┘  └─────────┘            │
│   Cluster 1     Cluster 2              │
│                                         │
└─────────────────────────────────────────┘
```

**Example: Customer Segmentation**
```python
import numpy as np
from sklearn.cluster import KMeans

# Customer data (unlabeled)
customers = np.array([
    [25, 50000],   # age, income
    [30, 60000],
    [45, 120000],
    [50, 130000],
    [28, 55000],
    [48, 125000]
])

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(customers)

print("Cluster assignments:", clusters)
# Output: [0, 0, 1, 1, 0, 1]
# Cluster 0: Young, lower income
# Cluster 1: Older, higher income

# Cluster centers
print("Cluster centers:\n", kmeans.cluster_centers_)
```

### 2. Dimensionality Reduction

**Compress data while preserving important information.**

```
HIGH-DIMENSIONAL DATA (1000 features)
        ↓ Compression
LOW-DIMENSIONAL DATA (2-3 features)
```

**Example: Principal Component Analysis (PCA)**
```python
from sklearn.decomposition import PCA
import numpy as np

# High-dimensional data: 1000 features
data = np.random.randn(100, 1000)  # 100 samples, 1000 features

# Reduce to 2 dimensions
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

print(f"Original shape: {data.shape}")      # (100, 1000)
print(f"Reduced shape: {reduced_data.shape}") # (100, 2)

# Explained variance
print(f"Variance captured: {pca.explained_variance_ratio_.sum():.2%}")
```

### 3. Anomaly Detection

**Identify unusual or outlier data points.**

```python
from sklearn.ensemble import IsolationForest

# Network traffic data
traffic = np.array([
    [100, 50],   # normal
    [110, 55],   # normal
    [105, 52],   # normal
    [900, 800],  # ANOMALY (DDoS attack?)
    [98, 48],    # normal
])

# Train anomaly detector
detector = IsolationForest(contamination=0.2)
predictions = detector.fit_predict(traffic)

# -1 = anomaly, 1 = normal
print("Anomalies:", predictions)
# Output: [1, 1, 1, -1, 1]
```

## Real-World Example: Autoencoder

**Learn compressed representation of images.**

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder: Compress 784 → 32
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Bottleneck (latent space)
            nn.ReLU()
        )
        
        # Decoder: Reconstruct 32 → 784
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Compress
        latent = self.encoder(x)
        # Reconstruct
        reconstructed = self.decoder(latent)
        return reconstructed

# Training (no labels needed!)
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for images in unlabeled_data:
        # Try to reconstruct the input
        reconstructed = model(images)
        
        # Loss: How different is reconstruction from original?
        loss = criterion(reconstructed, images)
        
        # Update to improve reconstruction
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Use learned representation
latent_repr = model.encoder(test_image)  # 32-dim vector
print(f"Compressed: {latent_repr.shape}")  # torch.Size([32])
```

## Analogy: Organizing a Library

Imagine organizing books without pre-defined categories:

```
SUPERVISED (with labels):
"Put all 'Mystery' books here"
"Put all 'Romance' books there"

UNSUPERVISED (no labels):
Read books and notice:
- Some books have detectives → Group 1
- Some books have love stories → Group 2
- Some books have space ships → Group 3

You discovered categories by finding patterns!
```

## Common Algorithms

### K-Means Clustering

```
Algorithm:
1. Pick K random cluster centers
2. Assign each point to nearest center
3. Move centers to mean of assigned points
4. Repeat steps 2-3 until convergence

Formula:
Minimize: Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
```

### Principal Component Analysis (PCA)

```
Algorithm:
1. Center the data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors (principal components)
4. Project data onto top components

Goal: Maximize variance in projected space
```

### Autoencoders

```
Architecture:
Input → Encoder → Latent (compressed) → Decoder → Output

Loss: Reconstruction error
L = ||x - decoder(encoder(x))||²

Learns: Efficient representation in latent space
```

## Visualization

```
SUPERVISED LEARNING:
┌──────┐     ┌───────┐
│ Data │────→│ Model │
│+Label│     │       │───→ Prediction
└──────┘     └───────┘
              ↑
         Minimize error with labels

UNSUPERVISED LEARNING:
┌──────┐     ┌───────┐
│ Data │────→│ Model │───→ Patterns/
│(only)│     │       │     Structure/
└──────┘     └───────┘     Clusters
              ↑
         Discover inherent structure
```

## Use Cases

| Application | What It Discovers |
|-------------|-------------------|
| Customer segmentation | Similar customer groups |
| Image compression | Essential features |
| Recommendation systems | Item similarities |
| Anomaly detection | Unusual patterns |
| Topic modeling | Themes in documents |
| Gene expression analysis | Related genes |

## Advanced: Generative Models

```python
# Variational Autoencoder (VAE)
# Can generate NEW data similar to training data

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()  # → mean, variance
        self.decoder = Decoder()  # latent → image
    
    def forward(self, x):
        # Encode to distribution
        mu, logvar = self.encoder(x)
        
        # Sample from distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization trick
        
        # Decode
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar

# After training on faces:
# Can generate NEW faces by sampling latent space!
random_latent = torch.randn(1, 32)
new_face = model.decoder(random_latent)
```

## Dimensionality Reduction Visualization

```
Original 3D data:          Projected to 2D:
      z                         y
      ↑                         ↑
     ●│ ●                       ● ●
    ● │  ●                     ●   ●
   ●  │   ●      PCA          ●     ●
  ●───┼────●●  ─────→        ●───────●●
      │  ●  ●                     ●  ●
      └──────→ x                  └────→ x

Information preserved, dimensions reduced!
```

## Key Challenges

### 1. Evaluation
No labels = harder to measure success.

**Solutions:**
- Silhouette score (clustering)
- Reconstruction error (autoencoders)
- Domain expert validation

### 2. Choosing Parameters
How many clusters? How many dimensions?

**Solutions:**
- Elbow method
- Cross-validation
- Information criteria (AIC, BIC)

### 3. Interpretability
What do discovered patterns mean?

**Solution:** Visualize and analyze clusters/features

## Comparison Table

| Aspect | Supervised | Unsupervised |
|--------|-----------|--------------|
| Data | Labeled (X, Y) | Unlabeled (X only) |
| Goal | Predict Y from X | Find structure in X |
| Evaluation | Accuracy, F1-score | Silhouette, reconstruction |
| Examples | Classification, Regression | Clustering, Dimensionality reduction |
| Cost | Expensive (labeling) | Cheaper (no labels) |

## Key Formulas

**K-Means Objective:**
```
J = Σₖ Σₓ∈Cₖ ||x - μₖ||²
```

**PCA (maximize variance):**
```
Cov(X) = (1/n)XᵀX
PC = eigenvectors of Cov(X)
```

**Autoencoder Loss:**
```
L = ||x - decoder(encoder(x))||² + regularization
```

## Key Takeaway

```
Unsupervised Learning = Finding Hidden Patterns

No teacher, no labels
Model explores data and discovers:
- Natural groupings (clustering)
- Essential features (dimensionality reduction)
- Unusual patterns (anomaly detection)
- Data generation (generative models)

Power: Works with abundant unlabeled data
Challenge: Harder to evaluate objectively

Real-world: Often combined with supervised learning
1. Unsupervised pre-training on unlabeled data
2. Supervised fine-tuning on small labeled dataset
```
