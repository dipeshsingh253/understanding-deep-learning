# Supervised Learning

## Definition

**Supervised Learning**: Training a model using labeled data where each input has a known correct output.

```
Input (X) + Label (Y) → Model learns mapping: X → Y
```

## Core Concept

```
┌──────────────────────────────────────────┐
│        SUPERVISED LEARNING               │
├──────────────────────────────────────────┤
│                                          │
│  Training Data:                          │
│  ┌─────────────┬──────────────┐         │
│  │   Input     │   Label      │         │
│  ├─────────────┼──────────────┤         │
│  │  Image: Cat │  "cat"       │         │
│  │  Image: Dog │  "dog"       │         │
│  │  Email text │  "spam"      │         │
│  │  House info │  $500,000    │         │
│  └─────────────┴──────────────┘         │
│                                          │
│  Model learns: Input → Prediction       │
│                                          │
└──────────────────────────────────────────┘
```

## How It Works

### Training Process

1. **Show labeled examples** to the model
2. **Model makes predictions** (initially random/wrong)
3. **Calculate error** between prediction and true label
4. **Adjust parameters** to reduce error
5. **Repeat** thousands of times

### Mathematical Formulation

Given dataset: `{(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}`

Goal: Learn function `f(x; θ)` where:
- `x` = input features
- `y` = true label
- `θ` = model parameters
- `f(x; θ) ≈ y` for all examples

Loss function measures error:
```
L(θ) = (1/n) Σ |f(xᵢ; θ) - yᵢ|²
```

Training minimizes `L(θ)` by adjusting `θ`.

## Types of Supervised Learning

```
SUPERVISED LEARNING
│
├─ 1. CLASSIFICATION (Discrete outputs)
│   ├─ Binary Classification (2 classes)
│   ├─ Multiclass Classification (3+ classes)
│   └─ Multilabel Classification (multiple labels per instance)
│
└─ 2. REGRESSION (Continuous outputs)
    ├─ Linear Regression (straight-line relationships)
    └─ Non-linear Regression (complex relationships)
```

### 1. Classification (Discrete outputs)

Predict **categories** or **classes**.

#### Subtypes of Classification

**a) Binary Classification** - Two classes (yes/no, spam/not spam)
**b) Multiclass Classification** - Three or more mutually exclusive classes
**c) Multilabel Classification** - Multiple labels can apply simultaneously

**Example: Binary Classification - Email Spam Detection**
```python
import numpy as np

# Training data
emails = [
    "Buy now! Limited offer!",  # spam
    "Meeting at 3pm tomorrow",   # not spam
    "Win a free iPhone!",        # spam
    "Quarterly report attached"  # not spam
]
labels = [1, 0, 1, 0]  # 1 = spam, 0 = not spam

# Simple model (simplified)
def classify_email(email, weights):
    # Extract features (word counts)
    features = extract_features(email)
    # Compute weighted sum
    score = np.dot(features, weights)
    # Classify
    return 1 if score > 0 else 0

# After training with labeled examples:
new_email = "Free money, click here!"
prediction = classify_email(new_email, trained_weights)
print(f"Spam: {prediction}")  # Output: 1 (spam)
```

**Example: Multiclass Classification - Digit Recognition**
```python
# 10 classes: digits 0-9
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Model predicts ONE class from 10 possible
prediction = model.predict(digit_image)
print(f"Predicted digit: {prediction}")  # Output: 7
```

**Example: Multilabel Classification - Movie Genre Tagging**
```python
# Movie can have MULTIPLE genres simultaneously
genres = ['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Romance']

# Multilabel output: [1, 0, 1, 0, 0]
# This movie is: Action + Drama
predictions = model.predict(movie_features)
# Output: [Action: Yes, Comedy: No, Drama: Yes, Sci-Fi: No, Romance: No]
```

### 2. Regression (Continuous outputs)

Predict **numerical values**.

#### Subtypes of Regression

**a) Linear Regression** - Models straight-line relationships
**b) Non-linear Regression** - Models complex, curved relationships

**Example: Linear Regression - House Price Prediction**
```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X_train = np.array([
    [1200, 3, 20],  # sqft, bedrooms, age
    [1800, 4, 10],
    [900,  2, 30],
    [2100, 5, 5]
])
y_train = np.array([300000, 450000, 250000, 550000])  # prices

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict new house
new_house = np.array([[1500, 3, 15]])
predicted_price = model.predict(new_house)
print(f"Predicted price: ${predicted_price[0]:,.0f}")
```

**Example: Non-linear Regression - Temperature Prediction**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Non-linear relationship (e.g., temperature over time of day)
X = np.array([0, 6, 12, 18, 24]).reshape(-1, 1)  # Hours
y = np.array([15, 12, 28, 22, 16])  # Temperature (curved pattern)

# Polynomial regression (degree 2) for non-linear fit
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Predict temperature at 9am
X_new = poly.transform([[9]])
predicted_temp = model.predict(X_new)
print(f"Predicted temperature: {predicted_temp[0]:.1f}°C")
```

## Real-World Example: Image Classification

```python
import torch
import torch.nn as nn

# Simple neural network for digit recognition (MNIST)
class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 128),  # 28x28 pixels flattened
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)     # 10 digits (0-9)
        )
    
    def forward(self, x):
        return self.network(x)

# Training loop (simplified)
model = DigitClassifier()
criterion = nn.CrossEntropyLoss()  # Loss function
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for images, labels in training_data:
        # Forward pass
        predictions = model(images)
        
        # Calculate loss
        loss = criterion(predictions, labels)
        
        # Backward pass (update parameters)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Use trained model
test_image = load_image("handwritten_7.png")
prediction = model(test_image)
digit = torch.argmax(prediction)
print(f"Predicted digit: {digit}")  # Output: 7
```

## Analogy: Teaching a Child

Supervised learning is like teaching a child to identify animals:

```
Parent: "This is a cat" (shows cat picture)
Child: Remembers features (whiskers, pointy ears, small)

Parent: "This is a dog" (shows dog picture)  
Child: Remembers features (floppy ears, larger, tail wags)

Parent: "What's this?" (shows new cat picture)
Child: "It's a cat!" (compares to learned features)
```

The parent provides **labels** (supervision). The child **learns patterns**.

## Training vs Inference

```
TRAINING PHASE:
┌──────────┐      ┌───────────┐      ┌────────┐
│  Input   │──→   │   Model   │──→   │ Output │
│  (Image) │      │ (Updates  │      │ (pred) │
└──────────┘      │ params)   │      └────────┘
                  └───────────┘           │
                        ↑                 │
                        │                 ↓
                        │          ┌─────────────┐
                        │          │ Compare to  │
                        └──────────│ True Label  │
                                   │ Adjust θ    │
                                   └─────────────┘

INFERENCE PHASE (after training):
┌──────────┐      ┌───────────┐      ┌────────┐
│  Input   │──→   │   Model   │──→   │ Output │
│ (new img)│      │ (Fixed θ) │      │ (pred) │
└──────────┘      └───────────┘      └────────┘
```

## Key Requirements

1. **Labeled Data**: Each example needs correct answer
2. **Quality Labels**: Accurate, consistent labels are crucial
3. **Sufficient Data**: More examples = better learning
4. **Representative Data**: Training data must cover real-world scenarios

## Common Applications

| Task | Input | Output |
|------|-------|--------|
| Email filtering | Email text | Spam/Not spam |
| Image recognition | Image pixels | Object class |
| Speech recognition | Audio wave | Text transcript |
| Sentiment analysis | Review text | Positive/Negative |
| Medical diagnosis | Patient data | Disease/Healthy |
| Stock prediction | Historical data | Future price |

## Challenges

### 1. Overfitting
Model memorizes training data but fails on new data.

```python
# Example: Overfitted model
# Training accuracy: 99%
# Test accuracy: 60% ← BAD!
```

**Solution**: Regularization, more data, simpler model

### 2. Insufficient Data
Not enough examples to learn patterns.

**Solution**: Data augmentation, transfer learning

### 3. Labeling Cost
Getting labeled data is expensive and time-consuming.

**Solution**: Active learning, semi-supervised learning

## Key Formulas

**Mean Squared Error (Regression):**
```
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

**Cross-Entropy Loss (Classification):**
```
L = -Σ yᵢ log(ŷᵢ)
```

**Parameter Update (Gradient Descent):**
```
θ ← θ - η∇L(θ)
```
where `η` = learning rate, `∇L(θ)` = gradient

## Key Takeaway

```
Supervised Learning = Learning from Examples

Teacher shows: "This input → This output"
Model learns: Pattern connecting inputs to outputs
Result: Model can predict outputs for new inputs

Success requires:
✓ Quality labeled data
✓ Appropriate model architecture
✓ Proper training process
✓ Validation on unseen data
```
