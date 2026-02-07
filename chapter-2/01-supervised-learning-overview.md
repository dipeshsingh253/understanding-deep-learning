# Supervised Learning Overview

## Core Definition

**Supervised learning** is about learning a function that maps inputs to outputs using labeled training data.

```
Given: Training data {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
Goal: Learn function f that predicts y from x
```

## The Complete Framework

```
┌────────────────────────────────────────────────────┐
│          SUPERVISED LEARNING FRAMEWORK             │
├────────────────────────────────────────────────────┤
│                                                    │
│  1. MODEL: y = f[x, ϕ]                           │
│     - x: input features                           │
│     - y: output prediction                        │
│     - ϕ: parameters (learned)                     │
│                                                    │
│  2. LOSS FUNCTION: L[ϕ]                          │
│     - Measures prediction quality                 │
│     - Lower = better                              │
│                                                    │
│  3. OPTIMIZATION: ϕ̂ = argmin L[ϕ]               │
│     - Find parameters that minimize loss          │
│     - Usually via gradient descent                │
│                                                    │
│  4. INFERENCE: ŷ = f[x*, ϕ̂]                     │
│     - Use trained model on new data               │
│                                                    │
└────────────────────────────────────────────────────┘
```

## 1. Model Definition: y = f[x, ϕ]

The model is a function with adjustable parameters.

### Components

**x (Input)**: The features we observe
```
Examples:
- House: [square_feet, bedrooms, age]
- Image: [pixel₁, pixel₂, ..., pixel₇₈₄]
- Text: [word₁, word₂, ..., word₁₀₀]
```

**ϕ (Parameters)**: The learned values
```
Examples:
- Linear regression: ϕ = [ϕ₀, ϕ₁]  (intercept, slope)
- Neural network: ϕ = [W₁, b₁, W₂, b₂, ...]  (weights, biases)
```

**y (Output)**: The prediction
```
Examples:
- House price: $450,000
- Image class: "cat"
- Sentiment: 0.85 (positive)
```

### Structured/Tabular Data

Most examples in this chapter use **structured data** - data organized in tables with rows and columns.

```
┌─────────────────────────────────────────────┐
│         TABULAR DATA EXAMPLE                │
├─────────────────────────────────────────────┤
│                                             │
│  Square Feet │ Bedrooms │ Age │ Price      │
│  ─────────────────────────────────────────  │
│      1200    │    3     │ 20  │ $300,000   │
│      1800    │    4     │ 10  │ $450,000   │
│       900    │    2     │ 30  │ $250,000   │
│      2100    │    5     │  5  │ $550,000   │
│                                             │
│  Each row = one training example            │
│  Each column = one feature                  │
│                                             │
└─────────────────────────────────────────────┘
```

## 2. Inference vs Training

### Inference (Prediction)

Use a **trained** model with **fixed** parameters to make predictions.

```python
# Inference: parameters are fixed
def predict(x, phi_0, phi_1):
    """Predict y for new input x"""
    return phi_0 + phi_1 * x

# Example: trained model with ϕ₀=100, ϕ₁=200
x_new = 15  # new house, 1500 sq ft
price = predict(x_new, phi_0=100, phi_1=200)
print(f"Predicted price: ${price * 1000}")
# Output: Predicted price: $3,100,000
```

**Key Point**: During inference, we DON'T change the parameters!

### Training (Learning)

Adjust parameters to minimize prediction errors on training data.

```python
# Training: parameters change
def train(training_data, epochs=1000, learning_rate=0.01):
    """Learn parameters from data"""
    phi_0 = 0.0  # Initialize
    phi_1 = 0.0
    
    for epoch in range(epochs):
        # Compute loss
        loss = compute_loss(training_data, phi_0, phi_1)
        
        # Compute gradients
        grad_phi_0 = compute_gradient_phi_0(training_data, phi_0, phi_1)
        grad_phi_1 = compute_gradient_phi_1(training_data, phi_0, phi_1)
        
        # Update parameters
        phi_0 = phi_0 - learning_rate * grad_phi_0
        phi_1 = phi_1 - learning_rate * grad_phi_1
    
    return phi_0, phi_1

# Train the model
phi_0_trained, phi_1_trained = train(data)
```

**Key Point**: During training, we ADJUST the parameters!

### Visual Comparison

```
INFERENCE:                      TRAINING:
────────────                    ─────────────
Input x                         Input x + Label y
    ↓                               ↓
Model f[x, ϕ]                   Model f[x, ϕ]
(ϕ fixed)                       (ϕ changes)
    ↓                               ↓
Output ŷ                        Output ŷ
                                    ↓
                                Compare to y
                                    ↓
                                Adjust ϕ
```

## 3. Parameters ϕ

Parameters are the **learnable numbers** that define the model's behavior.

### Simple Example: Linear Model

```
Model: y = ϕ₀ + ϕ₁x

Parameters:
- ϕ₀ (phi_0): intercept
- ϕ₁ (phi_1): slope

Different ϕ values → Different lines
```

### Visual: Parameter Space

```
Different parameter values create different models:

ϕ₀ = 0,  ϕ₁ = 1:   y = x           (45° line through origin)
ϕ₀ = 5,  ϕ₁ = 1:   y = 5 + x       (parallel, shifted up)
ϕ₀ = 0,  ϕ₁ = 2:   y = 2x          (steeper, through origin)
ϕ₀ = 3,  ϕ₁ = 0.5: y = 3 + 0.5x    (gentle slope, high intercept)
```

**Key Insight**: The model architecture is fixed (linear equation), but different parameter values create infinitely many different prediction functions!

## 4. Loss Function L[ϕ]

The loss function **measures how well** the model fits the training data.

### Definition

```
L[ϕ] = measure of prediction errors with parameters ϕ

Lower loss = Better fit
Higher loss = Worse fit
```

### Example: Mean Squared Error

For regression problems, we commonly use:

```
L[ϕ] = (1/N) Σᵢ₌₁ᴺ (f[xᵢ, ϕ] - yᵢ)²
       └─────────────┬─────────────┘
         Average squared error
```

**Components:**
- `f[xᵢ, ϕ]`: Model's prediction for input xᵢ
- `yᵢ`: True label
- `(f[xᵢ, ϕ] - yᵢ)²`: Squared error for example i
- `Σ`: Sum over all training examples
- `(1/N)`: Average

### Concrete Example

```python
# Training data: 3 houses
data = [
    (10, 250),  # 1000 sq ft → $250k
    (15, 350),  # 1500 sq ft → $350k
    (20, 450),  # 2000 sq ft → $450k
]

# Model: price = ϕ₀ + ϕ₁ × (sq_ft/100)
# Try: ϕ₀ = 50, ϕ₁ = 20

def compute_loss(data, phi_0, phi_1):
    total_error = 0
    for x, y_true in data:
        y_pred = phi_0 + phi_1 * x
        error = (y_pred - y_true) ** 2
        total_error += error
    return total_error / len(data)

loss = compute_loss(data, phi_0=50, phi_1=20)
print(f"Loss: {loss}")

# Let's trace through:
# Example 1: x=10, y=250
#   y_pred = 50 + 20*10 = 250
#   error = (250 - 250)² = 0
# Example 2: x=15, y=350
#   y_pred = 50 + 20*15 = 350
#   error = (350 - 350)² = 0
# Example 3: x=20, y=450
#   y_pred = 50 + 20*20 = 450
#   error = (450 - 450)² = 0
# Average: (0 + 0 + 0) / 3 = 0

# Perfect fit! Loss = 0
```

## 5. Optimization: ϕ̂ = argmin L[ϕ]

**Goal**: Find the parameters that minimize the loss.

```
ϕ̂ = argmin L[ϕ]
    └──────┬─────┘
    "argument that minimizes"

Read as: "ϕ-hat equals the parameters that minimize L"
```

### Visual Interpretation

```
Loss Surface:
    
    L[ϕ]
     ↑
     │     ╱╲
     │    ╱  ╲
     │   ╱    ╲
     │  ╱      ╲___
     │ ╱           ╲
     │╱      ★      ╲
     └────────────────→ ϕ
             ϕ̂
        (minimum)

Goal: Find ϕ̂ at the lowest point
```

### Gradient Descent

We find ϕ̂ using **gradient descent**:

```
1. Start with random ϕ
2. Compute gradient ∇L[ϕ] (direction of steepest increase)
3. Move in opposite direction: ϕ ← ϕ - α∇L[ϕ]
4. Repeat until convergence
```

**Analogy**: Like walking downhill in fog - you can only feel the slope under your feet, so you take small steps in the steepest downward direction.

## 6. Training vs Testing

### Training Set

Data used to **learn** the parameters.

```
Training: Adjust ϕ to minimize loss on training data
```

### Test Set

**Separate** data used to **evaluate** the final model.

```
Testing: Measure performance on unseen data
```

### Why Split?

```
┌─────────────────────────────────────────────┐
│  ALL DATA (100%)                            │
│  ├─ 80% Training Set                        │
│  │   └─ Used to learn parameters            │
│  │                                           │
│  └─ 20% Test Set                            │
│      └─ Used to evaluate generalization     │
└─────────────────────────────────────────────┘
```

**Key Point**: Model never sees test data during training!

### Goal: Generalization

We want models that perform well on **new, unseen** data, not just memorize training data.

```
Bad Model:                  Good Model:
Training accuracy: 99%      Training accuracy: 95%
Test accuracy: 60% ❌        Test accuracy: 93% ✅

Overfitting!                Generalizes well!
```

## The Complete Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                SUPERVISED LEARNING PIPELINE              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. COLLECT DATA                                         │
│     └─ Gather labeled examples {(xᵢ, yᵢ)}              │
│                                                          │
│  2. SPLIT DATA                                           │
│     ├─ Training set (80%)                                │
│     └─ Test set (20%)                                    │
│                                                          │
│  3. CHOOSE MODEL                                         │
│     └─ Define architecture: y = f[x, ϕ]                 │
│                                                          │
│  4. CHOOSE LOSS FUNCTION                                 │
│     └─ Define error measure: L[ϕ]                       │
│                                                          │
│  5. TRAIN                                                │
│     └─ Find ϕ̂ = argmin L[ϕ] via gradient descent       │
│                                                          │
│  6. TEST                                                 │
│     └─ Evaluate on test set                             │
│                                                          │
│  7. DEPLOY                                               │
│     └─ Use f[x, ϕ̂] for inference on new data           │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

## Key Formulas Summary

| Concept | Formula | Meaning |
|---------|---------|---------|
| Model | y = f[x, ϕ] | Prediction from input |
| Loss | L[ϕ] = Σ(ŷᵢ - yᵢ)² | Measure of error |
| Optimization | ϕ̂ = argmin L[ϕ] | Best parameters |
| Gradient Descent | ϕ ← ϕ - α∇L[ϕ] | Update rule |
| Prediction | ŷ = f[x*, ϕ̂] | Inference on new data |

## Quick Check

**Q1**: What's the difference between inference and training?
<details>
<summary>Answer</summary>

- **Inference**: Parameters are fixed, we only compute predictions
- **Training**: Parameters are adjusted to minimize loss
</details>

**Q2**: Why do we need a separate test set?
<details>
<summary>Answer</summary>

To evaluate whether the model **generalizes** to new, unseen data. Training accuracy alone can be misleading (model might just memorize).
</details>

**Q3**: What does ϕ̂ = argmin L[ϕ] mean?
<details>
<summary>Answer</summary>

"ϕ-hat equals the parameters that minimize the loss function" - i.e., the best parameters we can find through optimization.
</details>

## Key Takeaway

```
Supervised Learning Framework:
───────────────────────────────

1. MODEL: y = f[x, ϕ]
   └─ Predictions depend on parameters ϕ

2. LOSS: L[ϕ] 
   └─ Measures quality of predictions

3. TRAIN: ϕ̂ = argmin L[ϕ]
   └─ Find parameters that minimize loss

4. TEST: Evaluate on new data
   └─ Ensure generalization

This framework applies to ALL supervised learning,
from simple linear regression to GPT-4!
```
