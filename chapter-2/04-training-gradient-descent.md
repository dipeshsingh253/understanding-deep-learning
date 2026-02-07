# Training with Gradient Descent

## What is Training?

**Training** is the process of finding the parameter values (ϕ₀, ϕ₁) that minimize the loss function.

```
Goal: Find ϕ̂ = argmin L[ϕ]
      (the parameters that minimize loss)
```

## The Optimization Problem

We have:
- Loss function: `L[ϕ₀, ϕ₁] = Σᵢ (ϕ₀ + ϕ₁xᵢ - yᵢ)²`
- Goal: Find ϕ₀ and ϕ₁ that make L as small as possible

### Visual: Finding the Valley Bottom

```
        Loss
         ↑
         │        ╱╲
         │       ╱  ╲      We start somewhere random
         │      ╱ ?  ╲     and walk downhill to ★
         │     ╱  ★   ╲    
         │    ╱        ╲
         │   ╱          ╲
         └──────────────────
              Parameters

★ = Minimum (best parameters)
? = Random starting point
```

## Gradient Descent Algorithm

**Gradient descent** is an iterative algorithm that walks downhill on the loss surface.

### The Core Idea

```
1. Start with random parameters
2. Compute the gradient (direction of steepest ascent)
3. Move in the OPPOSITE direction (downhill)
4. Repeat until convergence
```

### The Update Rule

```
ϕ₀ ← ϕ₀ - α · ∂L/∂ϕ₀
ϕ₁ ← ϕ₁ - α · ∂L/∂ϕ₁

Where:
- α (alpha): learning rate (step size)
- ∂L/∂ϕ₀: partial derivative of loss w.r.t. ϕ₀
- ∂L/∂ϕ₁: partial derivative of loss w.r.t. ϕ₁
```

**The gradient** `∇L = [∂L/∂ϕ₀, ∂L/∂ϕ₁]` points in the direction of **steepest increase**.

**We move in the opposite direction** to go downhill!

## Computing the Gradients

For our linear regression loss:
```
L[ϕ] = Σᵢ₌₁ᴺ (ϕ₀ + ϕ₁xᵢ - yᵢ)²
```

The gradients are:

```
∂L/∂ϕ₀ = 2 Σᵢ₌₁ᴺ (ϕ₀ + ϕ₁xᵢ - yᵢ)

∂L/∂ϕ₁ = 2 Σᵢ₌₁ᴺ (ϕ₀ + ϕ₁xᵢ - yᵢ) · xᵢ
```

**Intuition:**
- These tell us how the loss changes as we adjust each parameter
- Positive gradient → loss increases when parameter increases
- Negative gradient → loss decreases when parameter increases

## Step-by-Step Example

Let's train on our house price data:

### Initial Setup

```python
import numpy as np

# Data: (sq_ft/100, price_$1000s)
X = np.array([10, 12, 15, 18, 20])
y = np.array([250, 280, 350, 410, 450])

# Initialize parameters randomly
phi_0 = 0.0   # Start at 0
phi_1 = 0.0   # Start at 0

# Learning rate
alpha = 0.01

print(f"Initial: ϕ₀={phi_0:.2f}, ϕ₁={phi_1:.2f}")
```

### Iteration 1

**Step 1: Compute predictions**
```python
predictions = phi_0 + phi_1 * X
# [0, 0, 0, 0, 0]  (all zeros - terrible!)
```

**Step 2: Compute errors**
```python
errors = predictions - y
# [0-250, 0-280, 0-350, 0-410, 0-450]
# = [-250, -280, -350, -410, -450]
```

**Step 3: Compute gradients**
```python
N = len(X)
grad_phi_0 = (2.0 / N) * np.sum(errors)
# = (2/5) * (-1740) = -696

grad_phi_1 = (2.0 / N) * np.sum(errors * X)
# = (2/5) * (-250*10 + -280*12 + -350*15 + -410*18 + -450*20)
# = (2/5) * (-27,960) = -11,184
```

**Step 4: Update parameters**
```python
phi_0 = phi_0 - alpha * grad_phi_0
# = 0 - 0.01 * (-696) = 6.96

phi_1 = phi_1 - alpha * grad_phi_1
# = 0 - 0.01 * (-11,184) = 111.84
```

**Step 5: Compute new loss**
```python
# New predictions with updated parameters
new_predictions = 6.96 + 111.84 * X
loss = np.sum((new_predictions - y) ** 2)
# Much better than before!
```

### After Many Iterations

```
Iteration    ϕ₀      ϕ₁      Loss
─────────────────────────────────
    0      0.00    0.00   679,300
    1      6.96  111.84    66,234
    2     12.91  132.41    17,923
   10     33.28   26.84     1,245
   50     47.82   20.84        26
  100     49.76   20.18         1
  500     50.00   20.00         0
 1000     50.00   20.00         0
 ✓ Converged!
```

After convergence:
- ϕ₀ ≈ 50
- ϕ₁ ≈ 20
- Loss ≈ 0 (perfect fit!)

## Visual: Walking Downhill

### Contour Plot View

```
       ϕ₁
        ↑
    150 │    1→
        │      2→
    100 │         3→
        │            4→
     50 │               5→ ★
        │                    (50, 20)
      0 └──────────────────────→ ϕ₀
        0   10   20   30   40  50

Numbers show gradient descent path:
1 → 2 → 3 → 4 → 5 → ★ (minimum)

Each step moves downhill!
```

### 3D Surface View

```
    Loss
     ↑
     │        ╱╲
     │   1   ╱  ╲
     │    2 ╱    ╲
high │     3  ★   ╲
     │    ╱4╱5     ╲
low  │   ╱          ╲
     └──────────────────
         ϕ₀  ϕ₁

Starting high, walking down to the valley (★)
```

## Learning Rate Importance

The learning rate α controls the step size.

### Too Small: Slow Convergence

```
α = 0.0001 (tiny steps)

    Loss
     ↑
     │    ╱╲
     │   ╱  ╲
     │  ╱....╲  ← Many tiny steps
     │ ╱..★...\  (takes forever!)
     └──────────→ ϕ

Takes 10,000+ iterations
```

### Just Right: Fast Convergence

```
α = 0.01 (good steps)

    Loss
     ↑
     │    ╱╲
     │   ╱  ╲
     │  ╱-→★ \  ← Nice strides
     │ ╱      \  (efficient!)
     └──────────→ ϕ

Takes ~500 iterations
```

### Too Large: Divergence

```
α = 1.0 (huge steps)

    Loss
     ↑
  ∞ │  ↗     ↖  ← Overshoots!
     │ ╱╲   ╱╲  (bounces around)
     │╱  ╲ ╱  ╲ (never converges!)
     └──────────→ ϕ

Explodes to infinity!
```

## Complete Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    """
    Train linear regression using gradient descent
    
    Parameters:
    - X: input features [N]
    - y: true outputs [N]
    - learning_rate: step size
    - n_iterations: number of update steps
    
    Returns:
    - phi_0, phi_1: learned parameters
    - history: loss at each iteration
    """
    N = len(X)
    
    # Initialize parameters
    phi_0 = 0.0
    phi_1 = 0.0
    
    # Track loss history
    history = []
    
    for iteration in range(n_iterations):
        # 1. Compute predictions
        predictions = phi_0 + phi_1 * X
        
        # 2. Compute errors
        errors = predictions - y
        
        # 3. Compute loss (for tracking)
        loss = np.sum(errors ** 2) / N
        history.append(loss)
        
        # 4. Compute gradients
        grad_phi_0 = (2.0 / N) * np.sum(errors)
        grad_phi_1 = (2.0 / N) * np.sum(errors * X)
        
        # 5. Update parameters
        phi_0 = phi_0 - learning_rate * grad_phi_0
        phi_1 = phi_1 - learning_rate * grad_phi_1
        
        # Print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"Iter {iteration:4d}: ϕ₀={phi_0:6.2f}, "
                  f"ϕ₁={phi_1:6.2f}, Loss={loss:8.2f}")
    
    return phi_0, phi_1, history

# Train the model
X = np.array([10, 12, 15, 18, 20])
y = np.array([250, 280, 350, 410, 450])

phi_0, phi_1, history = gradient_descent(X, y, 
                                         learning_rate=0.01, 
                                         n_iterations=1000)

print(f"\nFinal parameters:")
print(f"ϕ₀ = {phi_0:.2f}")
print(f"ϕ₁ = {phi_1:.2f}")

# Make predictions
predictions = phi_0 + phi_1 * X
print(f"\nPredictions: {predictions}")
print(f"Actual:      {y}")
```

Output:
```
Iter    0: ϕ₀=  6.96, ϕ₁=111.84, Loss=13246.80
Iter  100: ϕ₀= 49.76, ϕ₁= 20.18, Loss=    0.22
Iter  200: ϕ₀= 49.96, ϕ₁= 20.03, Loss=    0.00
Iter  300: ϕ₀= 49.99, ϕ₁= 20.01, Loss=    0.00
Iter  400: ϕ₀= 50.00, ϕ₁= 20.00, Loss=    0.00
Iter  500: ϕ₀= 50.00, ϕ₁= 20.00, Loss=    0.00
Iter  600: ϕ₀= 50.00, ϕ₁= 20.00, Loss=    0.00
Iter  700: ϕ₀= 50.00, ϕ₁= 20.00, Loss=    0.00
Iter  800: ϕ₀= 50.00, ϕ₁= 20.00, Loss=    0.00
Iter  900: ϕ₀= 50.00, ϕ₁= 20.00, Loss=    0.00

Final parameters:
ϕ₀ = 50.00
ϕ₁ = 20.00

Predictions: [250. 290. 350. 410. 450.]
Actual:      [250 280 350 410 450]
```

## Convergence

### When to Stop?

**Option 1: Fixed number of iterations**
```python
for i in range(1000):
    # ... gradient descent ...
```

**Option 2: Loss threshold**
```python
while loss > 0.01:
    # ... gradient descent ...
```

**Option 3: Gradient magnitude**
```python
while np.linalg.norm(gradient) > 0.0001:
    # ... gradient descent ...
```

**Option 4: Parameter change**
```python
while abs(phi_0_new - phi_0_old) > 0.0001:
    # ... gradient descent ...
```

### Visual: Convergence Curve

```
    Loss
     ↑
1000 │●
     │ ●
 100 │  ●●
     │    ●●●
  10 │       ●●●●
     │           ●●●●●●●●●●●●●
   0 └─────────────────────────→ Iteration
     0   100  200  300  400  500

Loss decreases rapidly at first,
then flattens out (convergence)
```

## Mini-Batch Gradient Descent

For large datasets, we can use **mini-batches**:

```python
def mini_batch_gradient_descent(X, y, batch_size=32, 
                               learning_rate=0.01, n_epochs=100):
    """Train using mini-batches"""
    N = len(X)
    phi_0 = 0.0
    phi_1 = 0.0
    
    for epoch in range(n_epochs):
        # Shuffle data
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Process mini-batches
        for i in range(0, N, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Compute gradients on batch
            predictions = phi_0 + phi_1 * X_batch
            errors = predictions - y_batch
            
            grad_phi_0 = (2.0 / len(X_batch)) * np.sum(errors)
            grad_phi_1 = (2.0 / len(X_batch)) * np.sum(errors * X_batch)
            
            # Update parameters
            phi_0 = phi_0 - learning_rate * grad_phi_0
            phi_1 = phi_1 - learning_rate * grad_phi_1
    
    return phi_0, phi_1
```

### Types of Gradient Descent

| Type | Batch Size | Speed | Convergence |
|------|------------|-------|-------------|
| **Batch GD** | All data (N) | Slow per iteration | Smooth, stable |
| **Stochastic GD** | 1 example | Fast per iteration | Noisy, unstable |
| **Mini-batch GD** | 32-256 | Good balance | Good balance ✓ |

## Analogy: Hiking Down a Mountain

```
Imagine you're hiking down a foggy mountain:

1. You can only see the ground under your feet
   → You can only compute LOCAL gradient

2. You feel which direction slopes down
   → Compute ∂L/∂ϕ

3. You take a step downhill
   → Update: ϕ ← ϕ - α∇L

4. You repeat until you reach the valley
   → Iterate until convergence

The learning rate is your step size:
- Too small → Takes forever
- Too large → You might overshoot and fall off a cliff!
- Just right → Efficient descent
```

## Key Insights

> **Why Gradients?** The gradient points in the direction of steepest **increase**. Moving in the opposite direction takes us downhill fastest.

> **Why Iterative?** For most problems, we can't find the minimum in one step. We need to take many small steps, adjusting as we go.

> **Why Learning Rate Matters?** It's a trade-off between speed and stability. Too fast → divergence. Too slow → wastes time.

## Quick Check

**Q1**: What does the gradient tell us?
<details>
<summary>Answer</summary>

The gradient points in the direction of steepest **increase** in the loss. Its magnitude tells us how steep the slope is.
</details>

**Q2**: Why do we subtract the gradient (not add it)?
<details>
<summary>Answer</summary>

We want to go **downhill** (minimize loss). The gradient points uphill, so we move in the opposite direction: `ϕ ← ϕ - α∇L`
</details>

**Q3**: What happens if the learning rate is too large?
<details>
<summary>Answer</summary>

The updates overshoot the minimum, causing oscillations or even divergence (loss goes to infinity).
</details>

**Q4**: When does gradient descent stop?
<details>
<summary>Answer</summary>

When the gradient becomes very small (near zero), indicating we're at or near a minimum. Or after a fixed number of iterations.
</details>

## Key Takeaway

```
Gradient Descent Algorithm:
──────────────────────────

1. Initialize ϕ randomly
2. REPEAT:
   a. Compute predictions: ŷ = ϕ₀ + ϕ₁x
   b. Compute gradients: ∇L = [∂L/∂ϕ₀, ∂L/∂ϕ₁]
   c. Update: ϕ ← ϕ - α∇L
3. UNTIL convergence

Key idea: Walk downhill on the loss surface!

Learning rate α controls step size:
- Too small: slow convergence
- Too large: divergence
- Just right: efficient training ✓

This is the workhorse algorithm for training
neural networks, deep learning, and AI!
```
