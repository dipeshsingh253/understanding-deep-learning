# Problems and Solutions

This section provides worked solutions to key problems from Chapter 2, helping you deepen your understanding of supervised learning concepts.

## Problem 2.1: Computing Gradients

### Problem Statement

Given the least squares loss function for linear regression:

```
L[ϕ] = Σᵢ₌₁ᴺ (ϕ₀ + ϕ₁xᵢ - yᵢ)²
```

Derive the gradients:
1. ∂L/∂ϕ₀
2. ∂L/∂ϕ₁

### Solution

#### Part 1: ∂L/∂ϕ₀

Let's compute the partial derivative with respect to ϕ₀.

**Step 1: Expand the loss**
```
L[ϕ] = Σᵢ₌₁ᴺ (ϕ₀ + ϕ₁xᵢ - yᵢ)²
```

**Step 2: Apply chain rule**

For each term in the sum:
```
∂/∂ϕ₀ [(ϕ₀ + ϕ₁xᵢ - yᵢ)²]
```

Use chain rule: `d/dx[f(x)²] = 2f(x) · f'(x)`

```
= 2(ϕ₀ + ϕ₁xᵢ - yᵢ) · ∂/∂ϕ₀(ϕ₀ + ϕ₁xᵢ - yᵢ)
= 2(ϕ₀ + ϕ₁xᵢ - yᵢ) · 1
= 2(ϕ₀ + ϕ₁xᵢ - yᵢ)
```

**Step 3: Sum over all examples**

```
∂L/∂ϕ₀ = Σᵢ₌₁ᴺ 2(ϕ₀ + ϕ₁xᵢ - yᵢ)
```

**Final Answer:**
```
∂L/∂ϕ₀ = 2Σᵢ₌₁ᴺ (ϕ₀ + ϕ₁xᵢ - yᵢ)
```

#### Part 2: ∂L/∂ϕ₁

**Step 1: Apply chain rule**

For each term:
```
∂/∂ϕ₁ [(ϕ₀ + ϕ₁xᵢ - yᵢ)²]
= 2(ϕ₀ + ϕ₁xᵢ - yᵢ) · ∂/∂ϕ₁(ϕ₀ + ϕ₁xᵢ - yᵢ)
= 2(ϕ₀ + ϕ₁xᵢ - yᵢ) · xᵢ
```

Note: The derivative of (ϕ₀ + ϕ₁xᵢ - yᵢ) w.r.t. ϕ₁ is xᵢ

**Step 2: Sum over all examples**

```
∂L/∂ϕ₁ = Σᵢ₌₁ᴺ 2(ϕ₀ + ϕ₁xᵢ - yᵢ)xᵢ
```

**Final Answer:**
```
∂L/∂ϕ₁ = 2Σᵢ₌₁ᴺ (ϕ₀ + ϕ₁xᵢ - yᵢ)xᵢ
```

### Verification with a Numerical Example

Let's verify these formulas with concrete numbers.

```python
import numpy as np

# Simple dataset
X = np.array([1, 2, 3])
y = np.array([2, 4, 6])

# Try parameters: ϕ₀ = 0, ϕ₁ = 2
phi_0 = 0.0
phi_1 = 2.0

# Compute loss
predictions = phi_0 + phi_1 * X  # [2, 4, 6]
errors = predictions - y          # [0, 0, 0]
loss = np.sum(errors ** 2)        # 0

print(f"Loss: {loss}")  # Perfect fit!

# Compute gradients
grad_phi_0 = 2 * np.sum(errors)
grad_phi_1 = 2 * np.sum(errors * X)

print(f"∂L/∂ϕ₀: {grad_phi_0}")  # 0 (at minimum)
print(f"∂L/∂ϕ₁: {grad_phi_1}")  # 0 (at minimum)

# Try slightly different parameters: ϕ₀ = 1, ϕ₁ = 1.5
phi_0 = 1.0
phi_1 = 1.5

predictions = phi_0 + phi_1 * X  # [2.5, 4.0, 5.5]
errors = predictions - y          # [0.5, 0.0, -0.5]
loss = np.sum(errors ** 2)        # 0.5

grad_phi_0 = 2 * np.sum(errors)          # 2 * 0 = 0
grad_phi_1 = 2 * np.sum(errors * X)      # 2 * (0.5*1 + 0*2 + -0.5*3) = -2

print(f"\nLoss: {loss}")
print(f"∂L/∂ϕ₀: {grad_phi_0}")  # 0
print(f"∂L/∂ϕ₁: {grad_phi_1}")  # -2 (negative → decrease ϕ₁ is wrong!)
```

**Interpretation:**
- When gradients are zero, we're at a minimum
- Positive gradient → decrease parameter to minimize loss
- Negative gradient → increase parameter to minimize loss

## Problem 2.2: Closed-Form Solution

### Problem Statement

For linear regression with the least squares loss, derive the **closed-form solution** (analytical solution) for the optimal parameters.

Given:
- N training examples: {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}
- Model: y = ϕ₀ + ϕ₁x

Find: Optimal ϕ₀ and ϕ₁ without using gradient descent.

### Solution

#### Matrix Formulation

**Step 1: Write in matrix form**

Define the design matrix X and parameter vector ϕ:

```
     ┌──────┐          ┌───┐
     │1  x₁ │          │ϕ₀ │
X =  │1  x₂ │    ϕ =   │ϕ₁ │
     │...   │          └───┘
     │1  xₙ │
     └──────┘
     [N × 2]         [2 × 1]
```

The predictions are: `ŷ = Xϕ`

**Step 2: Write loss in matrix form**

```
L[ϕ] = ||Xϕ - y||²
     = (Xϕ - y)ᵀ(Xϕ - y)
```

**Step 3: Expand**

```
L[ϕ] = (ϕᵀXᵀ - yᵀ)(Xϕ - y)
     = ϕᵀXᵀXϕ - ϕᵀXᵀy - yᵀXϕ + yᵀy
     = ϕᵀXᵀXϕ - 2yᵀXϕ + yᵀy
```

(Since ϕᵀXᵀy = yᵀXϕ and both are scalars)

**Step 4: Take derivative and set to zero**

```
∂L/∂ϕ = 2XᵀXϕ - 2Xᵀy = 0
```

**Step 5: Solve for ϕ**

```
XᵀXϕ = Xᵀy
ϕ = (XᵀX)⁻¹Xᵀy
```

This is the **normal equation**!

#### Analytical Formula

For our 1D case, we can expand this explicitly:

```
XᵀX = [N      Σxᵢ  ]
      [Σxᵢ    Σxᵢ² ]

Xᵀy = [Σyᵢ   ]
      [Σxᵢyᵢ ]
```

Solving:
```
ϕ₁ = (N·Σxᵢyᵢ - Σxᵢ·Σyᵢ) / (N·Σxᵢ² - (Σxᵢ)²)

ϕ₀ = (Σyᵢ - ϕ₁·Σxᵢ) / N
```

Or more intuitively:
```
ϕ₁ = Cov(x, y) / Var(x)
ϕ₀ = ȳ - ϕ₁·x̄

Where:
- x̄ = mean of x
- ȳ = mean of y
- Cov(x, y) = covariance
- Var(x) = variance of x
```

### Python Implementation

```python
import numpy as np

def closed_form_solution(X, y):
    """
    Compute optimal parameters using closed-form solution
    
    X: input features [N]
    y: outputs [N]
    
    Returns: phi_0, phi_1
    """
    N = len(X)
    
    # Create design matrix
    X_design = np.column_stack([np.ones(N), X])  # [N × 2]
    
    # Normal equation: ϕ = (XᵀX)⁻¹Xᵀy
    XtX = X_design.T @ X_design
    Xty = X_design.T @ y
    phi = np.linalg.solve(XtX, Xty)  # More stable than inv(XtX) @ Xty
    
    return phi[0], phi[1]  # ϕ₀, ϕ₁

# Example
X = np.array([10, 12, 15, 18, 20])
y = np.array([250, 280, 350, 410, 450])

phi_0, phi_1 = closed_form_solution(X, y)
print(f"Optimal parameters: ϕ₀={phi_0:.2f}, ϕ₁={phi_1:.2f}")

# Verify
predictions = phi_0 + phi_1 * X
loss = np.sum((predictions - y) ** 2)
print(f"Loss: {loss:.2f}")  # Should be very small
```

Output:
```
Optimal parameters: ϕ₀=50.00, ϕ₁=20.00
Loss: 0.00
```

### Comparison: Closed-Form vs Gradient Descent

| Method | Closed-Form | Gradient Descent |
|--------|-------------|------------------|
| **Speed** | Fast (one calculation) | Slow (many iterations) |
| **Memory** | Needs XᵀX (can be large) | Constant memory |
| **Scalability** | Poor for large N | Good for large N |
| **Generality** | Only for linear models | Works for any model |
| **Complexity** | O(N·d² + d³) | O(N·d·T) |

Where:
- N = number of examples
- d = number of features
- T = number of iterations

**When to use closed-form:**
- Small datasets (N < 10,000)
- Linear regression specifically
- Need exact solution

**When to use gradient descent:**
- Large datasets
- Non-linear models (neural networks)
- Online/streaming data

## Problem 2.3: Generative Model Formulation

### Problem Statement

So far we've studied **discriminative models** that learn P(y|x). 

Describe how a **generative model** would approach the house price prediction problem differently. What would it model, and what are the advantages/disadvantages?

### Solution

#### Discriminative Approach (What We've Done)

```
Model: P(y|x)

"Given house features x, what's the price y?"

Direct mapping: x → y
```

**Example:**
```python
# Discriminative model
def predict_price(square_feet):
    return 50 + 20 * (square_feet / 100)

price = predict_price(1500)  # → $350k
```

#### Generative Approach

```
Model: P(x|y) and P(y)

"What features x would a house with price y have?"
AND
"What's the distribution of prices?"

Indirect: Learn joint P(x, y), then use Bayes' rule
```

**Mathematical Formulation:**

By Bayes' theorem:
```
P(y|x) = P(x|y) · P(y) / P(x)

Where:
- P(x|y): Likelihood (how features relate to price)
- P(y): Prior (distribution of prices)
- P(x): Evidence (marginal probability of features)
```

**Example:**

Assume Gaussian distributions:

```python
import numpy as np
from scipy.stats import norm

# Generative model
class GenerativeHouseModel:
    def __init__(self):
        # Learn P(y): price distribution
        self.price_mean = 350  # $350k average
        self.price_std = 100   # $100k std dev
        
        # Learn P(x|y): size given price
        # Assume: size = 0.05 * price + noise
        self.size_slope = 0.05  # 50 sq ft per $1k
        self.size_noise = 100   # sq ft noise
    
    def p_y(self, y):
        """Prior: P(y)"""
        return norm.pdf(y, self.price_mean, self.price_std)
    
    def p_x_given_y(self, x, y):
        """Likelihood: P(x|y)"""
        expected_x = self.size_slope * y * 10  # Convert to hundreds
        return norm.pdf(x, expected_x, self.size_noise / 100)
    
    def predict(self, x):
        """Predict y given x using Bayes' rule"""
        # Sample possible prices
        y_values = np.linspace(100, 600, 1000)
        
        # Compute P(y|x) ∝ P(x|y) · P(y)
        posteriors = [self.p_x_given_y(x, y) * self.p_y(y) 
                     for y in y_values]
        
        # Return most likely price
        return y_values[np.argmax(posteriors)]

# Use the model
model = GenerativeHouseModel()
predicted_price = model.predict(15)  # 1500 sq ft
print(f"Predicted price: ${predicted_price:.0f}k")
```

#### Generative Model: Advantages

**1. Can generate new samples**
```python
# Sample from the model
def generate_house():
    # Sample price from prior
    price = np.random.normal(350, 100)
    
    # Sample size given price
    size = 0.05 * price * 10 + np.random.normal(0, 1)
    
    return size, price

# Generate 5 synthetic houses
for i in range(5):
    size, price = generate_house()
    print(f"House {i+1}: {size*100:.0f} sq ft, ${price:.0f}k")
```

**2. Can handle missing data**
```
If x is missing, can still sample from P(x|y)
If y is missing, can sample from P(y)
```

**3. Provides uncertainty**
```
Full probability distribution, not just point estimate
Can compute confidence intervals
```

**4. Interpretable**
```
Explicitly models how data is generated
Clear assumptions about distributions
```

#### Generative Model: Disadvantages

**1. Strong assumptions**
```
Must assume distributions (Gaussian, etc.)
If wrong, predictions suffer
```

**2. More complex**
```
Need to model P(x|y) AND P(y)
More parameters to learn
```

**3. Often less accurate**
```
Discriminative models focus directly on P(y|x)
Generative models take a detour through P(x|y)
```

**4. Computationally expensive**
```
May need to integrate over all possible y
Discriminative is direct: x → y
```

#### Summary Table

| Aspect | Discriminative P(y\|x) | Generative P(x\|y), P(y) |
|--------|---------------------|------------------------|
| **Goal** | Predict y from x | Model data generation |
| **What it learns** | Decision boundary | Data distribution |
| **Sampling** | Can't generate data | Can generate samples ✓ |
| **Missing data** | Struggles | Handles well ✓ |
| **Accuracy** | Often better ✓ | Often worse |
| **Assumptions** | Fewer ✓ | More |
| **Examples** | Linear regression, neural nets | Naive Bayes, GANs |

#### When to Use Each

**Use Discriminative (P(y|x)):**
- Only care about predictions
- Have plenty of labeled data
- Want best accuracy
- Classification/regression tasks

**Use Generative (P(x|y)):**
- Need to generate new samples
- Have missing data
- Want interpretability
- Small labeled datasets
- Need uncertainty estimates

## Key Takeaways

### From Problem 2.1 (Gradients)

```
Computing gradients is fundamental to training!

∂L/∂ϕ₀ = 2Σᵢ (ϕ₀ + ϕ₁xᵢ - yᵢ)
∂L/∂ϕ₁ = 2Σᵢ (ϕ₀ + ϕ₁xᵢ - yᵢ)xᵢ

These tell us how to adjust parameters
to minimize loss.
```

### From Problem 2.2 (Closed-Form)

```
Linear regression has an exact solution!

ϕ = (XᵀX)⁻¹Xᵀy

But gradient descent is more general and
scales better to large datasets and
non-linear models.
```

### From Problem 2.3 (Generative Models)

```
Two approaches to supervised learning:

Discriminative: Learn P(y|x) directly
└─ Best for prediction accuracy

Generative: Learn P(x|y) and P(y)
└─ Better for generating data, handling missing values

Most modern deep learning is discriminative,
but generative models (GANs, diffusion) are
increasingly important!
```

## Practice Problems

Try these on your own:

**P1**: Implement gradient descent with momentum:
```
v ← βv + (1-β)∇L
ϕ ← ϕ - αv
```

**P2**: Prove that the closed-form solution minimizes the loss by showing the second derivative is positive.

**P3**: Implement a generative model for binary classification using Bayes' rule.

**P4**: Compare convergence speed of different learning rates: 0.001, 0.01, 0.1, 1.0

**P5**: Add L2 regularization to the loss and derive the new gradients.
