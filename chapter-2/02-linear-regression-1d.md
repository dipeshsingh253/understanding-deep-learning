# 1D Linear Regression Model

## Overview

Linear regression is the simplest supervised learning model - a straight line that predicts one variable from another.

```
Model: y = ϕ₀ + ϕ₁x

Where:
- x: input feature
- y: output prediction
- ϕ₀: intercept (y-value when x=0)
- ϕ₁: slope (how much y changes per unit of x)
```

## The Model Equation

### Mathematical Form

```
y = ϕ₀ + ϕ₁x
```

This is just the **equation of a line** from high school algebra!

### Components

**ϕ₀ (phi-zero)**: The **intercept**
```
- Where the line crosses the y-axis
- Value of y when x = 0
- Shifts the line up or down
```

**ϕ₁ (phi-one)**: The **slope**
```
- How steep the line is
- Change in y for each unit change in x
- Positive slope → line goes up
- Negative slope → line goes down
```

## Visual Interpretation

### Different Parameter Values

```
Different ϕ values → Different lines:

      y
      ↑
   10 │         ╱ ϕ₀=0, ϕ₁=2 (steep)
    8 │       ╱
    6 │     ╱  ϕ₀=2, ϕ₁=1 (medium slope, raised)
    4 │   ╱────
    2 │ ╱      ϕ₀=5, ϕ₁=0.5 (gentle slope, high start)
    0 ├─────────────→ x
      0  2  4  6  8
```

Each choice of (ϕ₀, ϕ₁) defines a different line!

## Concrete Example: House Prices

Let's predict house prices from square footage.

### The Data

```python
# Training data: (square_feet/100, price_in_$1000s)
data = [
    (10, 250),  # 1000 sq ft → $250,000
    (15, 350),  # 1500 sq ft → $350,000
    (20, 450),  # 2000 sq ft → $450,000
    (12, 280),  # 1200 sq ft → $280,000
    (18, 410),  # 1800 sq ft → $410,000
]
```

### Finding the Best Line

We want to find ϕ₀ and ϕ₁ such that the line fits the data well.

```
Visualization:

Price ($1000s)
      ↑
  500 │              ✱ (20, 450)
      │          ✱ (18, 410)
  400 │       ✱ (15, 350)
      │    ✱ (12, 280)
  300 │  ✱ (10, 250)
      │
  200 │  ╱ Best fit line
      │╱
  100 │
      └─────────────────→ Sq Ft (100s)
      0   10   15   20

Best fit: y = 50 + 20x
(ϕ₀ = 50, ϕ₁ = 20)
```

### Making Predictions

```python
# Model: price = ϕ₀ + ϕ₁ × (sq_ft/100)
phi_0 = 50
phi_1 = 20

def predict_price(sq_ft_hundreds):
    return phi_0 + phi_1 * sq_ft_hundreds

# Predict price for 1600 sq ft house
price = predict_price(16)
print(f"Predicted price: ${price * 1000:,}")
# Output: Predicted price: $370,000

# Breakdown:
# y = 50 + 20 × 16
#   = 50 + 320
#   = 370 (in thousands)
```

## Family of Possible Lines

Before training, we don't know which parameters are best. There's a whole **family** of possible lines!

### Parameter Space

```
The space of all possible models:

ϕ₁ (slope)
    ↑
  3 │  ·  ·  ·  ·  ·
    │
  2 │  ·  ·  ·  ·  ·
    │
  1 │  ·  ·  ★  ·  ·  ← ϕ₀=50, ϕ₁=20 (best!)
    │
  0 │  ·  ·  ·  ·  ·
    │
 -1 │  ·  ·  ·  ·  ·
    └────────────────→ ϕ₀ (intercept)
    -10  0  50 100 150

Each point = one possible line
★ = best parameters (found by training)
```

### Examples of Different Lines

```python
# Very different predictions from different parameters!

# Model 1: Underestimate (ϕ₀=0, ϕ₁=15)
y1 = 0 + 15 * 16 = 240  # Predicts $240k

# Model 2: Best fit (ϕ₀=50, ϕ₁=20)
y2 = 50 + 20 * 16 = 370  # Predicts $370k ✓

# Model 3: Overestimate (ϕ₀=100, ϕ₁=25)
y3 = 100 + 25 * 16 = 500  # Predicts $500k

# For the same 1600 sq ft house!
```

## Why This Model?

### Advantages of Linear Models

```
✓ Simple to understand
✓ Fast to train
✓ Easy to interpret
✓ Works well for many problems
✓ Good starting point
```

### When Linear Models Work

Linear relationships exist in many real-world scenarios:

```python
# Examples of approximately linear relationships:
price = base + cost_per_sqft × square_feet
sales = baseline + conversion_rate × ad_spend
temperature = starting_temp + warming_rate × time
score = base_ability + practice_effect × hours_studied
```

### Limitations

```
✗ Can only model linear relationships
✗ Can't capture complex patterns
✗ May underfit complex data

Example of non-linear data that linear model can't fit:

  y
  ↑
  │    ✱        ✱
  │  ✱   ✱    ✱   ✱
  │ ✱      ✱✱      ✱
  │✱                ✱
  └──────────────────→ x
  
  This U-shape needs a non-linear model!
```

## Understanding the Parameters

### Intercept (ϕ₀)

```
What it means:
- Base value when input is zero
- Shifts entire line up/down
- Doesn't change slope

Example (house prices):
ϕ₀ = 50 means $50,000 base price
(price for a "zero" size house - not realistic, but mathematically useful)
```

### Slope (ϕ₁)

```
What it means:
- Rate of change
- How much output changes per unit input
- Steepness of line

Example (house prices):
ϕ₁ = 20 means $20,000 per 100 sq ft
or equivalently $200 per sq ft
```

## Practical Example with Real Numbers

```python
import numpy as np

# Training data
X = np.array([10, 12, 15, 18, 20])  # sq ft (in 100s)
y = np.array([250, 280, 350, 410, 450])  # price ($1000s)

# Model function
def linear_model(x, phi_0, phi_1):
    """Predict y from x using linear model"""
    return phi_0 + phi_1 * x

# Try some parameters
phi_0 = 50
phi_1 = 20

# Make predictions
predictions = linear_model(X, phi_0, phi_1)
print("Predictions:", predictions)
# Output: [250. 290. 350. 410. 450.]

# Compare to actual
print("Actual:     ", y)
# Output: [250 280 350 410 450]

# Pretty close! Let's check errors:
errors = predictions - y
print("Errors:     ", errors)
# Output: [0. 10. 0. 0. 0.]

# Only one prediction is off by $10k!
```

## Vectorized Form

For multiple training examples, we can write this compactly:

```
Matrix form:
┌──┐   ┌─────┐   ┌───┐
│y₁│   │1  x₁│   │ϕ₀│
│y₂│ = │1  x₂│ × │ϕ₁│
│y₃│   │1  x₃│   └───┘
│y₄│   │1  x₄│
└──┘   └─────┘

Or: y = Xϕ

Where X is the "design matrix" with a column of 1s
```

```python
import numpy as np

# Design matrix (add column of 1s for intercept)
X = np.array([
    [1, 10],  # [1, x₁]
    [1, 12],  # [1, x₂]
    [1, 15],  # [1, x₃]
    [1, 18],  # [1, x₄]
    [1, 20],  # [1, x₅]
])

# Parameters
phi = np.array([50, 20])  # [ϕ₀, ϕ₁]

# Predictions: matrix multiplication
y_pred = X @ phi  # Same as X.dot(phi)
print(y_pred)
# Output: [250 290 350 410 450]
```

## Key Insight

> **The Power of Parameters**: With just TWO numbers (ϕ₀ and ϕ₁), we can define infinitely many different lines! Training is the process of finding the specific two numbers that best fit our data.

## Quick Check

**Q1**: If ϕ₀ = 100 and ϕ₁ = 50, what does the model predict for x = 3?
<details>
<summary>Answer</summary>

y = 100 + 50 × 3 = 100 + 150 = 250
</details>

**Q2**: What happens to the line when you increase ϕ₀ but keep ϕ₁ the same?
<details>
<summary>Answer</summary>

The line shifts **upward** (parallel to the original). Same slope, different intercept.
</details>

**Q3**: What happens when ϕ₁ is negative?
<details>
<summary>Answer</summary>

The line slopes **downward** from left to right (negative correlation). As x increases, y decreases.
</details>

**Q4**: Can this model fit a curved relationship?
<details>
<summary>Answer</summary>

No! Linear models can only fit straight lines. For curves, you need polynomial regression or neural networks.
</details>

## Key Takeaway

```
1D Linear Regression:
────────────────────

Model: y = ϕ₀ + ϕ₁x

Parameters:
- ϕ₀: Intercept (where line starts)
- ϕ₁: Slope (how steep it is)

Each (ϕ₀, ϕ₁) pair defines a unique line.
Training finds the best pair for your data!

Simple but powerful:
✓ Foundation for more complex models
✓ Interpretable results
✓ Fast and reliable
```
