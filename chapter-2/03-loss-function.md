# Loss Function

## What is a Loss Function?

A **loss function** (also called cost function or objective function) measures how well your model fits the training data.

```
L[ϕ] = measure of prediction errors

Lower loss = Better fit
Higher loss = Worse fit
```

**Goal**: Find parameters ϕ that minimize the loss.

## Least Squares Loss

For linear regression, we use the **sum of squared errors**:

```
L[ϕ] = Σᵢ₌₁ᴺ (ϕ₀ + ϕ₁xᵢ - yᵢ)²
       └───────┬─────────┘
       error for example i
```

**Components:**
- `ϕ₀ + ϕ₁xᵢ`: Model's prediction for input xᵢ
- `yᵢ`: True output
- `(ϕ₀ + ϕ₁xᵢ - yᵢ)`: Error (prediction - truth)
- `²`: Squared
- `Σ`: Sum over all N training examples

## Why Square the Errors?

### Three Good Reasons

#### 1. Remove the Sign

```
Without squaring:
Example 1: error = +10  (overestimate by 10)
Example 2: error = -10  (underestimate by 10)
Sum: +10 + (-10) = 0  ← Errors cancel out! Bad!

With squaring:
Example 1: error² = 100
Example 2: error² = 100
Sum: 100 + 100 = 200  ← Errors add up! Good!
```

#### 2. Penalize Large Errors More

```
Small error: 2² = 4
Medium error: 5² = 25   ← 2.5× bigger error → 6× bigger penalty
Large error: 10² = 100  ← 5× bigger error → 25× bigger penalty

This is desirable! We want to avoid large mistakes.
```

#### 3. Smooth and Differentiable

```
Squared error is smooth:
    L
    ↑
    │    ╱╲
    │   ╱  ╲    Smooth curve
    │  ╱    ╲   → Easy to optimize with calculus
    │ ╱      ╲
    └──────────→ ϕ

Absolute error has a kink:
    L
    ↑
    │     ╱│╲
    │    ╱ │ ╲   Sharp corner at minimum
    │   ╱  │  ╲  → Harder to optimize
    │  ╱   │   ╲
    └───────────→ ϕ
           ↑
         kink!
```

**Calculus works better with smooth curves!**

## Concrete Example

Let's compute the loss for our house price example.

### The Data

```python
# Training data: (sq_ft/100, price_$1000s)
data = [
    (10, 250),  # House 1
    (15, 350),  # House 2
    (20, 450),  # House 3
]
```

### Computing Loss: Step by Step

Try parameters: ϕ₀ = 50, ϕ₁ = 20

```python
def compute_loss(data, phi_0, phi_1):
    """Compute least squares loss"""
    total = 0
    for x, y_true in data:
        # Prediction
        y_pred = phi_0 + phi_1 * x
        
        # Error
        error = y_pred - y_true
        
        # Squared error
        squared_error = error ** 2
        
        total += squared_error
    
    return total

# Let's trace through:
```

**House 1:** x=10, y=250
```
Prediction: ŷ = 50 + 20×10 = 250
Error: 250 - 250 = 0
Squared: 0² = 0
```

**House 2:** x=15, y=350
```
Prediction: ŷ = 50 + 20×15 = 350
Error: 350 - 350 = 0
Squared: 0² = 0
```

**House 3:** x=20, y=450
```
Prediction: ŷ = 50 + 20×20 = 450
Error: 450 - 450 = 0
Squared: 0² = 0
```

**Total Loss:**
```
L[ϕ] = 0 + 0 + 0 = 0

Perfect fit! Loss is zero.
```

### Different Parameters = Different Loss

Let's try worse parameters: ϕ₀ = 0, ϕ₁ = 15

```python
# House 1: x=10, y=250
y_pred = 0 + 15×10 = 150
error = 150 - 250 = -100
squared = 10,000

# House 2: x=15, y=350
y_pred = 0 + 15×15 = 225
error = 225 - 350 = -125
squared = 15,625

# House 3: x=20, y=450
y_pred = 0 + 15×20 = 300
error = 300 - 450 = -150
squared = 22,500

# Total loss:
L[ϕ] = 10,000 + 15,625 + 22,500 = 48,125

Much worse fit!
```

### Visual: Predictions vs Reality

```
Good parameters (ϕ₀=50, ϕ₁=20):

Price ($1000s)
    ↑
500 │           ✱
    │         ╱
400 │       ✱╱
    │     ╱
300 │   ✱╱
    │ ╱
200 │╱
    └─────────────→ Sq Ft (100s)
    
Points lie ON the line → Loss = 0

Bad parameters (ϕ₀=0, ϕ₁=15):

Price ($1000s)
    ↑
500 │           ✱ (prediction way off!)
    │          ↑
400 │        ✱ │
    │       ↑  │
300 │     ✱│   │
    │    ╱│   │
200 │  ╱  │   │
    │╱    │   │
100 │     ╱
    └─────────────→ Sq Ft (100s)
    
Points ABOVE the line → Large errors → High loss
```

## Loss Landscape

The loss is a function of the parameters. We can visualize it!

### 2D View: Loss vs One Parameter

If we fix ϕ₀ = 50 and vary ϕ₁:

```
L[ϕ₁]
  ↑
50k│     ╱╲
   │    ╱  ╲
25k│   ╱    ╲
   │  ╱      ╲
  0│ ╱   ★    ╲
   └────────────→ ϕ₁
      15  20  25
          ↑
     minimum at 20
```

### 3D Surface: Loss vs Both Parameters

```
        L[ϕ₀, ϕ₁]
         ↑
         │        ╱╲
         │       ╱  ╲
         │      ╱    ╲
         │     ╱  ★   ╲    ← Bowl shape
         │    ╱        ╲
         │   ╱          ╲
         └──────────────────
              ϕ₁
            ╱
          ╱
        ϕ₀

★ = Minimum (best parameters)
```

The loss surface is a **bowl** (or "paraboloid") in parameter space!

### Contour Plot View

Looking down from above:

```
       ϕ₁
        ↑
     25 │   ╭───╮
        │  ╭─────╮     Contour lines
     20 │ ╭───★───╮    (like elevation on a map)
        │╭─────────╮
     15 ││         │
        ╰───────────╯→ ϕ₀
         0   50  100

★ = Minimum (center of bowl)
Each contour = same loss value
Inner contours = lower loss
```

## Mathematical Form

For linear regression with N data points:

```
L[ϕ₀, ϕ₁] = Σᵢ₌₁ᴺ (ϕ₀ + ϕ₁xᵢ - yᵢ)²

Expanded:
L[ϕ₀, ϕ₁] = (ϕ₀ + ϕ₁x₁ - y₁)² + 
            (ϕ₀ + ϕ₁x₂ - y₂)² + 
            ... +
            (ϕ₀ + ϕ₁xₙ - yₙ)²
```

### Vectorized Form

```python
import numpy as np

def loss_vectorized(X, y, phi):
    """
    Compute loss using matrix operations
    
    X: design matrix [N × 2], each row is [1, xᵢ]
    y: true outputs [N]
    phi: parameters [ϕ₀, ϕ₁]
    """
    # Predictions: y_pred = X @ phi
    y_pred = X @ phi
    
    # Errors: (predictions - true values)
    errors = y_pred - y
    
    # Sum of squared errors
    loss = np.sum(errors ** 2)
    
    return loss

# Example
X = np.array([
    [1, 10],
    [1, 15],
    [1, 20]
])
y = np.array([250, 350, 450])
phi = np.array([50, 20])

loss = loss_vectorized(X, y, phi)
print(f"Loss: {loss}")  # Output: 0
```

## Alternative Loss Functions

### Mean Squared Error (MSE)

Average the squared errors:

```
MSE = (1/N) Σᵢ₌₁ᴺ (ϕ₀ + ϕ₁xᵢ - yᵢ)²
```

**Why average?** Makes loss independent of dataset size.

### Root Mean Squared Error (RMSE)

Take the square root:

```
RMSE = √(MSE)
```

**Why?** Same units as the output (e.g., dollars, not dollars²).

### Mean Absolute Error (MAE)

Use absolute values instead of squares:

```
MAE = (1/N) Σᵢ₌₁ᴺ |ϕ₀ + ϕ₁xᵢ - yᵢ|
```

**Pros:** More robust to outliers
**Cons:** Not differentiable at zero (harder to optimize)

## Visual Interpretation of Loss

### High Loss: Bad Fit

```
Price
  ↑
  │  ✱        ✱
  │      ✱          Large gaps between
  │  ✱        ✱     predictions and data
  │      ✱          → High loss
  │━━━━━━━━━━━━━
  └──────────────→ Sq Ft
```

### Low Loss: Good Fit

```
Price
  ↑
  │      ✱
  │    ✱   ✱        Points close to line
  │  ✱       ✱      → Low loss
  │━━━━━━━━━━━━━
  └──────────────→ Sq Ft
```

### Zero Loss: Perfect Fit

```
Price
  ↑
  │      ✱
  │    ✱   ✱        Points ON the line
  │  ✱━━━━━━✱      → Zero loss
  │              
  └──────────────→ Sq Ft
```

## Key Properties of Squared Loss

### 1. Always Non-negative

```
L[ϕ] ≥ 0 for all ϕ

Why? Squares are always ≥ 0
```

### 2. Zero at Perfect Fit

```
L[ϕ] = 0 ⟺ perfect predictions

(ϕ₀ + ϕ₁xᵢ = yᵢ for all i)
```

### 3. Convex for Linear Regression

```
Bowl-shaped → One global minimum → Easy to optimize!

No local minima to get stuck in.
```

## Practical Computation

```python
import numpy as np

def compute_loss_with_details(X, y, phi_0, phi_1):
    """Compute loss and show breakdown"""
    N = len(X)
    total_loss = 0
    
    print("Computing loss:")
    print("─" * 50)
    
    for i, (x, y_true) in enumerate(zip(X, y)):
        # Prediction
        y_pred = phi_0 + phi_1 * x
        
        # Error
        error = y_pred - y_true
        
        # Squared error
        sq_error = error ** 2
        
        print(f"Example {i+1}:")
        print(f"  x={x:5.1f}, y_true={y_true:5.1f}")
        print(f"  y_pred={y_pred:5.1f}")
        print(f"  error={error:5.1f}, squared={sq_error:8.1f}")
        
        total_loss += sq_error
    
    print("─" * 50)
    print(f"Total Loss: {total_loss:.1f}")
    print(f"MSE: {total_loss/N:.1f}")
    print(f"RMSE: {np.sqrt(total_loss/N):.2f}")
    
    return total_loss

# Example usage
X = np.array([10, 15, 20])
y = np.array([250, 350, 450])

print("\nGood parameters (ϕ₀=50, ϕ₁=20):")
compute_loss_with_details(X, y, 50, 20)

print("\n\nBad parameters (ϕ₀=0, ϕ₁=15):")
compute_loss_with_details(X, y, 0, 15)
```

Output:
```
Good parameters (ϕ₀=50, ϕ₁=20):
──────────────────────────────────────────────────
Example 1:
  x= 10.0, y_true=250.0
  y_pred=250.0
  error=  0.0, squared=     0.0
Example 2:
  x= 15.0, y_true=350.0
  y_pred=350.0
  error=  0.0, squared=     0.0
Example 3:
  x= 20.0, y_true=450.0
  y_pred=450.0
  error=  0.0, squared=     0.0
──────────────────────────────────────────────────
Total Loss: 0.0
MSE: 0.0
RMSE: 0.00

Bad parameters (ϕ₀=0, ϕ₁=15):
──────────────────────────────────────────────────
Example 1:
  x= 10.0, y_true=250.0
  y_pred=150.0
  error=-100.0, squared= 10000.0
Example 2:
  x= 15.0, y_true=350.0
  y_pred=225.0
  error=-125.0, squared= 15625.0
Example 3:
  x= 20.0, y_true=450.0
  y_pred=300.0
  error=-150.0, squared= 22500.0
──────────────────────────────────────────────────
Total Loss: 48125.0
MSE: 16041.7
RMSE: 126.66
```

## Quick Check

**Q1**: Why do we square the errors instead of just adding them?
<details>
<summary>Answer</summary>

1. Prevents positive and negative errors from canceling out
2. Penalizes large errors more heavily
3. Creates a smooth, differentiable function for optimization
</details>

**Q2**: What does a loss of zero mean?
<details>
<summary>Answer</summary>

Perfect fit! Every prediction exactly matches the true value. (Usually impossible with real data)
</details>

**Q3**: Can the loss be negative?
<details>
<summary>Answer</summary>

No! Squared errors are always ≥ 0, so their sum is always ≥ 0.
</details>

**Q4**: Why is the loss surface bowl-shaped for linear regression?
<details>
<summary>Answer</summary>

The loss is a quadratic function of the parameters (ϕ₀ and ϕ₁), which creates a parabolic/bowl shape. This is convex, meaning there's exactly one minimum.
</details>

## Key Takeaway

```
Least Squares Loss:
──────────────────

L[ϕ] = Σᵢ₌₁ᴺ (ϕ₀ + ϕ₁xᵢ - yᵢ)²

Why square?
✓ Removes sign (errors don't cancel)
✓ Penalizes large errors more
✓ Smooth for optimization

Properties:
✓ Always non-negative
✓ Zero at perfect fit
✓ Bowl-shaped (convex)
✓ One global minimum

Training = finding parameters that minimize this loss!
```
