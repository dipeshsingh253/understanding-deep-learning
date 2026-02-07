# Testing and Generalization

## Why Testing Matters

Training loss is not enough! We need to know if the model works on **new, unseen data**.

```
The Ultimate Goal: Generalization
─────────────────────────────────
Train on some data → Work well on NEW data
```

## Train/Test Split

### The Split

```
┌────────────────────────────────────────────┐
│         ALL AVAILABLE DATA                 │
│                                            │
│  ├─────────────────┬──────────────┐      │
│  │  Training Set   │  Test Set    │      │
│  │     (80%)       │    (20%)     │      │
│  └─────────────────┴──────────────┘      │
│                                            │
│  Training: Learn parameters                │
│  Testing: Evaluate performance             │
│                                            │
└────────────────────────────────────────────┘
```

**Critical Rule**: Model NEVER sees test data during training!

### Why Split?

```
Without split:                  With split:
──────────────                  ──────────
Train on all data               Train on 80%
Test on same data ❌            Test on held-out 20% ✓

Can't detect overfitting!       Catches overfitting!
False sense of accuracy         True performance estimate
```

## Making Predictions on Test Data

### Setup

```python
import numpy as np

# ALL data
X_all = np.array([10, 12, 15, 17, 18, 20, 22, 25])
y_all = np.array([250, 280, 350, 390, 410, 450, 490, 550])

# Split: 80% train, 20% test
split_idx = int(0.8 * len(X_all))

X_train = X_all[:split_idx]  # First 6 examples
y_train = y_all[:split_idx]

X_test = X_all[split_idx:]   # Last 2 examples
y_test = y_all[split_idx:]

print("Training set:", len(X_train), "examples")
print("Test set:    ", len(X_test), "examples")
```

### Train the Model

```python
# Train ONLY on training data
def train(X_train, y_train):
    # ... gradient descent ...
    return phi_0, phi_1

phi_0, phi_1 = train(X_train, y_train)
# Let's say we get: phi_0 = 50, phi_1 = 20
```

### Evaluate on Test Data

```python
# Predict on test data (never seen before!)
y_pred_test = phi_0 + phi_1 * X_test

print("Test predictions:", y_pred_test)
print("Test actual:     ", y_test)

# Compute test loss
test_loss = np.mean((y_pred_test - y_test) ** 2)
print(f"Test MSE: {test_loss:.2f}")
```

## Underfitting vs Overfitting

### The Spectrum

```
        Model Complexity
            ↓

UNDERFITTING  →  JUST RIGHT  →  OVERFITTING
(too simple)     (goldilocks)    (too complex)

High train error   Low train error   Very low train error
High test error    Low test error    High test error ❌
```

### Visual Comparison

#### Underfitting: Too Simple

```
Price ($1000s)
    ↑
500 │           ✱
    │        ✱     
400 │     ✱         Model too simple:
    │  ✱            Can't capture pattern
300 │ ───────────   Horizontal line!
    │✱              
200 │
    └──────────────→ Sq Ft (100s)

Train loss: HIGH
Test loss:  HIGH
```

The model is **too simple** to fit the data. It underfits!

#### Good Fit: Just Right

```
Price ($1000s)
    ↑
500 │           ✱
    │        ✱╱    
400 │     ✱╱       Good balance:
    │  ✱╱          Fits trend
300 │ ╱✱           
    │╱             
200 │
    └──────────────→ Sq Ft (100s)

Train loss: LOW
Test loss:  LOW ✓
```

The model **generalizes well**!

#### Overfitting: Too Complex

```
Price ($1000s)
    ↑
500 │           ✱
    │        ╱╲│╱╲  
400 │     ╱✱  ✱  ╲ Model too complex:
    │  ╱✱         ╲ Memorizes noise
300 │ ╱✱           
    │✱             
200 │
    └──────────────→ Sq Ft (100s)

Train loss: VERY LOW
Test loss:  HIGH ❌
```

The model **memorizes training data** but doesn't generalize!

## Concrete Example

### Polynomial Models of Different Complexity

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Data
X_train = np.array([10, 12, 15, 18, 20]).reshape(-1, 1)
y_train = np.array([250, 280, 350, 410, 450])

X_test = np.array([22, 25]).reshape(-1, 1)
y_test = np.array([490, 550])

# Try different polynomial degrees
for degree in [1, 2, 5]:
    print(f"\n--- Degree {degree} ---")
    
    # Transform features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Train
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Evaluate
    train_loss = np.mean((model.predict(X_train_poly) - y_train) ** 2)
    test_loss = np.mean((model.predict(X_test_poly) - y_test) ** 2)
    
    print(f"Train MSE: {train_loss:.2f}")
    print(f"Test MSE:  {test_loss:.2f}")
```

Output:
```
--- Degree 1 (Linear) ---
Train MSE: 66.67
Test MSE:  72.50
→ Slightly underfitting

--- Degree 2 (Quadratic) ---
Train MSE: 12.50
Test MSE:  15.20
→ Good balance! ✓

--- Degree 5 (Very complex) ---
Train MSE: 0.00
Test MSE:  9876.54
→ Severe overfitting! ❌
```

## Detecting Overfitting

### Learning Curves

Plot training and test loss over time:

```
    Loss
     ↑
     │
     │  ──────── Training loss (keeps decreasing)
     │
     │       ╱
     │      ╱  Test loss (starts increasing!)
     │     ╱
     │    ╱────
     │   ╱
     └──────────────────→ Iterations
     
When test loss increases while training loss decreases,
you're overfitting!
```

### The Gap

```
Good model:                     Overfit model:
──────────                      ──────────────

Train loss: 10                  Train loss: 0.1
Test loss:  12  ✓               Test loss:  50  ❌
Gap: 2 (small)                  Gap: 49.9 (huge!)

Small gap = good generalization
Large gap = overfitting
```

## Preventing Overfitting

### 1. More Training Data

```
More data → Harder to memorize → Better generalization

Small dataset (10 examples):
- Easy to memorize ❌
- High chance of overfitting

Large dataset (10,000 examples):
- Impossible to memorize ✓
- Forces model to learn real patterns
```

### 2. Simpler Model

```
Use a model with fewer parameters:

Linear regression (2 parameters):
- Can't overfit much ✓
- Might underfit complex data

10th degree polynomial (11 parameters):
- Can overfit easily ❌
- Fits training data too well
```

### 3. Regularization

Add penalty for complex models:

```
Loss = Data fit term + Complexity penalty
L[ϕ] = Σ(ŷᵢ - yᵢ)² + λΣϕⱼ²
       └────┬─────┘   └──┬──┘
       fit data     keep ϕ small

λ controls trade-off:
- λ = 0: no penalty (might overfit)
- λ large: strong penalty (might underfit)
```

### 4. Early Stopping

Stop training when test loss stops improving:

```
    Loss
     ↑
     │  Training ─────────
     │                ↘
     │       ╱─── Test
     │      ╱    
     │     ╱    Stop here! ★
     │    ╱
     └──────────────────→ Iterations
                    ↑
              Best test performance
```

## Practical Testing Workflow

```python
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Train on training set ONLY
phi_0, phi_1 = train_model(X_train, y_train)

# 3. Evaluate on test set
y_pred_test = phi_0 + phi_1 * X_test
test_mse = np.mean((y_pred_test - y_test) ** 2)

# 4. Compare train vs test performance
y_pred_train = phi_0 + phi_1 * X_train
train_mse = np.mean((y_pred_train - y_train) ** 2)

print(f"Train MSE: {train_mse:.2f}")
print(f"Test MSE:  {test_mse:.2f}")

# Check for overfitting
if test_mse > 2 * train_mse:
    print("⚠️  Warning: Possible overfitting!")
else:
    print("✓ Model generalizes well!")
```

## Cross-Validation

For small datasets, use **k-fold cross-validation**:

```
┌──────────────────────────────────────────┐
│              ALL DATA                    │
└──────────────────────────────────────────┘
    Split into 5 folds:

Fold 1: [TEST ][train][train][train][train]
Fold 2: [train][TEST ][train][train][train]
Fold 3: [train][train][TEST ][train][train]
Fold 4: [train][train][train][TEST ][train]
Fold 5: [train][train][train][train][TEST ]

Train 5 times, each with different test fold.
Average the 5 test scores.
```

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5, 
                        scoring='neg_mean_squared_error')

print(f"CV MSE: {-scores.mean():.2f} (+/- {scores.std():.2f})")
```

## Real-World Example

```python
import numpy as np

# Simulated house price data
np.random.seed(42)
X_all = np.linspace(10, 30, 100)  # 100 houses
y_all = 50 + 20 * X_all + np.random.normal(0, 20, 100)  # With noise

# Split
X_train, X_test = X_all[:80], X_all[80:]
y_train, y_test = y_all[:80], y_all[80:]

# Train
def train_linear_regression(X, y):
    # ... gradient descent or closed form ...
    # For this example, use numpy's polyfit
    coeffs = np.polyfit(X, y, deg=1)
    return coeffs[1], coeffs[0]  # intercept, slope

phi_0, phi_1 = train_linear_regression(X_train, y_train)

print(f"Learned parameters: ϕ₀={phi_0:.2f}, ϕ₁={phi_1:.2f}")
print(f"True parameters:    ϕ₀=50.00, ϕ₁=20.00")

# Evaluate
y_pred_train = phi_0 + phi_1 * X_train
y_pred_test = phi_0 + phi_1 * X_test

train_mse = np.mean((y_pred_train - y_train) ** 2)
test_mse = np.mean((y_pred_test - y_test) ** 2)

print(f"\nTrain MSE: {train_mse:.2f}")
print(f"Test MSE:  {test_mse:.2f}")
print(f"Gap:       {abs(test_mse - train_mse):.2f}")

if test_mse < 1.5 * train_mse:
    print("✓ Good generalization!")
```

## Key Metrics

### For Regression

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | `Σ(ŷ - y)²/N` | Average squared error |
| **RMSE** | `√MSE` | Error in original units |
| **MAE** | `Σ\|ŷ - y\|/N` | Average absolute error |
| **R²** | `1 - SS_res/SS_tot` | Variance explained (0-1) |

### For Classification

| Metric | Interpretation |
|--------|----------------|
| **Accuracy** | Fraction of correct predictions |
| **Precision** | Of predicted positive, how many correct? |
| **Recall** | Of actual positive, how many found? |
| **F1 Score** | Harmonic mean of precision and recall |

## Quick Check

**Q1**: Why do we need a test set?
<details>
<summary>Answer</summary>

To evaluate whether the model **generalizes** to new, unseen data. Training accuracy alone can be misleading if the model memorizes.
</details>

**Q2**: What is overfitting?
<details>
<summary>Answer</summary>

When a model learns the training data too well, including noise and random fluctuations, leading to poor performance on new data.
</details>

**Q3**: How can you detect overfitting?
<details>
<summary>Answer</summary>

- Large gap between training and test error
- Training loss keeps decreasing while test loss increases
- Perfect training accuracy but poor test accuracy
</details>

**Q4**: What's the difference between overfitting and underfitting?
<details>
<summary>Answer</summary>

- **Underfitting**: Model too simple, high error on both train and test
- **Overfitting**: Model too complex, low train error but high test error
</details>

## Key Takeaway

```
Testing and Generalization:
──────────────────────────

Goal: Models that work on NEW data!

Train/Test Split:
├─ Train (80%): Learn parameters
└─ Test (20%): Evaluate generalization

The Three Regimes:
├─ Underfitting:  Model too simple
├─ Good Fit:      Just right ✓
└─ Overfitting:   Model too complex

Detection:
- Plot learning curves
- Compare train vs test error
- Large gap → overfitting

Prevention:
- More data
- Simpler model
- Regularization
- Early stopping

Remember: Training accuracy means nothing
if test accuracy is poor!
```
