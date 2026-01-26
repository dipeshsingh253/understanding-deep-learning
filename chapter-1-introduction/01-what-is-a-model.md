# What is a Model?

## Core Definition

A **model** is a mathematical function with adjustable parameters that learns patterns from data.

```
Model = Architecture (fixed code) + Parameters (learned numbers)
```

## Key Components

```
┌─────────────────────────────────┐
│         NEURAL NETWORK          │
├─────────────────────────────────┤
│                                 │
│  PARAMETERS (Numbers - Change)  │
│  ├─ Weights                     │
│  ├─ Biases                      │
│  └─ Embeddings                  │
│                                 │
│  ARCHITECTURE (Code - Fixed)    │
│  ├─ Layer structure             │
│  ├─ Activation functions        │
│  └─ Forward pass logic          │
│                                 │
└─────────────────────────────────┘
```

## What Changes During Training?

**ONLY the parameter values (numbers)!**

### Before Training (Random):
```python
Parameter_1 = 0.0234 (random)
Parameter_2 = -0.1834 (random)
Parameter_3 = 0.9237 (random)
# Model is DUMB - outputs garbage
```

### After Training (Learned):
```python
Parameter_1 = 0.7492 (learned!)
Parameter_2 = -0.0923 (learned!)
Parameter_3 = 1.3374 (learned!)
# Model is SMART - outputs accurate predictions
```

**The architecture (code) stays the same. Only numbers change!**

## Analogy: Volume Knob

Think of a model like a stereo with billions of volume knobs:
- **Before training**: All knobs set randomly → Bad sound
- **Training**: Adjust each knob to improve sound quality
- **After training**: All knobs perfectly tuned → Great sound

Each knob = one parameter. GPT-3 has 175 billion knobs!

## Simple Example

```python
# A tiny model with 23 parameters
class TinyModel:
    def __init__(self):
        # PARAMETERS (will change during training)
        self.W1 = np.random.randn(3, 4)  # 12 numbers
        self.b1 = np.zeros(3)            # 3 numbers
        self.W2 = np.random.randn(2, 3)  # 6 numbers
        self.b2 = np.zeros(2)            # 2 numbers
        # Total: 23 parameters
    
    def forward(self, x):
        # ARCHITECTURE (stays fixed)
        h = np.maximum(0, self.W1 @ x + self.b1)  # ReLU
        output = self.W2 @ h + self.b2
        return output

# Before training
model = TinyModel()
print(model(input))  # Random output

# After training (parameters adjusted)
# Same code, different numbers → Smart output!
```

## Real-World Example: GPT-3

```
GPT-3 Model File (~700 GB):
├─ 175,000,000,000 parameters (numbers)
├─ Each is a floating-point number
└─ Stored in weight matrices and bias vectors

Architecture (few MB of code):
├─ 96 transformer layers
├─ Attention mechanism
├─ Feed-forward networks
└─ Layer normalization

The 175 billion numbers ARE the intelligence!
The code just processes them!
```

## Key Takeaway

```
Model = Structure of Numbers

- Structure: Defined by programmers (layers, connections)
- Numbers: Learned from data (weights, biases)
- Intelligence: Emerges from the specific pattern of numbers

Training = Finding the right values for billions of numbers
```