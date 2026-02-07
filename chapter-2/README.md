# Chapter 2: Supervised Learning

Welcome to Chapter 2! This chapter dives deep into supervised learning, covering the complete framework from model definition to training and testing. You'll learn about linear regression as a concrete example, understand loss functions, master gradient descent, and discover how models generalize to new data.

## 📚 Table of Contents

### Core Framework
1. **[Supervised Learning Overview](01-supervised-learning-overview.md)**
   - Model definition: y = f[x, ϕ]
   - Structured/tabular data
   - Inference vs training
   - Parameters ϕ
   - Loss function L[ϕ]
   - Optimization: ϕ̂ = argmin L[ϕ]

### Linear Regression - A Complete Example
2. **[1D Linear Regression Model](02-linear-regression-1d.md)**
   - Model: y = ϕ₀ + ϕ₁x
   - Parameters: intercept ϕ₀ and slope ϕ₁
   - Family of possible lines
   - Visual interpretation

3. **[Loss Function](03-loss-function.md)**
   - Least squares: L[ϕ] = Σ(ϕ₀ + ϕ₁xi - yi)²
   - Why square errors?
   - Loss landscape visualization
   - 3D surface and contour plots

4. **[Training with Gradient Descent](04-training-gradient-descent.md)**
   - Gradient descent algorithm
   - Walking downhill on loss surface
   - Iterative parameter updates
   - Visual training progression
   - Learning rate importance

5. **[Testing and Generalization](05-testing-generalization.md)**
   - Generalization to new data
   - Underfitting vs overfitting
   - Train/test split
   - Why testing matters

### Summary and Problems
6. **[Chapter Summary](06-summary.md)**
   - Complete framework recap
   - Roadmap for chapters 3-9
   - Loss vs cost function terminology
   - Generative vs discriminative models

7. **[Problems and Solutions](07-problems-solutions.md)**
   - Problem 2.1: Computing gradients ∂L/∂ϕ₀ and ∂L/∂ϕ₁
   - Problem 2.2: Closed-form solution for linear regression
   - Problem 2.3: Generative model formulation

## 🎯 What You'll Learn

- **Complete Framework**: Understand the full supervised learning pipeline
- **Concrete Examples**: See how theory applies to real data
- **Mathematical Foundation**: Learn the calculus behind optimization
- **Visual Intuition**: Understand loss landscapes and gradient descent
- **Practical Skills**: Know how to train and evaluate models

## 📝 Features

Each section includes:
- ✅ Clear, step-by-step explanations
- ✅ Concrete numerical examples
- ✅ Visual ASCII diagrams
- ✅ Mathematical formulas with context
- ✅ Python code examples
- ✅ Helpful analogies
- ✅ Key insights highlighted
- ✅ Quick check questions

## 🚀 How to Use This Chapter

1. **Sequential Learning**: Read sections in order for complete understanding
2. **Reference Guide**: Jump to specific topics as needed
3. **Practice Problems**: Work through the problems section
4. **Visual Learning**: Pay attention to diagrams and loss landscapes
5. **Code Examples**: Run and experiment with the code

## 💡 Quick Start

New to supervised learning?
1. Start with [Supervised Learning Overview](01-supervised-learning-overview.md)
2. Follow with [1D Linear Regression](02-linear-regression-1d.md)
3. Understand [Loss Functions](03-loss-function.md)
4. Master [Gradient Descent](04-training-gradient-descent.md)

Already familiar with basics?
→ Jump to [Problems and Solutions](07-problems-solutions.md) to test your understanding!

## 🔑 Key Concepts

This chapter introduces fundamental concepts that appear throughout deep learning:
- **Parameters** (ϕ) - The learned values
- **Loss function** (L[ϕ]) - Measures model quality
- **Optimization** - Finding best parameters
- **Gradient descent** - The optimization workhorse
- **Generalization** - Performance on new data

## 🌟 Chapter Highlights

> **Why This Matters**: This chapter builds the foundation for all modern deep learning. Every neural network, transformer, and diffusion model uses these same principles:
> - Define a model with parameters
> - Define a loss function
> - Use gradient descent to optimize
> - Test on held-out data

---

*This chapter serves as a comprehensive, practical guide to supervised learning fundamentals.*
