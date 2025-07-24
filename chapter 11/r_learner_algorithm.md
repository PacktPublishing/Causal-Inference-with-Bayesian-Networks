# R-Learner Algorithm: Mathematical Formulation

The R-Learner is a meta-learning algorithm for estimating Conditional Average Treatment Effects (CATE). This document provides a mathematical formulation of the algorithm steps.

## Problem Setup

Let's define the variables:
- $X \in \mathbb{R}^d$: Covariates (features)
- $T \in \{0, 1\}$: Treatment indicator (0 for control, 1 for treatment)
- $Y \in \mathbb{R}$: Outcome variable
- $\tau(x) = \mathbb{E}[Y | X=x, T=1] - \mathbb{E}[Y | X=x, T=0]$: Conditional Average Treatment Effect (CATE)

The goal is to estimate $\tau(x)$ from observational data.

## Algorithm Steps

### 1. First Stage: Fit Outcome and Propensity Models

In this stage, we fit two models:

1. **Outcome Model**: Estimates $m(x) = \mathbb{E}[Y | X=x]$
   ```
   m̂(x) = argmin_{m} \frac{1}{n} \sum_{i=1}^{n} (Y_i - m(X_i))^2
   ```

2. **Propensity Model**: Estimates $e(x) = \mathbb{P}(T=1 | X=x)$
   ```
   ê(x) = argmin_{e} \frac{1}{n} \sum_{i=1}^{n} (T_i - e(X_i))^2
   ```

### 2. Second Stage: Compute Residuals

We compute residuals for both the outcome and treatment:

1. **Outcome Residuals**: 
   ```
   R^Y_i = Y_i - m̂(X_i)
   ```

2. **Treatment Residuals**: 
   ```
   R^T_i = T_i - ê(X_i)
   ```

These residuals represent the parts of the outcome and treatment that cannot be explained by the covariates alone.

### 3. Third Stage: Regress Outcome Residuals on Treatment Residuals

In this final stage, we estimate the treatment effect by regressing the outcome residuals on the treatment residuals:

```
τ̂(x) = argmin_{τ} \frac{1}{n} \sum_{i=1}^{n} (R^Y_i - τ(X_i) \cdot R^T_i)^2
```

The intuition is that if the treatment has an effect, then the unexplained variation in the outcome (outcome residuals) should be related to the unexplained variation in the treatment (treatment residuals).

## Implementation Details

In practice, the R-Learner implementation includes several enhancements:

1. **Data Splitting**: The data is split into training and validation sets. The outcome and propensity models are fit on the training set, and residuals are computed on the validation set to avoid overfitting.

2. **Propensity Score Clipping**: To avoid numerical instability, propensity scores are often clipped to avoid extreme values (e.g., to the range [0.1, 0.9]).

3. **Alternative Prediction Method**: Instead of using propensity scores in prediction, a constant treatment residual of 1 can be used for all samples, which is less sensitive to propensity score estimation errors.

## Mathematical Justification

The R-Learner is based on the following decomposition of the outcome:

```
Y = m(X) + τ(X) \cdot (T - e(X)) + ε
```

where $ε$ is a noise term with $\mathbb{E}[ε | X, T] = 0$.

This decomposition leads to the R-Learner objective:

```
τ̂(x) = argmin_{τ} \mathbb{E}[(Y - m(X) - τ(X) \cdot (T - e(X)))^2 | X=x]
```

When $m(X)$ and $e(X)$ are estimated from data, we get the empirical objective in Stage 3.

## Advantages of the R-Learner

1. **Double Robustness**: The R-Learner is doubly robust, meaning it can consistently estimate the treatment effect if either the outcome model or the propensity model is correctly specified.

2. **Flexibility**: Any regression method can be used for each stage, allowing for complex, non-linear relationships.

3. **Efficiency**: By focusing on the residuals, the R-Learner can achieve better statistical efficiency than direct methods.