"""
X-Learner for Conditional Average Treatment Effect (CATE) Estimation
====================================================================

The X-Learner is a meta-learner approach for estimating heterogeneous treatment effects.
It was introduced by Künzel et al. (2019) in their paper "Metalearners for estimating
heterogeneous treatment effects using machine learning".

Key Characteristics:
-------------------
1. **Two-Stage Approach**: Uses a multi-stage estimation process
2. **Separate Models**: Fits separate models for control and treatment groups
3. **Crossover Estimation**: Estimates treatment effects by crossing over between groups
4. **Flexible Base Learners**: Can use any regression model that follows scikit-learn's API
5. **Robust to Imbalance**: Performs well when treatment and control groups have different sizes

Algorithm Steps:
--------------
1. **First Stage**: 
   - Fit separate outcome models for treatment and control groups
   - Use these models to predict counterfactual outcomes for each individual

2. **Second Stage**:
   - Compute "imputed" treatment effects for each individual
   - For treated individuals: actual outcome minus predicted control outcome
   - For control individuals: predicted treatment outcome minus actual outcome

3. **Third Stage**:
   - Train models to predict these imputed treatment effects
   - One model for the treatment group, one for the control group

4. **Final Estimation**:
   - Combine predictions from both models to get the final CATE estimate
   - Can use a simple average or a weighted average based on propensity scores

Advantages:
----------
- Robust to confounding when good predictive models are used
- Can handle different treatment and control group sizes effectively
- Often outperforms other meta-learners when treatment groups are imbalanced
- Provides more stable estimates in many practical scenarios

Limitations:
-----------
- More complex implementation than S-Learner or T-Learner
- Requires fitting four separate models (two in first stage, two in third stage)
- Performance depends on the quality of the base learners
- May struggle with very small sample sizes

References:
----------
Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating
heterogeneous treatment effects using machine learning. Proceedings of the National
Academy of Sciences, 116(10), 4156-4165.
"""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# Ensure compatibility with newer NumPy versions
if not hasattr(np, 'int'):
    np.int = np.int64

class XLearner:
    """
    X-Learner for estimating Conditional Average Treatment Effects (CATE).

    The X-Learner estimates treatment effects using a multi-stage approach:
    1. Fit separate models for control and treatment groups
    2. Estimate counterfactual outcomes and compute individual treatment effects
    3. Train models to predict these treatment effects
    4. Combine predictions to get the final CATE estimate

    Parameters
    ----------
    model : estimator object, optional (default=None)
        The base learner used for all four models. If None, uses:
        - LinearRegression for outcome models
        - GradientBoostingRegressor for treatment effect models
        The model must implement fit() and predict() methods.

    Attributes
    ----------
    model_c : estimator object
        The model used to predict outcomes for the control group.

    model_t : estimator object
        The model used to predict outcomes for the treatment group.

    te_model_c : estimator object
        The model used to predict treatment effects for the control group.

    te_model_t : estimator object
        The model used to predict treatment effects for the treatment group.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from metalearners.x_learner import XLearner
    >>> import numpy as np
    >>> # Generate synthetic data
    >>> X = np.random.normal(0, 1, size=(1000, 5))
    >>> t = np.random.binomial(1, 0.5, size=1000)
    >>> y = X[:, 0] + t * (2 + X[:, 1]) + np.random.normal(0, 1, size=1000)
    >>> # Initialize and fit X-Learner
    >>> xl = XLearner(RandomForestRegressor())
    >>> xl.fit(X, y, t)
    >>> # Estimate treatment effects
    >>> cate = xl.predict(X)
    """
    def __init__(self, model=None):
        if model is None:
            self.model_c = LinearRegression()
            self.model_t = LinearRegression()
            self.te_model_c = GradientBoostingRegressor()
            self.te_model_t = GradientBoostingRegressor()
        else:
            self.model_c = clone(model)
            self.model_t = clone(model)
            self.te_model_c = clone(model)
            self.te_model_t = clone(model)

    def fit(self, X, y, treatment):
        """
        Fit the X-Learner to estimate treatment effects.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The feature matrix.

        y : array-like, shape (n_samples,)
            The outcome variable.

        treatment : array-like, shape (n_samples,)
            The treatment indicator (0 for control, 1 for treatment).

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        The fitting process follows these steps:
        1. Fit separate models for control and treatment groups
        2. Predict counterfactual outcomes for each group
        3. Compute individual treatment effects
        4. Train models to predict these treatment effects
        """
        # Fit control and treatment models
        self.model_c.fit(X[treatment == 0], y[treatment == 0])
        self.model_t.fit(X[treatment == 1], y[treatment == 1])

        # Predict counterfactual outcomes
        control_outcome_t = self.model_c.predict(X[treatment == 1])
        treatment_outcome_c = self.model_t.predict(X[treatment == 0])

        # Treatment effect estimations
        tau_t = y[treatment == 1] - control_outcome_t
        tau_c = treatment_outcome_c - y[treatment == 0]

        # Train treatment effect models
        self.te_model_t.fit(X[treatment == 1], tau_t)
        self.te_model_c.fit(X[treatment == 0], tau_c)

        return self

    def predict(self, X):
        """
        Predict treatment effects for new data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The feature matrix for which to predict treatment effects.

        Returns
        -------
        cate : array-like, shape (n_samples,)
            The predicted Conditional Average Treatment Effects (CATE).

        Notes
        -----
        The prediction process:
        1. Predict treatment effects using the treatment group model
        2. Predict treatment effects using the control group model
        3. Average the two predictions to get the final CATE estimate

        This simple averaging can be replaced with weighted averaging
        based on propensity scores for potentially better performance.
        """
        # Predict treatment effects
        tau_hat_t = self.te_model_t.predict(X)
        tau_hat_c = self.te_model_c.predict(X)

        # Average treatment effect
        return (tau_hat_t + tau_hat_c) / 2
