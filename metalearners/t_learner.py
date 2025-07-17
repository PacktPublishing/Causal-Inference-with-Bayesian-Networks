#%% md
# T-Learner
#
# 1. **Importing Libraries**:
# 	-`clone` from `sklearn.base` is used to create a fresh copy of the machine learning model.
# 	- LinearRegression` from `sklearn.linear_model` is used as the default model if none is provided.
#
# 2. **Defining the TLearner Class**:
# 	- The `__init__` method initializes the class. It accepts an optional parameter `model`.
# 	- If no model is provided, it defaults to `LinearRegression`. Otherwise, it uses the provided model.
#
# 3. **Fit Method**:
# 	- `fit` method trains the models using the provided data:
# 	- `X`: Feature matrix.
# 	- `y`: Target variable (outcome).
# 	- `treatment`: Binary treatment indicator (1 for treatment, 0 for control).
# 	- `X[treatment == 0]` and `y[treatment == 0]` select samples that did not receive treatment.
# 	- `X[treatment == 1]` and `y[treatment == 1]` select samples that received treatment.
# 	- `clone(self.model)` ensures that separate copies of the model are used for control and treatment groups.
# 	- `self.model_c` is the model trained on the control group.
# 	- `self.model_t` is the model trained on the treatment group.
"""
T-Learner Implementation for Conditional Average Treatment Effect (CATE) Estimation

This module implements the T-Learner meta-learner approach for estimating
Conditional Average Treatment Effects (CATE). The T-Learner fits separate models
for the treatment and control groups, then estimates treatment effects by taking
the difference between predictions from these models.

The implementation follows scikit-learn's API conventions and supports various
base learners that implement the scikit-learn estimator interface.

Classes:
    TLearner: A meta-learner that uses separate models for treatment and control groups.

Example:
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from metalearners.t_learner import TLearner
    >>> 
    >>> # Generate synthetic data
    >>> X = np.random.normal(0, 1, size=(1000, 5))
    >>> t = np.random.binomial(1, 0.5, size=1000)
    >>> y = X[:, 0] + t * (2 * X[:, 1]) + np.random.normal(0, 0.1, size=1000)
    >>> 
    >>> # Initialize and fit T-Learner
    >>> tl = TLearner(RandomForestRegressor(n_estimators=100))
    >>> tl.fit(X, t, y)
    >>> 
    >>> # Estimate treatment effects
    >>> cate = tl.effect(X)
    >>> print(f"Average treatment effect: {cate.mean():.4f}")
"""

import numpy as np
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from typing import Optional, Union

# Ensure compatibility with newer NumPy versions
if not hasattr(np, 'int'):
    np.int = np.int64


class TLearner(BaseEstimator, RegressorMixin):
    """
    T-Learner for Conditional Average Treatment Effect (CATE) estimation.

    T-Learner fits separate models for the treatment and control groups, then
    estimates treatment effects by taking the difference between predictions
    from these models.

    Parameters
    ----------
    model : BaseEstimator, default=LinearRegression()
        The base learner model to use for both treatment and control groups.
        Should implement scikit-learn's estimator interface with fit() and predict() methods.
        If None, a LinearRegression model will be used.

    Attributes
    ----------
    model_t_ : BaseEstimator
        The fitted model for the treatment group.

    model_c_ : BaseEstimator
        The fitted model for the control group.

    is_fitted_ : bool
        Indicates whether the model has been fitted.

    Notes
    -----
    The T-Learner approach follows these steps:
    1. Split the data into treatment and control groups
    2. Train separate models on each group
    3. Estimate CATE by taking the difference between predictions from both models

    The main advantage of T-Learner is its ability to capture heterogeneous treatment
    effects when the treatment and control response functions are very different.
    However, it may be less efficient than other meta-learners when data is limited.
    """

    def __init__(self, model: Optional[BaseEstimator] = None):
        """
        Initialize the T-Learner.

        Parameters
        ----------
        model : BaseEstimator, optional (default=None)
            The base learner model to use for both treatment and control groups.
            If None, a LinearRegression model will be used.
        """
        self.model = model if model is not None else LinearRegression()
        self.model_t_ = None
        self.model_c_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> 'TLearner':
        """
        Fit the T-Learner on the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix.

        t : array-like of shape (n_samples,)
            The treatment assignment vector (0 for control, 1 for treatment).

        y : array-like of shape (n_samples,)
            The observed outcome vector.

        Returns
        -------
        self : TLearner
            The fitted estimator.

        Notes
        -----
        This method fits separate models for the treatment and control groups.
        It creates clones of the base model to ensure separate instances are used.
        """
        # Validate input
        X = np.asarray(X)
        t = np.asarray(t)
        y = np.asarray(y)

        if X.shape[0] != t.shape[0] or X.shape[0] != y.shape[0]:
            raise ValueError("X, t, and y must have the same number of samples")

        # Split data into treatment and control groups
        treatment_mask = t == 1
        control_mask = t == 0

        X_t = X[treatment_mask]
        y_t = y[treatment_mask]

        X_c = X[control_mask]
        y_c = y[control_mask]

        # Create and fit models for treatment and control groups
        self.model_t_ = clone(self.model)
        self.model_c_ = clone(self.model)

        self.model_t_.fit(X_t, y_t)
        self.model_c_.fit(X_c, y_c)

        self.is_fitted_ = True

        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate the Conditional Average Treatment Effect (CATE) for the given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix.

        Returns
        -------
        cate : ndarray of shape (n_samples,)
            The estimated treatment effect for each sample.

        Notes
        -----
        The CATE is estimated as the difference between the predicted outcomes
        under treatment and control conditions.
        """
        if not self.is_fitted_:
            raise ValueError("This TLearner instance is not fitted yet. "
                             "Call 'fit' before using this method.")

        X = np.asarray(X)

        # Predict outcomes under treatment and control conditions
        y_t_pred = self.model_t_.predict(X)
        y_c_pred = self.model_c_.predict(X)

        # Calculate treatment effect as the difference
        cate = y_t_pred - y_c_pred

        return cate

    def predict(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Predict outcomes for the given samples and treatment assignments.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix.

        t : array-like of shape (n_samples,)
            The treatment assignment vector (0 for control, 1 for treatment).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted outcomes.

        Notes
        -----
        This method uses the treatment model for samples with t=1 and
        the control model for samples with t=0.
        """
        if not self.is_fitted_:
            raise ValueError("This TLearner instance is not fitted yet. "
                             "Call 'fit' before using this method.")

        X = np.asarray(X)
        t = np.asarray(t)

        if X.shape[0] != t.shape[0]:
            raise ValueError("X and t must have the same number of samples")

        # Initialize predictions array
        y_pred = np.zeros(X.shape[0])

        # Predict using treatment model for treated units
        treatment_mask = t == 1
        if np.any(treatment_mask):
            y_pred[treatment_mask] = self.model_t_.predict(X[treatment_mask])

        # Predict using control model for control units
        control_mask = t == 0
        if np.any(control_mask):
            y_pred[control_mask] = self.model_c_.predict(X[control_mask])

        return y_pred
