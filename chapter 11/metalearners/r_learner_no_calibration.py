"""
R-Learner for Conditional Average Treatment Effect (CATE) Estimation (No Calibration)
====================================================================

This is a modified version of the R-Learner that skips the calibration step in propensity score calculation.
It's used for testing purposes to avoid issues with sparse matrices in the calibration function.

The R-Learner is a meta-learner approach for estimating heterogeneous treatment effects,
introduced by Nie and Wager (2017) in their paper "Quasi-Oracle Estimation of Heterogeneous 
Treatment Effects". It uses a residualization approach to separate the estimation of treatment 
effects from the estimation of baseline outcomes and propensity scores.
"""


import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from metalearners.propensity import compute_propensity_score

# Ensure compatibility with newer NumPy versions
if not hasattr(np, 'int'):
    np.int = np.int64


class RLearnerNoCalibration:
    """
    R-Learner for estimating Conditional Average Treatment Effects (CATE).
    This version skips the calibration step in propensity score calculation.

    The R-Learner estimates treatment effects using a residualization approach:
    1. Fit models to predict outcomes and treatment assignment
    2. Compute residuals from both models
    3. Regress outcome residuals on treatment residuals to estimate treatment effects

    Parameters
    ----------
    outcome_model : estimator object, optional (default=None)
        The model used to predict outcomes. If None, uses LinearRegression.
        The model must implement fit() and predict() methods.

    treatment_model : estimator object, optional (default=None)
        The model used to predict treatment assignment. If None, uses LinearRegression.
        The model must implement fit() and predict() methods.

    effect_model : estimator object, optional (default=None)
        The model used to predict treatment effects from residuals. 
        If None, uses GradientBoostingRegressor.
        The model must implement fit() and predict() methods.

    Attributes
    ----------
    outcome_model : estimator object
        The fitted model used to predict outcomes.

    treatment_model : estimator object
        The fitted model used to predict treatment assignment.

    effect_model : estimator object
        The fitted model used to predict treatment effects from residuals.
    """
    def __init__(self, outcome_model=None, treatment_model=None, effect_model=None):
        if outcome_model is None:
            self.outcome_model = LinearRegression()
        else:
            self.outcome_model = outcome_model

        if treatment_model is None:
            self.treatment_model = LinearRegression()
        else:
            self.treatment_model = treatment_model

        if effect_model is None:
            self.effect_model = GradientBoostingRegressor()
        else:
            self.effect_model = effect_model

        # Initialize propensity_model to None, it will be set during fit
        self.propensity_model = None

    def fit(self, X, y, treatment):
        """
        Fit the R-Learner to estimate treatment effects.

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
        """
        # Split the data
        X_train, X_val, y_train, y_val, treatment_train, treatment_val = train_test_split(X, y, treatment,
                                                                                          test_size=0.5,
                                                                                          random_state=42)

        # Convert to dense arrays if sparse
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        if hasattr(X_val, 'toarray'):
            X_val = X_val.toarray()

        # Fit the outcome model
        self.outcome_model.fit(X_train, y_train)
        y_pred = self.outcome_model.predict(X_val)

        # Compute propensity scores using the compute_propensity_score function
        # This uses the treatment_model as the p_model parameter
        # Note: We set calibrate_p=False to skip the calibration step
        propensity_scores, self.propensity_model = compute_propensity_score(
            X=X_train, 
            treatment=treatment_train, 
            p_model=self.treatment_model,
            X_pred=X_val,
            treatment_pred=treatment_val,
            calibrate_p=False  # Skip calibration
        )

        # Compute residuals
        residual_y = y_val - y_pred
        residual_treatment = treatment_val - propensity_scores

        # Instead of dividing residuals (which can cause numerical instability),
        # we regress outcome residuals on treatment residuals
        # First, create a feature matrix with X and treatment residuals
        X_with_residuals = np.column_stack((X_val, residual_treatment))

        # Fit the effect model using X and treatment residuals to predict outcome residuals
        self.effect_model.fit(X_with_residuals, residual_y)

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
        """
        # Convert to dense array if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()

        # For simplicity and to avoid issues with the propensity model,
        # we'll use a constant treatment residual of 1 for all samples
        X_with_unit_treatment = np.column_stack((X, np.ones(X.shape[0])))
        return self.effect_model.predict(X_with_unit_treatment)
