# Simple SLearner Implementation
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from typing import Optional, Union
import numpy as np

# Ensure compatibility with newer NumPy versions
if not hasattr(np, 'int'):
    np.int = np.int64


class SLearner(BaseEstimator, RegressorMixin):
    """S-Learner implementation for estimating heterogeneous treatment effects.

    The S-Learner (Single-Learner) approach combines treatment and features into a single
    predictor to estimate the outcome. CATE is then computed as the difference between
    predicted outcomes under treatment and control conditions.

    Parameters
    ----------
    model : BaseEstimator, default=RandomForestRegressor()
        The base learner that will be used for prediction. Must implement fit() and predict().
        Common choices include:
        - RandomForestRegressor()
        - LGBMRegressor()
        - XGBRegressor()
        - LinearRegression()
        The model should be from scikit-learn or follow scikit-learn's API.

    Attributes
    ----------
    model_instance_ : BaseEstimator
        The fitted model instance.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> # Initialize S-Learner with Random Forest
    >>> sl = SLearner(RandomForestRegressor(n_estimators=100))
    >>> # Fit the model
    >>> sl.fit(X=features, t=treatment, y=outcomes)
    >>> # Estimate CATE
    >>> cate_estimates = sl.effect(X=features)
    """

    def __init__(self, model: Optional[BaseEstimator] = None):
        if model is None:
            self.model = RandomForestRegressor()
        else:
            self.model = model

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray) -> 'SLearner':
        """Fit the S-Learner model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix.
        t : array-like of shape (n_samples,)
            The treatment assignments (0 or 1).
        y : array-like of shape (n_samples,)
            The observed outcomes.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Concatenate treatment with features
        X_with_t = np.column_stack([X, t])

        # Fit the model
        self.model_instance_ = self.model.fit(X_with_t, y)
        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        """Estimate the conditional average treatment effect (CATE).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix to estimate CATE for.

        Returns
        -------
        cate : array-like of shape (n_samples,)
            The estimated conditional average treatment effects.
        """
        # Create copies with treatment 0 and 1
        X_control = np.column_stack([X, np.zeros(X.shape[0])])
        X_treated = np.column_stack([X, np.ones(X.shape[0])])

        # Compute CATE as difference in predictions
        cate = self.model_instance_.predict(X_treated) - self.model_instance_.predict(X_control)
        return cate

    def predict(self, X: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Predict outcomes for given features and treatment assignments.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix.
        t : array-like of shape (n_samples,)
            The treatment assignments (0 or 1).

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted outcomes.
        """
        X_with_t = np.column_stack([X, t])
        return self.model_instance_.predict(X_with_t)
