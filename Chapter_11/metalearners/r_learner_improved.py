"""
Improved R-Learner for Conditional Average Treatment Effect (CATE) Estimation
====================================================================

This is an improved implementation of the R-Learner that addresses issues with propensity score
sensitivity and model complexity for highly non-linear treatment effects.

The R-Learner is a meta-learner approach for estimating heterogeneous treatment effects,
introduced by Nie and Wager (2017) in their paper "Quasi-Oracle Estimation of Heterogeneous 
Treatment Effects". It uses a residualization approach to separate the estimation of treatment 
effects from the estimation of baseline outcomes and propensity scores.

Key Characteristics:
-------------------
1. **Residualization Approach**: Uses residuals from outcome and treatment models
2. **Orthogonalization**: Separates treatment effect estimation from confounding
3. **Doubly Robust**: Provides consistent estimates if either the outcome model or propensity model is correct
4. **Flexible Base Learners**: Can use any regression model that follows scikit-learn's API
5. **Efficient Estimation**: Often achieves lower variance than other meta-learners

Improvements in this implementation:
----------------------------------
1. **Enhanced Model Complexity**: Default models have higher complexity to capture non-linear effects
2. **Propensity Score Handling**: Less sensitive to extreme propensity scores
3. **Prediction Options**: Multiple methods for using propensity scores in prediction
4. **Numerical Stability**: Better handling of edge cases and numerical issues

References:
----------
Nie, X., & Wager, S. (2017). Quasi-oracle estimation of heterogeneous treatment effects.
arXiv preprint arXiv:1712.04912.
"""


import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from metalearners.propensity import compute_propensity_score

# Ensure compatibility with newer NumPy versions
if not hasattr(np, 'int'):
    np.int = np.int64


class RLearnerImproved:
    """
    Improved R-Learner for estimating Conditional Average Treatment Effects (CATE).

    The R-Learner estimates treatment effects using a residualization approach:
    1. Fit models to predict outcomes and treatment assignment
    2. Compute residuals from both models
    3. Regress outcome residuals on treatment residuals to estimate treatment effects

    This improved implementation addresses issues with propensity score sensitivity
    and model complexity for highly non-linear treatment effects.

    Parameters
    ----------
    outcome_model : estimator object, optional (default=None)
        The model used to predict outcomes. If None, uses RandomForestRegressor with enhanced parameters.
        The model must implement fit() and predict() methods.

    treatment_model : estimator object, optional (default=None)
        The model used to predict treatment assignment. If None, uses RandomForestRegressor with enhanced parameters.
        The model must implement fit() and predict() methods.

    effect_model : estimator object, optional (default=None)
        The model used to predict treatment effects from residuals. 
        If None, uses RandomForestRegressor with enhanced parameters.
        The model must implement fit() and predict() methods.

    use_propensity_for_prediction : bool, optional (default=False)
        Whether to use propensity scores in the prediction step.
        If False, uses a constant treatment residual of 1 for all samples.
        If True, uses 1 - propensity_scores as the treatment residual.

    clip_propensity : bool, optional (default=True)
        Whether to clip propensity scores to avoid extreme values.
        Only relevant if use_propensity_for_prediction is True.

    clip_bounds : tuple, optional (default=(0.1, 0.9))
        Lower and upper bounds for clipping propensity scores.
        Only relevant if clip_propensity is True.

    Attributes
    ----------
    outcome_model : estimator object
        The fitted model used to predict outcomes.

    treatment_model : estimator object
        The fitted model used to predict treatment assignment.

    effect_model : estimator object
        The fitted model used to predict treatment effects from residuals.

    propensity_model : estimator object
        The fitted model used to predict propensity scores.
    """
    def __init__(self, outcome_model=None, treatment_model=None, effect_model=None,
                 use_propensity_for_prediction=False, clip_propensity=True, 
                 clip_bounds=(0.1, 0.9)):
        # Initialize with more complex models by default
        if outcome_model is None:
            self.outcome_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=5,
                random_state=42
            )
        else:
            self.outcome_model = outcome_model

        if treatment_model is None:
            self.treatment_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=5,
                random_state=42
            )
        else:
            self.treatment_model = treatment_model

        if effect_model is None:
            self.effect_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=5,
                random_state=42
            )
        else:
            self.effect_model = effect_model

        # Additional parameters for improved prediction
        self.use_propensity_for_prediction = use_propensity_for_prediction
        self.clip_propensity = clip_propensity
        self.clip_bounds = clip_bounds

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

        Notes
        -----
        The fitting process follows these steps:
        1. Split the data into training and validation sets
        2. Fit outcome model on the training set
        3. Compute propensity scores using the compute_propensity_score function
        4. Predict outcomes for the validation set
        5. Compute residuals for both outcomes and treatments
        6. Fit the effect model on the ratio of residuals

        The data splitting is necessary to avoid overfitting and ensure
        the residuals are computed on data not used for fitting the
        outcome and treatment models.
        """
        # Split the data
        X_train, X_val, y_train, y_val, treatment_train, treatment_val = train_test_split(
            X, y, treatment, test_size=0.5, random_state=42
        )

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
        # Note: We set calibrate_p=False to skip the calibration step and avoid issues with sparse matrices
        propensity_scores, self.propensity_model = compute_propensity_score(
            X=X_train, 
            treatment=treatment_train, 
            p_model=self.treatment_model,
            X_pred=X_val,
            treatment_pred=treatment_val,
            calibrate_p=False  # Skip calibration to avoid issues with sparse matrices
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

        Notes
        -----
        This improved implementation offers two methods for prediction:

        1. If use_propensity_for_prediction is False (default):
           - Uses a constant treatment residual of 1 for all samples
           - This is less sensitive to propensity score estimation errors
           - Works better when propensity scores are extreme or unreliable

        2. If use_propensity_for_prediction is True:
           - Computes propensity scores for the new data
           - Uses 1 - propensity_scores as the treatment residual
           - Optionally clips propensity scores to avoid extreme values
           - This is the traditional R-Learner approach
        """
        # Convert to dense array if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()

        if not self.use_propensity_for_prediction or self.propensity_model is None:
            # Use a constant treatment residual of 1 for all samples
            # This is less sensitive to propensity score estimation errors
            X_with_unit_treatment = np.column_stack((X, np.ones(X.shape[0])))
            return self.effect_model.predict(X_with_unit_treatment)
        else:
            # Compute propensity scores for the new data
            propensity_scores = self.propensity_model.predict(X)

            # Optionally clip propensity scores to avoid extreme values
            if self.clip_propensity:
                propensity_scores = np.clip(propensity_scores, self.clip_bounds[0], self.clip_bounds[1])

            # Create a feature matrix with X and a treatment residual of 1 - propensity_scores
            X_with_unit_treatment = np.column_stack((X, np.ones(X.shape[0]) - propensity_scores))

            # Predict treatment effect
            return self.effect_model.predict(X_with_unit_treatment)


def main():
    # Simulated data
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    treatment = np.random.binomial(1, 0.5, 100)

    # Create and fit the improved R-Learner
    rl = RLearnerImproved(use_propensity_for_prediction=False)
    rl.fit(X, y, treatment)

    # Predict treatment effects
    cate = rl.predict(X)
    print(f"Mean CATE: {cate.mean():.4f}")
    print(f"Std Dev CATE: {cate.std():.4f}")


if __name__ == "__main__":
    main()
