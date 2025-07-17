"""
R-Learner2 for Conditional Average Treatment Effect (CATE) Estimation
====================================================================

This is an improved implementation of the R-Learner that handles sparse matrices properly.
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

Algorithm Steps:
--------------
1. **First Stage**: 
   - Fit a model to predict outcomes (outcome model)
   - Fit a model to predict treatment assignment (propensity model)

2. **Second Stage**:
   - Compute residuals from both models
   - Outcome residuals: actual outcome minus predicted outcome
   - Treatment residuals: actual treatment minus predicted treatment probability

3. **Third Stage**:
   - Regress outcome residuals on treatment residuals
   - The coefficient represents the treatment effect
   - Can use any regression method for this stage

Advantages:
----------
- More efficient estimation than other meta-learners in many scenarios
- Robust to confounding when either outcome or propensity model is correctly specified
- Can achieve oracle rates of convergence under certain conditions
- Handles continuous treatments naturally
- Often performs well with limited data

Limitations:
-----------
- More complex implementation than S-Learner or T-Learner
- Sensitive to extreme propensity scores (values close to 0 or 1)
- Requires careful handling of division by zero in residual calculations
- Performance depends on the quality of both outcome and propensity models
- May be less stable with very small sample sizes

References:
----------
Nie, X., & Wager, S. (2017). Quasi-oracle estimation of heterogeneous treatment effects.
arXiv preprint arXiv:1712.04912.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import scipy.sparse as sp

# Ensure compatibility with newer NumPy versions
if not hasattr(np, 'int'):
    np.int = np.int64


def ensure_dense(X):
    """
    Convert input to dense numpy array if it's sparse.
    
    Parameters
    ----------
    X : array-like
        Input array which might be sparse
        
    Returns
    -------
    array-like
        Dense numpy array
    """
    if X is None:
        return None
    
    # Check if it's a scipy sparse matrix
    if sp.issparse(X):
        return X.toarray()
    
    # Check if it has toarray method (like some custom sparse implementations)
    if hasattr(X, 'toarray'):
        return X.toarray()
    
    # Already dense
    return X


class RLearner2:
    """
    Improved R-Learner for estimating Conditional Average Treatment Effects (CATE).
    
    This implementation properly handles sparse matrices and has improved numerical stability.
    
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
        The model must implement fit() and predict_proba() methods.
        
    effect_model : estimator object, optional (default=None)
        The model used to predict treatment effects from residuals. 
        If None, uses GradientBoostingRegressor.
        The model must implement fit() and predict() methods.
        
    random_state : int, optional (default=42)
        Random seed for reproducibility in data splitting.
        
    test_size : float, optional (default=0.5)
        Proportion of the data to use for validation in the data splitting step.
    
    Attributes
    ----------
    outcome_model_ : estimator object
        The fitted model used to predict outcomes.
        
    treatment_model_ : estimator object
        The fitted model used to predict treatment assignment.
        
    effect_model_ : estimator object
        The fitted model used to predict treatment effects from residuals.
    """
    
    def __init__(self, outcome_model=None, treatment_model=None, effect_model=None, 
                 random_state=42, test_size=0.5):
        # Initialize models with defaults if not provided
        self.outcome_model = outcome_model if outcome_model is not None else LinearRegression()
        self.treatment_model = treatment_model if treatment_model is not None else LinearRegression()
        self.effect_model = effect_model if effect_model is not None else GradientBoostingRegressor()
        
        # Store configuration
        self.random_state = random_state
        self.test_size = test_size
        
        # These will be set during fitting
        self.outcome_model_ = None
        self.treatment_model_ = None
        self.effect_model_ = None
        self.feature_count_ = None
    
    def _compute_propensity_scores(self, X, treatment):
        """
        Compute propensity scores using the treatment model.
        
        Parameters
        ----------
        X : array-like
            Feature matrix
            
        treatment : array-like
            Binary treatment indicator
            
        Returns
        -------
        array-like
            Propensity scores (probability of treatment)
        """
        # Ensure X is dense
        X = ensure_dense(X)
        
        # Clone the model to avoid modifying the original
        model = clone(self.treatment_model)
        
        # Fit the model
        model.fit(X, treatment)
        
        # Check if the model has predict_proba method (for classifiers)
        if hasattr(model, 'predict_proba'):
            # Get probability of treatment (class 1)
            propensity = model.predict_proba(X)[:, 1]
        else:
            # For regression models, just use the prediction
            propensity = model.predict(X)
        
        # Clip propensity scores to avoid extreme values
        eps = np.finfo(float).eps
        propensity = np.clip(propensity, eps, 1 - eps)
        
        return propensity, model
    
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
        # Ensure inputs are dense arrays
        X = ensure_dense(X)
        y = np.asarray(y)
        treatment = np.asarray(treatment)
        
        # Store feature count for prediction
        self.feature_count_ = X.shape[1]
        
        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
            X, y, treatment, test_size=self.test_size, random_state=self.random_state
        )
        
        # 1. Fit the outcome model on training data
        self.outcome_model_ = clone(self.outcome_model)
        self.outcome_model_.fit(X_train, y_train)
        
        # Predict outcomes on validation data
        y_pred = self.outcome_model_.predict(X_val)
        
        # 2. Compute propensity scores
        propensity, self.treatment_model_ = self._compute_propensity_scores(X_train, t_train)
        
        # Predict propensity scores on validation data
        if hasattr(self.treatment_model_, 'predict_proba'):
            p_val = self.treatment_model_.predict_proba(X_val)[:, 1]
        else:
            p_val = self.treatment_model_.predict(X_val)
        
        # Clip propensity scores to avoid extreme values
        eps = np.finfo(float).eps
        p_val = np.clip(p_val, eps, 1 - eps)
        
        # 3. Compute residuals
        residual_y = y_val - y_pred
        residual_t = t_val - p_val
        
        # 4. Fit the effect model on residuals
        # Create a feature matrix with X and treatment residuals
        X_with_residuals = np.column_stack((X_val, residual_t))
        
        # Fit the effect model
        self.effect_model_ = clone(self.effect_model)
        self.effect_model_.fit(X_with_residuals, residual_y)
        
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
        # Check if the model has been fitted
        if self.effect_model_ is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' first.")
        
        # Ensure X is dense
        X = ensure_dense(X)
        
        # Check feature count
        if X.shape[1] != self.feature_count_:
            raise ValueError(f"X has {X.shape[1]} features, but the model was trained with {self.feature_count_} features.")
        
        # Compute propensity scores for the new data
        if hasattr(self.treatment_model_, 'predict_proba'):
            propensity = self.treatment_model_.predict_proba(X)[:, 1]
        else:
            propensity = self.treatment_model_.predict(X)
        
        # Clip propensity scores to avoid extreme values
        eps = np.finfo(float).eps
        propensity = np.clip(propensity, eps, 1 - eps)
        
        # Create a feature matrix with X and a treatment residual of 1
        # (representing the effect of changing treatment from 0 to 1)
        X_with_unit_treatment = np.column_stack((X, np.ones(X.shape[0]) - propensity))
        
        # Predict treatment effect
        return self.effect_model_.predict(X_with_unit_treatment)


def main():
    """
    Simple demonstration of RLearner2 with synthetic data.
    """
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Features
    X = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Treatment assignment with confounding
    propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    treatment = np.random.binomial(1, propensity)
    
    # Treatment effect varies with X[:, 2]
    effect = 2 + 3 * X[:, 2]
    
    # Outcome depends on features and treatment effect
    y = X[:, 0] + 0.5 * X[:, 1] + treatment * effect + np.random.normal(0, 1, size=n_samples)
    
    # Initialize and fit R-Learner
    from sklearn.ensemble import RandomForestRegressor
    
    rl = RLearner2(
        outcome_model=RandomForestRegressor(n_estimators=100, random_state=42),
        treatment_model=RandomForestRegressor(n_estimators=100, random_state=42),
        effect_model=RandomForestRegressor(n_estimators=100, random_state=42)
    )
    
    rl.fit(X, y, treatment)
    
    # Predict treatment effects
    cate = rl.predict(X)
    
    # Compare with true effects
    print(f"Mean absolute error: {np.mean(np.abs(cate - effect)):.4f}")
    print(f"Correlation: {np.corrcoef(cate, effect)[0, 1]:.4f}")


if __name__ == "__main__":
    main()