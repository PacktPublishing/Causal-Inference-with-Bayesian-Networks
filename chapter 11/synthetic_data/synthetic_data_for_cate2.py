#%%
import pandas as pd
import numpy as np

# Ensure compatibility with newer NumPy versions
if not hasattr(np, 'int'):
    np.int = np.int64
#%%
def generate_synthetic_data_for_cate2(n=1000, d=5):
    r"""
    Generate synthetic data for Conditional Average Treatment Effect (CATE) estimation
    with enhanced heterogeneity to better differentiate meta-learners.

    This function creates a synthetic dataset with highly heterogeneous treatment effects
    that vary based on covariates. The data generation process is specifically designed
    to highlight differences between S-Learner and T-Learner approaches.

    Key Properties of the Generated Data:

    1. Highly Heterogeneous Treatment Effects:
       - The treatment effect varies dramatically based on covariates
       - Treatment effects range from strongly negative to strongly positive
       - Sharp thresholds create distinct subgroups with different effects

    2. Different Functional Forms for Treatment and Control:
       - Treatment group follows different functional patterns than control group
       - This favors T-Learner's separate models approach over S-Learner's single model
       - Control outcomes follow smoother patterns while treatment outcomes have discontinuities

    3. Complex Treatment-Covariate Interactions:
       - Treatment interacts with covariates in complex, non-linear ways
       - Some interactions create step functions and threshold effects
       - These interactions are difficult for S-Learner to capture with treatment as just another feature

    4. Strong Confounding with Non-linear Patterns:
       - Treatment assignment strongly depends on covariates
       - Propensity score has sharp thresholds creating selection bias
       - This challenges both learners but in different ways

    Parameters:
    -----------
    n : int, default=1000
        Number of samples to generate

    d : int, default=5
        Number of covariates/features to generate

    Returns:
    --------
    X : ndarray of shape (n, d)
        Matrix of covariates/features

    treatment : ndarray of shape (n,)
        Binary treatment assignment (1=treated, 0=control)

    y : ndarray of shape (n,)
        Outcome variable

    Notes:
    ------
    The true CATE for an individual with covariates X is given by the 'treatment_effects'
    formula in the code. This can be used as ground truth when evaluating estimation methods.
    To get the true CATE values, use the get_true_cate_model2 function.

    The data generation process includes:
    - Covariates X drawn from uniform distribution
    - Treatment assignment with strong, non-linear confounding:
      $\text{propensity} = \frac{1}{1 + \exp(-(-2 + 4X_1 + 2X_2^2 - 3X_3X_4))}$
    - Highly heterogeneous treatment effects with threshold functions:
      $\text{treatment\_effects} = 4.0 * (X_1 > 0.5) - 3.0 * (X_2 > 0.7) + 5.0 * (X_3 * X_4 > 0.5) - 2.0 * (X_5 < 0.3)$
    - Different functional forms for control and treatment outcomes
    - Heteroskedastic noise that varies with treatment status
    """
    np.random.seed(0)
    X = np.random.rand(n, d)

    # Generate treatment with strong, non-linear confounding
    # This creates a more complex propensity surface with sharper transitions
    propensity = 1 / (1 + np.exp(-(-2 + 4 * X[:, 0] + 2 * X[:, 1]**2 - 3 * X[:, 2] * X[:, 3])))
    treatment = np.random.binomial(1, propensity, n)

    # Create highly heterogeneous treatment effects with threshold functions
    # Using step functions creates distinct subgroups with very different effects
    treatment_effects = get_true_cate_model2(X)

    # Different functional forms for control and treatment groups
    # Control group has smoother, more continuous response surface
    control_outcome = (
        1.0 * np.sin(2 * X[:, 0]) +
        0.5 * X[:, 1]**2 +
        0.8 * X[:, 2] * X[:, 3] +
        0.3 * np.exp(X[:, 4])
    )

    # Treatment group has a different functional form with discontinuities
    # This makes it harder for a single model (S-learner) to capture both patterns
    treatment_outcome = (
        2.0 * (X[:, 0] > 0.3) * np.sin(3 * X[:, 0]) +
        1.5 * (X[:, 1] < 0.5) * X[:, 1]**3 +
        2.5 * np.tanh(X[:, 2] * X[:, 3]) +
        1.0 * (X[:, 4] > 0.6) * np.log(1 + X[:, 4])
    )

    # Different noise levels for treatment and control groups
    # Control group has constant noise
    control_noise = np.random.normal(0, 0.5, n)
    # Treatment group has heteroskedastic noise that depends on X1
    treatment_noise = np.random.normal(0, 0.5 + X[:, 0], n)

    # Generate outcomes
    y = np.zeros(n)
    control_mask = treatment == 0
    treatment_mask = treatment == 1

    # Control outcomes
    y[control_mask] = control_outcome[control_mask] + control_noise[control_mask]

    # Treatment outcomes = control outcomes + treatment effect + different noise
    y[treatment_mask] = (
        control_outcome[treatment_mask] + 
        treatment_effects[treatment_mask] + 
        treatment_outcome[treatment_mask] + 
        treatment_noise[treatment_mask]
    )

    return X, treatment, y

def get_true_cate_model2(features):
    """
    Calculate the true Conditional Average Treatment Effect (CATE) for model2.

    This function implements the true CATE formula used in generate_synthetic_data_for_cate2.

    Parameters:
    -----------
    features : ndarray of shape (n, d)
        Matrix of covariates/features

    Returns:
    --------
    true_cate : ndarray of shape (n,)
        True CATE values for each sample
    """
    true_cate = (
        4.0 * (features[:, 0] > 0.5) -  # Positive effect if X1 > 0.5
        3.0 * (features[:, 1] > 0.7) +  # Negative effect if X2 > 0.7
        5.0 * (features[:, 2] * features[:, 3] > 0.5) -  # Positive effect if X3*X4 > 0.5
        2.0 * (features[:, 4] < 0.3)    # Negative effect if X5 < 0.3
    )

    return true_cate
