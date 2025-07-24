#%%
import pandas as pd
import numpy as np
#%%
def generate_synthetic_data_for_cate(n=1000, d=5):
    """
    Generate synthetic data for Conditional Average Treatment Effect (CATE) estimation.

    This function creates a synthetic dataset with heterogeneous treatment effects
    that vary based on covariates. The data generation process is designed to exhibit
    several key properties that make it suitable for evaluating and comparing
    different CATE estimation methods.

    Key Properties of the Generated Data:

    1. Treatment Effects that Vary with Covariates:
       - The treatment effect varies based on all covariates (X1-X5)
       - Each covariate influences the treatment effect through different functional forms
       - This enables the evaluation of methods' ability to capture effect heterogeneity

    2. Non-linear Interactions between Treatment and Covariates:
       - Treatment effects include non-linear terms (sine, quadratic, exponential)
       - These non-linearities test methods' ability to capture complex relationships
       - Different methods (parametric vs. non-parametric) can be compared on their
         ability to capture these non-linearities

    3. Different Effect Sizes across Subgroups:
       - The magnitude of treatment effects varies substantially across different
         values of covariates
       - Some subgroups may have negative effects while others have positive effects
       - This tests methods' ability to identify subgroups with different responses

    4. Complex Confounding Relationships:
       - Treatment assignment depends non-linearly on covariates (X1, X2)
       - This creates confounding that must be addressed by CATE estimation methods
       - The non-linear confounding tests methods' ability to control for complex
         selection bias

    Applications for CATE Estimation Research:

    1. Method Comparison: Compare different machine learning approaches for CATE
       estimation (T-learner, S-learner, X-learner, causal forests, etc.)

    2. Model Selection: Evaluate different model specifications and hyperparameter
       settings for CATE estimation

    3. Feature Importance: Assess which covariates are most important for
       treatment effect heterogeneity

    4. Subgroup Identification: Discover subgroups with similar treatment effects
       for targeted interventions

    5. Robustness Testing: Evaluate how different methods perform under various
       data conditions (sample size, dimensionality, noise levels)

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
    To get the true CATE values, use the get_true_cate_model1 function.

    The data generation process includes:
    - Covariates X drawn from uniform distribution
    - Treatment assignment with non-linear confounding:
      $\text{propensity} = \frac{1}{1 + \exp(-(-1 + 2X_1 + X_2^2))}$
    - Heterogeneous treatment effects with various functional forms:
      $\text{treatment\_effects} = 2.0\sin(3X_1) + 1.5X_2^2 - 1.0X_3X_4 + 0.5\exp(X_5)$
    - Non-linear baseline effects:
      $\text{baseline} = 0.5\sin(3X_1) + 1.0X_2^2 + 0.8X_3X_4 + 0.3\exp(X_5)$
    - Heteroskedastic noise (noise level varies with X1):
      $\text{noise\_level} = 0.5 + 0.5X_1$
    """
    np.random.seed(0)
    X = np.random.rand(n, d)

    # Generate treatment with confounding
    propensity = 1 / (1 + np.exp(-(-1 + 2 * X[:, 0] + X[:, 1] ** 2)))  # Non-linear confounding
    treatment = np.random.binomial(1, propensity, n)

    # Create complex heterogeneous treatment effects
    treatment_effects = get_true_cate_model1(X)

    # Non-linear baseline effects
    baseline = (0.5 * np.sin(3 * X[:, 0]) +  # Non-linear term
                1.0 * X[:, 1] ** 2 +  # Quadratic term
                0.8 * X[:, 2] * X[:, 3] +  # Interaction term
                0.3 * np.exp(X[:, 4]))  # Exponential term

    # Different noise levels for different subgroups
    noise_level = 0.5 + 0.5 * X[:, 0]  # Heteroskedastic noise
    noise = np.random.normal(0, noise_level, n)

    # Generate outcome with all components
    y = (baseline +  # Non-linear baseline
         treatment * treatment_effects +  # Complex treatment effects
         noise)  # Heteroskedastic noise

    return X, treatment, y

def get_true_cate_model1(features):
    """
    Calculate the true Conditional Average Treatment Effect (CATE) for model1.

    This function implements the true CATE formula used in generate_synthetic_data_for_cate.

    Parameters:
    -----------
    features : ndarray of shape (n, d)
        Matrix of covariates/features

    Returns:
    --------
    true_cate : ndarray of shape (n,)
        True CATE values for each sample
    """
    true_cate = (2.0 * np.sin(3 * features[:, 0]) +  # Non-linear effect
                 1.5 * features[:, 1] ** 2 -  # Quadratic effect
                 1.0 * features[:, 2] * features[:, 3] +  # Interaction effect
                 0.5 * np.exp(features[:, 4]))  # Exponential effect

    return true_cate
