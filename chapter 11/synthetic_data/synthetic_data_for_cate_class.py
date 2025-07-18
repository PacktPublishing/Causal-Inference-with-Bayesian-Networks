import numpy as np

# Ensure compatibility with newer NumPy versions
if not hasattr(np, 'int'):
    np.int = np.int64

class synthetic_data_for_cate:
    """
    A class for generating synthetic data for Conditional Average Treatment Effect (CATE) estimation
    and calculating the true CATE values.

    This class provides a unified interface for different synthetic data generation models,
    allowing users to experiment with different data models without the need to write
    additional Python modules of the same structure.

    Attributes:
    -----------
    model_type : str
        The type of synthetic data model to use. Options are:
        - 'model1': Uses the model from synthetic_data_for_cate.py
        - 'model2': Uses the model from synthetic_data_for_cate2.py

    n : int
        Number of samples to generate

    d : int
        Number of covariates/features to generate

    X : ndarray of shape (n, d) or None
        Matrix of covariates/features. None until get_synthetic_data is called.

    treatment : ndarray of shape (n,) or None
        Binary treatment assignment (1=treated, 0=control). None until get_synthetic_data is called.

    y : ndarray of shape (n,) or None
        Outcome variable. None until get_synthetic_data is called.
    """

    def __init__(self, model_type='model2', n=1000, d=5):
        """
        Initialize the synthetic_data_for_cate class.

        Parameters:
        -----------
        model_type : str, default='model2'
            The type of synthetic data model to use. Options are:
            - 'model1': Uses the model from synthetic_data_for_cate.py
            - 'model2': Uses the model from synthetic_data_for_cate2.py

        n : int, default=1000
            Number of samples to generate

        d : int, default=5
            Number of covariates/features to generate
        """
        self.model_type = model_type
        self.n = n
        self.d = d
        self.X = None
        self.treatment = None
        self.y = None

    def get_synthetic_data(self):
        r"""
        Generate synthetic data for Conditional Average Treatment Effect (CATE) estimation.

        Model1 Equations:
        ----------------
        Propensity:
        $$p(X) = \frac{1}{1 + \exp(-(-1 + 2X_1 + X_2^2))}$$

        Treatment Effects:
        $$\tau(X) = 2.0\sin(3X_1) + 1.5X_2^2 - 1.0X_3X_4 + 0.5\exp(X_5)$$

        Baseline (Control Outcome):
        $$\mu_0(X) = 0.5\sin(3X_1) + 1.0X_2^2 + 0.8X_3X_4 + 0.3\exp(X_5)$$

        Noise:
        $$\epsilon \sim \mathcal{N}(0, 0.5 + 0.5X_1)$$

        Outcome:
        $$Y = \mu_0(X) + T \cdot \tau(X) + \epsilon$$

        Model2 Equations:
        ----------------
        Propensity:
        $$p(X) = \frac{1}{1 + \exp(-(-2 + 4X_1 + 2X_2^2 - 3X_3X_4))}$$

        Treatment Effects:
        $$\tau(X) = 4.0 \cdot \mathbb{1}(X_1 > 0.5) - 3.0 \cdot \mathbb{1}(X_2 > 0.7) + 5.0 \cdot \mathbb{1}(X_3X_4 > 0.5) - 2.0 \cdot \mathbb{1}(X_5 < 0.3)$$

        Baseline (Control Outcome):
        $$\mu_0(X) = 1.0\sin(2X_1) + 0.5X_2^2 + 0.8X_3X_4 + 0.3\exp(X_5)$$

        Noise:
        $$\epsilon \sim \mathcal{N}(0, 0.5 + 0.5X_1)$$

        Outcome:
        $$Y = \mu_0(X) + T \cdot \tau(X) + \epsilon$$

        Returns:
        --------
        X : ndarray of shape (n, d)
            Matrix of covariates/features

        treatment : ndarray of shape (n,)
            Binary treatment assignment (1=treated, 0=control)

        y : ndarray of shape (n,)
            Outcome variable
        """
        np.random.seed(0)
        self.X = np.random.rand(self.n, self.d)

        if self.model_type == 'model1':
            # Generate treatment with confounding
            propensity = 1 / (1 + np.exp(-(-1 + 2 * self.X[:, 0] + self.X[:, 1] ** 2)))  # Non-linear confounding
            self.treatment = np.random.binomial(1, propensity, self.n)

            # Create complex heterogeneous treatment effects
            treatment_effects = (2.0 * np.sin(3 * self.X[:, 0]) +  # Non-linear effect
                                1.5 * self.X[:, 1] ** 2 -  # Quadratic effect
                                1.0 * self.X[:, 2] * self.X[:, 3] +  # Interaction effect
                                0.5 * np.exp(self.X[:, 4]))  # Exponential effect

            # Non-linear baseline effects
            baseline = (0.5 * np.sin(3 * self.X[:, 0]) +  # Non-linear term
                        1.0 * self.X[:, 1] ** 2 +  # Quadratic term
                        0.8 * self.X[:, 2] * self.X[:, 3] +  # Interaction term
                        0.3 * np.exp(self.X[:, 4]))  # Exponential term

            # Different noise levels for different subgroups
            noise_level = 0.5 + 0.5 * self.X[:, 0]  # Heteroskedastic noise
            noise = np.random.normal(0, noise_level, self.n)

            # Generate outcome with all components
            self.y = (baseline +  # Non-linear baseline
                    self.treatment * treatment_effects +  # Complex treatment effects
                    noise)  # Heteroskedastic noise

        elif self.model_type == 'model2':
            # Generate treatment with strong, non-linear confounding
            propensity = 1 / (1 + np.exp(-(-2 + 4 * self.X[:, 0] + 2 * self.X[:, 1]**2 - 3 * self.X[:, 2] * self.X[:, 3])))
            self.treatment = np.random.binomial(1, propensity, self.n)

            # Create highly heterogeneous treatment effects with threshold functions
            treatment_effects = (
                4.0 * (self.X[:, 0] > 0.5) -  # Positive effect if X1 > 0.5
                3.0 * (self.X[:, 1] > 0.7) +  # Negative effect if X2 > 0.7
                5.0 * (self.X[:, 2] * self.X[:, 3] > 0.5) -  # Positive effect if X3*X4 > 0.5
                2.0 * (self.X[:, 4] < 0.3)    # Negative effect if X5 < 0.3
            )

            # Non-linear baseline effects (control outcome)
            baseline = (
                1.0 * np.sin(2 * self.X[:, 0]) +
                0.5 * self.X[:, 1]**2 +
                0.8 * self.X[:, 2] * self.X[:, 3] +
                0.3 * np.exp(self.X[:, 4])
            )

            # Different noise levels for different subgroups
            noise_level = 0.5 + 0.5 * self.X[:, 0]  # Heteroskedastic noise
            noise = np.random.normal(0, noise_level, self.n)

            # Generate outcome with all components
            self.y = (baseline +  # Non-linear baseline
                    self.treatment * treatment_effects +  # Complex treatment effects
                    noise)  # Heteroskedastic noise
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Choose 'model1' or 'model2'.")

        return self.X, self.treatment, self.y

    def get_true_cate(self, features=None):
        r"""
        Calculate the true Conditional Average Treatment Effect (CATE) for the given features.

        True CATE Equations:
        -------------------
        Model1:
        $$\tau(X) = 2.0\sin(3X_1) + 1.5X_2^2 - 1.0X_3X_4 + 0.5\exp(X_5)$$

        Model2:
        $$\tau(X) = 4.0 \cdot \mathbb{1}(X_1 > 0.5) - 3.0 \cdot \mathbb{1}(X_2 > 0.7) + 5.0 \cdot \mathbb{1}(X_3X_4 > 0.5) - 2.0 \cdot \mathbb{1}(X_5 < 0.3)$$

        Parameters:
        -----------
        features : ndarray of shape (n, d), default=None
            Matrix of covariates/features. If None, uses the features from the last call to get_synthetic_data.

        Returns:
        --------
        true_cate : ndarray of shape (n,)
            True CATE values for each sample
        """
        if features is None:
            if self.X is None:
                raise ValueError("No features available. Call get_synthetic_data first or provide features.")
            features = self.X

        # Convert to dense array if sparse
        if hasattr(features, 'toarray'):
            features = features.toarray()

        if self.model_type == 'model1':
            # True CATE for model1
            true_cate = (2.0 * np.sin(3 * features[:, 0]) +  # Non-linear effect
                         1.5 * features[:, 1] ** 2 -  # Quadratic effect
                         1.0 * features[:, 2] * features[:, 3] +  # Interaction effect
                         0.5 * np.exp(features[:, 4]))  # Exponential effect
        elif self.model_type == 'model2':
            # True CATE for model2
            true_cate = (
                4.0 * (features[:, 0] > 0.5) -  # Positive effect if X1 > 0.5
                3.0 * (features[:, 1] > 0.7) +  # Negative effect if X2 > 0.7
                5.0 * (features[:, 2] * features[:, 3] > 0.5) -  # Positive effect if X3*X4 > 0.5
                2.0 * (features[:, 4] < 0.3)    # Negative effect if X5 < 0.3
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Choose 'model1' or 'model2'.")

        return true_cate
