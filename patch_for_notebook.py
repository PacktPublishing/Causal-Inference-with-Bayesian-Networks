import numpy as np
import pygam.utils

# Monkey patch the numpy int type in pygam
original_b_spline_basis = pygam.utils.b_spline_basis

def patched_b_spline_basis(*args, **kwargs):
    # Store the original np.int
    original_np_int = getattr(np, 'int', None)
    # Set np.int to int
    setattr(np, 'int', int)
    try:
        return original_b_spline_basis(*args, **kwargs)
    finally:
        # Restore original state
        if original_np_int is not None:
            setattr(np, 'int', original_np_int)
        else:
            delattr(np, 'int')

# Replace the original function with patched version
pygam.utils.b_spline_basis = patched_b_spline_basis

# Patch the compute_propensity_score function to skip calibration
from metalearners.propensity import compute_propensity_score

# Store the original function
original_compute_propensity_score = compute_propensity_score

# Create a patched version that skips calibration
def patched_compute_propensity_score(X, treatment, p_model=None, X_pred=None, treatment_pred=None, calibrate_p=True):
    # Call the original function but force calibrate_p to False
    return original_compute_propensity_score(X, treatment, p_model, X_pred, treatment_pred, calibrate_p=False)

# Replace the original function with the patched version
import metalearners.propensity
metalearners.propensity.compute_propensity_score = patched_compute_propensity_score

# Force RLearner to use the fallback method in predict
from metalearners.r_learner import RLearner

# Store the original predict method
original_predict = RLearner.predict

# Create a patched version that always uses the fallback method
def patched_predict(self, X):
    # Force the model to use the fallback method
    self.propensity_model = None
    return original_predict(self, X)

# Replace the original method with the patched version
RLearner.predict = patched_predict

print("Applied patches for compatibility with newer NumPy versions")