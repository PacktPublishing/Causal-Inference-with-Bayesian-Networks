# Causal Inference with Machine Learning

This repository contains implementations of various meta-learners for causal inference, including:
- S-Learner
- T-Learner
- X-Learner
- R-Learner

## Compatibility Note

If you're using a newer version of NumPy (1.20+), you might encounter an `AttributeError: module 'numpy' has no attribute 'int'` when running the notebooks. This is because `np.int` was deprecated in NumPy 1.20 and removed in later versions.

### How to Fix

To fix this issue, follow these steps:

1. Add the following code at the beginning of your notebook (before any other imports):

```python
import sys
import os

# Add the project root to the Python path if needed
if not os.path.abspath('..') in sys.path:
    sys.path.append(os.path.abspath('..'))

# Import the patch
import patch_for_notebook
```

2. Run the notebook as usual.

The patch applies the following fixes:
- Patches pygam.utils.b_spline_basis to handle np.int compatibility
- Modifies the propensity score calculation to skip calibration (which can cause issues)
- Forces the R-Learner to use a fallback method for prediction that avoids compatibility issues

#### Example Usage in r_learner.ipynb

1. Add a new cell at the beginning of the notebook with the code above
2. Run all cells in the notebook
3. The notebook should now run without the AttributeError

## Notebooks

The repository includes several Jupyter notebooks demonstrating the use of different meta-learners:
- `r_learner.ipynb`: Demonstrates the R-Learner for CATE estimation
- `s_learner.ipynb`: Demonstrates the S-Learner for CATE estimation
- `t_learner.ipynb`: Demonstrates the T-Learner for CATE estimation
- `x_learner.ipynb`: Demonstrates the X-Learner for CATE estimation

## Synthetic Data

The repository includes modules for generating synthetic data with known treatment effects:
- `synthetic_data_for_cate_class.py`: A class-based implementation for generating synthetic data
- `synthetic_data_for_cate2.py`: A function-based implementation for generating synthetic data

## Requirements

- numpy
- scikit-learn
- matplotlib
- seaborn
- pygam
- xgboost (optional)
- lightgbm (optional)
