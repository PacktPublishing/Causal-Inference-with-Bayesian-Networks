"""
Synthetic Data for Causal Inference
===================================

This package provides tools for generating synthetic data for Conditional Average Treatment Effect (CATE) estimation.

The package includes:

1. Individual functions for generating synthetic data:
   - generate_synthetic_data_for_cate: Generates data with complex heterogeneous treatment effects
   - generate_synthetic_data_for_cate2: Generates data with highly heterogeneous treatment effects using threshold functions

2. Functions for calculating true CATE values:
   - get_true_cate_model1: Returns true CATE values for model1
   - get_true_cate_model2: Returns true CATE values for model2

3. A unified class interface:
   - synthetic_data_for_cate: A class that provides methods for generating synthetic data and calculating true CATE values

Usage Examples:
--------------

Using individual functions:

```python
from synthetic_data.synthetic_data_for_cate import generate_synthetic_data_for_cate, get_true_cate_model1
from synthetic_data.synthetic_data_for_cate2 import generate_synthetic_data_for_cate2, get_true_cate_model2

# Generate data using model1
X1, treatment1, y1 = generate_synthetic_data_for_cate()
true_cate1 = get_true_cate_model1(X1)

# Generate data using model2
X2, treatment2, y2 = generate_synthetic_data_for_cate2()
true_cate2 = get_true_cate_model2(X2)
```

Using the class interface:

```python
from synthetic_data.synthetic_data_for_cate_class import synthetic_data_for_cate

# Create an instance for model1
model1 = synthetic_data_for_cate(model_type='model1')
X1, treatment1, y1 = model1.get_synthetic_data()
true_cate1 = model1.get_true_cate()

# Create an instance for model2
model2 = synthetic_data_for_cate(model_type='model2')
X2, treatment2, y2 = model2.get_synthetic_data()
true_cate2 = model2.get_true_cate()

# Calculate true CATE for external features
external_features = ...  # Your features
true_cate = model2.get_true_cate(external_features)
```
"""

# Import functions from synthetic_data_for_cate.py
from .synthetic_data_for_cate import generate_synthetic_data_for_cate, get_true_cate_model1

# Import functions from synthetic_data_for_cate2.py
from .synthetic_data_for_cate2 import generate_synthetic_data_for_cate2, get_true_cate_model2

# Import class from synthetic_data_for_cate_class.py
from .synthetic_data_for_cate_class import synthetic_data_for_cate

__all__ = [
    'generate_synthetic_data_for_cate',
    'get_true_cate_model1',
    'generate_synthetic_data_for_cate2',
    'get_true_cate_model2',
    'synthetic_data_for_cate'
]