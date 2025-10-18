# Walkthrough: Minimal CFA Tutorial with semopy (semopy_demo_tutorial.ipynb)

This document provides a narrative walkthrough of the code and the underlying theory demonstrated in `semopy_demo_tutorial.ipynb`. It focuses on a minimal Confirmatory Factor Analysis (CFA) model estimated with the `semopy` library and explains how the notebook simulates data, specifies and fits the model, inspects results, computes fit indices, and generates a path diagram (Graphviz-first, with a matplotlib fallback).


## 1) What the tutorial demonstrates

- A one-factor CFA model where a single latent variable (eta) explains three observed indicators (y1, y2, y3).
- A lightweight simulation of data suitable for quick experimentation.
- A lavaan-like model specification string interpreted by `semopy`.
- Model fitting and extraction of parameter estimates and fit indices.
- Visualization of the path diagram using Graphviz (preferred) or matplotlib (fallback).

This is a pedagogical example for learning the workflow; it is not intended to be a substantive empirical study.


## 2) Theoretical background (brief)

- CFA posits that observed variables (indicators) are noisy reflections of an unobserved (latent) construct.
- In a one-factor CFA, each indicator loads on a single latent factor. Loadings quantify how strongly the indicator reflects the factor.
- Identification: A model needs a scale for the latent factor. Common choices are: (a) fix the factor variance to 1, or (b) fix one loading to 1. The tutorial uses the factor-variance-fixing approach for clarity.
- Residuals (unique variances) represent measurement error and indicator-specific variance not explained by the latent factor.


## 3) Data simulation in the notebook

The notebook defines a helper function `build_sample_data(n=120, seed=7)` that:
- Draws a standard normal latent factor `eta ~ N(0, 1)`.
- Constructs indicators with specified loadings and residual noise, e.g.,
  - y1 = 0.8 * eta + e1 (with e1 ~ N(0, 0.6^2))
  - y2 = 0.9 * eta + e2 (with e2 ~ N(0, 0.5^2))
  - y3 = 0.7 * eta + e3 (with e3 ~ N(0, 0.7^2))
- Returns a pandas DataFrame with columns y1, y2, y3.

This mirrors the CFA generative model and provides a dataset that should be well-behaved for demonstration purposes.

Explanation of the noise term (why variance equals `scale^2`):
- In the code, an indicator is generated as `y1 = 0.8 * eta + rng.normal(scale=0.6, size=n)`.
- Here `rng.normal(loc=0, scale=0.6, size=n)` draws `n` independent samples from a normal distribution with mean 0 and standard deviation 0.6. In NumPy, the parameter named `scale` is the standard deviation (σ), not the variance.
- Therefore the random error term `e1` is distributed as `e1 ~ Normal(0, σ^2)` with `σ = 0.6`, so `Var(e1) = 0.6^2 = 0.36`.
- This implements the generative equation `y1 = 0.8 * eta + e1` with `e1` independent of `eta` (drawn from the same generator but as a separate random sample), and with the intended residual variance. Because the model fixes `Var(eta) = 1`, the implied variance of `y1` is approximately `0.8^2 * 1 + 0.6^2 = 0.64 + 0.36 = 1.0` in the population (sample variance will fluctuate around this).


## 4) Model specification (lavaan-like syntax)

The model string used in the notebook is:

```
# Measurement model
eta =~ y1 + y2 + y3

# Fix latent variance to 1 for identifiability
eta ~~ 1*eta
```

- `=~` declares factor loadings (latent on the left, indicators on the right).
- `~~` can define (co)variances. Here it fixes the variance of `eta` to 1, setting the scale of the latent variable.

Note: semopy can also achieve identification by fixing one loading to 1 instead of the factor variance; both approaches are common.


## 5) Fitting the model

Core steps in the notebook:

```python
from semopy import Model
model = Model(MODEL_SPEC)
res = model.fit(data)
```

- `Model(MODEL_SPEC)` parses the specification and prepares a SEM object.
- `fit(data)` estimates parameters by minimizing the SEM objective function (typically maximum likelihood under multivariate normality). The result object may vary by semopy version, but the fitted model is stored in `model` either way.


## 6) Inspecting parameter estimates and fit indices

The notebook retrieves parameter estimates in a way that is robust to semopy version differences:

```python
try:
    estimates = model.inspect()
except Exception:
    estimates = semopy.inspect(model) if hasattr(semopy, 'inspect') else None
```

- The resulting table typically includes loadings, (co)variances, and other parameters with estimate columns (and sometimes standard errors, z-values, p-values, depending on version/features).

Fit statistics are computed similarly defensively:

```python
try:
    stats = model.calc_stats() if hasattr(model, 'calc_stats') else semopy.calc_stats(model)
except TypeError:
    stats = semopy.calc_stats(model, data)  # fallback signature in some versions
```

Common indices you may see include CFI, TLI, RMSEA, GFI/AGFI, degrees of freedom, etc. Exact availability depends on the semopy version and installed optional dependencies.

Interpretation:
- Loadings near or above ~0.7 generally indicate strong indicators, but context matters.
- Residual variances should be non-negative and typically moderate for good indicators.
- Fit indices closer to conventional targets (e.g., CFI/TLI ~>.95, RMSEA ~<.06–.08) indicate better fit, though thresholds are heuristic.

Clarifying chi-square, p-values, and sample size:
- What counts as a “low” chi-square? Low should be interpreted relative to the model’s degrees of freedom and, importantly, relative to the baseline (independence) model’s chi-square. If the baseline chi2 is large (e.g., ~127.7) and the fitted model’s chi2 is much smaller (e.g., ~8.3), the model has removed most of the misfit, which aligns with high relative fit indices (e.g., CFI near 1).
- Why can the chi-square p-value be close to 0 even when chi2 looks small? The chi2 test evaluates exact fit (H0: the model-implied covariance equals the population covariance exactly). Even tiny discrepancies can become statistically significant as n increases because the test roughly scales with sample size. Hence, a modest chi2 can still yield a very small p-value. This is why SEM reporting routinely supplements the chi2 test with approximate-fit indices (CFI/TLI, RMSEA, SRMR) and inspection of residuals.
- If we change sample size (e.g., from n = 120 to n = 129), will results “improve”?
  - Estimation precision improves with larger n (smaller standard errors), so parameter estimates become more stable.
  - The chi2 exact-fit test often becomes more likely to reject (p-values get smaller) because power increases with n—even for trivial misfit. By that single metric, results may appear to “worsen.”
  - Relative/approximate fit indices (CFI, TLI, RMSEA, SRMR) typically remain similar if the model is correct; RMSEA may decrease slightly with better precision, though behavior depends on df and random sampling variation.
  - In practice, judge fit holistically using multiple indices and substantive reasoning, not the chi2 p-value alone.


## 7) Visualizing the path diagram

The notebook parses `MODEL_SPEC` to extract nodes and edges, then attempts a Graphviz render first:

- Nodes: ellipses for latent variables (e.g., `eta`), boxes for observed indicators (`y1`, `y2`, `y3`).
- Edges: directed arrows for loadings; double-headed curved arrows (if any covariances are specified) with `constraint=false` for layout flexibility.
- Output: an image written to `bollen_semopy_report/plot/semopy_demo_model.png`.

If Graphviz (both the Python `graphviz` package and the `dot` executable) is not available, the notebook falls back to matplotlib:

- Ellipse and text patches are used to draw nodes, with equal aspect ratio and wider limits to avoid clipping.
- Directed arrows denote loadings; optional curved double-headed arrows can show covariances.

This dual approach ensures a readable diagram across environments while keeping dependencies light.


## 8) Identification details (why fix eta variance?)

CFA models are identified up to scale and sign of the latent variable. Without a constraint, there are infinitely many equivalent solutions where scaling of the latent factor and loadings changes jointly. Fixing `eta ~~ 1*eta` pins the latent variance, making the model estimable and providing interpretable loading magnitudes. Alternatively, you could fix one loading, e.g., `eta =~ 1*y1 + y2 + y3`; then the factor is on the scale of `y1`.


## 9) Reproducing and extending the example

- Run the notebook cells in order. It will simulate data, fit the model, show estimates and fit stats, and save the path diagram to `bollen_semopy_report/plot/semopy_demo_model.png`.
- To experiment:
  - Change `n` and `seed` in `build_sample_data` to alter sample size and random state.
  - Modify loadings or residual noise levels to see how estimates and fit respond.
  - Add indicators or specify additional covariances among residuals using `~~`.
  - Try the alternative identification scheme (fix a loading to 1) and compare results.

For a script-based variant, see `use_semopy_demo.py`, which implements a similar one-factor CFA with CLI options and an optional HTML report (if `semopy.report` is available).


## 10) Troubleshooting and version notes

- semopy 2.x has minor API differences across versions; the notebook includes try/except fallbacks for `inspect()` and `calc_stats()`.
- If Graphviz rendering fails, ensure both the `graphviz` Python package and the `dot` executable are installed and on your PATH. Otherwise the matplotlib fallback will be used automatically.
- If optimization fails to converge, try increasing sample size, adjusting starting values (advanced), or verifying the identification constraints.


## 11) References

- Bollen, K. A. (1989). Structural Equations with Latent Variables. Wiley.
- semopy documentation: https://semopy.com/
- Lavaan model syntax (conceptual reference): https://lavaan.ugent.be/
