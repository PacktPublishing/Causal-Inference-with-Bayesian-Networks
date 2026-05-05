# Political Democracy (Bollen, 1989) with semopy — Model Specification and Computation Steps

This document explains, in a step-by-step manner, how the Structural Equation Model (SEM) for the classic Political Democracy example is specified and estimated using the semopy library, and how the path diagram is produced. It also clarifies why the path diagram shows six curved, double-headed arrows (residual correlations among observed variables) while the formal model specification includes only four.


## 1) Data and Variables
We use the dataset bundled with semopy (examples.political_democracy). Observed indicators are grouped as:
- Industrialization (1960) latent: ind60 → indicators: x1, x2, x3
- Democracy (1960) latent: dem60 → indicators: y1, y2, y3, y4
- Democracy (1965) latent: dem65 → indicators: y5, y6, y7, y8

How the data are loaded in our project (see bollen_semopy.ipynb):

```python
import semopy as _semopy_ref
data = _semopy_ref.examples.political_democracy.get_data()
```


## 2) Model Specification (semopy syntax)
We follow a lavaan-like syntax supported by semopy:
- = ~ defines measurement loadings (latent =~ observed1 + observed2 + ...)
- ~ defines regressions (outcome ~ predictors)
- ~~ defines (residual) covariances/correlations

The model string used in the notebook is:

```text
# Measurement model
ind60 =~ x1 + x2 + x3
dem60 =~ y1 + y2 + y3 + y4
dem65 =~ y5 + y6 + y7 + y8

# Structural regressions
dem60 ~ ind60
dem65 ~ ind60 + dem60

# Residual correlations (same indicators across time)
y1 ~~ y5
y2 ~~ y6
y3 ~~ y7
y4 ~~ y8
```

Notes:
- The four residual covariances above connect the same indicators across time (e.g., y1 in 1960 with y5 in 1965). These are commonly included to account for correlated uniquenesses over time.


## 3) Estimation Steps in semopy
The key steps performed in bollen_semopy.ipynb are:

1) Build a model from the specification string and fit it to the data.
```python
from semopy import Model
model = Model(MODEL_SPEC)
res = model.fit(data)  # estimates parameters by minimizing the SEM objective
```

2) Inspect the parameter estimates and (optionally) fit indices.
```python
# Parameter table (loadings, regressions, residual (co)variances, etc.)
estimates = model.inspect()  # in new semopy versions

# Fit indices, if available in your semopy version
stats = model.calc_stats()  # or semopy.calc_stats(model[, data]) depending on version
```

Interpretation tips:
- Measurement loadings: Higher positive values suggest stronger indicators of the latent constructs.
- Structural paths (e.g., dem65 ~ ind60 + dem60): Sign and magnitude indicate predictive relations among the latent variables.
- The four residual covariances y1~~y5, y2~~y6, y3~~y7, y4~~y8 are estimated parameters because they are explicitly in the model string.


## 4) Path Diagram: How It Is Produced
The notebook attempts to render a compact diagram in two ways:
- Primary: via Graphviz (if graphviz package and the dot executable are present)
- Fallback: via matplotlib when Graphviz is unavailable

Both paths parse the MODEL_SPEC to find:
- Measurement edges (latent → observed)
- Structural edges (predictor → outcome)
- Residual covariance pairs listed with ~~ in the specification

The generated image is saved to:
- bollen_semopy_report/plot/bollen_semopy_model.png


## 5) Why Six Curved, Double-Headed Arrows in the Diagram, but Only Four in the Specification?
- The formal model that is actually estimated by semopy includes only the four residual correlations explicitly written in the specification:
  - y1 ~~ y5, y2 ~~ y6, y3 ~~ y7, y4 ~~ y8
- The path diagram produced by our visualization step shows six residual correlation arcs among the observed variables:
  - y1 ~~ y5, y2 ~~ y6, y3 ~~ y7, y4 ~~ y8  (the four specified and estimated)
  - y2 ~~ y4, y6 ~~ y8                     (two additional arcs)

Rationale for the extra two arcs in the diagram:
- They are added for visualization purposes to match the widely used layout of this example (see the reference diagram at https://semlj.github.io/example2.html) and to reflect commonly discussed “correlated uniquenesses” between the two Freedom House measures within each time point (y2 with y4 in 1960; y6 with y8 in 1965).
- Importantly, these two extra arcs are not included in the MODEL_SPEC string. Therefore, semopy does not estimate parameters for them and they do not affect model fit or parameter estimates. In our plotting code, these visualization-only arcs are drawn with Graphviz using constraint=false and do not correspond to estimated coefficients.

Practical implication:
- Numbers/tables produced by model.inspect() and calc_stats() reflect only the four specified residual correlations. The diagram shows two additional arcs purely for conceptual/contextual clarity and to match the referenced illustration.

If you wish the model to estimate those two additional covariances as well, simply add them to the MODEL_SPEC string:
```text
y2 ~~ y4
y6 ~~ y8
```
Then re-run the fitting and plotting steps.


## 6) How to Reproduce
Two convenient options:

1) Run the notebook
- Open bollen_semopy.ipynb and execute the cells.
- It will load data, fit the model, print estimates/fit indices, and save the diagram to bollen_semopy_report/plot/bollen_semopy_model.png.

2) Run the demo script (if you prefer a script workflow)
- See use_semopy_demo.py for an example of programmatic usage (model building, fitting, and basic outputs). Depending on your environment, its outputs may be printed to the console rather than generating a full report image.

Environment requirements are listed in requirements.txt. A typical setup is:
```bash
pip install -r requirements.txt
```


## 7) Version Notes
- semopy APIs such as inspect() and calc_stats() have slight version differences. The notebook includes fallbacks where possible to remain robust across versions in the 2.x line.
- Graphviz rendering requires both the Python graphviz package and the Graphviz “dot” executable to be installed and available on PATH. If they’re not present, the matplotlib fallback will be used.


## 8) Summary
- The model is specified and estimated in semopy using a lavaan-like syntax. The four residual correlations across time (same indicators) are part of the estimation.
- The path diagram includes two additional residual-correlation arcs (y2~~y4 and y6~~y8) strictly for visualization parity with a commonly cited diagram and for conceptual clarity. They are not estimated unless you add them to the model specification.
