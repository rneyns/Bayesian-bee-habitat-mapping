# Bayesian Nest Suitability Model

This repository contains code and tools for modeling the probability of bee nest presence based on various environmental predictors using a Bayesian logistic regression framework. The model incorporates both linear and nonlinear (quadratic) effects and supports spatial prediction across raster inputs.

## Overview

The project has two main components:

1. **Model Training**
   Fits a Bayesian logistic regression using PyMC based on tabular input data containing nest observations and environmental predictors.

2. **Spatial Prediction**
   Applies the fitted model to raster layers to generate spatial maps of predicted nest suitability and model uncertainty.

## Key Features

* Support for prior-informed and data-driven Bayesian logistic modeling
* Use of quadratic terms to capture U-shaped relationships
* Automatic rescaling (standardization) of predictors for stable inference
* Spatial inference using raster data and posterior sampling

## Inputs

### Tabular Data

A CSV file containing:

* `presence`: binary response (1 = nest present, 0 = absent)
* Predictors: clay, sand, silt, proximity, flood, agriculture, impervious, vegetation, insolation

Quadratic terms (`vegetation_squared`, `impervious_squared`) are automatically added and rescaled.

### Raster Stack

A multi-band raster containing aligned layers for all predictors, in the same order as the model expects.

## Outputs

* `predicted_probabilities.csv`: Posterior mean predictions for each row in the input CSV
* `presence_mean_map.tif`: Raster of posterior mean probability per pixel
* `presence_std_map.tif`: Raster of posterior standard deviation (uncertainty)
* `scaling_parameters.csv`: Saved scaling parameters for use in raster standardization

## Dependencies

* `pymc`
* `arviz`
* `matplotlib`, `seaborn`
* `rasterio`
* `pandas`, `numpy`, `scipy`

## Running the Model

To train and predict:

```bash
python bayesian_model.py
```

Ensure that:

* Input CSV and raster stack paths are correctly set in the script
* Python environment includes the required packages

## Visualization

To compare prior vs posterior for selected variables:

```python
az.plot_dist_comparison(trace, var_names=["beta_veg", "beta_veg2", "beta_building_bright", "beta_building_bright2"], kind="both")
```

## Notes

* Quadratic priors are informative but relaxed with wider standard deviations to allow model learning.
* Posterior diagnostics and convergence checks are recommended using `az.summary` and `az.plot_trace`.
* Predictor values must be scaled identically during training and prediction.

---

© 2025 Robbe Neyns
Vrije Universiteit Brussel
