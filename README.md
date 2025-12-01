# AR Analysis Workflow

This repository contains a **Jupyter Notebook** (`AR_analysis.ipynb`) and a set of custom Python functions in the `functions` folder for **autoregressive (AR) time series analysis**. The workflow implements almost everything from scratch using **NumPy, Pandas, Matplotlib, and SciPy (optimize)**.

---

## Notebook: `AR_analysis.ipynb`

The notebook provides a complete **AR(p) modeling workflow**:

1. **Data Preparation**
   - Load and preprocess time series data
   - Compute log-returns if needed

2. **AR(p) Process Overview**
   - Definition of AR(p) models
   - White noise innovations \(\varepsilon_t \sim WN(0, \sigma^2)\)
   - Stationarity conditions (roots of characteristic polynomial)
   - Consequences for mean, variance, and autocovariance

3. **Stationarity & Autocorrelation**
   - Characteristic polynomial
   - Yule-Walker equations
   - Theoretical vs sample autocovariance and autocorrelation
   - Gamma_0 and lag-k autocovariances

4. **Fitting AR(p) Models**
   - **OLS estimation**
     - Minimizing squared residuals
     - Derivation leads to normal equations
     - Matrix formulation with design matrix X and target Y
   - **Maximum Likelihood Estimation (MLE)**
     - Gaussian innovations
     - Student-t innovations
     - Numerical optimization using `scipy.optimize.minimize`
     - Handling heavy-tailed shocks

5. **Residual Analysis**
   - Compute residuals: \(\hat{\epsilon}_t = y_t - \hat{y}_t\)
   - Check distribution: histogram, QQ-plot
   - Check moments: mean ≈ 0, variance ≈ σ²
   - Check autocorrelation (ACF) of residuals

6. **Model Selection**
   - Compute **AIC and BIC** from negative log-likelihood
   - Compare Gaussian vs Student-t AR models
   - Discuss expectations:
     - Normal data → Gaussian model preferred
     - Heavy-tailed data → Student-t model may be preferred

7. **Practical Notes**
   - AR vs MA vs ARMA selection
   - Model diagnostics importance
   - Limitations: linearity and stationarity assumptions
   - Tips for heavy-tailed shocks or outliers

---

## Functions Folder: `functions/`

Contains custom Python scripts with helper functions used in the notebook:

- Implementations of:
  - AR(p) OLS fitting
  - MLE fitting for Gaussian and Student-t
  - Residual calculation and plotting
  - AIC/BIC calculation
  - Autocovariance and autocorrelation (theoretical and sample)
- Only depends on **NumPy, Pandas, Matplotlib, and SciPy** (optimize module for numerical MLE)

---

## Requirements

- Python ≥ 3.10
- Packages:
  - numpy
  - pandas
  - matplotlib
  - scipy

---

## Usage

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
