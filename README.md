# Indirect Inference Pricing

This repository contains a Python and MATLAB implementation aimed at **replicating and understanding the results** of the following papers:

- **Corsi, F. & Renò, R.**  
  *Discrete-Time Volatility Forecasting with Persistent Leverage Effects and the Pricing of Volatility Risk*

The focus is on:
- high-frequency volatility estimation
- the **LHAR-CJ model**
- **indirect inference** for stochastic volatility models
- option pricing under the physical (P) and risk-neutral (Q) measures

---

## Repository Structure

- `matlab_indirect_inference/`  
  MATLAB code used as reference and validation for the indirect inference procedure.

- `python_scripts/`  
  Core Python implementation:
  - volatility estimators (TSRV, RBV, jump-robust)
  - LHAR-CJ model
  - stochastic volatility models (GBM, Heston)
  - option pricing routines

- `results/`  
  Stores estimated parameters and numerical results.

- `*.ipynb` notebooks  
  Step-by-step workflow used to replicate the papers.

---

## Notes

- The code is written for **research and replication purposes**
- The virtual environment is not tracked
- Numerical accuracy and clarity are preferred over performance


