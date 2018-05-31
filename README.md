**M**ultiscale **G**eographically **W**eighted **R**egression (MGWR)
=======================================

This module provides functionality to calibrate multiscale (M)GWR as well as traditional GWR. It is
built upon the sparse generalized linear modeling (spglm) module. 

Features
--------

- GWR model calibration via iteratively weighted least squares for Gaussian,
  Poisson, and binomial probability models.
- GWR bandwidth selection via golden section search or equal interval search
- GWR-specific model diagnostics, including a multiple hypothesis test
  correction and local collinearity
- Monte Carlo test for spatial variability of parameter estimate surfaces
- GWR-based spatial prediction
- MGWR model calibration via GAM iterative backfitting for Gaussian model
- MGWR covariate-specific inference, including a multiple hypothesis test
  correction and local collinearity   
