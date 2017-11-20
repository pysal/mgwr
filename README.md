**G**eographically **W**eighted **R**egression
=======================================

This module provides geographically weighted regression functionality. It is
built upon the sparse generalized linear modeling (spglm) module. 

Features
--------

The gwr module currently features

- gwr model estimation via iteratively weighted least squares for Gaussian,
  Poisson, and binomial probability models.
- gwr bandwidth selection via golden section search
- gwr-specific model diagnostics, including a multiple hypothesis testing
  correction
- gwr-based spatial prediction

Future Work
-----------

- Additional probability models (gamma, negative binomial)
- Tests for spatial variability
- Multi-scale gwr



