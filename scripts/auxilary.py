import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import clone

def conditional_variance(X, Y, mu_hat, var_model):
    """
    Estimate the conditional variance Var(Y|X) using a random forest regressor.

    Parameters
    ----------
    X: np.ndarray, covariates
    Y: np.ndarray, outcomes
    mu_hat: function that takes X as input and returns the estimated E(Y|X)
    var_model: machine learning model class, estimate the conditional variance given covariates

    Returns
    -------
    function that takes X as input and returns the estimated conditional variance Var(Y|X)
    """
    pred = mu_hat(X)
    residuals = (Y - pred) ** 2
    model_var = clone(var_model)
    model_var.fit(X, residuals)
    return lambda x: model_var.predict(x)

def density_ratio(U_target, U_source, bandwidth=0.5):
    """
    Estimate the density ratio p_target(U) / p_source(U) using kernel density estimation.

    Parameters
    ----------
    U_target: np.ndarray, samples from the target distribution
    U_source: np.ndarray, samples from the source distribution
    bandwidth: float, bandwidth for kernel density estimation

    Returns
    -------
    function that takes U as input and returns the estimated density ratio
    """
    kde_target = stats.gaussian_kde(U_target.T, bw_method=bandwidth)
    kde_source = stats.gaussian_kde(U_source.T, bw_method=bandwidth)
    return lambda u: kde_target(u.T) / kde_source(u.T)