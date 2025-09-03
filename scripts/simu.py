import numpy as np
import pandas as pd

def SimuData(n, p_score, mu_0, mu_1, sigma_0, sigma_1):
    """
    Simulate data, including the following variables:
    - U1, U2: covariates
    - A: treatments
    - Y: outcomes

    Parameters
    ----------
    n : int, number of samples
    p_score : function of U1, U2, propensity score function
    mu_0 : function of U1, U2, mean function of Y when A=0
    mu_1 : function of U1, U2, mean function of Y when A=1
    sigma_0 : function of U1, U2, variance function of Y when A=0
    sigma_1 : function of U1, U2, variance deviation function of Y when A=1

    Returns
    -------
    pd.DataFrame
    """
    U = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.1], [0.1, 1]], size=n)
    U1 = U[:, 0]
    U2 = U[:, 1]
    A = np.random.binomial(1, p_score(U1, U2), n)
    Y = A*(mu_1(U1, U2)+np.random.normal(0, np.sqrt(sigma_1(U1, U2)), n)) + (1-A)*(mu_0(U1, U2)+np.random.normal(0, np.sqrt(sigma_0(U1, U2)), n))
    return pd.DataFrame({"U1": U1, "U2": U2, "A": A, "Y": Y})