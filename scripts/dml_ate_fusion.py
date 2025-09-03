import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from auxilary import conditional_variance, density_ratio    

def ATE_DR(u, a, y, p_hat, mu_hat_1, mu_hat_0):
    """
    Compute the estimated doubly robust score (EIF) of ATE at each observation.

    Parameters
    ----------
    u: list of floats, covariates
    a: int, treatment assignment
    y: float, outcome
    p_hat: return a float from u, propensity score
    mu_hat_1: return a float from u, predicted outcome under treatment
    mu_hat_0: return a float from u, predicted outcome under control

    Returns
    -------
    float
    """
    return (a * (y - mu_hat_1(u)) / p_hat(u) - (1 - a) * (y - mu_hat_0(u)) / (1 - p_hat(u)) + mu_hat_1(u) - mu_hat_0(u))

def DML_ATE(data, p_score_model, mean_model, n_splits=2):
    """
    Double machine learning for ATE estimation and standard error. The nuisance functions are estimated by random forest.

    Parameters
    ----------
    data: pd.DataFrame, including covariates, treatment assignment and outcome
    p_score_model: machine learning model class, estimate the propensity score given covariates
    mean_model: machine learning model class, estimate the conditional mean given covariates
    n_splits: int, number of splits for cross-fitting

    Returns
    -------
    float, ATE estimate
    float, standard error of the ATE estimate
    """
    # Split the data into n_splits parts
    data_splits = np.array_split(data.sample(frac=1), n_splits)
    ate_estimates = []
    var_estimates = []
    for i in range(n_splits):
        # Use all but the i-th part for training
        train_data = pd.concat([data_splits[j] for j in range(n_splits) if j != i])
        test_data = data_splits[i]
        X_train = train_data[["U1", "U2"]]
        A_train = train_data["A"]
        Y_train = train_data["Y"]
        X_test = test_data[["U1", "U2"]]
        A_test = test_data["A"]
        Y_test = test_data["Y"]
        # Fit the nuisance functions
        p_model = clone(p_score_model)
        mu_model_1 = clone(mean_model)
        mu_model_0 = clone(mean_model)
        p_model.fit(X_train, A_train)
        mu_model_1.fit(X_train[A_train == 1], Y_train[A_train == 1])
        mu_model_0.fit(X_train[A_train == 0], Y_train[A_train == 0])
        # Define the nuisance functions, trim the propensity scores between [0.01, 0.99]
        p_hat = lambda u: p_model.predict(u).clip(0.01, 0.99)
        mu_hat_1 = lambda u: mu_model_1.predict(u)
        mu_hat_0 = lambda u: mu_model_0.predict(u) 
        # Compute the doubly robust scores on the test data
        dr_scores = ATE_DR(X_test.values, A_test.values, Y_test.values, p_hat, mu_hat_1, mu_hat_0)
        ate_estimates.append(np.mean(dr_scores))
        var_estimates.append(np.var(dr_scores))
    return np.mean(ate_estimates), np.sqrt(np.mean(var_estimates) / len(data))

def ATE_EIF_fusion_target(u, a, y, p_hat, mu_hat_1, mu_hat_0, sigma_hat, omega_hat):
    """
    Compute the estimated EIF augmentation term of ATE at each target observation with external controls.

    Parameters
    ----------
    u: list of floats, covariates
    a: int, treatment assignment
    y: float, outcome
    p_hat: return a float from u, propensity score
    mu_hat_1: return a float from u, predicted outcome under treatment
    mu_hat_0: return a float from u, predicted outcome under control
    sigma_hat: return a positive float from u, estimated conditional variance Var(Y|A = 0, X) on target data
    omega_hat: return a positive float from u, estimated weighted inverse conditional variance from source data 

    Returns
    -------
    estimated EIF augmentation of ATE at each observation of the target data
    """
    g = (1-a)/(1-p_hat(u))*(y-mu_hat_0(u))
    proj = (1/omega_hat(u))/(1/sigma_hat(u)+omega_hat(u))
    return proj*g


def ATE_EIF_fusion_source(u, a, y, p_hat, mu_hat_0, sigma_hat, omega_hat, omega_source_hat, density_ratio_hat):
    """
    Compute the estimated EIF augmentation term of ATE at each source observation with external controls.

    Parameters
    ----------
    u: list of floats, covariates
    a: int, treatment assignment
    y: float, outcome
    p_hat: return a float from u, propensity score
    mu_hat_1: return a float from u, predicted outcome under treatment
    mu_hat_0: return a float from u, predicted outcome under control
    sigma_hat: return a positive float from u, estimated conditional variance Var(Y|A = 0, X) on target data
    omega_hat: return a positive float from u, estimated weighted inverse conditional variance from source data 
    omega_source_hat: return a positive float from u, estimated inverse conditional variance Var(Y|A = 0, X) on source data
    density_ratio_hat: return a positive float from u, estimated density ratio p_target(U) / p_source(U)
    
    Returns
    -------
    estimated EIF augmentation of ATE at each observation of the source data
    """
    g = (1-a)/(1-p_hat(u))*(y-mu_hat_0(u))
    proj = density_ratio_hat(u)*(1/omega_source_hat(u))/(1/sigma_hat(u)+omega_hat(u))
    return proj*g

def DML_ATE_fusion(target_data, source_data, p_score_model, mean_model, var_model, density_ratio, n_splits=2):
    """
    Double machine learning for ATE estimation with external control. The nuisance functions are estimated by random forest.

    Parameters
    ----------
    target_data: pd.DataFrame, including covariates, treatment assignment and outcome
    source_data: list of pd.DataFrame, including covariates, treatment assignment and outcome
    p_score_model: machine learning model class, estimate the propensity score given covariates
    mean_model: machine learning model class, estimate the conditional mean given covariates
    var_model: machine learning model class, estimate the conditional variance given covariates
    density_ratio: function that takes the target and source data and return the estimated density ratio p_target(U) / p_source(U) function
    n_splits: int, number of splits for cross-fitting

    Returns
    -------
    float, ATE estimate using only target data
    float, standard error of the ATE estimate using only target data
    """
    M = len(source_data)
    ate_estimates = []
    var_estimates = []

    # Split the data into n_splits parts
    target_data_splits = np.array_split(target_data.sample(frac=1), n_splits)
    source_data_splits = []
    for m in range(M):
        source_data_splits.append(np.array_split(source_data[m].sample(frac=1), n_splits))
    ate_estimates = []
    var_estimates = []
    ate_classic_estimates = []
    var_classic_estimates = []
    for i in range(n_splits):
        # Use all but the i-th part for training
        train_target_data = pd.concat([target_data_splits[j] for j in range(n_splits) if j != i])
        test_target_data = target_data_splits[i]
        train_source_data = []
        test_source_data = []
        for m in range(M):
            train_source_data.append(pd.concat([source_data_splits[m][j] for j in range(n_splits) if j != i]))
            test_source_data.append(source_data_splits[m][i])
        
        # Combine all data to estimate the nuisance function mu_0
        mu_model_0 = clone(mean_model)
        X_train = pd.concat([train_target_data[["U1", "U2"]]] + [train_source_data[m][["U1", "U2"]] for m in range(M)])
        A_train = pd.concat([train_target_data["A"]] + [train_source_data[m]["A"] for m in range(M)])
        Y_train = pd.concat([train_target_data["Y"]] + [train_source_data[m]["Y"] for m in range(M)])
        mu_model_0.fit(X_train[A_train == 0], Y_train[A_train == 0])
        mu_hat_0 = lambda u: mu_model_0.predict(u)

        # Fit the nuisance functions on target data
        X_target_train = train_target_data[["U1", "U2"]]
        A_target_train = train_target_data["A"]
        Y_target_train = train_target_data["Y"]

        # mu_model_0.fit(X_target_train[A_target_train == 0], Y_target_train[A_target_train == 0])
        # mu_hat_0 = lambda u: mu_model_0.predict(u)

        p_model = clone(p_score_model)
        mu_model_1 = clone(mean_model)
        p_model.fit(X_target_train, A_target_train)
        mu_model_1.fit(X_target_train[A_target_train == 1], Y_target_train[A_target_train == 1])
        p_hat = lambda u: p_model.predict(u).clip(0.01, 0.99)
        mu_hat_1 = lambda u: mu_model_1.predict(u)
        sigma_hat = conditional_variance(X_target_train[A_target_train == 0], Y_target_train[A_target_train == 0], mu_hat_0, var_model)

        # Fit the nuisance functions on each source data
        p_hats = []
        omega_source_hats = []
        density_ratio_hats = []
        for m in range(M):
            X_source_train = train_source_data[m][["U1", "U2"]]
            A_source_train = train_source_data[m]["A"]
            Y_source_train = train_source_data[m]["Y"]
            p_model_m = clone(p_score_model)
            p_model_m.fit(X_source_train, A_source_train)
            p_hats.append(lambda u, model=p_model_m: model.predict(u).clip(0.01, 0.99))
            sigma_source_hat = conditional_variance(X_source_train[A_source_train == 0], Y_source_train[A_source_train == 0], mu_hat_0, var_model)
            omega_source_hats.append(lambda u: 1/sigma_source_hat(u))
            density_ratio_hats.append(density_ratio(train_target_data[["U1", "U2"]].values, X_source_train.values))
        omega_hat = lambda u: sum((len(source_data[m])/len(target_data))*(1/density_ratio_hats[m](u))*omega_source_hats[m](u) for m in range(M))


        # Compute the doubly robust scores and augmentation terms on the target data
        X_target_test = test_target_data[["U1", "U2"]]
        A_test = test_target_data["A"]
        Y_test = test_target_data["Y"]
        dr_scores_target = ATE_DR(X_target_test.values, A_test.values, Y_test.values, p_hat, mu_hat_1, mu_hat_0)
        eif_aug_target = ATE_EIF_fusion_target(X_target_test.values, A_test.values, Y_test.values, p_hat, mu_hat_1, mu_hat_0, sigma_hat, omega_hat)
        efficiency_gain = np.mean(sigma_hat(X_target_test.values)**2/(sigma_hat(X_target_test.values)+1/omega_hat(X_target_test.values)))

        # Compute the augmentation terms on each source data
        eif_aug_sources = []
        for m in range(M):
            X_source_test = test_source_data[m][["U1", "U2"]]
            A_source_test = test_source_data[m]["A"]
            Y_source_test = test_source_data[m]["Y"]
            eif_aug_source_m = ATE_EIF_fusion_source(X_source_test.values, A_source_test.values, Y_source_test.values, p_hats[m], mu_hat_0, sigma_hat, omega_hat, omega_source_hats[m], density_ratio_hats[m])
            eif_aug_sources.append(eif_aug_source_m)
        
        # Combine all scores
        ate_estimates.append(np.mean(dr_scores_target)-np.mean(eif_aug_target)+sum(np.mean(eif_aug_sources[m]) for m in range(M)))
        var_estimates.append(np.var(dr_scores_target)-efficiency_gain)
    return np.mean(ate_estimates), np.sqrt(np.mean(var_estimates) / len(target_data))
    