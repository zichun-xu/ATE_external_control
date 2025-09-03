# Shift-Robust Causal Effect Estimation with External Controls (Python)

> **One-liner:** Fuse external source data to **reduce variance** in experiments **under distribution shift**, with **valid uncertainty** and **black-box ML**.

---

## Problem

Randomized experiments are the gold standard for causal effect estimation. However, in reality, sample sizes are often constrained by budget or time. Meanwhile, **external source data** such as, past experiments, adjacent markets, and holdouts, are available in large scale, but naively pooling them with experimental data results in **biases** estimates and **invalidates** CIs when there is **distribution shift**. In this project, we provide a shift-robust average treatment effect (ATE) estimator that borrows external data while preserving **valid uncertainty**. It uses double machine learning with cross-fitting that works with highly flexiable ML for nuisance models (outcome and propensity score).

---

## Approach

We assume only **one-sided transportability**, e.g., only the conditional mean outcome of the controls $$\(\mathbb{E}[Y(0)\mid X, A = 0] \)$$ is shared between the target data and external source data.   

Under such assumptions, we construct **double-robust, cross-fitted estimators** for the average treatment effect (ATE). Our estimator is **semiparametrically efficient**. 

## What this enables

- **Faster decision** and **reduced sample size to significance** (standard error reduction **13.7%** on synthetic data).
- Safely **fuse external source data** (maintain nominal CI coverage levels).  
See [`/notebooks`](./notebooks/ate_fusion.ipynb) for reproducible results.
