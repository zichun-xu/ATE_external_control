# Shift-Robust Experimentation via Data Fusion (Python)

> **One-liner:** Borrow strength from external data to **tighten confidence intervals** and **reduce variance** in experiments **under distribution shift**, with **valid uncertainty** and **black-box ML**.
---

## TL;DR (for product & research teams)

**Faster, smaller, safer experiments under shift.**  
This repo implements **shift-robust semiparametric estimators** that **borrow information from external cohorts** while preserving valid confidence intervals. Built on **double machine learning, orthogonal scores, and cross-fitting**; treats nuisance models as **black boxes** (scikit-learn, XGBoost/LightGBM, PyTorch/JAX, or R via rpy2).

- **Shorter A/B tests:** tighter CIs ⇒ fewer user-days to significance.  
- **Reuse past experiments / adjacent markets** without biasing lift.  
- **95% coverage** under stated assumptions + diagnostics for when *not* to borrow.

---

## Problem

Randomized experiments are the gold standard for causal effect estimation. However, in reality sample sizes are often constrained by budget or time. Meanwhile, **external source data** such as, past experiments, adjacent markets, and holdouts, are available in large scale. However, naively pooling them with a current RCT results in **biases** estimates and **invalidates** CIs when there is **distribution shift**.

---

## Approach

We assume only **limited transportability**, e.g., **one-sided transportability (controls exchangeable):** \( \mathbb{E}[Y(0)\mid X, A = 0] \) is shared between the target data and external source data.   

Under such assumptions, we construct **double-robust, cross-fitted estimators** for the average treatment effect (ATE)

## What this enables

- **Faster decision** and **greater effective sample size** (standard error reduction **13.7%** on synthetic data).
- Safely **fuse external source data** (maintain nominal CI coverage levels).  
See [`/notebooks`](./notebooks/ate_fusion.ipynb) for reproducible figures.
