"""Step 4 of Pipeline: ATE and CATE/Uplift Estimation.

Addresses Issue M5 & Section 5.1 of Review-2026-10-30:
- ATE via Augmented Inverse Probability Weighting (AIPW / Doubly Robust) and LinearDML.
- CATE/Uplift via CausalForestDML, DRLearner, and XLearner.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold


@dataclass
class ATEResult:
    """Container for point estimate, standard error, and 95% confidence interval of ATE."""

    ate: float
    se: float
    ci_lower: float
    ci_upper: float


def estimate_ate_aipw(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    adjustment_cols: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> ATEResult:
    """Cross-fitted Doubly Robust (AIPW) ATE estimator.
    If `adjustment_cols` is empty (Strategy S0), computes the unadjusted difference in means.
    """
    t = df[treatment_col].to_numpy(dtype=float)
    y = df[outcome_col].to_numpy(dtype=float)
    n = len(df)

    if not adjustment_cols:
        y1 = y[t == 1]
        y0 = y[t == 0]
        ate = float(np.mean(y1) - np.mean(y0))
        se = float(np.sqrt(np.var(y1, ddof=1) / len(y1) + np.var(y0, ddof=1) / len(y0)))
        return ATEResult(
            ate=ate,
            se=se,
            ci_lower=ate - 1.96 * se,
            ci_upper=ate + 1.96 * se,
        )

    X = df[adjustment_cols].to_numpy(dtype=float)
    psi = np.zeros(n, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, val_idx in kf.split(X):
        X_tr, t_tr, y_tr = X[train_idx], t[train_idx], y[train_idx]
        X_va, t_va, y_va = X[val_idx], t[val_idx], y[val_idx]

        # Propensity model e(X) = P(T=1 | X)
        prop_model = LogisticRegression(max_iter=1000, random_state=seed)
        prop_model.fit(X_tr, t_tr)
        e_hat = np.clip(prop_model.predict_proba(X_va)[:, 1], 0.02, 0.98)

        # Outcome models mu_1(X), mu_0(X)
        mu1_model = Ridge(alpha=1.0, random_state=seed)
        mu0_model = Ridge(alpha=1.0, random_state=seed)
        mu1_model.fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
        mu0_model.fit(X_tr[t_tr == 0], y_tr[t_tr == 0])

        mu1_hat = mu1_model.predict(X_va)
        mu0_hat = mu0_model.predict(X_va)

        # Doubly robust pseudo-outcome
        psi[val_idx] = (
            (mu1_hat - mu0_hat)
            + t_va * (y_va - mu1_hat) / e_hat
            - (1.0 - t_va) * (y_va - mu0_hat) / (1.0 - e_hat)
        )

    ate = float(np.mean(psi))
    se = float(np.std(psi, ddof=1) / np.sqrt(n))
    return ATEResult(
        ate=ate,
        se=se,
        ci_lower=ate - 1.96 * se,
        ci_upper=ate + 1.96 * se,
    )

