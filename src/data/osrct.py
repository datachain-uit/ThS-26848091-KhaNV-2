"""OSRCT: Observational Sampling from Randomized Controlled Trials.

Reference:
Gentzel, A. M., Pruthi, P., & Jensen, D. (2021).
"How and why to use experimental data to evaluate methods for observational causal inference."
ICML 2021 (PMLR 139).

Addresses Issue Q4, Q8 & Section 5.3 of Review-2026-11-07:
- Splits `pool_df` and `holdout_df` FIRST, and computes Ground Truth ATE (`rct_ground_truth_ate`)
  directly on `pool_df` (the exact population from which OSRCT samples are drawn), while recording
  `full_rct_ate` as a reference.
- Applies `np.log1p` transformation to skewed monetary/count covariates (e.g. `history`) prior to
  z-score standardization so extreme values do not distort selection propensities.
- Supports downsampling imbalanced RCTs (e.g. Lenta 75/25, Criteo 85/15) to 50/50 prior to OSRCT.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@dataclass
class OSRCTSplitResult:
    """Holds the RCT pool, holdout test set, and pool-anchored ground-truth ATE."""

    rct_ground_truth_ate: float      # Computed on rct_pool_df (used for |Bias| and RMSE)
    rct_ground_truth_se: float       # Standard error on rct_pool_df
    full_rct_ate: float              # Computed on full df (reported for reference)
    full_rct_se: float
    rct_holdout_df: pd.DataFrame
    rct_pool_df: pd.DataFrame


def balance_rct_arms_50_50(
    df: pd.DataFrame,
    treatment_col: str,
    seed: int = 42,
) -> pd.DataFrame:
    """For imbalanced RCTs (e.g. Lenta 0.75, Criteo 0.85), downsample the majority arm to 50/50
    before running OSRCT so the covariate distribution of the retained sample is not shifted.
    """
    rng = np.random.default_rng(seed)
    idx_1 = df.index[df[treatment_col] == 1].to_numpy()
    idx_0 = df.index[df[treatment_col] == 0].to_numpy()
    n_min = min(len(idx_1), len(idx_0))
    keep_1 = rng.choice(idx_1, size=n_min, replace=False)
    keep_0 = rng.choice(idx_0, size=n_min, replace=False)
    balanced_idx = np.sort(np.concatenate([keep_1, keep_0]))
    return df.loc[balanced_idx].reset_index(drop=True)


def split_rct_and_compute_ground_truth(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    holdout_ratio: float = 0.30,
    seed: int = 42,
) -> OSRCTSplitResult:
    """Step 1 of OSRCT (Review-2026-11-07 Section 5.3):
    - Split `pool_df` (70%) and `holdout_df` (30%) FIRST.
    - Compute Ground Truth ATE (`rct_ground_truth_ate`) on `pool_df`.
    - Also compute `full_rct_ate` on the full dataset for reference.
    """
    full_y1 = df.loc[df[treatment_col] == 1, outcome_col].to_numpy(dtype=float)
    full_y0 = df.loc[df[treatment_col] == 0, outcome_col].to_numpy(dtype=float)
    full_ate = float(full_y1.mean() - full_y0.mean())
    full_se = float(np.sqrt(full_y1.var(ddof=1) / len(full_y1) + full_y0.var(ddof=1) / len(full_y0)))

    pool_df, holdout_df = train_test_split(
        df,
        test_size=holdout_ratio,
        stratify=df[treatment_col],
        random_state=seed,
    )

    y1 = pool_df.loc[pool_df[treatment_col] == 1, outcome_col].to_numpy(dtype=float)
    y0 = pool_df.loc[pool_df[treatment_col] == 0, outcome_col].to_numpy(dtype=float)
    ate_gt = float(y1.mean() - y0.mean())
    se_gt = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))

    return OSRCTSplitResult(
        rct_ground_truth_ate=ate_gt,
        rct_ground_truth_se=se_gt,
        full_rct_ate=full_ate,
        full_rct_se=full_se,
        rct_holdout_df=holdout_df.reset_index(drop=True),
        rct_pool_df=pool_df.reset_index(drop=True),
    )


def sample_osrct_biased_dataset(
    rct_pool_df: pd.DataFrame,
    treatment_col: str,
    biasing_covariates_B: List[str],
    bias_strength: float = 1.0,
    weights: Optional[Dict[str, float]] = None,
    log_transform_cols: Optional[List[str]] = None,
    seed: int = 2026,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Generate an observational dataset from an RCT pool via biased rejection sampling (Gentzel et al., 2021)."""
    rng = np.random.default_rng(seed)
    X_B = rct_pool_df[biasing_covariates_B].to_numpy(dtype=float).copy()

    if log_transform_cols is None:
        log_transform_cols = ["history", "monetary_total", "avg_basket_value"]

    for idx, col in enumerate(biasing_covariates_B):
        if col in log_transform_cols:
            X_B[:, idx] = np.log1p(np.maximum(X_B[:, idx], 0.0))

    means = np.mean(X_B, axis=0)
    stds = np.std(X_B, axis=0)
    stds = np.where(stds == 0, 1.0, stds)
    Z_B = (X_B - means) / stds

    if weights is None:
        w = np.ones(len(biasing_covariates_B), dtype=float) / np.sqrt(len(biasing_covariates_B))
    else:
        w = np.array([weights.get(col, 1.0) for col in biasing_covariates_B], dtype=float)

    linear_score = bias_strength * (Z_B @ w)
    prob_select_if_treated = expit(linear_score)

    t_vec = rct_pool_df[treatment_col].to_numpy(dtype=int)
    selection_prob = np.where(t_vec == 1, prob_select_if_treated, 1.0 - prob_select_if_treated)

    u = rng.uniform(0.0, 1.0, size=len(rct_pool_df))
    keep_mask = u < selection_prob

    obs_df = rct_pool_df.loc[keep_mask].reset_index(drop=True)

    diagnostics = {
        "bias_strength": float(bias_strength),
        "n_pool": int(len(rct_pool_df)),
        "n_observational": int(len(obs_df)),
        "retention_rate": float(np.mean(keep_mask)),
        "min_propensity": float(np.min(prob_select_if_treated)),
        "max_propensity": float(np.max(prob_select_if_treated)),
        "extreme_propensity_share": float(
            np.mean((prob_select_if_treated < 0.02) | (prob_select_if_treated > 0.98))
        ),
        "treated_share_obs": float(obs_df[treatment_col].mean()),
    }
    return obs_df, diagnostics
