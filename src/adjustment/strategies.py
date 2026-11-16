r"""Step 3 of Pipeline: Covariate / Adjustment Set Selection Strategies (S0 - S6).

Implements Section 5.2 of Review-2026-11-07:
- `optimal_adjustment_set`: Exact O-set of Henckel, Perković & Maathuis (JRSS-B 2022) on a DAG:
    O = pa(cn(T, Y)) \ forb(T, Y)
- `outcome_adaptive_lasso_selection`: Outcome-Adaptive Lasso following Shortreed & Ertefaie (Biometrics 2017):
    Step 1: Fit outcome regression Y ~ T + X to obtain coefficient estimates beta_hat_j.
    Step 2: Compute adaptive penalty weights w_j = |beta_hat_j|^(-gamma) for each covariate X_j.
    Step 3: Fit penalized logistic propensity models over a grid of lambda values and select lambda
            that minimizes the weighted Absolute Mean Difference (wAMD).
- `an_y_minus_de_t_set`: Valid backdoor adjustment set using ancestors of Y minus descendants of T
  (kept with an explicit honest name for reference comparison).
"""

from typing import Dict, List, Optional, Set
import networkx as nx
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge

from src.data.datasets import ThreeTierDataset


def optimal_adjustment_set(dag: nx.DiGraph, x: str, y: str) -> List[str]:
    """O-set (Henckel, Perković & Maathuis, 2022) on a DAG: O = pa(cn(x, y)) \\ forb(x, y)."""
    if y not in nx.descendants(dag, x):
        raise ValueError("Y không phải hậu duệ của X")
    an_y = nx.ancestors(dag, y) | {y}
    cn = {v for v in nx.descendants(dag, x) if v in an_y}      # causal nodes between X and Y
    forb = {x}
    for v in cn:
        forb |= nx.descendants(dag, v) | {v}                   # forbidden set
    pa_cn = set().union(*(set(dag.predecessors(v)) for v in cn))
    return sorted(pa_cn - forb)


def an_y_minus_de_t_set(
    dag: nx.DiGraph,
    treatment: str,
    outcome: str,
    candidate_nodes: Optional[Set[str]] = None,
) -> List[str]:
    """Valid backdoor set taking ancestors of Y (excluding paths through T) minus descendants of T."""
    descendants_of_t = nx.descendants(dag, treatment) | {treatment, outcome}
    ancestors_of_y = nx.ancestors(dag, outcome)
    valid_candidates = ancestors_of_y - descendants_of_t
    if candidate_nodes is not None:
        valid_candidates = valid_candidates & set(candidate_nodes)

    dag_without_t_out = dag.copy()
    dag_without_t_out.remove_edges_from(list(dag.out_edges(treatment)))
    ancestors_of_y_without_t = nx.ancestors(dag_without_t_out, outcome)
    return sorted(valid_candidates & ancestors_of_y_without_t)


def compute_wamd(
    X: np.ndarray,
    t: np.ndarray,
    beta_abs: np.ndarray,
    e_hat: np.ndarray,
) -> float:
    """Compute the weighted Absolute Mean Difference (wAMD) Criterion (Shortreed & Ertefaie, 2017):
    wAMD(lambda) = sum_j |beta_hat_j| * | E_w[X_j | T=1] - E_w[X_j | T=0] |
    using Inverse Probability of Treatment Weights (IPTW) derived from e_hat.
    """
    e_clipped = np.clip(e_hat, 1e-3, 1.0 - 1e-3)
    w1 = t / e_clipped
    w0 = (1.0 - t) / (1.0 - e_clipped)

    w1_norm = w1 / np.sum(w1)
    w0_norm = w0 / np.sum(w0)

    mean_1 = X.T @ w1_norm
    mean_0 = X.T @ w0_norm
    amd = np.abs(mean_1 - mean_0)
    return float(np.sum(beta_abs * amd))


def outcome_adaptive_lasso_selection(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    candidate_cols: List[str],
    gamma: float = 2.0,
    c_grid: Optional[List[float]] = None,
    seed: int = 42,
) -> List[str]:
    """S5 Baseline: Outcome-Adaptive Lasso (Shortreed & Ertefaie, Biometrics 2017).

    Three-step algorithm:
    1. Fit main-effects outcome regression Y ~ T + X (standardized X) to obtain |beta_hat_j|.
    2. Construct adaptive weights w_j = |beta_hat_j|^(-gamma). Scaling X_j by 1 / w_j = |beta_hat_j|^gamma
       transforms a standard L1-penalized logistic regression on T into an adaptive L1 problem:
         min_{alpha} -loglik(T | X, alpha) + lambda * sum_j w_j |alpha_j|
       Thus pure instruments (where beta_hat_j ~ 0) receive a huge penalty w_j and are excluded,
       while confounders and precision variables (where |beta_hat_j| is large) are retained.
    3. Select regularization strength C = 1/lambda across a grid by minimizing wAMD.
    """
    if not candidate_cols:
        return []

    X_raw = df[candidate_cols].to_numpy(dtype=float)
    t = df[treatment_col].to_numpy(dtype=float)
    y = df[outcome_col].to_numpy(dtype=float)

    means = np.mean(X_raw, axis=0)
    stds = np.std(X_raw, axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    X_std = (X_raw - means) / stds

    # Step 1: Outcome regression Y ~ T + X_std
    design_yx = np.column_stack([t, X_std])
    outcome_reg = Ridge(alpha=1e-2, random_state=seed).fit(design_yx, y)
    beta_hat = outcome_reg.coef_[1:]  # coefficients for X_std
    beta_abs = np.maximum(np.abs(beta_hat), 1e-6)
    # Normalize adaptive weights by max(|beta_hat|) so penalty stays invariant to outcome scale
    beta_rel = beta_abs / np.max(beta_abs)

    # Step 2: Adaptive scaling X_tilde_j = X_std_j * (beta_rel_j ** gamma)
    scale_factors = np.maximum(beta_rel ** gamma, 1e-6)
    X_tilde = X_std * scale_factors

    if c_grid is None:
        c_grid = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

    best_wamd = float("inf")
    best_selected: List[str] = []

    # Step 3: Grid search over C = 1/lambda minimizing wAMD
    for c_val in c_grid:
        clf = LogisticRegression(
            l1_ratio=1.0,
            C=c_val,
            solver="saga",
            max_iter=2000,
            random_state=seed,
        )
        clf.fit(X_tilde, t.astype(int))
        alpha_tilde = clf.coef_.ravel()
        alpha_orig = alpha_tilde * scale_factors
        selected_mask = np.abs(alpha_orig) > 1e-4

        # Refit unpenalized propensity on selected covariates + precision variables to evaluate wAMD
        if np.any(selected_mask):
            prop_eval = LogisticRegression(max_iter=1000, random_state=seed)
            prop_eval.fit(X_std[:, selected_mask], t.astype(int))
            e_hat = prop_eval.predict_proba(X_std[:, selected_mask])[:, 1]
        else:
            e_hat = np.full(len(t), np.mean(t), dtype=float)

        wamd = compute_wamd(X_std, t, beta_abs, e_hat)
        if wamd < best_wamd:
            best_wamd = wamd
            best_selected = [col for col, keep in zip(candidate_cols, selected_mask) if keep]

    return best_selected


def get_adjustment_set(
    strategy_code: str,
    dataset: ThreeTierDataset,
    expert_dag: Optional[nx.DiGraph] = None,
    learned_dag: Optional[nx.DiGraph] = None,
    obs_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Return the selected adjustment set for strategies S0 through S6."""
    code = strategy_code.upper()
    if code == "S0":
        return []
    if code == "S1":
        return list(dataset.all_available_features)
    if code == "S2":
        return list(dataset.all_pretreatment_features)
    if code == "S3":
        if expert_dag is None:
            raise ValueError("Strategy S3 requires `expert_dag`.")
        o_set = optimal_adjustment_set(
            expert_dag,
            x=dataset.treatment_col,
            y=dataset.outcome_col,
        )
        avail = set(dataset.all_available_features)
        return [v for v in o_set if v in avail]
    if code == "S4":
        if learned_dag is None:
            raise ValueError("Strategy S4 requires `learned_dag`.")
        o_set = optimal_adjustment_set(
            learned_dag,
            x=dataset.treatment_col,
            y=dataset.outcome_col,
        )
        avail = set(dataset.all_available_features)
        return [v for v in o_set if v in avail]
    if code == "S5":
        target_df = obs_df if obs_df is not None else dataset.df
        return outcome_adaptive_lasso_selection(
            target_df,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            candidate_cols=dataset.all_pretreatment_features,
        )
    if code == "S6":
        return list(dataset.biasing_covariates_B)

    raise ValueError(f"Unknown adjustment strategy code: {strategy_code}")
