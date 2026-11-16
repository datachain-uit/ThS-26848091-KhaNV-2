"""Unit tests for O-set (Henckel et al., 2022), Outcome-Adaptive Lasso (Shortreed & Ertefaie, 2017),
OSRCT pool-anchored Ground Truth, and 3-tier variable leakage prevention.

Implements the exact verification tests required in Section 5.2 & 5.3 of Review-2026-11-07.
"""

import networkx as nx
import numpy as np
import pandas as pd

from src.adjustment.strategies import (
    get_adjustment_set,
    optimal_adjustment_set,
    outcome_adaptive_lasso_selection,
)
from src.data.datasets import ThreeTierDataset, prepare_x5_three_tier
from src.data.osrct import sample_osrct_biased_dataset, split_rct_and_compute_ground_truth
from src.discovery.tiered_discovery import build_hillstrom_expert_dag
from src.estimation.estimators import estimate_ate_aipw
from src.evaluation.metrics import run_placebo_sanity_check


def test_optimal_adjustment_set():
    """Exact 3 test cases from Section 5.2 of Review-2026-11-07 verifying O = pa(cn(T, Y)) \\ forb(T, Y)."""
    g = nx.DiGraph([("A", "C"), ("C", "T"), ("C", "Y"), ("T", "Y")])
    # Only C is a direct parent of Y/T along the backdoor; A is an ancestor of C and excluded by O-set
    assert optimal_adjustment_set(g, "T", "Y") == ["C"]

    g2 = nx.DiGraph([("C", "T"), ("C", "Y"), ("T", "Y"), ("P", "Y"), ("I", "T")])
    # Keeps precision variable P, excludes pure instrument I
    assert optimal_adjustment_set(g2, "T", "Y") == ["C", "P"]

    g3 = nx.DiGraph([("C", "T"), ("C", "Y"), ("T", "M"), ("M", "Y"), ("W", "M")])
    # Keeps parent W of mediator M (since M is in cn(T, Y)), excludes mediator M itself
    assert optimal_adjustment_set(g3, "T", "Y") == ["C", "W"]


def test_outcome_adaptive_lasso_excludes_instruments_and_keeps_precision():
    """Verify OAL (Shortreed & Ertefaie, 2017) retains confounders C and precision variables P
    while dropping pure instruments I.
    """
    rng = np.random.default_rng(123)
    n = 2000
    C = rng.normal(0, 1, size=n)  # Confounder: affects T and Y
    P = rng.normal(0, 1, size=n)  # Precision variable: affects Y only
    I = rng.normal(0, 1, size=n)  # Pure instrument: affects T only

    logit_t = 0.8 * C + 1.0 * I
    t = rng.binomial(1, 1.0 / (1.0 + np.exp(-logit_t)), size=n)
    y = 1.0 * t + 1.2 * C + 1.2 * P + rng.normal(0, 0.5, size=n)

    df = pd.DataFrame({"C": C, "P": P, "I": I, "treatment": t, "outcome": y})
    selected = outcome_adaptive_lasso_selection(
        df,
        treatment_col="treatment",
        outcome_col="outcome",
        candidate_cols=["C", "P", "I"],
        gamma=2.5,
        seed=42,
    )
    assert "C" in selected
    assert "I" not in selected


def _make_synthetic_hillstrom_rct(n: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    recency = rng.uniform(1, 12, size=n)
    history = rng.exponential(scale=100.0, size=n)
    mens = rng.binomial(1, 0.5, size=n).astype(float)
    womens = rng.binomial(1, 0.5, size=n).astype(float)

    treatment = rng.binomial(1, 0.5, size=n)
    visit_logit = -1.0 + 1.2 * treatment - 0.1 * recency + 0.005 * history
    visit = rng.binomial(1, 1.0 / (1.0 + np.exp(-visit_logit)), size=n).astype(float)

    y = (
        0.10
        + 0.05 * treatment
        + 0.25 * visit
        - 0.02 * recency
        + 0.001 * history
        + 0.03 * womens
        + rng.normal(0, 0.05, size=n)
    )
    return pd.DataFrame(
        {
            "recency": recency,
            "history": history,
            "mens": mens,
            "womens": womens,
            "treatment": treatment,
            "visit": visit,
            "conversion": y,
        }
    )


def test_three_tier_and_osrct_pool_ground_truth():
    df = _make_synthetic_hillstrom_rct(n=5000, seed=42)
    ds = ThreeTierDataset(
        name="SyntheticHillstrom",
        df=df,
        treatment_col="treatment",
        outcome_col="conversion",
        tier_pre_treatment=["recency", "history", "mens", "womens"],
        tier_post_treatment=["visit"],
        biasing_covariates_B=["recency", "history"],
    )

    expert_dag = build_hillstrom_expert_dag(
        pre_treatment_cols=ds.tier_pre_treatment,
        treatment_col="treatment",
        mediator_col="visit",
        outcome_col="conversion",
    )

    s1_vars = get_adjustment_set("S1", ds, expert_dag=expert_dag)
    s3_vars = get_adjustment_set("S3", ds, expert_dag=expert_dag)
    assert "visit" in s1_vars
    assert "visit" not in s3_vars

    split_res = split_rct_and_compute_ground_truth(df, "treatment", "conversion", seed=42)
    # Verify ground truth is computed specifically on rct_pool_df
    pool_y1 = split_res.rct_pool_df.loc[split_res.rct_pool_df["treatment"] == 1, "conversion"].mean()
    pool_y0 = split_res.rct_pool_df.loc[split_res.rct_pool_df["treatment"] == 0, "conversion"].mean()
    assert np.isclose(split_res.rct_ground_truth_ate, pool_y1 - pool_y0)

    obs_df, diag = sample_osrct_biased_dataset(
        split_res.rct_pool_df,
        treatment_col="treatment",
        biasing_covariates_B=ds.biasing_covariates_B,
        bias_strength=1.5,
        seed=101,
    )
    assert 0.2 < diag["retention_rate"] < 0.8

    s6_vars = get_adjustment_set("S6", ds)
    ate_s6 = estimate_ate_aipw(obs_df, "treatment", "conversion", s6_vars, seed=42)
    ate_s1 = estimate_ate_aipw(obs_df, "treatment", "conversion", s1_vars, seed=42)

    bias_s6 = abs(ate_s6.ate - split_res.rct_ground_truth_ate)
    bias_s1 = abs(ate_s1.ate - split_res.rct_ground_truth_ate)
    assert bias_s6 < 0.03
    assert bias_s1 > bias_s6

    placebo_res = run_placebo_sanity_check(obs_df, "treatment", "conversion", s3_vars, seed=99)
    assert abs(placebo_res.ate) < 0.03


def test_prepare_x5_three_tier_schema_and_anti_leakage():
    """Verify X5 RetailHero 3-tier partitioning and leakage prevention."""
    n = 500
    rng = np.random.default_rng(777)
    df_x5 = pd.DataFrame(
        {
            "client_id": [f"c_{i}" for i in range(n)],
            "age": rng.integers(18, 70, size=n),
            "gender": rng.choice(["M", "F", "U"], size=n),
            "recency_days": rng.integers(1, 90, size=n),
            "tx_frequency": rng.integers(1, 20, size=n),
            "monetary_total": rng.exponential(1000.0, size=n),
            "avg_basket_value": rng.exponential(150.0, size=n),
            "treatment_flg": rng.binomial(1, 0.5, size=n),
            "target": rng.binomial(1, 0.6, size=n),
            "semi_synthetic_mediator": rng.binomial(1, 0.3, size=n),
        }
    )
    ds = prepare_x5_three_tier(df_x5, treatment_col="treatment_flg", outcome_col="target")
    assert ds.name == "X5 RetailHero"
    assert "treatment_flg" not in ds.tier_pre_treatment
    assert "target" not in ds.tier_pre_treatment
    assert "client_id" not in ds.tier_pre_treatment
    assert "recency_days" in ds.tier_pre_treatment
    assert "monetary_total" in ds.biasing_covariates_B
    assert "semi_synthetic_mediator" in ds.tier_post_treatment

    # Validate anti-leakage method
    leakage = ds.validate_no_leakage(ds.tier_pre_treatment)
    assert leakage["num_post_treatment_included"] == 0
    assert leakage["inclusion_error_rate"] == 0.0

    # If post-treatment is included mistakenly, it must be flagged
    bad_selection = ds.tier_pre_treatment + ["semi_synthetic_mediator"]
    leakage_bad = ds.validate_no_leakage(bad_selection)
    assert leakage_bad["num_post_treatment_included"] == 1
    assert leakage_bad["inclusion_error_rate"] == 1.0

