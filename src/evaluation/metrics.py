"""Step 5 of Pipeline: Evaluation Metrics, Statistical Tests & Sanity Checks.

Implements Section 5.2b, 5.2e & 5.3 of Review-2026-11-07:
- ATE Metrics: |Bias|, Monte Carlo SE of |Bias| (Morris, White & Crowther, 2019), RMSE, 95% CI Coverage, CI Width
- CATE/Uplift Metrics on Held-out RCT Test Set: Normalized AUUC & Qini Coefficient (cross-checked with scikit-uplift)
- Statistical Hypothesis Testing: One-sided / two-sided Paired Wilcoxon signed-rank test + Holm step-down adjustment
"""

from typing import Dict, List, Literal
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.estimation.estimators import ATEResult, estimate_ate_aipw


def summarize_ate_iterations(
    results: List[ATEResult],
    ground_truth_ate: float,
) -> Dict[str, float]:
    """Aggregate ATE estimation metrics across R OSRCT iterations, including Monte Carlo SE (Morris et al., 2019)."""
    ates = np.array([r.ate for r in results], dtype=float)
    lowers = np.array([r.ci_lower for r in results], dtype=float)
    uppers = np.array([r.ci_upper for r in results], dtype=float)
    r_count = len(ates)

    errors = ates - ground_truth_ate
    abs_biases = np.abs(errors)
    covered = (lowers <= ground_truth_ate) & (ground_truth_ate <= uppers)
    widths = uppers - lowers

    std_abs_bias = float(np.std(abs_biases, ddof=1)) if r_count > 1 else 0.0
    mcse_abs_bias = float(std_abs_bias / np.sqrt(r_count)) if r_count > 1 else 0.0

    return {
        "num_iterations": int(r_count),
        "mean_ate": float(np.mean(ates)),
        "std_ate": float(np.std(ates, ddof=1)) if r_count > 1 else 0.0,
        "mean_abs_bias": float(np.mean(abs_biases)),
        "std_abs_bias": std_abs_bias,
        "mcse_abs_bias": mcse_abs_bias,
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ci_coverage_95": float(np.mean(covered)),
        "mean_ci_width": float(np.mean(widths)),
    }


def compute_auuc(
    y_true: np.ndarray,
    t_true: np.ndarray,
    uplift_pred: np.ndarray,
    normalize_by_random: bool = True,
) -> float:
    """Compute Area Under the Uplift Curve (AUUC) on the held-out RCT test set.
    Uses `sklift.metrics.uplift_auc_score` when available, or exact trapezoidal integration
    subtracting the random targeting baseline area when `normalize_by_random=True`.
    """
    try:
        from sklift.metrics import uplift_auc_score

        return float(uplift_auc_score(y_true=y_true, uplift=uplift_pred, treatment=t_true))
    except Exception:
        order = np.argsort(-uplift_pred)
        y_sorted = y_true[order].astype(float)
        t_sorted = t_true[order].astype(float)

        cum_t = np.cumsum(t_sorted)
        cum_c = np.cumsum(1.0 - t_sorted)
        cum_y_t = np.cumsum(y_sorted * t_sorted)
        cum_y_c = np.cumsum(y_sorted * (1.0 - t_sorted))

        rate_t = np.divide(cum_y_t, cum_t, out=np.zeros_like(cum_y_t), where=cum_t > 0)
        rate_c = np.divide(cum_y_c, cum_c, out=np.zeros_like(cum_y_c), where=cum_c > 0)

        n = len(y_true)
        fractions = np.arange(1, n + 1, dtype=float) / n
        uplift_curve = (rate_t - rate_c) * fractions
        raw_area = float(np.trapezoid(uplift_curve, fractions))
        if normalize_by_random:
            random_area = 0.5 * float(uplift_curve[-1])
            return raw_area - random_area
        return raw_area


def compute_qini(
    y_true: np.ndarray,
    t_true: np.ndarray,
    uplift_pred: np.ndarray,
) -> float:
    """Compute Normalized Qini Area Under Curve (`sklift.metrics.qini_auc_score`)."""
    from sklift.metrics import qini_auc_score

    return float(qini_auc_score(y_true=y_true, uplift=uplift_pred, treatment=t_true))


def run_placebo_sanity_check(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    adjustment_cols: List[str],
    seed: int = 42,
) -> ATEResult:
    """Permute the treatment assignment column randomly (Placebo treatment)."""
    rng = np.random.default_rng(seed)
    placebo_df = df.copy()
    placebo_df[treatment_col] = rng.permutation(placebo_df[treatment_col].to_numpy())
    return estimate_ate_aipw(
        placebo_df,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        adjustment_cols=adjustment_cols,
        seed=seed,
    )


def paired_wilcoxon_test(
    errors_a: List[float],
    errors_b: List[float],
    alternative: Literal["two-sided", "less", "greater"] = "less",
) -> Dict[str, float]:
    """Run paired Wilcoxon signed-rank test (default one-sided `alternative='less'` testing if A < B)."""
    stat, p_val = wilcoxon(errors_a, errors_b, zero_method="wilcox", alternative=alternative)
    return {"statistic": float(stat), "p_value": float(p_val)}


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down multiple testing correction to a list of p-values."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj_p = min(1.0, p_values[idx] * (m - rank))
        running_max = max(running_max, adj_p)
        adjusted[idx] = running_max
    return adjusted.tolist()
