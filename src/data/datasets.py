"""Data loading and 3-tier variable partitioning (Addresses Issue N5 & Q5 in Review-2026-10-30)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd


@dataclass
class ThreeTierDataset:
    """Container for a causal dataset with explicit 3-tier variable partitioning.

    Why 3 tiers? (Review-2026-10-30, Issue N5):
    If the candidate feature pool only contains pre-treatment variables, then
    adjusting for all variables (S2) is already a valid backdoor adjustment set.
    To test Hypothesis H1 (bias reduction via causal graph filtering out mediators/colliders)
    and Hypothesis H2 (variance/RMSE reduction via optimal adjustment O-set filtering out
    instruments/pure treatment predictors), we explicitly track:
      - tier_pre_treatment: valid pre-treatment covariates (demographics + historical RFM)
      - tier_treatment_only: instruments or pure treatment predictors (excluded in O-set)
      - tier_post_treatment: mediators / descendants of treatment (e.g., `visit` in Hillstrom,
                             `exposure` and `visit` in Criteo) that induce post-treatment bias if adjusted.
    """

    name: str
    df: pd.DataFrame
    treatment_col: str
    outcome_col: str
    tier_pre_treatment: List[str]
    tier_post_treatment: List[str] = field(default_factory=list)
    tier_treatment_only: List[str] = field(default_factory=list)
    biasing_covariates_B: List[str] = field(default_factory=list)

    @property
    def all_available_features(self) -> List[str]:
        """Return all available candidate features (Pre-treatment + Treatment-only + Post-treatment).
        Used by naive strategy S1 ('kitchen-sink' adjustment).
        """
        return (
            self.tier_pre_treatment
            + self.tier_treatment_only
            + self.tier_post_treatment
        )

    @property
    def all_pretreatment_features(self) -> List[str]:
        """Return all variables measured prior to treatment (time zero).
        Used by traditional valid strategy S2 (VanderWeele, 2019).
        """
        return self.tier_pre_treatment + self.tier_treatment_only

    def validate_no_leakage(self, selected_features: List[str]) -> Dict[str, float | int]:
        """Check whether any post-treatment variables were mistakenly included (inclusion error)."""
        post_set = set(self.tier_post_treatment)
        included_post = [f for f in selected_features if f in post_set]
        return {
            "num_post_treatment_included": len(included_post),
            "inclusion_error_rate": (
                len(included_post) / len(post_set) if post_set else 0.0
            ),
            "included_post_vars": included_post,
        }


def prepare_hillstrom_three_tier(
    raw_df: pd.DataFrame,
    treatment_arm: str = "Womens E-Mail",
    control_arm: str = "No E-Mail",
    outcome_col: str = "conversion",
) -> ThreeTierDataset:
    """Prepare the Hillstrom (MineThatData) RCT dataset for a single treatment-control pair.

    Follows Section 5.2a & 5.2c of Review-2026-10-30:
    - Analyzes each arm pair separately (e.g. Womens E-Mail vs No E-Mail) rather than pooling.
    - Encodes categorical variables (`zip_code`, `channel`, `history_segment`) into numeric dummies.
    - Retains `visit` as the explicit post-treatment mediator between email and conversion/spend.
    """
    sub = raw_df[raw_df["segment"].isin([treatment_arm, control_arm])].copy()
    sub["treatment"] = (sub["segment"] == treatment_arm).astype(int)

    # One-hot encode categorical pre-treatment variables (drop_first=True to avoid collinearity)
    cat_cols = ["zip_code", "channel"]
    existing_cats = [c for c in cat_cols if c in sub.columns]
    sub_encoded = pd.get_dummies(sub, columns=existing_cats, drop_first=True, dtype=float)

    base_pre = ["recency", "history", "mens", "womens", "newbie"]
    dummy_pre = [
        c
        for c in sub_encoded.columns
        if any(c.startswith(f"{cat}_") for cat in existing_cats)
    ]
    pre_treatment_cols = [c for c in base_pre if c in sub_encoded.columns] + dummy_pre
    post_treatment_cols = ["visit"] if "visit" in sub_encoded.columns else []
    biasing_B = [c for c in ["recency", "history", "mens", "womens"] if c in sub_encoded.columns]

    return ThreeTierDataset(
        name=f"Hillstrom ({treatment_arm} vs {control_arm})",
        df=sub_encoded.reset_index(drop=True),
        treatment_col="treatment",
        outcome_col=outcome_col,
        tier_pre_treatment=pre_treatment_cols,
        tier_post_treatment=post_treatment_cols,
        tier_treatment_only=[],
        biasing_covariates_B=biasing_B,
    )


def prepare_x5_three_tier(
    df: pd.DataFrame,
    treatment_col: str = "treatment_flg",
    outcome_col: str = "target",
    post_treatment_cols: Optional[List[str]] = None,
) -> ThreeTierDataset:
    """Prepare X5 RetailHero dataset into a ThreeTierDataset container.

    Follows Section 5.2 & Data Card 2 of Review-2026-11-07:
    - Pre-treatment covariates (Tier 1 & Tier 2): Demographics (age, gender) and RFM features
      aggregated strictly prior to communication time zero (recency_days, tx_frequency,
      monetary_total, avg_basket_value, points_received_mean, points_spent_ratio, unique_stores_visited).
    - Biasing covariates B: core RFM variables used to induce selection bias in OSRCT.
    - Post-treatment covariates (Tier 3): intermediate indicators if available or semi-synthetic mediator.
    """
    df_clean = df.copy()
    if df_clean[treatment_col].dtype != int:
        df_clean[treatment_col] = df_clean[treatment_col].astype(int)
    if df_clean[outcome_col].dtype != int:
        df_clean[outcome_col] = df_clean[outcome_col].astype(int)

    # Encode categorical gender if present
    if "gender" in df_clean.columns and df_clean["gender"].dtype == object:
        df_clean = pd.get_dummies(df_clean, columns=["gender"], drop_first=True, dtype=float)

    exclude_non_features = {
        "client_id",
        "customer_id",
        treatment_col,
        outcome_col,
        "first_issue_date",
        "first_redeem_date",
        "treatment_datetime",
    }
    post_cols = list(post_treatment_cols) if post_treatment_cols is not None else []
    if not post_cols and "semi_synthetic_mediator" in df_clean.columns:
        post_cols = ["semi_synthetic_mediator"]

    all_candidate_cols = [
        c
        for c in df_clean.columns
        if c not in exclude_non_features and c not in post_cols
    ]

    # Pre-treatment variables are all non-excluded columns not in post_cols
    pre_treatment_cols = all_candidate_cols

    # Core biasing candidates in retail RFM
    preferred_biasing = [
        "monetary_total",
        "tx_frequency",
        "recency_days",
        "age",
        "avg_basket_value",
        "purchase_sum",
        "regular_points_received",
    ]
    biasing_B = [c for c in preferred_biasing if c in pre_treatment_cols]
    if not biasing_B:
        biasing_B = pre_treatment_cols[: min(4, len(pre_treatment_cols))]

    return ThreeTierDataset(
        name="X5 RetailHero",
        df=df_clean.reset_index(drop=True),
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        tier_pre_treatment=pre_treatment_cols,
        tier_post_treatment=post_cols,
        tier_treatment_only=[],
        biasing_covariates_B=biasing_B,
    )


