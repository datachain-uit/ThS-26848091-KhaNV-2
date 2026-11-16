"""Step 1 & 2 of Pipeline: Expert DAG and Tiered Causal Discovery.

Implements temporal tier constraints (Background Knowledge) following:
- Section 5.1 & 5.3C of Review-2026-10-30
- Andrews, Spirtes & Cooper (AISTATS 2020)
- Bang & Didelez (UAI 2023)
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import networkx as nx
import pandas as pd


@dataclass
class TemporalTierSpec:
    """Defines temporal tiers of the retail marketing funnel:
    Tier 0: Demographics / Customer profile (zip_code, channel, newbie, age, gender)
    Tier 1: Historical pre-treatment RFM behavior (recency, history, mens, womens)
    Tier 2: Marketing Treatment (email / SMS campaign)
    Tier 3: Post-treatment Mediator / Funnel step (visit, exposure)
    Tier 4: Final Outcome (conversion, spend)

    Rule: Variables in a later tier CANNOT cause variables in an earlier tier.
    """

    tier_map: Dict[str, int]
    required_edges: List[Tuple[str, str]]
    forbidden_edges: List[Tuple[str, str]]

    def get_forbidden_by_tiers(self) -> Set[Tuple[str, str]]:
        """Generate all pairs (u, v) where tier(u) > tier(v), which are temporally impossible."""
        forbidden = set(self.forbidden_edges)
        nodes = list(self.tier_map.keys())
        for u in nodes:
            for v in nodes:
                if u != v and self.tier_map[u] > self.tier_map[v]:
                    forbidden.add((u, v))
        return forbidden


def build_hillstrom_expert_dag(
    pre_treatment_cols: List[str],
    treatment_col: str = "treatment",
    mediator_col: str = "visit",
    outcome_col: str = "conversion",
) -> nx.DiGraph:
    """Step 1: Construct the Expert Causal DAG for Hillstrom retail funnel.

    Causal structure:
    - Pre-treatment covariates X -> Treatment T (in observational/OSRCT setting)
    - Pre-treatment covariates X -> Mediator M (`visit`) and Outcome Y (`conversion`)
    - Treatment T -> Mediator M (`visit`) -> Outcome Y (`conversion`)
    - Direct path T -> Y (`conversion`)
    """
    dag = nx.DiGraph()
    for col in pre_treatment_cols:
        dag.add_edge(col, treatment_col)
        if mediator_col:
            dag.add_edge(col, mediator_col)
        dag.add_edge(col, outcome_col)

    if mediator_col:
        dag.add_edge(treatment_col, mediator_col)
        dag.add_edge(mediator_col, outcome_col)
    dag.add_edge(treatment_col, outcome_col)
    return dag

